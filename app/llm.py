"""Thin wrapper around the Anthropic Messages API, bound to one agent session per call."""

import json
import logging
import time
from typing import TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.sessions import AgentSession
from app.trace import Trace, null_trace

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

MAX_TOKENS = 16_000
RETRYABLE = (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError)


class LLMError(RuntimeError):
    """The model call failed after retries, or the model refused."""


class MalformedAgentResponse(RuntimeError):
    """The model's output could not be validated against the expected schema."""


def block_to_dict(block) -> dict:
    if isinstance(block, dict):
        return block
    return block.model_dump(mode="json", exclude_none=True)


def response_text(response) -> str:
    return "".join(b.text for b in response.content if b.type == "text")


def extract_json(text: str) -> str:
    """Return the outermost JSON object in a reply, tolerating prose or code fences."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in the response")
    return text[start : end + 1]


class LLMClient:
    def __init__(self, settings: Settings, client=None, sleep=time.sleep, trace: Trace | None = None):
        self.model = settings.anthropic_model
        self.max_retries = settings.llm_max_retries
        # The SDK's own retries are disabled so that the retry policy lives in one place.
        self._client = client or anthropic.Anthropic(api_key=settings.anthropic_api_key, max_retries=0)
        self._sleep = sleep
        self.trace = trace or null_trace()

    def send(
        self,
        session: AgentSession,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        output_schema: dict | None = None,
    ):
        """Send the session's history and append the assistant reply to that same session.

        `output_schema` turns on structured outputs: the reply text must be JSON matching it.
        """
        kwargs = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": session.system_prompt,
            "messages": session.history(),
            # Each session resends its growing history on every call; caching the prefix makes
            # the repeated part much cheaper and faster.
            "cache_control": {"type": "ephemeral"},
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        if output_schema:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": output_schema}}

        agent, session_id = session.agent_name, session.session_id
        mode = "structured_output" if output_schema else ("tools" if tools else "text")
        started = time.perf_counter()
        response, attempt = None, 0
        while response is None:
            attempt += 1
            try:
                response = self._client.messages.create(**kwargs)
            except RETRYABLE as exc:
                if attempt >= self.max_retries:
                    error = LLMError(f"{agent}: model call failed after {attempt} attempts: {exc}")
                    self.trace.emit("llm_call", agent=agent, session_id=session_id, status="error", error=error,
                                    model=self.model, attempts=attempt)
                    raise error from exc
                delay = 2 ** (attempt - 1)
                log.warning("%s: transient model error (%s), retrying in %ss", agent, type(exc).__name__, delay)
                self.trace.emit("llm_retry", agent=agent, session_id=session_id, status="warning", error=exc,
                                attempt=attempt, delay_s=delay)
                self._sleep(delay)
            except anthropic.APIStatusError as exc:
                error = LLMError(f"{agent}: model call rejected ({exc.status_code}): {exc}")
                self.trace.emit("llm_call", agent=agent, session_id=session_id, status="error", error=error,
                                model=self.model, attempts=attempt)
                raise error from exc

        usage = getattr(response, "usage", None)
        self.trace.emit(
            "llm_call",
            agent=agent,
            session_id=session_id,
            status="error" if response.stop_reason == "refusal" else "ok",
            duration_ms=(time.perf_counter() - started) * 1000,
            model=self.model,
            mode=mode,
            retries=attempt - 1,
            stop_reason=response.stop_reason,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", None),
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", None),
            tool_calls=[b.name for b in response.content if b.type == "tool_use"],
        )
        if response.stop_reason == "refusal":
            raise LLMError(f"{session.agent_name}: the model declined the request")
        # Keep every block (including thinking blocks) so the history can be replayed unchanged.
        session.add_assistant([block_to_dict(b) for b in response.content])
        return response

    def call_structured(self, session: AgentSession, content: str, model_cls: type[T], check=None) -> T:
        """Ask for JSON matching `model_cls`; allow one repair attempt before failing.

        `check` is an optional callable returning a list of extra problems (e.g. bad citations).
        """
        schema = json.dumps(model_cls.model_json_schema())
        session.add_user(
            f"{content}\n\nReply with a single JSON object (no prose, no code fences) "
            f"that validates against this JSON schema:\n{schema}"
        )
        error = ""
        for attempt in range(2):
            response = self.send(session)
            try:
                parsed = model_cls.model_validate_json(extract_json(response_text(response)))
                problems = check(parsed) if check else []
                if not problems:
                    return parsed
                error = "; ".join(problems)
            except (ValidationError, ValueError) as exc:
                error = str(exc)
            self.trace.emit(
                "malformed_response", agent=session.agent_name, session_id=session.session_id,
                status="warning" if attempt == 0 else "error", error=error[:500],
                expected=model_cls.__name__, repair_requested=attempt == 0,
            )
            if attempt == 0:
                session.add_user(
                    f"Your previous reply was invalid: {error}\n"
                    "Reply again with only the corrected JSON object."
                )
        raise MalformedAgentResponse(f"{session.agent_name} returned invalid {model_cls.__name__}: {error}")
