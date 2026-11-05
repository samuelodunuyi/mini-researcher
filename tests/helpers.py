"""Fake Anthropic client and reply builders shared by the tests."""

import copy
import itertools
import json
from types import SimpleNamespace

from anthropic.types import TextBlock, ToolUseBlock

from app.config import Settings
from app.llm import LLMClient

_ids = itertools.count(1)


def make_settings(**overrides) -> Settings:
    return Settings(anthropic_api_key="test-key", anthropic_model="test-model", tavily_api_key="test-tavily", **overrides)


def usage(input_tokens=100, output_tokens=20, cache_read=50, cache_write=10):
    return SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                           cache_read_input_tokens=cache_read, cache_creation_input_tokens=cache_write)


def text_reply(text: str):
    return SimpleNamespace(content=[TextBlock(type="text", text=text)], stop_reason="end_turn", usage=usage())


def json_reply(obj, prose: str = ""):
    return text_reply(f"{prose}\n{json.dumps(obj)}" if prose else json.dumps(obj))


def tool_reply(*calls, prose: str = ""):
    """calls: (name, input) tuples."""
    blocks = [TextBlock(type="text", text=prose)] if prose else []
    blocks += [ToolUseBlock(type="tool_use", id=f"toolu_{next(_ids)}", name=n, input=i) for n, i in calls]
    return SimpleNamespace(content=blocks, stop_reason="tool_use", usage=usage())


class FakeAnthropic:
    """Returns scripted replies in order and records every request it receives."""

    def __init__(self, replies=()):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.messages = self

    def queue(self, *replies):
        self.replies.extend(replies)

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if not self.replies:
            raise AssertionError("FakeAnthropic ran out of scripted replies")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def make_llm(fake: FakeAnthropic, trace=None, **settings) -> LLMClient:
    return LLMClient(make_settings(**settings), client=fake, sleep=lambda s: None, trace=trace)


def read_trace(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def last_tool_results(session) -> list[dict]:
    return [b for b in session.history()[-1]["content"] if isinstance(b, dict) and b.get("type") == "tool_result"]
