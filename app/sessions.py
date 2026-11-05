"""Isolated, in-memory agent sessions.

Each agent gets its own session holding its own instructions and conversation history.
Nothing is shared between sessions: information moves between agents only as explicit,
validated hand-off objects passed by the workflow.
"""

import copy
import uuid
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"
AGENT_NAMES = ("orchestrator", "researcher", "creator")


def load_prompt(agent_name: str) -> str:
    return (PROMPTS_DIR / f"{agent_name}.md").read_text(encoding="utf-8")


class AgentSession:
    def __init__(self, agent_name: str, system_prompt: str):
        self.agent_name = agent_name
        self.session_id = str(uuid.uuid4())
        self.system_prompt = system_prompt
        self._messages: list[dict] = []

    def add_user(self, content: str | list) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str | list) -> None:
        self._messages.append({"role": "assistant", "content": content})

    def history(self) -> list[dict]:
        """A deep copy, so callers can never mutate the session's own history."""
        return copy.deepcopy(self._messages)

    def __len__(self) -> int:
        return len(self._messages)


class SessionStore:
    """Holds the three sessions for one run."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self._sessions = {name: AgentSession(name, load_prompt(name)) for name in AGENT_NAMES}

    def get(self, agent_name: str) -> AgentSession:
        return self._sessions[agent_name]

    def all(self) -> list[AgentSession]:
        return list(self._sessions.values())
