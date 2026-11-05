"""Runtime configuration loaded from environment variables (and an optional .env file)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

REQUIRED_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "TAVILY_API_KEY")


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


# Research depth presets: fixed, tested bundles of the limits the code enforces. The brief's
# maximum of two research rounds holds at every level; "deep" goes wider, not longer.
DEPTH_PRESETS: dict[str, dict] = {
    "quick": {
        "source_target": "4-6", "max_research_rounds": 1, "min_plan_tasks": 3, "max_plan_tasks": 4,
        "max_follow_up_tasks": 0, "task_max_searches": 2, "task_max_page_opens": 2,
        "page_char_limit": 8_000, "pdf_char_limit": 20_000,
    },
    "standard": {
        "source_target": "8-15", "max_research_rounds": 2, "min_plan_tasks": 4, "max_plan_tasks": 5,
        "max_follow_up_tasks": 3, "task_max_searches": 4, "task_max_page_opens": 3,
        "page_char_limit": 12_000, "pdf_char_limit": 30_000,
    },
    "deep": {
        "source_target": "20-30", "max_research_rounds": 2, "min_plan_tasks": 6, "max_plan_tasks": 8,
        "max_follow_up_tasks": 5, "task_max_searches": 5, "task_max_page_opens": 5,
        "page_char_limit": 16_000, "pdf_char_limit": 40_000,
    },
}


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str
    anthropic_model: str
    tavily_api_key: str
    runs_dir: Path = Path("artifacts/runs")
    # Research depth preset (see DEPTH_PRESETS); the defaults below are "standard".
    depth: str = "standard"
    source_target: str = "8-15"  # sources the agents aim for across the run; guidance, not a hard limit
    # Workflow limits
    max_research_rounds: int = 2
    min_plan_tasks: int = 4
    max_plan_tasks: int = 5
    max_follow_up_tasks: int = 3
    run_timeout_s: int = 1800
    # Per research task caps
    task_max_model_turns: int = 12
    task_max_searches: int = 4
    task_max_page_opens: int = 3
    task_timeout_s: int = 180
    # Search, browser and retries (retry settings are total attempts)
    search_max_results: int = 5
    page_timeout_ms: int = 20_000
    page_char_limit: int = 12_000  # page text the model sees, per HTML page
    pdf_char_limit: int = 30_000  # per PDF; documents are longer and often the primary source
    page_raw_char_limit: int = 300_000  # text kept from the browser before relevance selection
    llm_max_retries: int = 3
    browser_max_retries: int = 2

    def __repr__(self) -> str:  # never expose secrets
        return f"Settings(model={self.anthropic_model!r}, runs_dir={str(self.runs_dir)!r})"


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer") from None
    if value < 1:
        raise ConfigError(f"{name} must be at least 1")
    return value


def load_settings(env_file: str | None = ".env", depth: str | None = None) -> Settings:
    """Load settings. Precedence: individual environment variables > depth preset > defaults.

    The depth comes from the `depth` argument (the CLI's --depth), else RESEARCH_DEPTH, else "standard".
    """
    if env_file:
        load_dotenv(env_file, override=False)
        # An empty OS-level variable would otherwise hide the value in .env.
        for name, value in dotenv_values(env_file).items():
            if value and not os.getenv(name, "").strip():
                os.environ[name] = value
    missing = [name for name in REQUIRED_VARS if not os.getenv(name, "").strip()]
    if missing:
        raise ConfigError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill them in."
        )
    depth = (depth or os.getenv("RESEARCH_DEPTH", "") or "standard").strip().lower()
    if depth not in DEPTH_PRESETS:
        raise ConfigError(f"Unknown research depth {depth!r}; choose one of: {', '.join(DEPTH_PRESETS)}")
    preset = DEPTH_PRESETS[depth]
    # Every integer setting can be overridden by an environment variable of the same name in upper case.
    limits = {
        name: _int_env(name.upper(), preset.get(name, field.default))
        for name, field in Settings.__dataclass_fields__.items()
        if field.type in (int, "int")
    }
    if limits["min_plan_tasks"] > limits["max_plan_tasks"]:
        raise ConfigError("MIN_PLAN_TASKS cannot be greater than MAX_PLAN_TASKS")
    return Settings(
        anthropic_api_key=os.environ["ANTHROPIC_API_KEY"].strip(),
        anthropic_model=os.environ["ANTHROPIC_MODEL"].strip(),
        tavily_api_key=os.environ["TAVILY_API_KEY"].strip(),
        runs_dir=Path(os.getenv("RUNS_DIR", "artifacts/runs")),
        depth=depth,
        source_target=preset["source_target"],
        **limits,
    )
