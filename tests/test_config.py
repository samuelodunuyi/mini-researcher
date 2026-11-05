import pytest

from app.config import REQUIRED_VARS, ConfigError, load_settings


@pytest.fixture
def env(monkeypatch):
    for name in REQUIRED_VARS:
        monkeypatch.setenv(name, f"value-for-{name}")
    return monkeypatch


def test_loads_required_vars_and_defaults(env):
    s = load_settings(env_file=None)
    assert s.anthropic_model == "value-for-ANTHROPIC_MODEL"
    assert s.max_research_rounds == 2
    assert s.task_max_searches == 4


def test_missing_vars_named_in_error(env):
    env.delenv("TAVILY_API_KEY")
    env.setenv("ANTHROPIC_API_KEY", "   ")
    with pytest.raises(ConfigError) as exc:
        load_settings(env_file=None)
    msg = str(exc.value)
    assert "ANTHROPIC_API_KEY" in msg and "TAVILY_API_KEY" in msg
    assert "ANTHROPIC_MODEL" not in msg


def test_error_and_repr_never_show_secrets(env):
    env.setenv("ANTHROPIC_API_KEY", "sk-secret-123")
    s = load_settings(env_file=None)
    assert "sk-secret-123" not in repr(s)


def test_env_file_fills_empty_os_variables(env, tmp_path):
    env.setenv("TAVILY_API_KEY", "")
    env_file = tmp_path / ".env"
    env_file.write_text("TAVILY_API_KEY=from-file\nANTHROPIC_MODEL=file-model\n")
    s = load_settings(env_file=str(env_file))
    assert s.tavily_api_key == "from-file"
    assert s.anthropic_model == "value-for-ANTHROPIC_MODEL"  # a real OS value still wins


def test_limit_override(env):
    env.setenv("MAX_RESEARCH_ROUNDS", "1")
    assert load_settings(env_file=None).max_research_rounds == 1


@pytest.mark.parametrize("value", ["two", "0"])
def test_invalid_limit_rejected(env, value):
    env.setenv("TASK_MAX_SEARCHES", value)
    with pytest.raises(ConfigError, match="TASK_MAX_SEARCHES"):
        load_settings(env_file=None)
