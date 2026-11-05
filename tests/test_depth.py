"""Research depth presets: values, precedence, the CLI option, plan ranges and quick's single round."""

import pytest
from helpers import FakeAnthropic, json_reply, make_llm, read_trace
from test_workflow import MORE, PLAN, StubResearcher, make_workflow

from app import cli
from app.agents.orchestrator import Orchestrator
from app.config import DEPTH_PRESETS, REQUIRED_VARS, ConfigError, Settings, load_settings
from app.llm import MalformedAgentResponse
from app.sessions import AgentSession


@pytest.fixture
def env(monkeypatch):
    for name in REQUIRED_VARS:
        monkeypatch.setenv(name, "x" * 10)
    for name in ("RESEARCH_DEPTH", "TASK_MAX_PAGE_OPENS"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


# ---- preset values and precedence ----

def test_standard_is_the_default_and_matches_the_sample_run_settings(env):
    s = load_settings(env_file=None)
    assert s.depth == "standard"
    assert (s.min_plan_tasks, s.max_plan_tasks, s.max_research_rounds, s.max_follow_up_tasks,
            s.task_max_searches, s.task_max_page_opens, s.page_char_limit, s.pdf_char_limit) == (4, 5, 2, 3, 4, 3, 12000, 30000)
    # The dataclass defaults are the standard preset, so code that builds Settings directly behaves the same.
    defaults = Settings(anthropic_api_key="k", anthropic_model="m", tavily_api_key="t")
    assert all(getattr(defaults, k) == v for k, v in DEPTH_PRESETS["standard"].items())


@pytest.mark.parametrize("depth", list(DEPTH_PRESETS))
def test_each_preset_applies_and_never_exceeds_two_rounds(env, depth):
    s = load_settings(env_file=None, depth=depth)
    assert s.depth == depth
    assert all(getattr(s, k) == v for k, v in DEPTH_PRESETS[depth].items())
    assert 1 <= s.max_research_rounds <= 2 and s.min_plan_tasks <= s.max_plan_tasks


def test_quick_has_one_round_and_no_follow_ups(env):
    s = load_settings(env_file=None, depth="quick")
    assert (s.max_research_rounds, s.max_follow_up_tasks, s.task_max_page_opens, s.source_target) == (1, 0, 2, "4-6")


def test_precedence_argument_then_env_then_default(env):
    env.setenv("RESEARCH_DEPTH", "deep")
    assert load_settings(env_file=None).depth == "deep"
    assert load_settings(env_file=None, depth="quick").depth == "quick"


def test_single_env_var_overrides_the_preset(env):
    env.setenv("TASK_MAX_PAGE_OPENS", "4")
    s = load_settings(env_file=None, depth="quick")
    assert s.task_max_page_opens == 4 and s.max_research_rounds == 1


def test_unknown_depth_is_a_clear_error(env):
    with pytest.raises(ConfigError, match="Unknown research depth 'extreme'.*quick, standard, deep"):
        load_settings(env_file=None, depth="extreme")


# ---- CLI option ----

def test_cli_passes_depth(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "run", lambda topic, depth=None: seen.update(topic=topic, depth=depth) or 0)
    assert cli.main(["run", "--topic", "t", "--depth", "quick"]) == 0
    assert seen == {"topic": "t", "depth": "quick"}
    cli.main(["run", "--topic", "t"])
    assert seen["depth"] is None  # falls back to RESEARCH_DEPTH, then standard


def test_cli_rejects_unknown_depth(capsys):
    with pytest.raises(SystemExit):
        cli.main(["run", "--topic", "t", "--depth", "extreme"])
    assert "invalid choice" in capsys.readouterr().err


# ---- plan range enforcement for each level ----

def plan_with(n):
    return {"topic": "t", "tasks": [{"id": f"T{i}", "question": f"q{i}", "evidence_required": "e"} for i in range(1, n + 1)]}


def orchestrator(fake, depth):
    p = DEPTH_PRESETS[depth]
    return Orchestrator(AgentSession("orchestrator", "p"), make_llm(fake), p["min_plan_tasks"], p["max_plan_tasks"],
                        p["source_target"], p["max_follow_up_tasks"])


@pytest.mark.parametrize("depth", list(DEPTH_PRESETS))
def test_plan_range_is_enforced_for_each_level(depth):
    low, high = DEPTH_PRESETS[depth]["min_plan_tasks"], DEPTH_PRESETS[depth]["max_plan_tasks"]
    fake = FakeAnthropic([json_reply(plan_with(high + 1)), json_reply(plan_with(low - 1))])
    with pytest.raises(MalformedAgentResponse, match=f"between {low} and {high}"):
        orchestrator(fake, depth).create_plan("topic")
    assert f"with {low} to {high} tasks" in fake.calls[0]["messages"][0]["content"]
    assert DEPTH_PRESETS[depth]["source_target"] in fake.calls[0]["messages"][0]["content"]

    ok = FakeAnthropic([json_reply(plan_with(low))])
    assert len(orchestrator(ok, depth).create_plan("topic").tasks) == low


# ---- quick skips round 2 ----

QUICK = DEPTH_PRESETS["quick"]


def test_quick_never_runs_a_second_round_and_accepts_when_checks_pass(tmp_path):
    researcher = StubResearcher()
    wf, fake, _ = make_workflow(tmp_path, [json_reply(PLAN), json_reply(MORE)], researcher,
                                pdf_writer=lambda html, path: path.write_bytes(b"x"),
                                max_research_rounds=QUICK["max_research_rounds"],
                                max_follow_up_tasks=QUICK["max_follow_up_tasks"])
    state = wf.run("run-test", "topic")
    assert len(researcher.calls) == 1
    assert "No more research rounds are allowed" in fake.calls[1]["messages"][-1]["content"]
    assert state["decisions"][-1].outcome.value == "ACCEPT"
    records = read_trace(tmp_path / "run" / "trace.jsonl")
    override = next(r for r in records if r["event"] == "round_limit_override")
    assert override["metadata"]["reason"] == "round limit reached and deterministic checks passed"


def test_quick_drops_failing_tasks_instead_of_a_second_round(tmp_path):
    class Thin(StubResearcher):
        def run_tasks(self, tasks):
            result = super().run_tasks(tasks)
            return result.model_copy(update={"findings": [f for f in result.findings if f.task_id != "T2"]})

    researcher = Thin()
    wf, _, _ = make_workflow(tmp_path, [json_reply(PLAN), json_reply(MORE)], researcher,
                             pdf_writer=lambda html, path: path.write_bytes(b"x"),
                             max_research_rounds=1, max_follow_up_tasks=0)
    state = wf.run("run-test", "topic")
    assert len(researcher.calls) == 1
    assert state["decisions"][-1].outcome.value == "ACCEPT"
    assert "round limit reached and deterministic checks failed: excluded T2" in state["decisions"][-1].rationale


def test_depth_is_recorded_in_trace_and_summary(tmp_path):
    import json

    from app.workflow import Workflow

    wf, _, _ = make_workflow(tmp_path, [json_reply(PLAN), json_reply({"outcome": "STOP_INSUFFICIENT", "rationale": "x"})],
                             creator_replies=())
    wf.settings = Settings(**{**wf.settings.__dict__, "depth": "deep"})  # frozen dataclass: build a copy
    assert isinstance(wf, Workflow)
    wf.run("run-test", "topic")
    started = read_trace(tmp_path / "run" / "trace.jsonl")[0]
    assert started["event"] == "run_started" and started["metadata"]["depth"] == "deep"
    assert started["metadata"]["limits"]["plan_tasks"] == "4-5"
    assert json.loads((tmp_path / "run" / "run_summary.json").read_text())["depth"] == "deep"
