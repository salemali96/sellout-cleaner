"""Agent tests with a scripted fake Anthropic client: no network, deterministic."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest
import yaml

from src import fixes
from src.agent import graph, tools
from src.agent.llm import MODEL, LLM, LLMError
from src.agent.memory import Memory
from src.config import load_mapping
from tests.conftest import raw_row, weeks

# ---- fake client ----------------------------------------------------------------------------

Script = Callable[[dict[str, Any]], Any] | Any


class FakeClient:
    def __init__(self, script: list[Script]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw: Any) -> Any:
        self.calls.append(kw)
        if not self.script:
            raise AssertionError("unexpected LLM call")
        step = self.script.pop(0)
        return step(kw) if callable(step) else step


def text(t: str, stop: str = "end_turn") -> Any:
    return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="text", text=t)], stop_details=None)


def tool_calls(*calls: tuple[str, dict[str, Any]]) -> Any:
    blocks = [SimpleNamespace(type="tool_use", id=f"tu_{i}", name=n, input=a) for i, (n, a) in enumerate(calls)]
    return SimpleNamespace(stop_reason="tool_use", content=blocks, stop_details=None)


def last_tool_results(kw: dict[str, Any]) -> list[dict[str, Any]]:
    return kw["messages"][-1]["content"]


# ---- fixtures -------------------------------------------------------------------------------

PROMO = {"promo": "PR1", "promo_type": "TPR", "promo_start": "2025-01-26", "promo_end": "2025-01-06"}  # inverted


def dirty_rows() -> list[dict[str, str]]:
    return weeks(4, **PROMO) + [
        raw_row(sku="SKU2", promo="PR-GHOST"),                            # row 4: orphan promo
        raw_row(sku="SKU2", week="2025-W02"),                             # row 5
        raw_row(sku="SKU2", week="2025-W04", units="-500", value="-5"),   # row 6: obviously broken row
    ]


@pytest.fixture
def session(make_csv, rules, tmp_path: Path):
    def _make(script: list[Script], approve: bool | None = True, mapping_path: Path | None = None,
              memory: Path | None = None, **kw: Any) -> tuple[graph.Session, FakeClient, list[dict[str, Any]]]:
        asked: list[dict[str, Any]] = []

        def approver(op: dict[str, Any]) -> bool:
            asked.append(op)
            assert approve is not None, "approver should not be asked"
            return approve

        if mapping_path is None:
            mapping_path = tmp_path / "schema_mapping.yaml"
            mapping_path.write_text(Path("config/schema_mapping.yaml").read_text())
        kw.setdefault("explain", False)  # the reporter makes one more LLM call; tested on its own
        client = FakeClient(script)
        s = graph.Session(file=make_csv(dirty_rows()), out_dir=tmp_path / "out", mapping_path=mapping_path,
                          rules=rules, llm=LLM(client), approver=approver, memory=Memory(memory), **kw)
        return s, client, asked

    return _make


# ---- profiler ---------------------------------------------------------------------------------

def mapping_reply(columns: dict[str, str], confidence: float = 0.95) -> Any:
    return text(json.dumps({"mappings": [{"raw_column": r, "standard_column": s, "confidence": confidence,
                                          "reason": "header and samples match"} for r, s in columns.items()],
                            "notes": ""}))


def test_profiler_writes_mapping_with_table_split(session, tmp_path: Path) -> None:
    std = load_mapping().columns
    bad = {**std, "Brand": "product_id"}  # first answer maps two raw columns to product_id
    s, client, _ = session([mapping_reply(bad), mapping_reply({**std, "Region": "region"}, 0.6)],
                           mapping_path=tmp_path / "new_mapping.yaml", force_profile=True)
    out = graph.profiler(s, {"events": []})
    assert out["mapping_source"] == "profiler"
    written = load_mapping(s.mapping_path)
    assert written.columns == std
    assert written.tables["sellout"].key == ["product_id", "market_id", "week_start_date"]
    assert "LOW CONFIDENCE" in s.mapping_path.read_text()
    assert len(client.calls) == 2 and "mapped more than once" in client.calls[1]["messages"][-1]["content"]
    fmt = client.calls[0]["output_config"]["format"]
    assert fmt["type"] == "json_schema" and fmt["schema"]["additionalProperties"] is False


def test_profiler_reuses_valid_existing_mapping(session) -> None:
    s, client, _ = session([])
    assert graph.profiler(s, {"events": []})["mapping_source"] == "existing"
    assert client.calls == []


def test_profile_columns_is_deterministic_and_small(make_csv, rules) -> None:
    prof = tools.profile_columns(make_csv(dirty_rows()), rules)
    by = {p["raw_column"]: p for p in prof}
    assert by["Sales Qty"]["numeric_rate"] == 1.0 and by["Promo ID"]["fill_rate"] < 1
    assert len(by["SKU Code"]["samples"]) <= 8


# ---- full graph ---------------------------------------------------------------------------------

def cleaner_round_one() -> list[Script]:
    return [
        tool_calls(("get_issue_samples", {"rule_id": "promo.invalid_date_range", "limit": 5})),
        tool_calls(("fix_swap_promo_dates", {"promo_ids": ["PR1"], "all_flagged": False, "rationale": "end before start"}),
                   ("fix_add_promo_stubs", {"promo_ids": [], "all_flagged": True, "rationale": "referenced, never defined"}),
                   ("fix_delete_rows", {"row_nrs": [6], "reason": "value/units nonsensical",
                                        "rationale": "-500 units for 5 EGP"})),
    ]  # lean mode: a turn of only successful fixes ends the cleaner, no follow-up "summary" request


def test_graph_end_to_end_with_approval(session) -> None:
    s, client, asked = session(cleaner_round_one(), approve=True)
    state = graph.run_agent(s)
    assert state["status"] == "done" and state["validation"]["passed"] and state["iteration"] == 1
    assert [p["status"] for p in state["plan"]] == ["applied", "applied", "approved"]
    assert [a["name"] for a in asked] == ["delete_rows"]
    log = s.result.log
    assert log["reconciliation"]["passed"]
    assert "promo.invalid_date_range" not in log["issue_counts"]
    assert "cross_table.orphan_promo" not in log["issue_counts"]
    promo = pl.read_parquet(s.out_dir / "promo.parquet")
    assert set(promo["promo_id"]) == {"PR1", "PR-GHOST"}
    q = pl.read_parquet(s.out_dir / "quarantine.parquet")
    assert q.filter(pl.col("row_nr") == 6)["rule_id"].to_list() == ["manual.delete_rows"]
    assert json.loads((s.out_dir / "run_log.json").read_text())["agent"]["plan"][2]["status"] == "approved"
    # API contract: model, strict tools, refusal fallback
    kw = client.calls[0]
    assert kw["model"] == MODEL and kw["fallbacks"] == "default"
    assert all(t["strict"] for t in kw["tools"])


def test_rejected_risky_fix_is_not_applied(session) -> None:
    s, _, _ = session(cleaner_round_one(), approve=False)
    state = graph.run_agent(s)
    assert state["plan"][2]["status"] == "rejected"
    q = pl.read_parquet(s.out_dir / "quarantine.parquet")
    assert 6 not in q["row_nr"].to_list()
    assert s.result.log["reconciliation"]["passed"]


def test_memory_answers_repeat_questions(session, tmp_path: Path) -> None:
    mem = tmp_path / "memory.json"
    s1, _, asked1 = session(cleaner_round_one(), approve=True, memory=mem)
    graph.run_agent(s1)
    s2, _, asked2 = session(cleaner_round_one(), approve=None, memory=mem)  # approver must not be called
    state = graph.run_agent(s2)
    assert len(asked1) == 1 and asked2 == []
    approvals = [e for e in state["events"] if e["node"] == "approval"]
    assert approvals[-1]["decisions"][0]["by"] == "memory"


def test_validator_sends_work_back_to_cleaner(session, monkeypatch) -> None:
    monkeypatch.setattr(fixes, "swap_promo_dates", lambda promo, ids: (promo, fixes.FixRecord("swap_promo_dates", False, 0)))

    def round_two(kw: dict[str, Any]) -> Any:
        brief = kw["messages"][0]["content"]
        assert "validator rejected" in brief and "swap_promo_dates did not resolve" in brief
        return tool_calls(("withdraw_fix", {"plan_index": 0, "rationale": "did not work; needs a human"}))

    script = [tool_calls(("fix_swap_promo_dates", {"promo_ids": ["PR1"], "all_flagged": False, "rationale": "swap"})),
              round_two]
    s, _, _ = session(script)
    state = graph.run_agent(s)
    validators = [e for e in state["events"] if e["node"] == "validator"]
    assert [v["passed"] for v in validators] == [False, True]
    assert state["iteration"] == 2 and state["plan"][0]["status"] == "withdrawn"


def test_tool_errors_are_returned_to_the_model(session) -> None:
    def check_error(kw: dict[str, Any]) -> Any:
        res = last_tool_results(kw)[0]
        assert res["is_error"] and "already defined" in res["content"]
        return text("nothing to do")

    s, _, _ = session([tool_calls(("fix_add_promo_stubs", {"promo_ids": ["PR1"], "all_flagged": False, "rationale": "x"})), check_error])
    state = graph.run_agent(s)
    assert state["plan"] == [] and state["status"] == "done"


# ---- safety rails -----------------------------------------------------------------------------

def test_data_layer_refuses_unapproved_risky_fix(prep, rules, mapping) -> None:
    p = prep(weeks(2))
    op = fixes.FixOp("delete_rows", {"row_nrs": [0], "reason": "x"}, risky=True, approved=False)
    with pytest.raises(fixes.ApprovalRequired):
        fixes.apply_plan(p.ctx.fixed, [op], rules, mapping)


def test_refusal_raises() -> None:
    llm = LLM(FakeClient([text("", stop="refusal")]))
    with pytest.raises(LLMError):
        llm.create(system="s", messages=[{"role": "user", "content": "x"}])


def test_written_mapping_is_valid_yaml(tmp_path: Path) -> None:
    m = tools.mapping_from_columns({"A": "product_id", "B": "market_id", "C": "week_raw", "D": "units", "E": "value"})
    p = tools.write_mapping(m, tmp_path / "m.yaml", "# header\n")
    data = yaml.safe_load(p.read_text())
    assert data["tables"]["product"]["key"] == ["product_id"]
    assert load_mapping(p).tables["product"].attributes == []


# ---- UI flow: pause before approval, resume after the human decides ---------------------------

def test_graph_pauses_for_approval_and_resumes(session) -> None:
    from langgraph.checkpoint.memory import MemorySaver

    decisions: dict[str, bool] = {}
    s, _, _ = session(cleaner_round_one(), approve=None)
    s.approver = lambda op: decisions[op["name"]]
    app = graph.build_graph(s, MemorySaver(), interrupt_before=["approval"])
    cfg = {"configurable": {"thread_id": "t"}, "recursion_limit": 50}
    app.invoke({"events": []}, cfg)
    assert app.get_state(cfg).next == ("approval",) and s.result is None
    assert [p["status"] for p in s.ws.plan] == ["applied", "applied", "pending"]
    decisions["delete_rows"] = True
    app.invoke(None, cfg)
    state = app.get_state(cfg)
    assert not state.next and state.values["status"] == "done" and s.result.log["reconciliation"]["passed"]


# ---- reporter, usage, duplicate finder, analyst -----------------------------------------------

BRIEF = {"headline": "Clean with one open point", "summary": "7 rows in, 6 out.",
         "findings": [{"title": "<b>Row 6</b>", "detail": "-500 units", "severity": "attention", "action": "check"}],
         "next_steps": ["Ask the retailer about row 6"]}


def test_reporter_writes_brief_into_log_and_report(session) -> None:
    def brief_call(kw: dict[str, Any]) -> Any:
        assert '"row_counts"' in kw["messages"][0]["content"] and kw["output_config"]["format"]["type"] == "json_schema"
        return text(json.dumps(BRIEF))

    s, _, _ = session([*cleaner_round_one(), brief_call], explain=True)
    state = graph.run_agent(s)
    assert state["brief"]["headline"] == BRIEF["headline"] and state["events"][-1]["node"] == "reporter"
    log = json.loads((s.out_dir / "run_log.json").read_text())
    assert log["agent"]["brief"] == BRIEF
    assert log["agent"]["usage"]["cleaner"]["calls"] == 2 and log["agent"]["usage"]["reporter"]["calls"] == 1
    page = (s.out_dir / "dq_report.html").read_text()
    assert "Clean with one open point" in page and "&lt;b&gt;Row 6&lt;/b&gt;" in page  # escaped, not injected


def test_reporter_failure_keeps_the_run(session) -> None:
    s, _, _ = session([*cleaner_round_one(), text("", stop="refusal")], explain=True)
    state = graph.run_agent(s)
    assert state["status"] == "done" and state["brief"] is None and "LLMError" in state["events"][-1]["error"]
    assert s.result.log["reconciliation"]["passed"]


def test_usage_is_summed_per_stage() -> None:
    reply = text("ok")
    reply.usage = SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=3, cache_creation_input_tokens=None)
    llm = LLM(FakeClient([reply, reply]))
    llm.stage = "cleaner"
    for _ in range(2):
        llm.create(system="s", messages=[{"role": "user", "content": "x"}])
    assert llm.usage == {"cleaner": {"calls": 2, "input_tokens": 20, "output_tokens": 10,
                                     "cache_read_input_tokens": 6, "cache_creation_input_tokens": 0}}


def test_find_duplicate_dimensions_tool(prep, rules, mapping) -> None:
    rows = weeks(2) + [raw_row(store="M-A", store_name="City  Star-Mall"), raw_row(store="M-B", store_name="city star mall")]
    ws = tools.Workspace(Path("x.csv"), rules, mapping, prep(rows))
    ws.materialize()
    out, is_error = tools.call_tool(ws, "find_duplicate_dimensions", {"table": "market"})
    groups = json.loads(out)["groups"]
    assert not is_error and [sorted(r["market_id"] for r in g) for g in groups] == [["M-A", "M-B"]]


@pytest.fixture
def analyst_con(session):
    from src.agent import analyst

    s, _, _ = session(cleaner_round_one())
    graph.run_agent(s)
    return analyst.connect(s.out_dir)


@pytest.mark.parametrize("sql", [
    "DROP TABLE sellout", "SELECT 1; SELECT 2", "COPY sellout TO '/tmp/leak.csv'",
    "SELECT * FROM read_csv('/etc/passwd')", "SET enable_external_access = true", "ATTACH '/tmp/x.db'",
])
def test_analyst_sql_is_read_only_and_sandboxed(analyst_con, sql: str) -> None:
    from src.agent import analyst

    with pytest.raises(tools.ToolError):
        analyst.run_sql(analyst_con, sql)


def test_analyst_answers_with_sql(analyst_con) -> None:
    from src.agent import analyst

    def final(kw: dict[str, Any]) -> Any:
        first, second = last_tool_results(kw)
        assert json.loads(first["content"])["rows"] == [[6]] and second["is_error"]
        assert kw["messages"][0]["content"] == "earlier q" and "sellout (6 rows)" in kw["system"][0]["text"]
        return text("There are 6 clean rows.")

    script = [tool_calls(("run_sql", {"sql": "SELECT count(*) FROM sellout", "purpose": "count"}),
                         ("run_sql", {"sql": "DELETE FROM sellout", "purpose": "bad"})), final]
    a = analyst.Analyst(analyst_con, LLM(FakeClient(script))).ask("how many rows?", [("earlier q", "earlier a")])
    assert a.text == "There are 6 clean rows." and [("error" in q) for q in a.queries] == [False, True]
    assert analyst_con.execute("SELECT count(*) FROM sellout").fetchone()[0] == 6


def test_save_and_remove_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os
    import stat

    from src import env

    monkeypatch.setenv(env.KEY, "old")
    p = tmp_path / ".env"
    p.write_text("# note\nOTHER=1\nANTHROPIC_API_KEY=old\n")
    env.save_key("  sk-ant-new-key-1234567890  ", p)
    assert env.parse_env(p.read_text()) == {"OTHER": "1", env.KEY: "sk-ant-new-key-1234567890"}
    assert stat.S_IMODE(p.stat().st_mode) == 0o600 and os.environ[env.KEY] == "sk-ant-new-key-1234567890"
    assert env.masked_key() == "sk-ant-...7890" and "# note" in p.read_text()
    with pytest.raises(ValueError):
        env.save_key("two words", p)
    env.remove_key(p)
    assert env.parse_env(p.read_text())[env.KEY] == "" and env.KEY not in os.environ


# ---- token / request savings ----------------------------------------------------------------

def test_lean_brief_carries_the_evidence(session) -> None:
    def check_brief(kw: dict[str, Any]) -> Any:
        brief = kw["messages"][0]["content"]
        ev = json.loads(brief.split("\n")[1])
        by = {r["rule_id"]: r for r in ev["issues"]}
        assert by["promo.invalid_date_range"]["affected_ids"] == ["PR1"]
        assert "top_values" in by["cross_table.orphan_promo"] or "affected_ids" in by["cross_table.orphan_promo"]
        assert "in ONE" in brief and "several tool calls in the same reply" in brief
        return text("nothing")

    s, _, _ = session([check_brief])
    graph.run_agent(s)


def test_identical_issues_reuse_the_plan_without_ai(session, tmp_path: Path) -> None:
    mem = tmp_path / "memory.json"
    s1, c1, _ = session(cleaner_round_one(), approve=True, memory=mem)
    graph.run_agent(s1)
    s2, c2, asked = session([], approve=None, memory=mem)        # no scripted replies: any AI call fails the test
    state = graph.run_agent(s2)
    assert c2.calls == [] and asked == []
    assert [p["status"] for p in state["plan"]] == ["applied", "applied", "approved"]
    assert state["validation"]["passed"] and "no AI call" in state["summaries"][0]
    s3, c3, _ = session(cleaner_round_one(), approve=True, memory=mem, reuse_plan=False)
    graph.run_agent(s3)
    assert len(c3.calls) == 2                                      # switched off: the model is asked again


def test_fingerprint_is_order_independent() -> None:
    a = pl.DataFrame({"rule_id": ["r1", "r2"], "row_key": ["1", "2"], "column": ["c", "c"], "value": ["x", None]})
    assert tools.issues_fingerprint(a) == tools.issues_fingerprint(a.reverse())
    assert tools.issues_fingerprint(a) != tools.issues_fingerprint(a.head(1))


def test_lean_mode_skips_the_summary_request_and_off_keeps_it(session) -> None:
    s, c, _ = session(cleaner_round_one())
    state = graph.run_agent(s)
    assert len(c.calls) == 2 and "Applied: swap_promo_dates, add_promo_stubs" in state["summaries"][0]
    assert "Waiting for approval: delete_rows" in state["summaries"][0]
    s2, c2, _ = session([*cleaner_round_one(), text("All done.")], lean=False)
    state2 = graph.run_agent(s2)
    assert len(c2.calls) == 3 and state2["summaries"] == ["All done."]
