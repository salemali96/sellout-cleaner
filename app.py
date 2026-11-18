"""Sellout Cleaner: Streamlit UI.

    uv run streamlit run app.py

Connect an API key in the sidebar, pick or upload a raw CSV, check the column mapping, run the rules-only
cleaning or the AI agents (with approve/reject for risky fixes), then explore, question and download the result.
"""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

import polars as pl
import streamlit as st

from src import env, evaluate, pipeline, writer
from src.config import load_mapping, load_rules
from src.agent import providers
from src.ingest import raw_columns, scan_raw
from src.ui import style

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "output"
MAPPING_PATH = ROOT / "config" / "schema_mapping.yaml"
MEMORY_PATH = ROOT / "data" / "agent_memory.json"

# Hosted (one server, many visitors, e.g. a Hugging Face Space): every visitor gets a private folder and private
# AI settings. The host's own key, if set as a server secret, is the default; visitors can paste their own instead.
# Streamlit Community Cloud runs apps from /mount/src/<repo>, so it is detected without any setting.
HOSTED = os.environ.get("SELLOUT_HOSTED", "").lower() in ("1", "true", "yes") or str(ROOT).startswith("/mount/src/")
HOSTED_MAX_DEMO_ROWS = 200_000  # free hosting has about 1 GB of memory
WORKSPACES = Path(tempfile.gettempdir()) / "sellout-workspaces"
WORKSPACE_TTL_HOURS = 24
HOSTED_HIDDEN = ("ollama", "custom")  # no local Ollama on a server; no visitor-chosen URLs called from the server

env.load_env()
st.set_page_config(page_title="Sellout Cleaner", page_icon="🧹", layout="wide")
st.markdown(style.CSS, unsafe_allow_html=True)


def remove_old_workspaces() -> None:
    cutoff = time.time() - WORKSPACE_TTL_HOURS * 3600
    for old in WORKSPACES.iterdir():
        if old.is_dir() and old.stat().st_mtime < cutoff:
            shutil.rmtree(old, ignore_errors=True)


def workspace() -> Path:
    """This visitor's private folder (uploads, outputs, mapping, agent memory), removed after a day unused."""
    known = st.session_state.get("workspace")
    if known and Path(known).exists():
        os.utime(known)
        return Path(known)
    WORKSPACES.mkdir(parents=True, exist_ok=True)
    remove_old_workspaces()
    path = WORKSPACES / uuid.uuid4().hex
    (path / "raw").mkdir(parents=True)
    (path / "output").mkdir()
    shutil.copy(MAPPING_PATH, path / "schema_mapping.yaml")
    st.session_state.workspace = str(path)
    return path


if HOSTED:
    WORK = workspace()
    RAW_DIR, OUT_DIR = WORK / "raw", WORK / "output"
    MAPPING_PATH, MEMORY_PATH = WORK / "schema_mapping.yaml", WORK / "agent_memory.json"
    providers.use_session(st.session_state.setdefault("ai_settings", {}))


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


# ---- helpers ----------------------------------------------------------------------------------

def has_key() -> bool:
    return providers.is_ready()


def out_dir(mode: str) -> Path:
    return OUT_DIR / mode


def truth_for(file: Path) -> Path | None:
    for p in (file.with_name(f"{file.stem}_truth.json"), file.with_name("ground_truth.json")):
        if p.exists() and (p.name != "ground_truth.json" or file.stem == "sellout"):
            return p
    return None


def load_log(mode: str, file: Path) -> dict[str, Any] | None:
    p = out_dir(mode) / "run_log.json"
    if not p.exists():
        return None
    log = json.loads(p.read_text(encoding="utf-8"))
    return log if Path(log["input"]["file"]).resolve() == file.resolve() else None


def zip_outputs(folder: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in folder.rglob("*"):
            if p.is_file() and ".cache" not in p.parts:
                z.write(p, p.relative_to(folder))
    return buf.getvalue()


# ---- AI connection (any provider, several free) ------------------------------------------------

def provider_label(pid: str) -> str:
    p = providers.PROVIDERS[pid]
    return f"{p.label} · {p.tier}"


def connect_provider(p: providers.Provider, key: str, base_url: str, remember: bool) -> None:
    """Verify the new settings with a free model-list call; save them only if the provider accepts them."""
    try:
        key = env.validate_key(key) if key else ""
    except ValueError as e:
        st.error(str(e))
        return
    status, message, models = providers.check(p, key or None, base_url or None)
    if status == "rejected":
        st.error(message)
        return
    providers.connect(p, key or None, None, base_url or None, remember=remember)
    st.session_state.setdefault("models", {})[p.id] = models
    st.session_state.flash = message if status == "ok" else f"Saved, but: {message}"
    st.rerun()


def model_picker(p: providers.Provider, prefix: str) -> None:
    cache = st.session_state.setdefault("models", {})
    if p.id not in cache:  # load once per session (free call), so every model is in the dropdown
        cache[p.id] = providers.check(p)[2]
    models = cache.get(p.id) or []
    current = providers.model_of(p)
    options = list(dict.fromkeys([current, *models])) if current else models
    if options:
        pick = st.selectbox("Model", options, index=0, key=f"{prefix}_model",
                            help="Loaded from your provider. Bigger models follow instructions better.")
    else:
        pick = st.text_input("Model", current, key=f"{prefix}_model_txt")
    if pick and pick != current:
        providers.connect(p, None, pick, None, remember=True)
        st.rerun()
    if not models and st.button("Retry loading models", key=f"{prefix}_models"):
        status, message, found = providers.check(p)
        st.session_state.setdefault("models", {})[p.id] = found
        (st.toast if status == "ok" else st.error)(message)
        st.rerun()


def key_form(p: providers.Provider, prefix: str) -> None:
    with st.form(f"{prefix}_form_{p.id}", border=False):
        key = st.text_input("API key", type="password", placeholder="Paste your key") if p.key_var else ""
        url = ""
        if p.id in ("custom", "ollama"):
            url = st.text_input("Base URL", providers.base_url_of(p) or "", placeholder="https://.../v1")
        if HOSTED:
            remember = False
            st.caption("🔒 Your key stays in this browser session only. It is not saved on the server, other "
                       "visitors cannot use it, and it is gone when you close the tab.")
        else:
            remember = st.checkbox("Remember on this computer", value=True,
                                   help="Saves to the project's .env file (private to your user, never in git). "
                                        "Untick to keep it only until the app restarts.")
        ready = st.form_submit_button("Connect", type="primary", width="stretch")
        if ready and (key or not p.key_var):
            connect_provider(p, key, url, remember)
    if p.key_url:
        st.caption(f"{'Get a free key' if 'free' in p.tier else 'Get a key'}: [{p.key_url.split('//')[1].split('/')[0]}]({p.key_url})")


def key_panel(prefix: str) -> None:
    ids = [i for i in providers.PROVIDERS if not (HOSTED and i in HOSTED_HIDDEN)]
    active = providers.current()
    index = ids.index(active.id) if active.id in ids else 0
    pid = st.selectbox("AI provider", ids, index=index, format_func=provider_label, key=f"{prefix}_prov")
    p = providers.PROVIDERS[pid]
    if pid != active.id and p.key_var and providers.is_ready(p):
        providers.select(p)  # key saved earlier (or the host's shared key): switch straight to it
        st.rerun()
    if p.note:
        st.caption(p.note)
    if not (providers.is_ready(p) and pid == active.id):
        key_form(p, prefix)
        return
    shared = providers.is_shared(p)
    st.markdown(f"🟢 **Connected** · `{providers.masked(p)}`")
    model_picker(p, prefix)
    c1, c2 = st.columns(2)
    if c1.button("Test", key=f"{prefix}_test", width="stretch"):
        status, message, found = providers.check(p)
        st.session_state.setdefault("models", {})[p.id] = found
        (st.success if status == "ok" else st.error)(message)
    if p.key_var and not shared and c2.button("Disconnect", key=f"{prefix}_rm", width="stretch"):
        providers.disconnect(p)
        st.rerun()
    if shared:
        with st.expander("Use my own key instead"):
            st.caption("The shared key has a daily limit for everyone. Your own free key gives you your own limit.")
            key_form(p, prefix)


# ---- sidebar ---------------------------------------------------------------------------------

def generate_demo(rows: int, seed: int) -> Path:
    out = RAW_DIR / f"demo_{rows}.csv"
    cmd = [sys.executable, str(ROOT / "scripts" / "make_dirty_data.py"), "--rows", str(rows), "--seed", str(seed),
           "--out", str(out), "--truth", str(out.with_name(f"{out.stem}_truth.json"))]
    subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True)
    return out


def pick_file() -> Path | None:
    source = st.radio("Source", ["Existing file", "Upload CSV", "Generate demo data"], label_visibility="collapsed")
    if source == "Upload CSV":
        up = st.file_uploader("CSV file", type=["csv", "txt"])
        if up is not None:
            dest = RAW_DIR / "uploads" / Path(up.name).name
            if not dest.exists() or dest.stat().st_size != up.size:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(up.getbuffer())
            st.session_state.file = str(dest)
    elif source == "Generate demo data":
        rows = st.number_input("Rows", 1_000, HOSTED_MAX_DEMO_ROWS if HOSTED else 5_000_000, 50_000, step=10_000)
        seed = st.number_input("Seed", 0, 10_000, 42)
        if st.button("Generate", width="stretch"):
            with st.spinner("Generating dirty data..."):
                st.session_state.file = str(generate_demo(int(rows), int(seed)))
    else:
        files = sorted(RAW_DIR.rglob("*.csv"))
        if not files:
            st.info("No CSV yet. Upload one or generate demo data." if HOSTED else
                    "No CSV in data/raw yet. Upload one or generate demo data.")
        else:
            names = [str(p.relative_to(RAW_DIR)) for p in files]
            current = st.session_state.get("file")
            idx = names.index(str(Path(current).relative_to(RAW_DIR))) if current and Path(current) in files else 0
            st.session_state.file = str(RAW_DIR / st.selectbox("File", names, index=idx))
    f = st.session_state.get("file")
    return Path(f) if f and Path(f).exists() else None


def sidebar() -> Path | None:
    with st.sidebar:
        html("<div class='brand'>🧹 Sellout <span>Cleaner</span></div>")
        html(style.section("AI connection"))
        key_panel("side")
        html(style.section("Data"))
        if HOSTED:
            st.caption(f"Your files and results are private to this session and deleted after "
                       f"{WORKSPACE_TTL_HOURS} hours. Download what you want to keep.")
        return pick_file()


# ---- results view (shared by both modes) -----------------------------------------------------

def show_summary(log: dict[str, Any]) -> None:
    rc, rec = log["row_counts"], log["reconciliation"]
    total = log["timings"].get("total", sum(log["timings"].values()))
    html(style.kpis([
        ("Raw rows", f"{rc['raw']:,}", ""),
        ("Clean sellout rows", f"{rc['sellout']:,}", "ok"),
        ("Quarantined", f"{rc['quarantine']:,}", "warn" if rc["quarantine"] else ""),
        ("Products · markets · promos", f"{rc['product']} · {rc['market']} · {rc['promo']}", ""),
        ("Issues found", f"{sum(log['issue_counts'].values()):,}", ""),
        ("Totals check", "Passed" if rec["passed"] else "FAILED", "ok" if rec["passed"] else "err"),
        ("Run time", f"{total:.1f} s", ""),
    ]))
    if not rec["passed"]:
        st.error("Totals check failed: raw rows, units or value do not equal output + quarantine. See Reconciliation.")


def show_reconciliation(rec: dict[str, Any]) -> None:
    rows = [{"measure": m, "raw": rec["raw"][m],
             "added by fixes": rec["units_added_by_fixes"] if m == "units" else 0.0,
             "clean output": rec["output"][m], "quarantine": rec["quarantine"][m],
             "difference": rec["differences"][m]} for m in ("rows", "units", "value")]
    st.dataframe(pl.DataFrame(rows), hide_index=True, width="stretch")


def show_issues(folder: Path) -> pl.DataFrame:
    issues = pl.read_parquet(folder / "issues.parquet")
    summary = (issues.group_by("rule_id", "table", "severity").agg(pl.len().alias("count"))
               .sort("count", descending=True))
    if summary.is_empty():
        st.success("No issues left.")
        return issues
    left, right = st.columns([3, 2])
    left.dataframe(summary, hide_index=True, width="stretch")
    right.bar_chart(summary, x="rule_id", y="count", color="severity", horizontal=True)
    rule = st.selectbox("Show examples for", summary["rule_id"].to_list(), key=f"rule_{folder.name}")
    st.dataframe(issues.filter(pl.col("rule_id") == rule).head(200), hide_index=True, width="stretch")
    return issues


def show_ground_truth(issues: pl.DataFrame, folder: Path, truth: Path) -> None:
    market = pl.read_parquet(folder / "market.parquet")
    scores = evaluate.compare(issues, json.loads(truth.read_text(encoding="utf-8")), market)
    perfect = scores.filter((pl.col("precision") == 1) & (pl.col("recall") == 1)).height
    st.caption(f"This file has a list of injected issues ({truth.name}). "
               f"{perfect} of {scores.height} issue types detected perfectly.")
    st.dataframe(scores, hide_index=True, width="stretch")


def show_table_explorer(folder: Path) -> None:
    name = st.radio("Table", ["sellout", "product", "market", "promo", "quarantine"], horizontal=True,
                    key=f"table_{folder.name}")
    if name == "sellout":
        lf = writer.scan_sellout(folder)
        c1, c2 = st.columns(2)
        prod = c1.text_input("Filter product_id", key=f"p_{folder.name}")
        mkt = c2.text_input("Filter market_id", key=f"m_{folder.name}")
        if prod:
            lf = lf.filter(pl.col("product_id") == prod.strip())
        if mkt:
            lf = lf.filter(pl.col("market_id") == mkt.strip())
        df = lf.sort("week_start_date").head(1000).collect()
    else:
        df = pl.read_parquet(folder / f"{name}.parquet")
        if name == "quarantine" and "rule_id" in df.columns:
            pick = st.multiselect("Reason", sorted(df["rule_id"].unique().to_list()), key=f"q_{folder.name}")
            df = df.filter(pl.col("rule_id").is_in(pick)) if pick else df
        df = df.head(1000)
    where = "the Download tab" if HOSTED or not folder.is_relative_to(ROOT) else f"{folder.relative_to(ROOT)}/"
    st.caption(f"Showing up to 1,000 rows. Full files are in {where}.")
    st.dataframe(df, hide_index=True, width="stretch")


def show_downloads(mode: str, folder: Path) -> None:
    c = st.columns(3)
    c[0].download_button("All outputs (.zip)", zip_outputs(folder), f"sellout_{mode}_outputs.zip", "application/zip",
                         width="stretch", key=f"dl_zip_{mode}")
    c[1].download_button("DQ report (.html)", (folder / "dq_report.html").read_bytes(), f"dq_report_{mode}.html",
                         "text/html", width="stretch", key=f"dl_report_{mode}")
    c[2].download_button("Quarantine (.csv)", pl.read_parquet(folder / "quarantine.parquet").write_csv(),
                         f"quarantine_{mode}.csv", "text/csv", width="stretch", key=f"dl_quarantine_{mode}")


def show_results(mode: str, log: dict[str, Any], file: Path) -> None:
    folder = out_dir(mode)
    show_summary(log)
    tabs = st.tabs(["Issues", "Clean tables", "Reconciliation & fixes", "Timings", "DQ report", "Download"])
    with tabs[0]:
        issues = show_issues(folder)
        truth = truth_for(file)
        if truth:
            with st.expander("Detections vs injected ground truth"):
                show_ground_truth(issues, folder, truth)
    with tabs[1]:
        show_table_explorer(folder)
    with tabs[2]:
        show_reconciliation(log["reconciliation"])
        fixes = pl.DataFrame([{**f, "detail": json.dumps(f.get("detail") or {}, ensure_ascii=False)} for f in log["fixes"]])
        st.dataframe(fixes, hide_index=True, width="stretch")
    with tabs[3]:
        t = pl.DataFrame({"step": list(log["timings"]), "seconds": list(log["timings"].values())})
        st.bar_chart(t.filter(pl.col("step") != "total"), x="step", y="seconds", horizontal=True)
        ct = log.get("check_timings") or {}
        if ct:
            st.dataframe(pl.DataFrame({"check": list(ct), "seconds": list(ct.values())}).sort("seconds", descending=True),
                         hide_index=True, width="stretch")
    with tabs[4]:
        # data: URL = opaque origin, so the report (which contains file values) is isolated from the app
        report = base64.b64encode((folder / "dq_report.html").read_bytes()).decode()
        st.iframe(f"data:text/html;base64,{report}", height=900)
    with tabs[5]:
        show_downloads(mode, folder)


# ---- tab: data & mapping ---------------------------------------------------------------------

def tab_data(file: Path) -> None:
    rules = load_rules()
    lf, fmt = scan_raw(file, rules.ingest, OUT_DIR / ".cache")
    cols = raw_columns(lf)
    html(style.kpis([("File size", f"{file.stat().st_size / 1e6:,.1f} MB", ""), ("Encoding", fmt.encoding, ""),
                     ("Delimiter", repr(fmt.delimiter), ""), ("Columns", len(cols), "")]))
    html(style.section("First 100 rows"))
    st.dataframe(lf.head(100).collect(), hide_index=True, width="stretch")
    html(style.section("Column mapping"))
    mapping_editor(file, cols)


def saved_mapping() -> dict[str, str]:
    try:
        return load_mapping(MAPPING_PATH).columns if MAPPING_PATH.exists() else {}
    except Exception:  # noqa: BLE001 - an unreadable mapping counts as no mapping
        return {}


def mapping_fits(file: Path) -> bool:
    from src.agent import tools

    raw_cols = raw_columns(scan_raw(file, load_rules().ingest, OUT_DIR / ".cache")[0])
    return not tools.validate_mapping({r: c for r, c in saved_mapping().items() if r in raw_cols}, raw_cols)


def ai_map(file: Path, raw_cols: list[str]) -> None:
    """Ask the Profiler agent for a mapping of this file's columns (one AI call) and save it."""
    from types import SimpleNamespace

    from src.agent import graph, tools

    with st.spinner("The Profiler is reading the columns and sample values..."):
        try:
            proposal = graph.propose_mapping(SimpleNamespace(llm=providers.make_llm("low")),
                                             tools.profile_columns(file, load_rules()), raw_cols)
        except Exception as e:  # noqa: BLE001 - provider errors, or no valid mapping after retries
            st.error(providers.explain_error(e) if getattr(e, "status_code", None) else f"Could not map: {e}")
            return
    tools.write_mapping(tools.mapping_from_columns(proposal.columns), MAPPING_PATH,
                        f"# Proposed by the Profiler agent for {file.name}. Review before trusting it.\n")
    unsure = f" It was unsure about {len(proposal.low_confidence)} column(s): check them." if proposal.low_confidence else ""
    st.session_state.flash = f"The Profiler mapped {len(proposal.columns)} columns.{unsure}"
    st.rerun()


def mapping_editor(file: Path, raw_cols: list[str]) -> None:
    from src.agent import tools

    mapped = saved_mapping()
    fits = not tools.validate_mapping({r: c for r, c in mapped.items() if r in raw_cols}, raw_cols)
    guess = None if fits else tools.guess_mapping(raw_cols)
    if guess:  # every header is a known name: map it with code, no AI request
        tools.write_mapping(tools.mapping_from_columns(guess), MAPPING_PATH,
                            f"# Matched by column names (no AI call) for {file.name}.\n")
        st.session_state.flash = f"Recognised all {len(guess)} column names, no AI needed. Check the mapping below."
        st.rerun()
    if not fits:
        with st.container(border=True):
            st.markdown("**This file's column names don't match the saved mapping yet**, so the tool doesn't know "
                        "which column is the product, the store, the week, the units or the sales value.")
            if has_key():
                st.caption("Let the Profiler agent read the column names and sample values and propose a mapping "
                           "(one AI call), or set the 'standard column' for each row below yourself.")
                if st.button("🧭 Map columns with AI", type="primary"):
                    ai_map(file, raw_cols)
            else:
                st.caption("Connect an AI provider in the sidebar and the Profiler can map them for you. Or set the "
                           "'standard column' for each row below yourself. Running the AI agents also maps them.")
    choices = ["ignore", *tools.STANDARD_COLUMNS]
    df = pl.DataFrame({"raw column": raw_cols, "standard column": [mapped.get(c, "ignore") for c in raw_cols],
                       "meaning": [tools.STANDARD_COLUMNS.get(mapped.get(c, ""), "") for c in raw_cols]})
    st.caption("Which raw column feeds which standard column. Edit and save, or let the Profiler agent propose one "
               "(AI agents tab, 'Re-map columns with AI').")
    edited = st.data_editor(df, hide_index=True, width="stretch", disabled=["raw column", "meaning"],
                            column_config={"standard column": st.column_config.SelectboxColumn(options=choices)})
    cols = {r["raw column"]: r["standard column"] for r in edited.to_dicts() if r["standard column"] != "ignore"}
    errors = tools.validate_mapping(cols, raw_cols)
    if errors:
        st.warning("Not ready to clean yet:\n" + "\n".join(f"- {e}" for e in errors))
    if cols != mapped and st.button("Save mapping", disabled=bool(errors)):
        tools.write_mapping(tools.mapping_from_columns(cols), MAPPING_PATH, "# Edited in the Sellout Cleaner app.\n")
        st.session_state.flash = ("Saved the mapping for this session." if HOSTED else
                                  "Saved config/schema_mapping.yaml (previous version kept as .bak).")
        st.rerun()


# ---- tab: rules-only cleaning ----------------------------------------------------------------

def tab_rules(file: Path) -> None:
    st.write("Runs every check and safe fix with plain code. No AI, no API key, no cost. "
             "Error rows go to quarantine; warnings are kept and reported.")
    if not mapping_fits(file):
        st.info("First tell the tool what each column means: open the **📄 Data** tab and use **Map columns with "
                "AI**, or set the mapping yourself. (The AI agents tab maps them automatically.)")
        return
    if st.button("Run cleaning", type="primary"):
        with st.spinner("Cleaning..."):
            try:
                pipeline.run(file, out_dir("rules"), load_rules(), load_mapping(MAPPING_PATH))
            except Exception as e:  # noqa: BLE001 - show any pipeline error in the UI
                st.error(f"Run failed: {e}")
                return
    log = load_log("rules", file)
    if log:
        show_results("rules", log, file)


# ---- tab: AI agents --------------------------------------------------------------------------

ORDER = [a[0] for a in style.AGENTS]


def start_agent(file: Path, effort: str, max_iter: int, remap: bool, explain: bool, reuse: bool) -> None:
    from langgraph.checkpoint.memory import MemorySaver

    from src.agent import graph
    from src.agent.memory import Memory, signature

    decisions: dict[str, bool] = {}
    session = graph.Session(
        file=file, out_dir=out_dir("agent"), mapping_path=MAPPING_PATH, rules=load_rules(), llm=providers.make_llm(effort),
        approver=lambda op: decisions.get(signature(op["name"], op["params"]), False),
        memory=Memory(MEMORY_PATH), force_profile=remap, explain=explain, reuse_plan=reuse,
        max_iterations=max_iter)
    app = graph.build_graph(session, MemorySaver(), interrupt_before=["approval"])
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex}, "recursion_limit": 50}
    st.session_state.agent = {"session": session, "app": app, "cfg": cfg, "decisions": decisions, "file": str(file),
                              "steps": {"profiler": "running"}, "error": None}
    advance_agent({"events": []})


def pending_ops(agent: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    s = agent["session"]
    return [(i, p["op"]) for i, p in enumerate(s.ws.plan if s.ws else [])
            if p["status"] == "pending" and s.memory.decision(p["op"]["name"], p["op"]["params"]) is None]


def next_step(agent: dict[str, Any], node: str, update: dict[str, Any]) -> str | None:
    """Which agent works next, so the pipeline view can light it up."""
    if node == "validator":
        s = agent["session"]
        retry = not update["validation"]["passed"] and agent["app"].get_state(agent["cfg"]).values.get("iteration", 0) < s.max_iterations
        return "cleaner" if retry else "writer"
    i = ORDER.index(node)
    return ORDER[i + 1] if i + 1 < len(ORDER) else None


def advance_agent(inp: dict[str, Any] | None) -> None:
    """Run the graph until it finishes or pauses for human approval, updating the pipeline view live."""
    agent = st.session_state.agent
    app, cfg, steps = agent["app"], agent["cfg"], agent["steps"]
    view = st.session_state.get("flow_slot") or st.empty()
    try:
        while True:
            view.markdown(style.flow(steps), unsafe_allow_html=True)
            for chunk in app.stream(inp, cfg, stream_mode="updates"):
                for node, update in chunk.items():
                    if node not in ORDER:
                        continue
                    steps[node] = "fail" if node == "validator" and not update["validation"]["passed"] else "done"
                    upcoming = next_step(agent, node, update)
                    if upcoming:
                        steps[upcoming] = "running"
                    view.markdown(style.flow(steps), unsafe_allow_html=True)
            inp = None
            if not app.get_state(cfg).next:
                return
            if pending_ops(agent):
                steps["approval"] = "wait"
                return
    except Exception as e:  # noqa: BLE001 - API/auth errors are shown in the UI, not as a crash
        agent["error"] = providers.explain_error(e)
        for node, state in steps.items():
            if state == "running":
                steps[node] = "fail"


def show_pending(agent: dict[str, Any]) -> None:
    from src.agent.memory import signature

    s = agent["session"]
    st.warning("The Cleaner wants to make risky changes. Nothing is applied until you decide.")
    choices: dict[int, bool] = {}
    for i, op in pending_ops(agent):
        with st.container(border=True):
            st.markdown(f"**{op['name']}**")
            st.write(op["rationale"])
            st.json(op["params"], expanded=False)
            rows = op["params"].get("row_nrs")
            if rows:
                st.dataframe(s.ws.prepared.ctx.fixed.filter(pl.col("row_nr").is_in(rows)).head(50), hide_index=True)
            choices[i] = st.radio("Decision", ["Reject", "Approve"], horizontal=True,
                                  key=f"op_{agent['cfg']['configurable']['thread_id']}_{i}") == "Approve"
    if st.button("Submit decisions and continue", type="primary"):
        for i, ok in choices.items():
            op = s.ws.plan[i]["op"]
            agent["decisions"][signature(op["name"], op["params"])] = ok
        agent["steps"]["approval"] = "running"
        advance_agent(None)
        st.rerun()


def event_text(e: dict[str, Any]) -> str:
    node = e["node"]
    if node == "profiler":
        low = e.get("low_confidence") or []
        return e.get("action", "") + (f" · {len(low)} low-confidence column(s) to review" if low else "")
    if node == "modeler":
        return f"{e.get('rows', 0):,} rows loaded · {e.get('issues', 0):,} issues found · tables: {e.get('tables')}"
    if node == "cleaner":
        return f"{len(e.get('tool_calls', []))} tool call(s). {e.get('summary', '')}"
    if node == "approval":
        d = e.get("decisions", [])
        return ", ".join(f"{x['fix']}: {'approved' if x['approved'] else 'rejected'} (by {x['by']})" for x in d) or "nothing to decide"
    if node == "validator":
        return "all fixes verified" if e.get("passed") else "sent back: " + "; ".join(e.get("failures", []))
    if node == "writer":
        return f"outputs written · totals check {'passed' if e.get('reconciliation') else 'FAILED'}"
    return e.get("action", "") + (f" ({e['error']})" if e.get("error") else "")


def show_activity(info: dict[str, Any]) -> None:
    icons = {a[0]: f"{a[1]} {a[2]}" for a in style.AGENTS}
    usage = info.get("usage") or {}
    for e in info.get("events", []):
        tokens = usage.get(e["node"])
        with st.expander(f"{icons.get(e['node'], e['node'])} · {event_text(e)[:110]}"):
            st.write(event_text(e))
            for call in e.get("tool_calls", []):
                st.markdown(f"{'⚠️' if call.get('error') else '🔧'} `{call['tool']}`")
                st.json(call["input"], expanded=False)
            if tokens and e["node"] in ("profiler", "cleaner", "reporter"):
                st.caption(f"{tokens['calls']} API call(s) in total for this agent · "
                           f"{tokens['input_tokens'] + tokens['cache_read_input_tokens'] + tokens['cache_creation_input_tokens']:,} "
                           f"tokens in · {tokens['output_tokens']:,} tokens out")


def show_agent_run(info: dict[str, Any]) -> None:
    """What the agents did in a finished run (from run_log.json)."""
    if info.get("brief"):
        html(style.brief(info["brief"]))
    plan = info.get("plan") or []
    left, right = st.columns([3, 2])
    with left:
        html(style.section("Fix plan"))
        if plan:
            st.dataframe(pl.DataFrame([{"status": p["status"], "fix": p["op"]["name"], "risky": p["op"].get("risky", False),
                                        "why": p["op"].get("rationale", "")} for p in plan]), hide_index=True, width="stretch")
        else:
            st.caption("The Cleaner proposed no fixes.")
        for f in (info.get("validation") or {}).get("failures", []):
            st.error(f)
    with right:
        html(style.section("AI usage (tokens)"))
        usage = info.get("usage") or {}
        if usage:
            st.dataframe(pl.DataFrame([{"agent": k, "calls": v["calls"],
                                        "in": v["input_tokens"] + v["cache_read_input_tokens"] + v["cache_creation_input_tokens"],
                                        "of which cached": v["cache_read_input_tokens"],
                                        "out": v["output_tokens"]} for k, v in usage.items()]), hide_index=True, width="stretch")
        else:
            st.caption("No AI calls were needed.")
    html(style.section("What each agent did"))
    show_activity(info)


def agent_controls(file: Path) -> None:
    c = st.columns([1, 1, 1.4, 1.4])
    effort = c[0].selectbox("Effort", ["low", "medium", "high"], index=1,
                            help="Claude only: higher = more thorough reasoning, slower and more tokens.")
    max_iter = c[1].number_input("Max cleaner rounds", 1, 5, 2)
    remap = c[2].checkbox("Re-map columns with AI", help="Ask the Profiler to propose a new column mapping.")
    explain = c[3].checkbox("Write a plain-language summary", value=True, help="The Reporter agent (one extra AI call).")
    reuse = st.checkbox("Reuse the earlier fix plan when the issues are identical (saves the Cleaner's AI calls)",
                        value=True, help="Same file, mapping and rules as a past successful run: its fixes are replayed "
                                         "and re-verified by the Validator. Untick to make the Cleaner decide again.")
    p = providers.current()
    where = "stays on this computer" if p.id == "ollama" else f"is sent to {p.label}"
    st.caption(f"Using **{p.label}** · `{providers.model_of(p)}`. Column samples and issue examples from this file "
               f"{where}." + ("" if p.id in ("ollama", "anthropic", "openai") else
                              " Free tiers may let the provider use what you send to improve their models: check "
                              "their terms before sending confidential sales data."))
    if st.button("▶ Run the agents", type="primary", disabled=not has_key()):
        try:
            start_agent(file, effort, int(max_iter), remap, explain, reuse)
        except Exception as e:  # noqa: BLE001 - e.g. provider not configured
            st.error(f"Could not start: {e}")
            return
        st.rerun()


def tab_agent(file: Path) -> None:
    agent = st.session_state.get("agent")
    live = agent if agent and agent["file"] == str(file) else None
    log = load_log("agent", file)
    done_states = dict.fromkeys(ORDER, "done") if log and not live else {}
    st.session_state.flow_slot = st.empty()
    st.session_state.flow_slot.markdown(style.flow(live["steps"] if live else done_states), unsafe_allow_html=True)
    if not has_key():
        with st.container(border=True):
            st.markdown("**Connect an AI provider to switch the agents on.** Gemini, Groq, OpenRouter and "
                        "Mistral have free keys; Ollama runs free on your own computer.")
            key_panel("agent")
    agent_controls(file)
    if live and live["error"]:
        st.error(f"The run stopped: {live['error']}")
        st.caption("Nothing was written. Change the model or provider in the sidebar if needed, then run again.")
        return
    if live and live["app"].get_state(live["cfg"]).next:
        show_pending(live)
        return
    if log:
        show_agent_run(log.get("agent") or {})
        show_results("agent", log, file)


# ---- tab: ask your data ----------------------------------------------------------------------

SUGGESTIONS = ["Top 5 products by sales value", "Which markets have the most quarantined rows?",
               "Total units per month", "Which promo type sold the most units?"]


@st.cache_resource(max_entries=4, show_spinner=False)
def analyst_db(folder: str, stamp: float) -> Any:  # noqa: ARG001 - stamp invalidates the cache after a new run
    from src.agent import analyst

    return analyst.connect(Path(folder))


def ask(folder: Path, question: str, chat: list[dict[str, Any]]) -> None:
    from src.agent import analyst

    con = analyst_db(str(folder), (folder / "run_log.json").stat().st_mtime).cursor()
    history = [(m["q"], m["a"]) for m in chat[-6:]]
    try:
        answer = analyst.Analyst(con, providers.make_llm("low")).ask(question, history)
        chat.append({"q": question, "a": answer.text, "queries": answer.queries})
    except Exception as e:  # noqa: BLE001 - show API errors in the chat
        st.error(providers.explain_error(e))


def show_query_result(q: dict[str, Any]) -> None:
    try:
        st.dataframe(pl.DataFrame(q["rows"], schema=q["columns"], orient="row", strict=False), hide_index=True,
                     width="stretch")
    except Exception:  # noqa: BLE001 - odd shapes (duplicate column names, mixed types) still get shown
        st.json({"columns": q["columns"], "rows": q["rows"][:50]}, expanded=False)


def show_chat(chat: list[dict[str, Any]]) -> None:
    for m in chat:
        st.chat_message("user").write(m["q"])
        with st.chat_message("assistant", avatar="📊"):
            st.markdown(m["a"])
            ok = [q for q in m["queries"] if "error" not in q]
            if ok:
                with st.expander(f"How I got this ({len(m['queries'])} SQL quer{'y' if len(m['queries']) == 1 else 'ies'})"):
                    for q in m["queries"]:
                        st.code(q["sql"], language="sql")
                        if "error" in q:
                            st.caption(f"Rejected: {q['error']}")
                        else:
                            show_query_result(q)


def tab_ask(file: Path) -> None:
    runs = {m: label for m, label in (("agent", "AI agents run"), ("rules", "Rules-only run")) if load_log(m, file)}
    if not runs:
        st.info("Clean this file first (Clean or AI agents tab). Then you can ask questions about the result.")
        return
    st.write("Ask anything about the cleaned data. The Analyst agent writes SQL, runs it read-only on the output "
             "tables and answers with the numbers. It cannot change or export anything.")
    if not has_key():
        with st.container(border=True):
            st.markdown("**Connect an AI provider to ask questions** (free options available)")
            key_panel("ask")
        return
    mode = st.radio("Answer from", list(runs), format_func=runs.get, horizontal=True) if len(runs) > 1 else next(iter(runs))
    folder = out_dir(mode)
    chat = st.session_state.setdefault("chat", {}).setdefault(f"{mode}:{file}", [])
    show_chat(chat)
    cols = st.columns(len(SUGGESTIONS))
    picked = next((s for c, s in zip(cols, SUGGESTIONS, strict=True) if c.button(s, key=f"sug_{s}", width="stretch")), None)
    question = st.chat_input("Ask about the cleaned data...") or picked
    if question:
        with st.spinner("Analyst is querying the data..."):
            ask(folder, question, chat)
        st.rerun()


# ---- tab: how it works -----------------------------------------------------------------------

HOW = """
**In one sentence:** code does the checking and fixing, AI makes the judgment calls, you approve anything risky,
and a totals check proves no data was lost.

1. **Read** the file (encoding and separator are detected) without loading it all into memory.
2. **Map** its columns to standard names. Edit the mapping yourself or let the Profiler propose it.
3. **Standardize**: trim spaces, turn "N/A" and "-" into real blanks, convert every week format to its Monday,
   convert cases to units.
4. **Split** into four tables: products, markets, promos, and weekly sales that link to them.
5. **Check** everything: duplicates, conflicting numbers for the same product-store-week, missing weeks,
   negative units, unusual prices, promos that are used but never defined.
6. **Fix**: the Cleaner picks fixes from a fixed list of functions. It cannot edit data directly. Merges and
   deletions wait for your approval, and your answer is remembered.
7. **Verify**: the Validator re-runs every check. A fix that did not work goes back to the Cleaner.
8. **Write** the clean tables. Rows with errors go to quarantine with the reason, never deleted.
9. **Prove**: rows, units and value in output + quarantine must equal the raw file exactly.
10. **Explain**: the Reporter writes a plain-language summary, and the Analyst answers your questions with SQL.
"""


def tab_how() -> None:
    html(style.flow())
    st.markdown(HOW)


# ---- page ------------------------------------------------------------------------------------

def main() -> None:
    file = sidebar()
    flash = st.session_state.pop("flash", None)
    if flash:
        st.toast(flash)
    p = providers.current()
    chips = [(f"AI: {p.label} · {providers.model_of(p)}", "ok") if has_key() else ("AI not connected", "warn")]
    if file:
        chips += [(file.name, ""), (f"{file.stat().st_size / 1e6:,.1f} MB", "")]
    html(style.hero("Sellout Cleaner", "Messy retailer sales file in. Four clean tables, a quarantine and a proof "
                    "that nothing was lost out.", chips))
    if file is None:
        st.info("Choose, upload or generate a sellout CSV in the sidebar to start.")
        tab_how()
        return
    data, rules_tab, agent_tab, ask_tab, how = st.tabs(
        ["📄 Data", "⚙️ Clean (rules only)", "🤖 AI agents", "💬 Ask your data", "❓ How it works"])
    with data:
        tab_data(file)
    with rules_tab:
        tab_rules(file)
    with agent_tab:
        tab_agent(file)
    with ask_tab:
        tab_ask(file)
    with how:
        tab_how()


main()
