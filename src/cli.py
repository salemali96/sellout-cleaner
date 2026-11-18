"""Command line entry point.

    uv run python -m src.cli run --file data/raw/sellout.csv [--out data/output] [--truth data/raw/ground_truth.json]
    uv run python -m src.cli agent --file data/raw/sellout.csv [--approve ask|all|none] [--remap]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

from src import evaluate, pipeline
from src.config import load_mapping, load_rules
from src.env import load_env


def _print_run(log: dict) -> None:
    print(f"\ninput: {log['input']['file']}  (encoding={log['input']['encoding']!r}, "
          f"delimiter={log['input']['delimiter']!r})")
    print("\nTimings (s)")
    for k, v in log["timings"].items():
        print(f"  {k:16s} {v:8.3f}")
    print("\nRow counts")
    for k, v in log["row_counts"].items():
        print(f"  {k:16s} {v:>10,}")
    print("\nIssues")
    for k, v in log["issue_counts"].items():
        print(f"  {k:34s} {v:>8,}")
    print("\nFixes")
    for f in log["fixes"]:
        print(f"  {f['name']:20s} {f['rows_affected']:>8,} rows  {f['detail'] or ''}")
    rec = log["reconciliation"]
    print(f"\nReconciliation: {'PASSED' if rec['passed'] else 'FAILED'}")
    for m in ["rows", "units", "value"]:
        added = f" + fixes {rec['units_added_by_fixes']:,.0f}" if m == "units" else ""
        print(f"  {m:6s} raw {rec['raw'][m]:>18,.2f}{added} = output {rec['output'][m]:>18,.2f}"
              f" + quarantine {rec['quarantine'][m]:>14,.2f}   diff {rec['differences'][m]}")


def cmd_run(args: argparse.Namespace) -> int:
    rules = load_rules(args.rules)
    mapping = load_mapping(args.mapping)
    result = pipeline.run(args.file, args.out, rules, mapping)
    _print_run(result.log)
    print(f"\noutputs in {args.out}  (dq_report.html, run_log.json)")
    if args.truth:
        truth = json.loads(args.truth.read_text(encoding="utf-8"))
        scores = evaluate.compare(result.issues, truth, result.prepared.ctx.tables["market"])
        with pl.Config(tbl_rows=50, tbl_width_chars=160, tbl_hide_dataframe_shape=True):
            print("\nGround truth comparison\n", scores)
    return 0 if result.log["reconciliation"]["passed"] else 2


def _approver(mode: str):  # noqa: ANN202
    def ask(op: dict) -> bool:
        if not sys.stdin.isatty():
            print(f"  [non-interactive] rejecting risky fix {op['name']}")
            return False
        print(f"\nRISKY FIX: {op['name']}\n  params: {json.dumps(op['params'], ensure_ascii=False)[:500]}"
              f"\n  rationale: {op['rationale']}")
        return input("  approve? [y/N] ").strip().lower() in ("y", "yes")

    return {"ask": ask, "all": lambda op: True, "none": lambda op: False}[mode]


def cmd_agent(args: argparse.Namespace) -> int:
    from src.agent import graph, providers
    from src.agent.memory import Memory

    provider = providers.PROVIDERS[args.provider] if args.provider else providers.current()
    print(f"AI provider: {provider.label} · model {args.model or providers.model_of(provider)}")

    session = graph.Session(
        file=args.file, out_dir=args.out, mapping_path=args.mapping, rules=load_rules(args.rules),
        llm=providers.make_llm(args.effort, provider, args.model), approver=_approver(args.approve), memory=Memory(args.memory),
        force_profile=args.remap, max_iterations=args.max_iterations,
    )
    state = graph.run_agent(session)
    print(f"\nmapping: {state.get('mapping_source')}  |  cleaner rounds: {state.get('iteration')}  |  "
          f"status: {state.get('status')}")
    for i, p in enumerate(state.get("plan", [])):
        print(f"  [{i}] {p['status']:9s} {p['op']['name']:24s} {p['op']['rationale']}")
    for summary in state.get("summaries", []):
        print(f"\ncleaner: {summary}")
    if not state["validation"]["passed"]:
        print("\nvalidation failures:", *state["validation"]["failures"], sep="\n  ")
    assert session.result is not None
    _print_run(session.result.log)
    return 0 if session.result.log["reconciliation"]["passed"] else 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sellout")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the cleaning pipeline")
    r.add_argument("--file", type=Path, required=True)
    r.add_argument("--out", type=Path, default=Path("data/output"))
    r.add_argument("--rules", type=Path, default=None)
    r.add_argument("--mapping", type=Path, default=None)
    r.add_argument("--truth", type=Path, default=None, help="ground_truth.json to score detections against")
    r.set_defaults(func=cmd_run)
    a = sub.add_parser("agent", help="run the LLM agent (profiler -> modeler -> cleaner -> validator)")
    a.add_argument("--file", type=Path, required=True)
    a.add_argument("--out", type=Path, default=Path("data/output"))
    a.add_argument("--rules", type=Path, default=None)
    a.add_argument("--mapping", type=Path, default=Path("config/schema_mapping.yaml"),
                   help="written by the profiler (previous version kept as .bak)")
    a.add_argument("--remap", action="store_true", help="re-profile even if the mapping already fits the file")
    a.add_argument("--approve", choices=["ask", "all", "none"], default="ask", help="how risky fixes are decided")
    a.add_argument("--memory", type=Path, default=Path("data/agent_memory.json"))
    a.add_argument("--max-iterations", type=int, default=2)
    a.add_argument("--provider", choices=["gemini", "groq", "openrouter", "mistral", "huggingface", "ollama", "anthropic", "openai",
                                          "custom"], default=None, help="default: LLM_PROVIDER in .env")
    a.add_argument("--model", default=None, help="default: <PROVIDER>_MODEL in .env, else the provider's default")
    a.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default="medium")
    a.set_defaults(func=cmd_agent)
    args = ap.parse_args(argv)
    load_env()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
