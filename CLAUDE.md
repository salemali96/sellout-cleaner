# Sellout Cleaning Agent

## What this project does
Takes a large, messy, flat sellout CSV and turns it into 4 clean Parquet tables:
- `product` (dimension): one row per product
- `market` (dimension): one row per market/store/channel
- `promo` (dimension): one row per promo event
- `sellout` (fact): one row per product x market x week, partitioned by year/week

Bad rows are never silently dropped. They go to `quarantine.parquet` with `rule_id` and `reason`.
Every run produces a data quality report (`dq_report.html`) and a JSON run log.

## Architecture (hybrid)
- **Deterministic code** (Polars/DuckDB) does all heavy checks and transformations. Fast, testable, no LLM.
- **The LLM agent** only does judgment work: column mapping, deciding the table split, choosing fixes,
  explaining issues, and asking the human before risky fixes.
- Agent flow (LangGraph): profiler -> modeler -> cleaner -> validator (validator can send work back to cleaner).

## Tech stack
- Python 3.11+, managed with `uv`
- Polars (lazy `scan_csv` for big files), DuckDB for SQL checks, PyArrow for Parquet
- Pydantic for schemas and config models
- LangGraph + Anthropic API for the agent (API key read from `ANTHROPIC_API_KEY` env var, never hard-coded)
- Streamlit for the UI, pytest for tests

## Project layout
```
config/schema_mapping.yaml   # raw column -> standard column (agent generates, human edits)
config/rules.yaml            # thresholds, dedup strategy, week start day, enabled rules
src/ingest.py                # encoding/delimiter detection, chunked reading
src/split.py                 # wide file -> 4 tables
src/checks/                  # generic, product, market, promo, sellout, cross_table
src/fixes.py                 # fix functions the agent can call
src/writer.py                # parquet, partitioning, quarantine
src/report.py                # DQ report
src/agent/                   # tools.py, graph.py, memory.py
app.py                       # Streamlit UI
tests/                       # pytest, uses generated dirty data with known issues
```

## Coding rules
- Every check is a function returning a Polars DataFrame of issues with columns:
  `rule_id, table, severity (error|warning), row_key, column, value, reason`.
- Checks never mutate data. Fixes are separate functions and every fix is logged.
- Never load the full CSV eagerly. Use lazy Polars and only `collect()` at the end.
- Reconciliation is mandatory: raw totals (rows, units, value) = output + quarantine.
- Thresholds live in `config/rules.yaml`, never hard-coded in check functions.
- Write a pytest test for every check using a small dirty fixture with the issue injected.
- Type hints everywhere. Keep functions small.

## Business rules
- Week = ISO week. `week_start_date` is the Monday. Normalize all formats (2025-W03, 202503, dates).
- Sellout primary key: (product_id, market_id, week_start_date).
- Duplicate key with identical values -> drop. With different values -> strategy from rules.yaml
  (default: quarantine both).
- Negative units/value -> flag as warning (possible returns), do not drop.
- avg_price = value / units; flag if outside +/- 50% of the product median price.
- Product/market merges and row deletions are RISKY: require human approval.

## Commands
- Install: `uv sync`
- Tests: `uv run pytest`
- Generate test data: `uv run python scripts/make_dirty_data.py --rows 500000`
- Run pipeline: `uv run python -m src.cli run --file data/raw/sellout.csv`
- Score against injected issues: add `--truth data/raw/ground_truth.json` to the run command
- Run agent: `uv run python -m src.cli agent --file data/raw/sellout.csv` (needs `ANTHROPIC_API_KEY`)
- AI provider + key: set in the app sidebar, or in `.env` (git-ignored, loaded by `src/env.py`): `LLM_PROVIDER` plus that
  provider's key var. CLI: `--provider gemini|groq|openrouter|mistral|huggingface|ollama|anthropic|openai|custom --model ...`
- UI: `uv run streamlit run app.py`
- Hosted mode (`SELLOUT_HOSTED=1`, set in `Dockerfile`): per-visitor workspace in the temp dir (24h TTL) and per-visitor
  AI settings via `providers.use_session` (host's key from env/secrets is the default, never shown). Deploy: `DEPLOY.md`
  (Hugging Face Docker Space, port 7860).

## Current phase
Phase 2: LLM agent (Phase 1 done: ingestion, split, core checks, Parquet output, quarantine, reconciliation, tests).
- `src/agent/graph.py`: LangGraph flow profiler -> modeler -> cleaner -> approval -> validator -> writer
  (validator routes back to cleaner on failure, up to `max_iterations`).
- The agent only calls tools in `src/agent/tools.py`. Fix tools append `FixOp`s to a plan; data changes happen
  only in `src.fixes.apply_plan`, which raises `ApprovalRequired` for unapproved risky ops.
- Human decisions on risky fixes are remembered in `data/agent_memory.json` (`src/agent/memory.py`), as are fix plans
  of successful runs keyed by an issue fingerprint: identical issues replay the plan with no LLM call (`reuse_plan`).
- Column mapping by name first: `tools.guess_mapping` (HEADER_SYNONYMS) maps known headers with no AI call; the Profiler
  LLM runs only for unknown/ambiguous headers or `--remap`.
- Token economy (`Session.lean`, default on): the cleaner's first message carries code-computed evidence
  (`tools.evidence`), fixes take `all_flagged` instead of long id lists, and a turn of only successful fixes ends
  the cleaner without a follow-up request.
- Models: any provider via `src/agent/providers.py`. Claude (`claude-opus-5-5`, native SDK) or any OpenAI-compatible API
  (Gemini, Groq, OpenRouter, Mistral, Hugging Face, Ollama, custom) through `OpenAICompatClient`, which translates the Anthropic
  Messages shape so agent code is provider-neutral. Tests use scripted fake clients for both shapes.
- Agents: profiler, cleaner (LLM) + reporter (plain-language brief) + analyst (`src/agent/analyst.py`, read-only
  sandboxed DuckDB SQL over outputs, used by the app's 'Ask your data' tab).
