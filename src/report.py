"""Data quality report (dq_report.html) built from the run log and issues."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import polars as pl

CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--muted:#6b6b70;--line:#e3e3e8;--err:#b42318;--warn:#b54708;--ok:#067647;--card:#f7f7f9}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececf0;--muted:#a0a0a8;--line:#2c2c31;--err:#f97066;
--warn:#fdb022;--ok:#47cd89;--card:#1c1c20}}
body{background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,Segoe UI,sans-serif;margin:0;padding:24px 16px}
main{max-width:1100px;margin:auto}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px}
.muted{color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:16px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}
.card b{display:block;font-size:20px;font-variant-numeric:tabular-nums}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid var(--line);padding:6px 8px;
text-align:left;vertical-align:top}td.n{text-align:right;font-variant-numeric:tabular-nums}
.error{color:var(--err);font-weight:600}.warning{color:var(--warn);font-weight:600}.ok{color:var(--ok);font-weight:600}
.scroll{overflow-x:auto}details{margin:4px 0}code{font-size:12px}
"""


def _table(rows: list[dict[str, Any]], num: set[str] = frozenset()) -> str:
    if not rows:
        return "<p class='muted'>None.</p>"
    cols = list(rows[0])
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = ""
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            text = f"{v:,.2f}" if isinstance(v, float) else (f"{v:,}" if isinstance(v, int) else html.escape(str(v)))
            cls = " class='n'" if c in num or isinstance(v, (int, float)) else ""
            if c == "severity":
                cls = f" class='{html.escape(str(v))}'"
            cells.append(f"<td{cls}>{text}</td>")
        body += "<tr>" + "".join(cells) + "</tr>"
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def issue_summary(issues: pl.DataFrame) -> pl.DataFrame:
    return (
        issues.group_by(["rule_id", "table", "severity"]).agg(pl.len().alias("count"))
        .sort(["severity", "count"], descending=[False, True])
    )


def _brief(brief: dict[str, Any] | None) -> str:
    if not brief:
        return ""
    e = html.escape
    items = "".join(f"<li><b class='{'error' if f['severity'] == 'critical' else 'warning' if f['severity'] == 'attention' else 'ok'}'>"
                    f"{e(f['title'])}</b><br>{e(f['detail'])}<br><span class='muted'>Action: {e(f['action'])}</span></li>"
                    for f in brief.get("findings", []))
    steps = "".join(f"<li>{e(x)}</li>" for x in brief.get("next_steps", []))
    return (f"<h2>Summary (written by the AI reporter)</h2><div class='card'><b>{e(brief.get('headline', ''))}</b>"
            f"<p>{e(brief.get('summary', ''))}</p><ul>{items}</ul>"
            + (f"<p class='muted'>Next steps</p><ol>{steps}</ol>" if steps else "") + "</div>")


def render(log: dict[str, Any], issues: pl.DataFrame, samples_per_rule: int = 5) -> str:
    rec = log["reconciliation"]
    status = "<span class='ok'>PASSED</span>" if rec["passed"] else "<span class='error'>FAILED</span>"
    counts = log["row_counts"]
    cards = "".join(
        f"<div class='card'><span class='muted'>{html.escape(k)}</span><b>{v:,}</b></div>" for k, v in counts.items()
    )
    recon_rows = [
        {"measure": m, "raw": rec["raw"][m], "output": rec["output"][m], "quarantine": rec["quarantine"][m],
         "added_by_fixes": rec["units_added_by_fixes"] if m == "units" else 0.0, "difference": rec["differences"][m]}
        for m in ["rows", "units", "value"]
    ]
    samples = ""
    for rid in issue_summary(issues)["rule_id"].to_list():
        sub = issues.filter(pl.col("rule_id") == rid).head(samples_per_rule).drop("rule_id", "table")
        samples += f"<details><summary><code>{html.escape(rid)}</code></summary>{_table(sub.to_dicts())}</details>"
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DQ Report</title><style>{CSS}</style></head><body><main>
<h1>Data quality report</h1>
<p class="muted">{html.escape(log['input']['file'])} · {html.escape(log['started_at'])} · reconciliation {status}</p>
<div class="cards">{cards}</div>
{_brief((log.get('agent') or {}).get('brief'))}
<h2>Reconciliation (raw = output + quarantine)</h2>{_table(recon_rows)}
<h2>Issues by rule</h2>{_table(issue_summary(issues).to_dicts())}
<h2>Fixes applied</h2>{_table([{**f, 'detail': str(f['detail'])} for f in log['fixes']])}
<h2>Sample issues</h2>{samples or "<p class='muted'>None.</p>"}
<h2>Timings (s)</h2>{_table([{'step': k, 'seconds': float(v)} for k, v in log['timings'].items()])}
</main></body></html>"""


def write_report(log: dict[str, Any], issues: pl.DataFrame, path: Path) -> Path:
    path.write_text(render(log, issues), encoding="utf-8")
    return path
