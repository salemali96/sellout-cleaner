"""Look and feel of the Streamlit app: CSS plus small HTML builders. Every dynamic value is escaped."""

from __future__ import annotations

from html import escape
from typing import Any

CSS = """
<style>
:root{--card:#121626;--card2:#171c30;--line:#252c45;--muted:#9aa3bd;--fg:#e8eaf2;--a1:#8b7cff;--a2:#22d3ee;
--ok:#34d399;--warn:#fbbf24;--err:#f87171}
.block-container{padding-top:2.2rem;max-width:1380px}
#MainMenu,footer{visibility:hidden}
.hero{position:relative;overflow:hidden;border:1px solid var(--line);border-radius:20px;padding:26px 30px;margin-bottom:18px;
background:radial-gradient(900px 260px at 0% 0%,rgba(139,124,255,.28),transparent 60%),
radial-gradient(700px 240px at 100% 0%,rgba(34,211,238,.18),transparent 60%),var(--card)}
.hero h1{margin:0;padding:0;font-size:34px;font-weight:800;letter-spacing:-.02em;
background:linear-gradient(90deg,#fff 0%,#c9c2ff 45%,var(--a2) 100%);-webkit-background-clip:text;background-clip:text;color:transparent}
.hero p{margin:6px 0 0;color:var(--muted);font-size:15px}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}
.chip{border:1px solid var(--line);background:rgba(255,255,255,.04);border-radius:999px;padding:4px 12px;font-size:12.5px;color:var(--fg)}
.chip.ok{border-color:rgba(52,211,153,.5);color:var(--ok)}.chip.warn{border-color:rgba(251,191,36,.5);color:var(--warn)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:6px 0 14px}
.kpi{background:linear-gradient(180deg,var(--card2),var(--card));border:1px solid var(--line);border-radius:16px;padding:14px 16px}
.kpi .l{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.06em}
.kpi .v{font-size:26px;font-weight:750;margin-top:4px;font-variant-numeric:tabular-nums;color:var(--fg)}
.kpi.ok .v{color:var(--ok)}.kpi.warn .v{color:var(--warn)}.kpi.err .v{color:var(--err)}
.kpi.accent{border-color:rgba(139,124,255,.55)}
.flow{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:8px 0 16px}
.step{position:relative;background:var(--card);border:1px solid var(--line);border-radius:16px;padding:14px;min-height:132px;transition:all .25s}
.step .ico{font-size:24px}.step .nm{font-weight:700;margin-top:6px;color:var(--fg)}
.step .ds{color:var(--muted);font-size:12.5px;line-height:1.35;margin-top:3px}
.step .who{position:absolute;top:12px;right:12px;font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;
border:1px solid var(--line);border-radius:999px;padding:1px 8px;color:var(--muted)}
.step .who.ai{color:#c9c2ff;border-color:rgba(139,124,255,.6)}.step .who.human{color:var(--warn);border-color:rgba(251,191,36,.5)}
.step .st{font-size:12px;margin-top:8px;color:var(--muted)}
.step.done{border-color:rgba(52,211,153,.45)}.step.done .st{color:var(--ok)}
.step.running{border-color:var(--a1);box-shadow:0 0 0 1px var(--a1),0 0 28px rgba(139,124,255,.35);animation:pulse 1.6s ease-in-out infinite}
.step.running .st{color:#c9c2ff}
.step.wait{border-color:var(--warn);box-shadow:0 0 24px rgba(251,191,36,.25)}.step.wait .st{color:var(--warn)}
.step.fail{border-color:var(--err)}.step.fail .st{color:var(--err)}
@keyframes pulse{0%,100%{box-shadow:0 0 0 1px var(--a1),0 0 14px rgba(139,124,255,.25)}50%{box-shadow:0 0 0 1px var(--a1),0 0 34px rgba(139,124,255,.55)}}
.brief{border:1px solid rgba(139,124,255,.45);border-radius:18px;padding:20px 22px;margin:8px 0 16px;
background:linear-gradient(135deg,rgba(139,124,255,.14),rgba(34,211,238,.06) 60%),var(--card)}
.brief .tag{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#c9c2ff}
.brief h3{margin:4px 0 6px;padding:0;font-size:21px;color:var(--fg)}.brief p{color:var(--fg);margin:0 0 12px}
.finding{border-left:3px solid var(--line);padding:6px 0 6px 12px;margin:8px 0}
.finding.critical{border-color:var(--err)}.finding.attention{border-color:var(--warn)}.finding.info{border-color:var(--ok)}
.finding b{color:var(--fg)}.finding .d{color:var(--fg);font-size:14px}.finding .a{color:var(--muted);font-size:13px}
.brief ol{margin:6px 0 0 18px;color:var(--fg)}
.section{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:14px 0 4px}
.stTabs [data-baseweb="tab-list"]{gap:6px}
.stTabs [data-baseweb="tab"]{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:8px 16px;height:auto}
.stTabs [aria-selected="true"]{border-color:var(--a1);background:rgba(139,124,255,.14)}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none}
div[data-testid="stSidebarContent"]{background:linear-gradient(180deg,#0f1322,#0b0d14)}
.brand{font-weight:800;font-size:19px;letter-spacing:-.01em}.brand span{color:var(--a2)}
@media (prefers-reduced-motion:reduce){.step.running{animation:none}}
</style>
"""

# (node, icon, name, who, what it does)
AGENTS: list[tuple[str, str, str, str, str]] = [
    ("profiler", "🧭", "Profiler", "AI", "Maps your columns to the standard schema"),
    ("modeler", "🧱", "Modeler", "Code", "Loads, normalizes and splits into 4 tables"),
    ("cleaner", "🧹", "Cleaner", "AI", "Reads the issues and picks fixes"),
    ("approval", "🙋", "You", "Human", "Approve or reject risky fixes"),
    ("validator", "🔎", "Validator", "Code", "Re-runs every check on the result"),
    ("writer", "💾", "Writer", "Code", "Writes tables, quarantine and totals check"),
    ("reporter", "📝", "Reporter", "AI", "Explains the result in plain language"),
]
STATE_TEXT = {"idle": "Waiting", "running": "Working...", "done": "Done", "wait": "Needs your decision", "fail": "Failed"}


def hero(title: str, subtitle: str, chips: list[tuple[str, str]]) -> str:
    tags = "".join(f"<span class='chip {escape(tone)}'>{escape(text)}</span>" for text, tone in chips)
    return f"<div class='hero'><h1>{escape(title)}</h1><p>{escape(subtitle)}</p><div class='chips'>{tags}</div></div>"


def kpis(items: list[tuple[str, Any, str]]) -> str:
    cells = "".join(f"<div class='kpi {escape(tone)}'><div class='l'>{escape(label)}</div><div class='v'>{escape(str(value))}</div></div>"
                    for label, value, tone in items)
    return f"<div class='kpis'>{cells}</div>"


def flow(states: dict[str, str] | None = None) -> str:
    states = states or {}
    cards = ""
    for node, icon, name, who, what in AGENTS:
        state = states.get(node, "idle")
        cards += (f"<div class='step {state}'><span class='who {who.lower()}'>{who}</span><div class='ico'>{icon}</div>"
                  f"<div class='nm'>{name}</div><div class='ds'>{what}</div><div class='st'>{STATE_TEXT[state]}</div></div>")
    return f"<div class='flow'>{cards}</div>"


def brief(b: dict[str, Any]) -> str:
    findings = "".join(
        f"<div class='finding {escape(str(f.get('severity', 'info')))}'><b>{escape(str(f.get('title', '')))}</b>"
        f"<div class='d'>{escape(str(f.get('detail', '')))}</div><div class='a'>→ {escape(str(f.get('action', '')))}</div></div>"
        for f in b.get("findings", []))
    steps = "".join(f"<li>{escape(str(x))}</li>" for x in b.get("next_steps", []))
    return (f"<div class='brief'><div class='tag'>📝 Reporter agent · summary</div><h3>{escape(str(b.get('headline', '')))}</h3>"
            f"<p>{escape(str(b.get('summary', '')))}</p>{findings}"
            + (f"<div class='section'>Next steps</div><ol>{steps}</ol>" if steps else "") + "</div>")


def section(text: str) -> str:
    return f"<div class='section'>{escape(text)}</div>"
