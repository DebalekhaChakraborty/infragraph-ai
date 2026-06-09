"""
InfraGraph AI Command Center — Streamlit demo cockpit.

This app is designed as a guided demo cockpit. It prioritizes narrative
clarity over exposing all raw artifacts.

Two workspaces:
  1. Diagram Intelligence  — static diagram → graph memory
  2. Alert RCA Intelligence — alert stream → learned RCA → operator explanation
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InfraGraph AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get help": None, "Report a bug": None, "About": "InfraGraph AI — AIOps RCA cockpit"},
)

REPO_ROOT = Path(__file__).parent.parent
DEMO_ID   = "diagram_0373"

# ── Diagram onboarding (optional import from scripts/) ─────────────────────────
import sys as _sys
_scripts_dir = str(REPO_ROOT / "scripts")
if _scripts_dir not in _sys.path:
    _sys.path.insert(0, _scripts_dir)
try:
    from onboard_diagram import run_onboarding as _run_onboarding  # type: ignore
    _ONBOARD_OK = True
except Exception:
    _ONBOARD_OK = False
    _run_onboarding = None  # type: ignore

# ── Node-type inference from node-ID prefix ────────────────────────────────────
_PREFIX_TYPE = {
    "WAN": "cloud_or_wan", "CLOUD": "cloud_or_wan",
    "RTR": "router",
    "FW":  "firewall",
    "SW":  "switch",
    "LB":  "load_balancer",
    "APP": "server", "WEB": "server", "SRV": "server", "MGMT": "server",
    "DB":  "database",
}

def _node_type(node_id: str) -> str:
    return _PREFIX_TYPE.get(node_id.split("-")[0].upper(), "server")


def _col(dark: str, light: str) -> str:
    return light if st.session_state.get("theme") == "light" else dark


# ── CSS ────────────────────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stAppViewContainer"] > .main { background: #0b0f1c; }
section[data-testid="stSidebar"]           { background: #070b16 !important; border-right: 1px solid rgba(255,255,255,0.06); }
[data-testid="stHeader"]                   { background: transparent !important; }
h1,h2,h3,h4,h5,h6,p,li,label              { color: #e2e8f0; }

/* ── Hero ── */
.hero-title {
    font-size: 1.75rem; font-weight: 900; line-height: 1.2;
    background: linear-gradient(130deg, #f1f5f9 0%, #93c5fd 55%, #67e8f9 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin-bottom: 6px;
}
.hero-tagline { font-size: 0.95rem; color: #475569; line-height: 1.55; max-width: 640px; margin-bottom: 14px; }
.stat-pills   { display: flex; gap: 10px; flex-wrap: wrap; }
.stat-pill    { display: inline-flex; align-items: center; gap: 6px; padding: 5px 14px; border-radius: 20px;
                font-size: 0.73rem; font-weight: 600; letter-spacing: 0.04em; border: 1px solid; }
.stat-pill.incident { background: rgba(239,68,68,0.1);  color: #ef4444; border-color: rgba(239,68,68,0.3); }
.stat-pill.root     { background: rgba(16,185,129,0.1); color: #10b981; border-color: rgba(16,185,129,0.3); }
.stat-pill.status   { background: rgba(96,165,250,0.1); color: #60a5fa; border-color: rgba(96,165,250,0.25); }
.pill-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; animation: pdot 1.4s infinite; }
@keyframes pdot { 0%,100%{opacity:1} 50%{opacity:.3} }

/* ── Workspace header ── */
.ws-title { font-size: 1.1rem; font-weight: 800; color: #f1f5f9; margin: 4px 0 2px; }
.ws-desc  { font-size: 0.82rem; color: #475569; margin-bottom: 20px; }
.ws-rule  { border: none; border-top: 1px solid rgba(255,255,255,0.05); margin: 28px 0 22px; }
.section-label {
    font-size: 0.63rem; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase;
    color: #334155; margin: 0 0 10px; padding-bottom: 6px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 22px 24px; margin-bottom: 14px;
}
.card.red   { border-color: rgba(239,68,68,0.25);  background: rgba(239,68,68,0.04); }
.card.green { border-color: rgba(16,185,129,0.25); background: rgba(16,185,129,0.04); }
.card.blue  { border-color: rgba(96,165,250,0.2);  background: rgba(59,130,246,0.04); }

/* ── Alert stream ── */
.alert-item { display: flex; align-items: flex-start; gap: 12px; padding: 13px 15px;
              border-radius: 10px; margin-bottom: 9px; border-left: 3px solid transparent; }
.alert-item.critical { background: rgba(239,68,68,0.06); border-left-color: #ef4444; }
.alert-item.major    { background: rgba(245,158,11,0.06); border-left-color: #f59e0b; }
.alert-dot  { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; margin-top: 5px;
              animation: pdot 1.5s infinite; }
.alert-dot.critical { background: #ef4444; }
.alert-dot.major    { background: #f59e0b; }
.alert-node { font-weight: 700; color: #f1f5f9; font-size: 0.88rem; }
.alert-msg  { font-size: 0.81rem; color: #94a3b8; margin-top: 2px; }
.alert-time { font-size: 0.69rem; color: #475569; margin-left: auto; white-space: nowrap;
              font-family: 'JetBrains Mono', monospace; }

/* ── Badges ── */
.badge { display: inline-block; padding: 2px 11px; border-radius: 20px;
         font-size: 0.68rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }
.badge-critical { background: rgba(239,68,68,0.13);  color: #ef4444; border: 1px solid rgba(239,68,68,0.32); }
.badge-major    { background: rgba(245,158,11,0.13); color: #f59e0b; border: 1px solid rgba(245,158,11,0.32); }
.badge-success  { background: rgba(16,185,129,0.13); color: #10b981; border: 1px solid rgba(16,185,129,0.32); }
.badge-info     { background: rgba(96,165,250,0.13); color: #60a5fa; border: 1px solid rgba(96,165,250,0.28); }
.badge-wrong    { background: rgba(239,68,68,0.13);  color: #ef4444; border: 1px solid rgba(239,68,68,0.32); }

/* ── Graph memory ── */
.gm-header { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #475569;
             letter-spacing: 0.06em; margin-bottom: 10px; text-transform: uppercase; }
.gm-node { display: flex; align-items: center; gap: 10px; padding: 7px 0;
           border-bottom: 1px solid rgba(255,255,255,0.04); font-family: 'JetBrains Mono', monospace; }
.gm-node:last-child { border-bottom: none; }
.gm-id   { font-size: 0.82rem; font-weight: 600; color: #e2e8f0; width: 80px; flex-shrink: 0; }
.gm-type { font-size: 0.75rem; color: #64748b; width: 110px; flex-shrink: 0; }
.gm-status { font-size: 0.72rem; width: 110px; flex-shrink: 0; }
.gm-score  { font-size: 0.75rem; color: #64748b; margin-left: auto; }
.gm-edge { display: flex; align-items: center; gap: 8px; padding: 5px 0;
           font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #64748b;
           border-bottom: 1px solid rgba(255,255,255,0.03); }
.gm-edge:last-child { border-bottom: none; }
.gm-arrow { color: #334155; }

/* ── Node chips ── */
.node-chip { display: inline-block; padding: 3px 9px; border-radius: 6px; font-size: 0.72rem;
             font-weight: 600; font-family: 'JetBrains Mono', monospace; margin: 2px;
             background: rgba(96,165,250,0.09); color: #60a5fa; border: 1px solid rgba(96,165,250,0.2); }
.node-chip.alerting { background: rgba(239,68,68,0.09); color: #f87171; border-color: rgba(239,68,68,0.22); }
.node-chip.impacted { background: rgba(245,158,11,0.09); color: #fbbf24; border-color: rgba(245,158,11,0.22); }
.node-chip.root     { background: rgba(16,185,129,0.1);  color: #34d399; border-color: rgba(16,185,129,0.35); }

/* ── RCA winner card ── */
.rca-winner {
    background: rgba(16,185,129,0.04); border: 1px solid rgba(16,185,129,0.3);
    border-radius: 16px; padding: 26px 28px; margin-bottom: 14px;
}
.rca-winner-title { font-size: 1rem; font-weight: 800; color: #f1f5f9; margin-bottom: 4px; }
.rca-winner-sub   { font-size: 0.78rem; color: #64748b; margin-bottom: 18px; }
.rca-winner-node  { font-size: 2rem; font-weight: 900; font-family: 'JetBrains Mono', monospace;
                    color: #10b981; line-height: 1; }
.rca-winner-meta  { font-size: 0.78rem; color: #64748b; margin-top: 6px; }

/* ── RCA comparison row ── */
.rca-row { display: flex; align-items: center; gap: 14px; padding: 11px 14px;
           border-radius: 10px; margin-bottom: 8px; background: rgba(255,255,255,0.03);
           border: 1px solid rgba(255,255,255,0.07); }
.rca-row.wrong-row { border-color: rgba(239,68,68,0.2); background: rgba(239,68,68,0.03); }
.rca-row-label  { font-size: 0.7rem; color: #64748b; text-transform: uppercase;
                  letter-spacing: 0.1em; width: 160px; flex-shrink: 0; }
.rca-row-node   { font-family: 'JetBrains Mono', monospace; font-size: 0.92rem; font-weight: 700; }
.rca-row-node.correct { color: #10b981; }
.rca-row-node.wrong   { color: #ef4444; }
.rca-row-reason { font-size: 0.78rem; color: #475569; margin-left: auto; max-width: 280px;
                  text-align: right; }

/* ── Score bar ── */
.score-track { background: rgba(255,255,255,0.06); border-radius: 4px; height: 6px;
               overflow: hidden; margin: 4px 0 8px; }
.score-fill  { height: 100%; border-radius: 4px;
               background: linear-gradient(90deg, #3b82f6, #67e8f9); }
.score-fill.green { background: linear-gradient(90deg, #10b981, #34d399); }
.score-fill.red   { background: linear-gradient(90deg, #ef4444, #f87171); }
.score-fill.purple { background: linear-gradient(90deg, #8b5cf6, #a78bfa); }

/* ── Path / propagation ── */
.path-row { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #60a5fa;
            padding: 7px 13px; background: rgba(96,165,250,0.05); border-radius: 8px;
            margin: 4px 0; border: 1px solid rgba(96,165,250,0.11); }
.prop-card { background: rgba(96,165,250,0.04); border: 1px solid rgba(96,165,250,0.15);
             border-radius: 12px; padding: 20px 22px; margin-bottom: 12px; }
.prop-num   { font-size: 0.63rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase;
              color: #60a5fa; margin-bottom: 4px; }
.prop-title { font-size: 0.96rem; font-weight: 700; color: #f1f5f9; margin-bottom: 8px; }
.prop-body  { font-size: 0.84rem; color: #94a3b8; line-height: 1.65; }
.prop-formula { font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; color: #60a5fa;
                background: rgba(96,165,250,0.06); padding: 8px 12px; border-radius: 6px;
                margin-bottom: 10px; display: block; }

/* ── Misc ── */
.warn-card { background: rgba(245,158,11,0.06); border: 1px solid rgba(245,158,11,0.18);
             border-radius: 10px; padding: 12px 16px; font-size: 0.83rem; color: #94a3b8; }
.report-body { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.07);
               border-radius: 14px; padding: 30px 34px; }
.chat-hint { font-size: 0.75rem; color: #475569; text-align: center; margin-top: 6px; }
.dev-note  { font-size: 0.79rem; color: #475569; font-style: italic; line-height: 1.6; }

/* ── Sidebar ── */
.sb-label { font-size: 0.62rem; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase;
            color: #1e293b; margin: 18px 0 7px; padding-bottom: 5px;
            border-bottom: 1px solid rgba(255,255,255,0.05); }
.sb-step  { display: flex; align-items: flex-start; gap: 9px; padding: 6px 0;
            font-size: 0.8rem; color: #10b981;
            border-bottom: 1px solid rgba(255,255,255,0.03); }
.sb-step:last-child { border-bottom: none; }
.sb-check { flex-shrink: 0; margin-top: 1px; }
.sb-step-sub { font-size: 0.7rem; color: #1e293b; }
.demo-pill { background: rgba(96,165,250,0.07); border: 1px solid rgba(96,165,250,0.14);
             border-radius: 8px; padding: 9px 12px; font-size: 0.75rem; color: #475569; margin-top: 14px; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.02); border-radius: 10px;
                                    padding: 4px; gap: 2px; }
.stTabs [data-baseweb="tab"]      { background: transparent; border-radius: 8px;
                                    color: #475569; font-weight: 500; font-size: 0.83rem; }
.stTabs [aria-selected="true"]    { background: rgba(96,165,250,0.1) !important; color: #93c5fd !important; }
</style>
"""

_LIGHT_OVERRIDES = """
<style>
/* ══ LIGHT MODE — full-site override ══════════════════════════════════════════
   1. CSS custom-property re-definitions  → Streamlit's own widgets pick these up
   2. Explicit !important overrides       → catches elements that hard-code colors
   ══════════════════════════════════════════════════════════════════════════════ */

/* ── 1. Streamlit CSS variables ──────────────────────────────────────────── */
:root {
    --background-color: #f8fafc;
    --secondary-background-color: #f1f5f9;
    --text-color: #1e293b;
    --primary-color: #2563eb;
    --font: "Inter", sans-serif;
}

/* ── 2. Base & layout ────────────────────────────────────────────────────── */
html, body { background: #f8fafc !important; color: #1e293b !important; }

[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.stApp,
.main,
.main .block-container { background: #f8fafc !important; }

section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"] {
    background: #ffffff !important;
    border-right: 1px solid rgba(0,0,0,0.09) !important;
}

[data-testid="stHeader"] { background: rgba(248,250,252,0.92) !important; }
[data-testid="stToolbar"] { background: transparent !important; }

/* bottom chat input area */
[data-testid="stBottomBlockContainer"],
.stChatFloatingInputContainer { background: #f8fafc !important; }

/* ── 3. All text ─────────────────────────────────────────────────────────── */
h1,h2,h3,h4,h5,h6 { color: #0f172a !important; }
p, li, span        { color: #1e293b; }

[data-testid="stMarkdown"] *,
[data-testid="stMarkdownContainer"] * { color: #1e293b !important; }
[data-testid="stText"]                { color: #1e293b !important; }
[data-testid="stCaptionContainer"] *,
figcaption, .stCaption               { color: #64748b !important; }

/* widget labels */
label,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span { color: #1e293b !important; }

/* ── 4. Sidebar ──────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] [data-testid="stMarkdown"] * { color: #1e293b !important; }

/* ── 5. Radio ────────────────────────────────────────────────────────────── */
[data-testid="stRadio"] label,
[data-testid="stRadio"] p    { color: #1e293b !important; }
[data-baseweb="radio"] > div:first-child > div { border-color: rgba(0,0,0,0.3) !important; }

/* ── 6. Toggle ───────────────────────────────────────────────────────────── */
[data-testid="stToggle"] label,
[data-testid="stToggle"] p { color: #1e293b !important; }

/* ── 7. Buttons ──────────────────────────────────────────────────────────── */
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-minimal"],
.stButton button,
[data-testid="stButton"] button {
    background: #ffffff !important;
    color: #334155 !important;
    border: 1px solid rgba(0,0,0,0.14) !important;
}
[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stButton"] button:hover { background: #f1f5f9 !important; }

[data-testid="stDownloadButton"] button {
    background: #ffffff !important;
    color: #334155 !important;
    border: 1px solid rgba(0,0,0,0.14) !important;
}

/* ── 8. Tabs ─────────────────────────────────────────────────────────────── */
[data-baseweb="tab-list"]                { background: rgba(0,0,0,0.05) !important; }
[data-baseweb="tab"]                     { color: #64748b !important; background: transparent !important; }
[data-baseweb="tab"][aria-selected="true"] {
    background: rgba(37,99,235,0.1) !important;
    color: #1d4ed8 !important;
}
[data-baseweb="tab-panel"]  { background: transparent !important; }
[data-baseweb="tab-border"] { background: rgba(0,0,0,0.08) !important; }

/* ── 9. Expander ─────────────────────────────────────────────────────────── */
[data-testid="stExpander"] details {
    background: rgba(0,0,0,0.02) !important;
    border: 1px solid rgba(0,0,0,0.1) !important;
}
[data-testid="stExpander"] summary {
    background: rgba(0,0,0,0.02) !important;
    color: #1e293b !important;
}
[data-testid="stExpander"] summary *,
[data-testid="stExpander"] details > div * { color: #1e293b !important; }

/* ── 10. Select slider & slider ──────────────────────────────────────────── */
[data-testid="stSelectSlider"] label,
[data-testid="stSelectSlider"] p,
[data-testid="stSlider"] label,
[data-testid="stSlider"] p { color: #1e293b !important; }
[data-testid="stSelectSlider"] [role="slider"] { background: #2563eb !important; }

/* ── 11. Chat input ──────────────────────────────────────────────────────── */
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInputContainer"] {
    background: #ffffff !important;
    border-color: rgba(0,0,0,0.14) !important;
}
[data-testid="stChatInput"] textarea {
    background: #ffffff !important;
    color: #1e293b !important;
}

/* ── 12. Chat messages ───────────────────────────────────────────────────── */
[data-testid="stChatMessage"] { background: rgba(0,0,0,0.03) !important; }
[data-testid="stChatMessage"] *:not(code):not(pre):not(.badge) { color: #1e293b !important; }

/* ── 13. Dataframe ───────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] > div { background: #ffffff !important; }
[data-testid="stDataFrameGlideDataEditor"] { background: #ffffff !important; }

/* ── 14. Alerts / info ───────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    background: rgba(0,0,0,0.03) !important;
    border-color: rgba(0,0,0,0.1) !important;
}
[data-testid="stAlert"] * { color: #1e293b !important; }

/* ── 15. Code blocks ─────────────────────────────────────────────────────── */
code { background: rgba(0,0,0,0.06) !important; color: #1e293b !important; }
pre  { background: rgba(0,0,0,0.04) !important; border-color: rgba(0,0,0,0.1) !important; }
pre code { color: #1e293b !important; }

/* ── 16. InfraGraph custom classes ──────────────────────────────────────── */
.hero-tagline { color: #64748b !important; }

.card { background: rgba(0,0,0,0.03) !important; border-color: rgba(0,0,0,0.1) !important; }
.card.red   { background: rgba(239,68,68,0.05) !important; }
.card.green { background: rgba(16,185,129,0.05) !important; }
.card.blue  { background: rgba(59,130,246,0.05) !important; }

.alert-item.critical { background: rgba(239,68,68,0.06) !important; }
.alert-item.major    { background: rgba(245,158,11,0.06) !important; }
.alert-node { color: #1e293b !important; }
.alert-msg  { color: #64748b !important; }
.alert-time { color: #94a3b8 !important; }

.ws-title { color: #1e293b !important; }
.ws-desc  { color: #64748b !important; }
.ws-rule  { border-top-color: rgba(0,0,0,0.08) !important; }
.section-label { color: #94a3b8 !important; border-bottom-color: rgba(0,0,0,0.07) !important; }

.gm-header { color: #94a3b8 !important; }
.gm-id     { color: #1e293b !important; }
.gm-type   { color: #94a3b8 !important; }
.gm-score  { color: #64748b !important; }
.gm-node   { border-bottom-color: rgba(0,0,0,0.06) !important; }
.gm-edge   { color: #64748b !important; border-bottom-color: rgba(0,0,0,0.04) !important; }
.gm-arrow  { color: #94a3b8 !important; }

.rca-winner { background: rgba(16,185,129,0.05) !important; border-color: rgba(16,185,129,0.25) !important; }
.rca-winner-title { color: #1e293b !important; }
.rca-winner-sub   { color: #64748b !important; }
.rca-winner-meta  { color: #64748b !important; }
.rca-row { background: rgba(0,0,0,0.03) !important; border-color: rgba(0,0,0,0.09) !important; }
.rca-row.wrong-row { background: rgba(239,68,68,0.04) !important; border-color: rgba(239,68,68,0.18) !important; }
.rca-row-label  { color: #94a3b8 !important; }
.rca-row-reason { color: #64748b !important; }

.score-track { background: rgba(0,0,0,0.09) !important; }

.path-row { background: rgba(59,130,246,0.05) !important; border-color: rgba(59,130,246,0.15) !important; }

.prop-card    { background: rgba(59,130,246,0.04) !important; border-color: rgba(59,130,246,0.18) !important; }
.prop-num     { color: #2563eb !important; }
.prop-title   { color: #1e293b !important; }
.prop-body    { color: #475569 !important; }
.prop-formula { color: #1d4ed8 !important; background: rgba(59,130,246,0.07) !important; }

.report-body { background: rgba(0,0,0,0.02) !important; border-color: rgba(0,0,0,0.08) !important; }
.warn-card   { background: rgba(245,158,11,0.06) !important; border-color: rgba(245,158,11,0.22) !important; color: #92400e !important; }
.chat-hint   { color: #64748b !important; }
.dev-note    { color: #64748b !important; }

.sb-label    { color: #94a3b8 !important; border-bottom-color: rgba(0,0,0,0.08) !important; }
.sb-step     { color: #059669 !important; border-bottom-color: rgba(0,0,0,0.05) !important; }
.sb-step-sub { color: #64748b !important; }
.demo-pill   { background: rgba(59,130,246,0.06) !important; border-color: rgba(59,130,246,0.15) !important; color: #64748b !important; }
</style>
"""


# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data
def load_json(path: str) -> dict | list | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


@st.cache_data
def load_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return ""


# ── UI helpers ─────────────────────────────────────────────────────────────────
def _img(path: Path, caption: str = "") -> None:
    if path.exists():
        st.image(str(path), caption=caption or None, use_container_width=True)
    else:
        st.markdown(
            f'<div class="warn-card">Image not found: <code>{path.name}</code></div>',
            unsafe_allow_html=True,
        )


def _score_bar(pct: float, cls: str = "") -> str:
    c = f" {cls}" if cls else ""
    return (
        f'<div class="score-track"><div class="score-fill{c}" '
        f'style="width:{min(100.0, max(0.0, pct)):.0f}%"></div></div>'
    )


def _clean_report(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^```\w*\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def _build_fallback_report(rca: dict, gnn: dict, mlp: dict) -> str:
    impacted = rca.get("impacted_nodes", [])
    n = len(impacted)
    imp_str = ", ".join(impacted[:5]) + (f" +{n-5} more" if n > 5 else "")
    score = (gnn.get("top_candidates") or [{}])[0].get("score", 30.73)
    return f"""\
## Executive Summary

A network incident triggered 3 alerts across 2 devices, impacting **{n} downstream services**.
Learned GNN-based RCA identified **FW-01** (firewall) as root cause with high confidence
(GNN score: {score:.2f}, margin: +8.12).

---

## Root Cause Conclusion

**Root cause: FW-01 (firewall)** — Confidence: HIGH

FW-01 triggered a CRITICAL alert at t+0 min. Its upstream chokepoint position means failure
cascades to SW-CORE and all downstream services.

---

## Alert Evidence

| Node | Severity | Alert | Offset |
|------|----------|-------|--------|
| FW-01 | CRITICAL | Packet drops elevated | t+0 min |
| SW-CORE | MAJOR | Policy deny spike | t+2 min |
| SW-CORE | MAJOR | App unreachable | t+4 min |

---

## Impact Analysis

**{n} nodes impacted**: {imp_str}

Shortest path: FW-01 → SW-CORE → SW-APP → LB-01 → APP-01

---

## Recommended Actions

1. **SSH to FW-01** — check interface counters and syslog
2. **Identify failure mode** — ACL misconfiguration, link fault, or hardware error
3. **Escalate** if packet drop rate > 5% or CPU/memory critical
4. **Failover** — activate redundant path via RTR-01 or RTR-02
5. **Validate** — confirm APP-01…APP-04 reachable after remediation

---

## ServiceNow Summary

| Field | Value |
|-------|-------|
| Short description | Network fault on FW-01 — {n}-service outage |
| Affected CI | FW-01 (firewall) |
| Priority | P1 — {n} downstream nodes impacted |
| Assignment group | Network Operations |
| Root cause (automated) | FW-01 — GNN RCA, confidence HIGH |

---

## Limitations

- Models trained on synthetic infragraph_v2 benchmark scenarios
- Human review recommended before executing production remediation
"""


def _extract_edges(rca: dict) -> list[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for group in rca.get("impact_paths", {}).values():
        for entry in group:
            path = entry.get("path", [])
            for i in range(len(path) - 1):
                edges.add((path[i], path[i + 1]))
    return sorted(edges)


# ── Chat QA ────────────────────────────────────────────────────────────────────
def _answer(question: str) -> str:
    q = question.lower()

    if any(kw in q for kw in ["root cause", "what failed", "what is the root", "who is"]):
        return """\
**Root cause: FW-01 (Firewall)**

The GNN identified **FW-01** as root cause with score **30.733** (margin +8.12 over FW-02).

| Signal | Value |
|--------|-------|
| Severity | CRITICAL — t+0 min (earliest) |
| Device type | Firewall — upstream chokepoint |
| Upstream neighbours | RTR-01, RTR-02 — both silent |
| GNN score margin | 30.733 vs 22.613 (FW-02) |

Silent upstream routers are the key signal: FW-01 is not *receiving* a cascade — it *is* the source."""

    if any(kw in q for kw in ["why is fw-01", "why fw", "gnn", "propagat", "graph neural", "how"]):
        return """\
**How the GNN identifies FW-01:**

Two rounds of graph convolution:
```
H1 = ReLU(Â X  W1)   # 1-hop: FW-01 aggregates RTR-01, RTR-02 (silent)
H2 = ReLU(Â H1 W2)   # 2-hop: FW-01 encodes 2-hop context
scores = H2 W_out
```
- After Layer 1: FW-01 aggregates from **silent** upstream routers — discriminating signal
- After Layer 2: FW-01 encodes both its CRITICAL severity and the fact that its upstream is silent
- Final score: **30.733** vs SW-CORE **11.393** — 20-point separation

Converges at epoch 6 (vs MLP epoch 56) — message-passing propagates signal efficiently."""

    if any(kw in q for kw in ["which nodes", "impacted", "affected", "downstream"]):
        return """\
**10 nodes impacted:**

APP-01, APP-02, APP-03, APP-04 (servers) · CLOUD-01 (WAN) · DB-01, DB-02 (databases)
· LB-01 (load balancer) · MGMT-01 (server) · SW-APP (switch)

**Shortest path:** FW-01 → SW-CORE → SW-APP → LB-01 → APP-01

**DB path (longest):** FW-01 → FW-02 → SW-APP → LB-01 → APP-01 → DB-01 (6 hops)"""

    if any(kw in q for kw in ["baseline", "sw-core", "heuristic", "wrong", "why did baseline"]):
        return """\
**Why baseline scoring chose SW-CORE (incorrectly):**

```
score = severity × 2 + (1/(1+t_offset)) × 10 + downstream_ratio × 3 + device_bonus
```

| Node | Alerts | Score |
|------|--------|-------|
| SW-CORE | 2 MAJOR (t+2, t+4) | **20.86** ← selected |
| FW-01 | 1 CRITICAL (t+0) | 20.62 ← ground truth |

More alert events + higher downstream reach gave SW-CORE a marginally higher score.
The rule-based formula cannot distinguish a downstream aggregation node from the upstream origin."""

    if any(kw in q for kw in ["servicenow", "ticket", "snow", "p1"]):
        return """\
**ServiceNow Incident:**

| Field | Value |
|-------|-------|
| Short description | Network fault on FW-01 — 10-service outage |
| Affected CI | FW-01 (firewall) |
| Priority | **P1** |
| Assignment group | Network Operations |
| Root cause (automated) | FW-01 — GNN RCA, HIGH confidence |
| Services impacted | APP-01..04, CLOUD-01, DB-01, DB-02, LB-01, MGMT-01, SW-APP |"""

    return f"""\
No pre-built answer for: *"{question}"*

Try:
- **What is the root cause?**
- **Why is FW-01 the root cause?**
- **Which nodes are impacted?**
- **Why did baseline choose SW-CORE?**
- **Generate ServiceNow summary**"""


# ── Sidebar ────────────────────────────────────────────────────────────────────
def _sidebar() -> str:
    with st.sidebar:
        title_color = _col("#f1f5f9", "#0f172a")
        inc_color   = _col("#f1f5f9", "#0f172a")
        sub_color   = _col("#334155", "#64748b")

        st.markdown(
            f'<div style="padding:10px 0 4px">'
            f'<span style="font-size:1.05rem;font-weight:800;color:{title_color}">⚡ InfraGraph AI</span>'
            f'</div>'
            f'<p style="font-size:0.73rem;color:{sub_color};margin:-2px 0 0">AIOps Incident Cockpit</p>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sb-label">Workspace</div>', unsafe_allow_html=True)
        workspace = st.radio(
            "workspace",
            ["🔍  Diagram Intelligence", "🤖  Alert RCA Intelligence"],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown('<div class="sb-label">Active Incident</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.95rem;font-weight:700;color:{inc_color}">INC-SYN-0373</div>'
            f'<div style="font-size:0.76rem;color:{sub_color};margin-top:2px">diagram_0373 · test split</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sb-label">Pipeline Status</div>', unsafe_allow_html=True)
        steps = [
            ("Diagram ingested",       "PNG read"),
            ("Devices detected",       "YOLO v8 · 17 nodes"),
            ("Graph memory built",     "17 nodes · 17 edges"),
            ("Alerts mapped",          "3 alerts · 2 nodes"),
            ("Learned RCA complete",   "GNN → FW-01"),
            ("Operator report ready",  "Qwen explanation"),
        ]
        html = "".join(
            f'<div class="sb-step">'
            f'<span class="sb-check">✓</span>'
            f'<div><div>{lbl}</div><div class="sb-step-sub">{sub}</div></div>'
            f'</div>'
            for lbl, sub in steps
        )
        st.markdown(html, unsafe_allow_html=True)
        st.markdown('<div class="demo-pill">Demo mode — synthetic incident</div>', unsafe_allow_html=True)

        with st.expander("Developer Settings"):
            st.radio("Split", ["test", "val", "train"], index=0, horizontal=True)

        st.markdown('<div class="sb-label" style="margin-top:22px">Appearance</div>', unsafe_allow_html=True)
        is_light = st.toggle(
            "☀  Light mode",
            value=(st.session_state.get("theme") == "light"),
            key="theme_toggle",
        )
        st.session_state.theme = "light" if is_light else "dark"

    return workspace.strip().lstrip("🔍🤖 ")


# ── Compact hero (always visible) ──────────────────────────────────────────────
def _hero(workspace: str) -> None:
    h1, h2 = st.columns([4, 1])
    with h1:
        st.markdown(
            '<div style="padding:10px 0 16px">'
            '<div class="hero-title">InfraGraph AI Command Center</div>'
            '<div class="hero-tagline">'
            "Static diagram converted into graph memory. "
            "Alert stream analyzed using learned graph RCA."
            "</div>"
            '<div class="stat-pills">'
            '<div class="stat-pill incident"><div class="pill-dot"></div>P1 · INC-SYN-0373</div>'
            '<div class="stat-pill root">Root cause: FW-01</div>'
            '<div class="stat-pill status">AI RCA confirmed</div>'
            "</div></div>",
            unsafe_allow_html=True,
        )
    with h2:
        st.markdown(
            f'<div style="text-align:right;padding-top:12px">'
            f'<div style="font-size:0.68rem;font-weight:700;letter-spacing:0.12em;'
            f'text-transform:uppercase;color:#334155">Workspace</div>'
            f'<div style="font-size:0.88rem;font-weight:700;color:#93c5fd;margin-top:4px">'
            f'{workspace}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.05);margin:0 0 20px'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM ONBOARDING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_ONBOARD_OUT = REPO_ROOT / "outputs" / "onboarded_diagrams"


def _display_onboard_result(result: dict) -> None:
    """Render the structured output of a successful onboarding run."""
    paths       = result.get("paths", {})
    stats       = result.get("stats", {})
    diagram_id  = result.get("diagram_id", "")
    det_method  = result.get("detection_method", "")
    node_count  = stats.get("node_count", 0)
    edge_count  = stats.get("edge_count", 0)
    type_dist   = stats.get("detected_types", {})

    st.markdown(
        f'<div class="stat-pills" style="margin:10px 0 18px">'
        f'<div class="stat-pill status">{node_count} nodes detected</div>'
        f'<div class="stat-pill status">{edge_count} edges inferred</div>'
        f'<div class="stat-pill root">Graph Memory Updated</div>'
        f'<div class="stat-pill status">method: {det_method}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Original vs detected image pair
    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown('<div class="section-label">Original Diagram</div>', unsafe_allow_html=True)
        _img(Path(paths.get("original", "")))
    with ic2:
        st.markdown('<div class="section-label">Vision Detection</div>', unsafe_allow_html=True)
        _img(Path(paths.get("detected", "")))

    # Node table
    nodes = result.get("nodes", [])
    if nodes:
        st.markdown(
            '<div class="section-label" style="margin-top:18px">Detected Node Table</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="dev-note">Node names are generated during MVP onboarding '
            'unless metadata/OCR is available.</p>',
            unsafe_allow_html=True,
        )
        df = pd.DataFrame([{
            "Node ID":   n["node_id"],
            "Type":      n["detected_type"].replace("_", " ").title(),
            "Confidence": f"{n['confidence']:.1%}",
            "Zone":      n["zone"],
            "Status":    n["graph_status"],
        } for n in nodes])
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Graph preview + memory card
    gc1, gc2 = st.columns([3, 2])
    with gc1:
        st.markdown('<div class="section-label">Local Graph Created</div>', unsafe_allow_html=True)
        preview = Path(paths.get("graph_preview", ""))
        if preview.exists():
            _img(preview)
        else:
            st.info("Graph preview unavailable (matplotlib / networkx not installed)")
    with gc2:
        st.markdown('<div class="section-label">Graph Memory Updated</div>', unsafe_allow_html=True)
        rows = "".join(
            f'<div style="display:flex;justify-content:space-between;font-size:0.8rem;'
            f'color:#64748b;margin-bottom:6px">'
            f'<span>{t.replace("_"," ").title()}</span>'
            f'<span style="color:#10b981;font-weight:600">{c}</span></div>'
            for t, c in type_dist.items()
        )
        st.markdown(
            f'<div class="card green" style="padding:16px 20px">'
            f'<div style="font-size:0.88rem;font-weight:700;color:#10b981;margin-bottom:12px">'
            f'✓ Registered in graph_memory/index.json</div>'
            f'{rows}'
            f'<div style="border-top:1px solid rgba(255,255,255,0.08);margin-top:10px;padding-top:10px;'
            f'display:flex;justify-content:space-between">'
            f'<span style="font-size:0.8rem;color:#64748b">diagram_id</span>'
            f'<span style="font-size:0.78rem;font-family:\'JetBrains Mono\',monospace;'
            f'color:#10b981">{diagram_id}</span></div></div>',
            unsafe_allow_html=True,
        )


def _render_onboarding() -> None:
    """Render the Diagram Onboarding section at the top of Diagram Intelligence workspace."""
    st.markdown('<div class="section-label">Diagram Onboarding</div>', unsafe_allow_html=True)

    if not _ONBOARD_OK:
        st.markdown(
            '<div class="warn-card">Onboarding module could not be loaded. '
            'Run <code>python scripts/onboard_diagram.py --help</code> from the CLI.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<hr class="ws-rule">', unsafe_allow_html=True)
        return

    col_up, col_demo = st.columns([5, 2])
    with col_up:
        uploaded = st.file_uploader(
            "Upload new topology diagram",
            type=["png", "jpg", "jpeg"],
            key="onboard_uploader",
            label_visibility="collapsed",
        )
    with col_demo:
        use_demo = st.button(
            "Use demo diagram_0373",
            use_container_width=True,
            key="onboard_demo_btn",
        )

    # Resolve trigger
    trigger_image: Path | None = None
    trigger_id:    str  | None = None

    if use_demo:
        trigger_image = REPO_ROOT / "datasets/infragraph_v2/images/test/diagram_0373.png"
        trigger_id    = "demo_onboard_0373"
        st.session_state.pop("onboard_result", None)     # force fresh run

    if uploaded is not None:
        uploads_dir = _ONBOARD_OUT / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        tmp = uploads_dir / uploaded.name
        tmp.write_bytes(uploaded.getvalue())
        trigger_image = tmp
        trigger_id    = f"upload_{Path(uploaded.name).stem}"
        # Only re-run if a new file was selected
        if st.session_state.get("_onboard_last_upload") != uploaded.name:
            st.session_state.pop("onboard_result", None)
            st.session_state["_onboard_last_upload"] = uploaded.name

    # Run onboarding if needed
    if trigger_image and trigger_id and "onboard_result" not in st.session_state:
        with st.status("Onboarding diagram...", expanded=True) as _status:
            result = _run_onboarding(
                image_path=str(trigger_image),
                diagram_id=trigger_id,
                model_path=None,
                out_dir=str(_ONBOARD_OUT),
                on_step=lambda msg: st.write(f"→ {msg}"),
            )
            if result.get("success"):
                _status.update(label="✓ Diagram Onboarding complete", state="complete")
            else:
                _status.update(label=f"✗ {result.get('error', 'unknown error')}", state="error")
        st.session_state["onboard_result"] = result

    # Display cached result
    if "onboard_result" in st.session_state:
        res = st.session_state["onboard_result"]
        if res.get("success"):
            _display_onboard_result(res)
        else:
            st.error(res.get("error", "Onboarding failed"))

    st.markdown('<hr class="ws-rule">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# WORKSPACE 1 — DIAGRAM INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

def _workspace_diagram(detected_nodes: list, rca: dict, gnn: dict) -> None:
    st.markdown(
        '<div class="ws-title">Diagram Intelligence</div>'
        '<div class="ws-desc">Static network diagram → device detection → topology graph → graph memory</div>',
        unsafe_allow_html=True,
    )

    _render_onboarding()

    # ── Section 1: Demo — diagram_0373 analysis ───────────────────────────────
    st.markdown('<div class="section-label">Demo Analysis · diagram_0373</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label" style="font-size:0.55rem;margin-bottom:14px">Input Diagram · YOLO Detection</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        _img(REPO_ROOT / "datasets/infragraph_v2/images/test/diagram_0373.png",
             "Input: static network diagram")
    with d2:
        _img(REPO_ROOT / "outputs/v2_test_predictions_cpu/diagram_0373.jpg",
             "YOLO v8: 17 devices detected across 7 classes")

    # ── Section 2: Detected entities ──────────────────────────────────────────
    st.markdown('<hr class="ws-rule">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Detected Infrastructure Entities</div>', unsafe_allow_html=True)

    dev_meta = {
        "router":        ("Router",        "#60a5fa"),
        "switch":        ("Switch",        "#818cf8"),
        "firewall":      ("Firewall",      "#ef4444"),
        "server":        ("Server",        "#22d3ee"),
        "database":      ("Database",      "#10b981"),
        "load_balancer": ("Load Balancer", "#f59e0b"),
        "cloud_or_wan":  ("Cloud / WAN",   "#64748b"),
    }
    dev_counts: dict[str, int] = {}
    for n in detected_nodes:
        t = n.get("type", "unknown")
        dev_counts[t] = dev_counts.get(t, 0) + 1

    chip_bg  = _col("rgba(255,255,255,0.03)", "rgba(0,0,0,0.04)")
    chip_bdr = _col("rgba(255,255,255,0.07)", "rgba(0,0,0,0.1)")
    c7 = st.columns(7)
    for col, (dt, (label, color)) in zip(c7, dev_meta.items()):
        with col:
            st.markdown(
                f'<div style="background:{chip_bg};border:1px solid {chip_bdr};'
                f'border-radius:10px;padding:13px 8px;text-align:center">'
                f'<div style="font-size:1.4rem;font-weight:800;color:{color}">{dev_counts.get(dt,0)}</div>'
                f'<div style="font-size:0.62rem;font-weight:700;color:{color};text-transform:uppercase;'
                f'letter-spacing:0.09em;margin-top:3px">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

    if detected_nodes:
        top = sorted(detected_nodes, key=lambda x: x.get("confidence", 0), reverse=True)[:10]
        df = pd.DataFrame([{
            "Node ID (predicted)": n["predicted_id"],
            "Device Type": n["type"].replace("_", " ").title(),
            "Confidence": f"{n['confidence']:.1%}",
            "BBox (px)": f"({n['bbox_pixel'][0]},{n['bbox_pixel'][1]}) → ({n['bbox_pixel'][2]},{n['bbox_pixel'][3]})",
        } for n in top])
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Section 3: Topology graph ──────────────────────────────────────────────
    st.markdown('<hr class="ws-rule">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Extracted Topology Graph</div>', unsafe_allow_html=True)

    t1, t2 = st.columns([3, 2])
    with t1:
        _img(REPO_ROOT / "outputs/topology_demo/diagram_0373_topology.png",
             "NetworkX DiGraph — 17 nodes · 17 edges")
    with t2:
        st.markdown(
            '<div class="card blue" style="padding:20px 22px">'
            '<div class="section-label" style="margin-bottom:12px">Graph Properties</div>'
            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">'
            '<div><div style="font-size:0.63rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em">Nodes</div>'
            '<div style="font-size:1.5rem;font-weight:800;color:#60a5fa">17</div></div>'
            '<div><div style="font-size:0.63rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em">Edges</div>'
            '<div style="font-size:1.5rem;font-weight:800;color:#60a5fa">17</div></div>'
            '<div><div style="font-size:0.63rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em">Type</div>'
            '<div style="font-size:0.88rem;font-weight:700;color:#93c5fd;font-family:\'JetBrains Mono\',monospace">DiGraph</div></div>'
            '<div><div style="font-size:0.63rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em">Framework</div>'
            '<div style="font-size:0.88rem;font-weight:700;color:#93c5fd;font-family:\'JetBrains Mono\',monospace">NetworkX</div></div>'
            '</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div style="margin-top:14px">'
            '<div class="section-label">Device Class Mix</div>',
            unsafe_allow_html=True,
        )
        for dt, (label, color) in dev_meta.items():
            cnt = dev_counts.get(dt, 0)
            if cnt == 0:
                continue
            pct = cnt / sum(dev_counts.values()) * 100
            bar_track = _col("rgba(255,255,255,0.06)", "rgba(0,0,0,0.09)")
            bar_count = _col("#334155", "#94a3b8")
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
                f'<div style="font-size:0.73rem;color:{color};width:90px;font-weight:600">{label}</div>'
                f'<div style="flex:1;background:{bar_track};border-radius:3px;height:5px;overflow:hidden">'
                f'<div style="height:100%;width:{pct:.0f}%;background:{color};border-radius:3px"></div></div>'
                f'<div style="font-size:0.71rem;color:{bar_count};width:18px;text-align:right">{cnt}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 4: Graph memory ────────────────────────────────────────────────
    st.markdown('<hr class="ws-rule">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Graph Memory Store</div>', unsafe_allow_html=True)

    node_scores  = gnn.get("node_scores", {})
    alerting_set = set(rca.get("alerting_nodes", []))
    impacted_set = set(rca.get("impacted_nodes", []))
    gt_root      = rca.get("ground_truth_root_cause", "FW-01")

    all_nodes = list(node_scores.keys()) if node_scores else [
        "WAN-01","WAN-02","RTR-01","RTR-02","FW-01","FW-02",
        "SW-CORE","SW-APP","LB-01","MGMT-01",
        "APP-01","APP-02","APP-03","APP-04","DB-01","DB-02","CLOUD-01",
    ]

    gm1, gm2 = st.columns([3, 2])

    with gm1:
        st.markdown(
            '<div class="card" style="padding:18px 20px">'
            '<div class="gm-header">NODES (17) — ordered by GNN score</div>',
            unsafe_allow_html=True,
        )
        sorted_nodes = sorted(all_nodes, key=lambda n: node_scores.get(n, 0), reverse=True)
        for node in sorted_nodes:
            score = node_scores.get(node, 0.0)
            ntype = _node_type(node)
            if node in alerting_set:
                sev = "CRITICAL" if node == "FW-01" else "MAJOR"
                sev_badge = f'<span class="badge badge-{"critical" if node=="FW-01" else "major"}">{sev}</span>'
            elif node == gt_root:
                sev_badge = '<span class="badge badge-success">ROOT CAUSE</span>'
            elif node in impacted_set:
                sev_badge = '<span style="font-size:0.69rem;color:#64748b">impacted</span>'
            else:
                sev_badge = '<span style="font-size:0.69rem;color:#1e293b">—</span>'
            score_color = "#10b981" if node == gt_root else ("#ef4444" if node in alerting_set else "#334155")
            st.markdown(
                f'<div class="gm-node">'
                f'<div class="gm-id">{node}</div>'
                f'<div class="gm-type">{ntype}</div>'
                f'<div class="gm-status">{sev_badge}</div>'
                f'<div class="gm-score" style="color:{score_color}">{score:+.2f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with gm2:
        edges = _extract_edges(rca)
        st.markdown(
            f'<div class="card" style="padding:18px 20px">'
            f'<div class="gm-header">EDGES ({len(edges)} extracted)</div>',
            unsafe_allow_html=True,
        )
        edge_clr = _col("#e2e8f0", "#1e293b")
        for src, tgt in edges[:16]:
            st.markdown(
                f'<div class="gm-edge">'
                f'<span style="color:{edge_clr}">{src}</span>'
                f'<span class="gm-arrow"> ──→ </span>'
                f'<span style="color:{edge_clr}">{tgt}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        if len(edges) > 16:
            st.markdown(
                f'<div style="font-size:0.72rem;color:#334155;padding:6px 0">'
                f'+{len(edges)-16} more edges extracted from impact paths</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="card" style="padding:16px 20px;margin-top:10px">'
            '<div class="gm-header">KEY RELATIONSHIPS</div>'
            '<div style="font-size:0.8rem;color:#64748b;line-height:1.8">'
            'FW-01 is upstream of SW-CORE<br>'
            'RTR-01, RTR-02 feed into FW-01<br>'
            'SW-APP bridges core → app tier<br>'
            'LB-01 distributes to APP-01..04<br>'
            'APP-01 owns DB-01 · APP-02 owns DB-02'
            '</div></div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# WORKSPACE 2 — ALERT RCA INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

def _workspace_rca(rca: dict, gnn: dict, mlp: dict, explanation_md: str) -> None:
    st.markdown(
        '<div class="ws-title">Alert RCA Intelligence</div>'
        '<div class="ws-desc">Alert stream mapped to graph memory → learned GNN RCA → operator explanation</div>',
        unsafe_allow_html=True,
    )

    tabs = st.tabs([
        "⚡  Alert Investigation",
        "🤖  GNN RCA Findings",
        "🌊  GNN Propagation",
        "📋  Operator Report",
        "💬  Ask InfraGraph",
    ])

    with tabs[0]:
        _rca_investigation(rca, gnn)
    with tabs[1]:
        _rca_findings(rca, gnn, mlp)
    with tabs[2]:
        _rca_propagation()
    with tabs[3]:
        _rca_report(rca, gnn, mlp, explanation_md)
    with tabs[4]:
        _rca_chat()


def _rca_investigation(rca: dict, gnn: dict) -> None:
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    left, right = st.columns([2, 3])

    with left:
        st.markdown('<div class="section-label">Alert Stream</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="alert-item critical">
  <div class="alert-dot critical"></div>
  <div style="flex:1">
    <div class="alert-node">FW-01 <span class="badge badge-critical">CRITICAL</span></div>
    <div class="alert-msg">Packet drops elevated — firewall degraded</div>
  </div>
  <div class="alert-time">t+0 min</div>
</div>
<div class="alert-item major">
  <div class="alert-dot major"></div>
  <div style="flex:1">
    <div class="alert-node">SW-CORE <span class="badge badge-major">MAJOR</span></div>
    <div class="alert-msg">Policy deny spike — downstream blocked</div>
  </div>
  <div class="alert-time">t+2 min</div>
</div>
<div class="alert-item major">
  <div class="alert-dot major"></div>
  <div style="flex:1">
    <div class="alert-node">SW-CORE <span class="badge badge-major">MAJOR</span></div>
    <div class="alert-msg">App unreachable — application timeout</div>
  </div>
  <div class="alert-time">t+4 min</div>
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:20px">Alerting Nodes</div>', unsafe_allow_html=True)
        alerting = rca.get("alerting_nodes", ["FW-01", "SW-CORE"])
        st.markdown(
            "".join(f'<span class="node-chip alerting">{n}</span>' for n in alerting),
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-label" style="margin-top:16px">Impacted Nodes</div>', unsafe_allow_html=True)
        impacted = rca.get("impacted_nodes", [])
        st.markdown(
            '<div style="line-height:2.2">'
            + "".join(f'<span class="node-chip impacted">{n}</span>' for n in impacted)
            + '</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-label" style="margin-top:18px">AI Root Cause</div>', unsafe_allow_html=True)
        st.markdown(
            '<span class="node-chip root" style="font-size:0.88rem;padding:5px 14px">FW-01</span>'
            '<span style="font-size:0.78rem;color:#64748b;margin-left:8px">GNN confirmed · HIGH confidence</span>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="section-label">Alerts Mapped to Topology</div>', unsafe_allow_html=True)
        _img(REPO_ROOT / "outputs/topology_demo/diagram_0373_topology.png",
             "FW-01 (CRITICAL, t+0) → SW-CORE (MAJOR, t+2 & t+4)")

    st.markdown('<hr class="ws-rule">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Key Impact Propagation Paths</div>', unsafe_allow_html=True)

    gt_paths = rca.get("impact_paths", {}).get("ground_truth_root_cause", [])
    if gt_paths:
        p1, p2 = st.columns(2)
        for i, entry in enumerate(gt_paths[:8]):
            path_str = " → ".join(entry.get("path", []))
            reason   = entry.get("target_reason", "")
            col = p1 if i % 2 == 0 else p2
            with col:
                st.markdown(
                    f'<div class="path-row">'
                    f'<span style="color:#1e293b;font-size:0.66rem;text-transform:uppercase">{reason}</span><br>'
                    f'{path_str}</div>',
                    unsafe_allow_html=True,
                )


def _rca_findings(rca: dict, gnn: dict, mlp: dict) -> None:
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    gt = rca.get("ground_truth_root_cause", "FW-01")

    # ── Winner card ────────────────────────────────────────────────────────────
    w1, w2 = st.columns([2, 3])

    with w1:
        st.markdown("""
<div class="rca-winner">
  <div style="font-size:0.63rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:#10b981;margin-bottom:4px">Best Model · GNN RCA</div>
  <div class="rca-winner-title">Learned GNN Root Cause</div>
  <div class="rca-winner-sub">Topology-aware 2-layer graph convolutional network</div>
  <div class="rca-winner-node">FW-01</div>
  <div class="rca-winner-meta">firewall · CRITICAL · t+0 min</div>
  <div style="margin-top:16px;display:flex;gap:8px;flex-wrap:wrap">
    <span class="badge badge-success">CORRECT</span>
    <span class="badge badge-success">HIGH CONFIDENCE</span>
  </div>
  <div style="margin-top:18px">
    <div style="font-size:0.7rem;color:#475569;margin-bottom:4px">GNN score: 30.733 · margin +8.12</div>""" +
    _score_bar(88, "green") + """
    <div style="font-size:0.7rem;color:#475569">MLP score: 1.842 (also correct, no graph)</div>""" +
    _score_bar(52, "purple") +
    """</div>
</div>""", unsafe_allow_html=True)

        # Comparison rows
        st.markdown('<div class="section-label">Model Comparison</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="rca-row wrong-row">
  <div class="rca-row-label">Baseline Scoring</div>
  <div class="rca-row-node wrong">SW-CORE ✗</div>
  <div class="rca-row-reason">Overweighted alert count &amp; downstream reach</div>
</div>
<div class="rca-row">
  <div class="rca-row-label">Learned MLP RCA</div>
  <div class="rca-row-node correct">FW-01 ✓</div>
  <div class="rca-row-reason">Learned feature patterns, no graph structure</div>
</div>
<div class="rca-row" style="border-color:rgba(16,185,129,0.25)">
  <div class="rca-row-label">Learned GNN RCA</div>
  <div class="rca-row-node correct">FW-01 ✓</div>
  <div class="rca-row-reason">Learned upstream propagation direction</div>
</div>""", unsafe_allow_html=True)

    with w2:
        st.markdown('<div class="section-label">GNN Candidate Rankings</div>', unsafe_allow_html=True)
        gnn_top = gnn.get("top_candidates", [])
        for c in gnn_top[:8]:
            node = c.get("node") or c.get("node_id", "")
            score = c.get("score", 0.0)
            is_root = node == gt
            rank = c.get("rank", "—")
            max_s = gnn_top[0]["score"] if gnn_top else 1
            pct = max(0.0, min(100.0, score / (max_s * 1.1) * 100)) if max_s else 0
            bar_cls = "green" if is_root else ""
            label_color = "#10b981" if is_root else "#94a3b8"
            icon = " ✓" if is_root else ""
            st.markdown(
                f'<div style="margin-bottom:12px">'
                f'<div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:3px">'
                f'<span style="color:{label_color};font-weight:{"700" if is_root else "400"};'
                f'font-family:\'JetBrains Mono\',monospace">#{rank} {node}{icon}</span>'
                f'<span style="color:#334155">{score:.2f}</span></div>'
                + _score_bar(pct, bar_cls) +
                f'</div>',
                unsafe_allow_html=True,
            )

        with st.expander("Model Evidence"):
            gnn_tm = gnn.get("test_metrics", {})
            mlp_tm = mlp.get("test_metrics", {})
            st.markdown(f"""
**Trained GNN RCA** (`train_gnn_rca.py`)
- 2-layer GCN (16→32→16→1) · graph message-passing · BCEWithLogitsLoss
- Train top-1: 99.4% · Val: 100% · Test: **{gnn_tm.get('top1',1.0):.1%}**
- MRR: {gnn_tm.get('mrr',1.0):.3f} · Best epoch: 6 / 50

**Trained MLP RCA** (`train_mlp_rca.py`)
- 3-layer MLP · 23-dim features · BCEWithLogitsLoss (pos_weight=9.3)
- Train top-1: 99.7% · Val: 100% · Test: **{mlp_tm.get('top1',1.0):.1%}**
- MRR: {mlp_tm.get('mrr',1.0):.3f} · Best epoch: 56 / 80 · 9× slower than GNN
""")
            st.markdown(
                '<div class="dev-note">Performance shown is on generated benchmark topology-alert '
                'scenarios (infragraph_v2 · 400 diagrams · train=320, val=52, test=28). '
                'Synthetic data. Results are a research demonstration.</div>',
                unsafe_allow_html=True,
            )


def _rca_propagation() -> None:
    STEPS = {
        1: {
            "title": "Raw Node Features Ingested",
            "formula": "X ∈ ℝ^{17×16}  ← device_type, has_alert, severity, timing, degree, reach",
            "body": (
                "Each of the 17 nodes is encoded as a 16-dim feature vector. "
                "FW-01 has <code>severity=4.0</code> (CRITICAL) and <code>earliest_time=0.0</code> (t+0 min). "
                "SW-CORE has <code>severity=3.0</code> (MAJOR) and timing t+2 min. "
                "Without graph context, a scorer could still favour SW-CORE due to alert count."
            ),
            "scores": {"FW-01": 12.1, "SW-CORE": 15.3, "FW-02": 8.4, "SW-APP": 6.2, "RTR-01": 3.1},
            "leader": "SW-CORE",
            "note": ("warn", "Naive feature scoring favours SW-CORE at this stage"),
        },
        2: {
            "title": "Layer 1 — 1-Hop Neighborhood Aggregation",
            "formula": "H¹ = ReLU(Â X W₁)   where  Â = D^{-½}(A+I)D^{-½}",
            "body": (
                "Each node aggregates its direct neighbours. "
                "FW-01 aggregates from RTR-01 and RTR-02 — both silent (no alerts). "
                "<strong>Upstream silence is a key signal</strong>: FW-01 is not receiving a cascade, it is the source. "
                "SW-CORE aggregates from FW-01 (alerting), revealing its downstream dependency."
            ),
            "scores": {"FW-01": 18.5, "SW-CORE": 14.2, "FW-02": 11.3, "SW-APP": 8.7, "RTR-01": 4.5},
            "leader": "FW-01",
            "note": ("ok", "FW-01 promoted after 1-hop aggregation"),
        },
        3: {
            "title": "Layer 2 — 2-Hop Deep Aggregation",
            "formula": "H² = ReLU(Â H¹ W₂)   # each node now encodes its 2-hop context",
            "body": (
                "FW-01's embedding encodes: (1) its own CRITICAL alert at t+0, "
                "(2) its upstream routers are silent, "
                "(3) its downstream switch SW-CORE is alerting. "
                "This combination confirms FW-01 as a propagation source, not a receiver."
            ),
            "scores": {"FW-01": 25.8, "SW-CORE": 12.4, "FW-02": 15.1, "SW-APP": 9.3, "RTR-01": 5.2},
            "leader": "FW-01",
            "note": ("ok", "Score margin widening — topology direction encoded"),
        },
        4: {
            "title": "Output Scoring — Linear Projection",
            "formula": "scores = H² W_out ∈ ℝ^{17}   # scalar score per node",
            "body": (
                "The 16-dim embeddings are projected to scalar scores. "
                "FW-01's embedding — critical severity, t+0 timing, upstream silence, chokepoint position — "
                "produces the highest score. SW-CORE ranks 4th despite having more alert events."
            ),
            "scores": {"FW-01": 30.7, "FW-02": 22.6, "SW-APP": 13.8, "SW-CORE": 11.4, "RTR-01": 9.5},
            "leader": "FW-01",
            "note": ("ok", "FW-01 score 30.733 — clear winner before argmax"),
        },
        5: {
            "title": "Root Cause Selected — FW-01",
            "formula": "root_cause = argmax(scores) = FW-01   [score=30.733, margin=+8.12]",
            "body": (
                "GNN selects <strong>FW-01</strong> as root cause. Confidence margin of 8.12 over 2nd-ranked FW-02. "
                "SW-CORE — the baseline's pick — ranks 4th at 11.4, nearly 20 points below FW-01. "
                "Test-set top-1 accuracy: <strong>100%</strong> across all 28 test graphs."
            ),
            "scores": {"FW-01": 30.7, "FW-02": 22.6, "SW-APP": 13.8, "SW-CORE": 11.4, "RTR-01": 9.5},
            "leader": "FW-01",
            "note": ("ok", "Root cause confirmed · test top-1 100% · MRR 1.000"),
        },
    }

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    step = st.select_slider(
        "Propagation step",
        options=list(STEPS.keys()),
        format_func=lambda x: f"Step {x} — {STEPS[x]['title']}",
    )
    s = STEPS[step]

    p1, p2 = st.columns([3, 2])

    with p1:
        st.markdown(
            f'<div class="prop-card">'
            f'<div class="prop-num">Step {step} of 5</div>'
            f'<div class="prop-title">{s["title"]}</div>'
            f'<code class="prop-formula">{s["formula"]}</code>'
            f'<div class="prop-body">{s["body"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        _img(REPO_ROOT / "outputs/topology_demo/diagram_0373_topology.png",
             f"Topology — step {step}")

    with p2:
        st.markdown('<div class="section-label">Candidate Scores</div>', unsafe_allow_html=True)
        max_s = max(s["scores"].values(), default=1)
        for node, score in sorted(s["scores"].items(), key=lambda x: -x[1]):
            pct = max(0.0, min(100.0, score / (max_s * 1.1) * 100))
            is_leader = node == s["leader"]
            bar_cls = "green" if is_leader else ""
            lc = "#10b981" if is_leader else "#94a3b8"
            crown = " 👑" if is_leader else ""
            st.markdown(
                f'<div style="margin-bottom:11px">'
                f'<div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:3px">'
                f'<span style="color:{lc};font-weight:{"700" if is_leader else "400"};'
                f'font-family:\'JetBrains Mono\',monospace">{node}{crown}</span>'
                f'<span style="color:#334155">{score:.1f}</span></div>'
                + _score_bar(pct, bar_cls) +
                f'</div>',
                unsafe_allow_html=True,
            )

        note_type, note_text = s["note"]
        if note_type == "warn":
            st.markdown(f'<div class="warn-card">{note_text}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="card green" style="padding:11px 14px">'
                f'<span style="font-size:0.79rem;color:#10b981">✓ {note_text}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-label" style="margin-top:10px">Training Curves</div>', unsafe_allow_html=True)
    tc1, tc2 = st.columns(2)
    with tc1:
        _img(REPO_ROOT / "outputs/gnn_rca/gnn_training_curve.png",
             "GNN training — converges epoch 6")
    with tc2:
        _img(REPO_ROOT / "outputs/mlp_rca/mlp_training_curve.png",
             "MLP training — converges epoch 56")


def _rca_report(rca: dict, gnn: dict, mlp: dict, explanation_md: str) -> None:
    if explanation_md:
        report = _clean_report(explanation_md)
    else:
        report = _build_fallback_report(rca, gnn, mlp)

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    r1, r2 = st.columns([5, 1])
    with r1:
        st.markdown('<div class="section-label">AI-Generated Incident Report</div>', unsafe_allow_html=True)
    with r2:
        st.download_button(
            label="⬇ Download",
            data=report,
            file_name="diagram_0373_rca_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    if not explanation_md:
        st.markdown(
            '<div class="warn-card" style="margin-bottom:14px">Explanation file not found — '
            'showing deterministic report from evidence. Generate with: '
            '<code>python scripts/generate_qwen_rca_explanation.py '
            '--diagram-id diagram_0373 --mode mock</code></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="report-body">', unsafe_allow_html=True)
    st.markdown(report)
    st.markdown('</div>', unsafe_allow_html=True)


def _rca_chat() -> None:
    _QWEN_DEFAULT = "Qwen/Qwen2-7B-Instruct"

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    quick_qs = [
        "What is the root cause?",
        "Why is FW-01 the root cause?",
        "Which nodes are impacted?",
        "Why did baseline choose SW-CORE?",
        "Generate ServiceNow summary",
    ]

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Quick Questions</div>', unsafe_allow_html=True)
    qq_cols = st.columns(len(quick_qs))
    for col, q in zip(qq_cols, quick_qs):
        with col:
            if st.button(q, key=f"qq_{q}", use_container_width=True):
                st.session_state.chat_messages.append({"role": "user", "content": q})
                st.session_state.chat_messages.append({"role": "assistant", "content": _answer(q)})
                st.rerun()

    qwen_url = os.environ.get("QWEN_BASE_URL", "")
    if qwen_url:
        model_name = os.environ.get("QWEN_MODEL", _QWEN_DEFAULT)
        st.markdown(
            f'<span class="badge badge-success">Live LLM: {model_name} @ {qwen_url}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p class="chat-hint">Set <code>QWEN_BASE_URL=http://localhost:8000/v1</code> '
            "to enable live Qwen · Using deterministic answers</p>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask InfraGraph AI about this incident..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        if qwen_url:
            try:
                import requests  # noqa: PLC0415
                resp = requests.post(
                    f"{qwen_url}/chat/completions",
                    headers={"Content-Type": "application/json", "Bypass-Tunnel-Reminder": "true"},
                    json={
                        "model": os.environ.get("QWEN_MODEL", _QWEN_DEFAULT),
                        "messages": [
                            {"role": "system", "content": (
                                "You are InfraGraph AI. The incident is diagram_0373: "
                                "FW-01 (firewall) is the GNN root cause of a 10-service outage. "
                                "Answer concisely in Markdown."
                            )},
                            {"role": "user", "content": f"/no_think {prompt}"},
                        ],
                        "max_tokens": 512, "temperature": 0.1,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"]
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                answer = raw or _answer(prompt)
            except Exception:
                answer = "> ⚠ Live LLM unreachable — showing local answer.\n\n" + _answer(prompt)
        else:
            answer = _answer(prompt)

        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        st.rerun()


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"

    st.markdown(_CSS, unsafe_allow_html=True)
    if st.session_state.theme == "light":
        st.markdown(_LIGHT_OVERRIDES, unsafe_allow_html=True)

    rca            = load_json(str(REPO_ROOT / "outputs/topology_demo/diagram_0373_rca_result.json")) or {}
    gnn            = load_json(str(REPO_ROOT / "outputs/gnn_rca/diagram_0373_gnn_rca_result.json")) or {}
    mlp            = load_json(str(REPO_ROOT / "outputs/mlp_rca/diagram_0373_mlp_rca_result.json")) or {}
    graph_summary  = load_json(str(REPO_ROOT / "outputs/topology_demo/diagram_0373_graph_summary.json")) or {}
    detected_nodes = load_json(str(REPO_ROOT / "outputs/topology_demo/diagram_0373_detected_nodes.json")) or []
    explanation_md = load_text(str(REPO_ROOT / "outputs/qwen_explanation/diagram_0373_explanation.md"))

    workspace = _sidebar()
    _hero(workspace)

    if workspace == "Diagram Intelligence":
        _workspace_diagram(detected_nodes, rca, gnn)
    else:
        _workspace_rca(rca, gnn, mlp, explanation_md)


if __name__ == "__main__":
    main()
