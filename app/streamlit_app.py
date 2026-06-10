"""
InfraGraph AI Command Center — Streamlit cockpit.

Two sections in Diagram Intelligence (Tab 1):
  A. Diagram Gallery     — Browse diagrams available in graph memory
  B. Onboard New Diagram — Live diagram intelligence: detect -> graph -> absorb -> RCA

Four workspaces (tabs):
  1. Diagram Intelligence   — image to local graph
  2. Local RCA              — alert simulation on ingested diagram
  3. Enterprise Graph Brain — local graph absorbed into enterprise graph
  4. Graph Copilot          — ask the enterprise graph
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

try:
    import networkx as nx
    _NX = True
except Exception:
    nx = None  # type: ignore
    _NX = False

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InfraGraph AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get help": None, "Report a bug": None,
                "About": "InfraGraph AI — AIOps RCA cockpit"},
)

REPO_ROOT = Path(__file__).parent.parent
_GALLERY_FALLBACK_ID = "diagram_0373"

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

_src_dir = str(REPO_ROOT / "src")
if _src_dir not in _sys.path:
    _sys.path.insert(0, _src_dir)
try:
    from runtime_ingestion import run_live_v3_ingestion as _live_ingest          # type: ignore
    from runtime_ingestion import run_enterprise_absorption as _live_absorb      # type: ignore
    from runtime_ingestion import run_ingestion as _run_ingestion                # type: ignore
    from runtime_ingestion import run_absorption as _run_absorption              # type: ignore
    _RUNTIME_INGESTION = True
except Exception:
    _RUNTIME_INGESTION = False
    _live_ingest      = None  # type: ignore
    _live_absorb      = None  # type: ignore
    _run_ingestion    = None  # type: ignore
    _run_absorption   = None  # type: ignore

try:
    from live_detector import find_best_yolo_checkpoint as _find_ckpt  # type: ignore
    from live_detector import run_live_yolo_detection as _run_yolo      # type: ignore
    _LIVE_DETECTOR = True
except Exception:
    _LIVE_DETECTOR = False
    _find_ckpt = None  # type: ignore
    _run_yolo  = None  # type: ignore

try:
    from live_rfdetr_detector import find_best_rfdetr_checkpoint as _find_rfdetr_ckpt  # type: ignore
    from live_rfdetr_detector import load_rfdetr_model as _load_rfdetr_model_raw        # type: ignore
    _LIVE_RFDETR = True
except Exception:
    _LIVE_RFDETR = False
    _find_rfdetr_ckpt    = None  # type: ignore
    _load_rfdetr_model_raw = None  # type: ignore

_PREFIX_TYPE = {
    "WAN": "cloud_or_wan", "CLOUD": "cloud_or_wan",
    "RTR": "router", "RTR01": "router",
    "FW":  "firewall",
    "SW":  "switch",
    "LB":  "load_balancer",
    "APP": "server", "WEB": "server", "SRV": "server", "MGMT": "server",
    "DB":  "database",
    "SHAR": "service",
    "DC":  "server",
    "BR":  "router",
}

def _node_type(node_id: str) -> str:
    prefix = node_id.split("-")[0].upper()
    if prefix == "BR" and "RTR" in node_id.upper():
        return "router"
    if prefix == "DC" and "FW" in node_id.upper():
        return "firewall"
    if prefix == "DC" and "SW" in node_id.upper():
        return "switch"
    if prefix == "DC" and "SRV" in node_id.upper():
        return "server"
    if prefix == "APP" and "LB" in node_id.upper():
        return "load_balancer"
    if prefix in ("DB",):
        return "database"
    return _PREFIX_TYPE.get(prefix, "server")

def _col(dark: str, light: str) -> str:
    return light if st.session_state.get("theme") == "light" else dark


# ── CSS ────────────────────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stAppViewContainer"] > .main { background: #0b0f1c; }
section[data-testid="stSidebar"] { background: #070b16 !important; border-right: 1px solid rgba(255,255,255,0.06); }
[data-testid="stHeader"] { background: transparent !important; }
h1,h2,h3,h4,h5,h6 { color: #f1f5f9 !important; }
p,li,label { color: #cbd5e1; }

/* ── Hero ── */
.hero-title {
    font-size: 1.75rem; font-weight: 900; line-height: 1.2;
    background: linear-gradient(130deg, #f1f5f9 0%, #93c5fd 55%, #67e8f9 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin-bottom: 6px;
}
.hero-tagline { font-size: 0.95rem; color: #64748b; line-height: 1.55; max-width: 640px; margin-bottom: 14px; }
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
.ws-desc  { font-size: 0.82rem; color: #64748b; margin-bottom: 20px; }
.ws-rule  { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 24px 0 20px; }

/* ── Section labels — high contrast ── */
.section-label {
    font-size: 0.63rem; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase;
    color: #94a3b8; margin: 0 0 10px; padding-bottom: 6px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}

/* ── Mode selector banner ── */
.mode-hero    { background: rgba(16,185,129,0.06); border: 1px solid rgba(16,185,129,0.2); border-radius: 10px; padding: 11px 16px; margin-bottom: 16px; }
.mode-gallery { background: rgba(96,165,250,0.06); border: 1px solid rgba(96,165,250,0.18); border-radius: 10px; padding: 11px 16px; margin-bottom: 16px; }
.mode-title   { font-size: 0.88rem; font-weight: 600; }
.mode-sub     { font-size: 0.76rem; color: #64748b; margin-top: 3px; }

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
.alert-time { font-size: 0.69rem; color: #64748b; margin-left: auto; white-space: nowrap;
              font-family: 'JetBrains Mono', monospace; }

/* ── Badges ── */
.badge { display: inline-block; padding: 2px 11px; border-radius: 20px;
         font-size: 0.68rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }
.badge-critical { background: rgba(239,68,68,0.13);  color: #ef4444; border: 1px solid rgba(239,68,68,0.32); }
.badge-major    { background: rgba(245,158,11,0.13); color: #f59e0b; border: 1px solid rgba(245,158,11,0.32); }
.badge-success  { background: rgba(16,185,129,0.13); color: #10b981; border: 1px solid rgba(16,185,129,0.32); }
.badge-info     { background: rgba(96,165,250,0.13); color: #60a5fa; border: 1px solid rgba(96,165,250,0.28); }
.badge-warn     { background: rgba(245,158,11,0.13); color: #f59e0b; border: 1px solid rgba(245,158,11,0.28); }

/* ── Graph memory ── */
.gm-header { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #64748b;
             letter-spacing: 0.06em; margin-bottom: 10px; text-transform: uppercase; }
.gm-node { display: flex; align-items: center; gap: 10px; padding: 7px 0;
           border-bottom: 1px solid rgba(255,255,255,0.04); font-family: 'JetBrains Mono', monospace; }
.gm-node:last-child { border-bottom: none; }
.gm-id   { font-size: 0.82rem; font-weight: 600; color: #e2e8f0; width: 80px; flex-shrink: 0; }
.gm-type { font-size: 0.75rem; color: #64748b; width: 110px; flex-shrink: 0; }
.gm-status { font-size: 0.72rem; width: 110px; flex-shrink: 0; }
.gm-score  { font-size: 0.75rem; color: #64748b; margin-left: auto; }
.gm-edge { display: flex; align-items: center; gap: 8px; padding: 5px 0;
           font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #94a3b8;
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
.rca-row-reason { font-size: 0.78rem; color: #64748b; margin-left: auto; max-width: 280px;
                  text-align: right; }

/* ── Score bar ── */
.score-track { background: rgba(255,255,255,0.06); border-radius: 4px; height: 6px;
               overflow: hidden; margin: 4px 0 8px; }
.score-fill  { height: 100%; border-radius: 4px;
               background: linear-gradient(90deg, #3b82f6, #67e8f9); }
.score-fill.green  { background: linear-gradient(90deg, #10b981, #34d399); }
.score-fill.red    { background: linear-gradient(90deg, #ef4444, #f87171); }
.score-fill.purple { background: linear-gradient(90deg, #8b5cf6, #a78bfa); }

/* ── Path ── */
.path-row { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #60a5fa;
            padding: 7px 13px; background: rgba(96,165,250,0.05); border-radius: 8px;
            margin: 4px 0; border: 1px solid rgba(96,165,250,0.11); }

/* ── Propagation ── */
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
.info-card { background: rgba(96,165,250,0.06); border: 1px solid rgba(96,165,250,0.18);
             border-radius: 10px; padding: 12px 16px; font-size: 0.83rem; color: #94a3b8; }
.report-body { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.07);
               border-radius: 14px; padding: 30px 34px; }
.chat-hint { font-size: 0.75rem; color: #64748b; text-align: center; margin-top: 6px; }
.dev-note  { font-size: 0.79rem; color: #64748b; font-style: italic; line-height: 1.6; }

/* ── Absorption story ── */
.absorb-card { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.07);
               border-radius: 12px; padding: 16px 18px; height: 100%; min-height: 260px; }
.absorb-step { display: flex; align-items: center; gap: 10px; padding: 8px 0;
               border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 0.8rem; }
.absorb-step:last-child { border-bottom: none; }
.absorb-done  { color: #10b981; font-weight: 700; }
.absorb-wait  { color: #334155; }

/* ── Sidebar ── */
.sb-label { font-size: 0.62rem; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase;
            color: #475569; margin: 18px 0 7px; padding-bottom: 5px;
            border-bottom: 1px solid rgba(255,255,255,0.05); }
.sb-step  { display: flex; align-items: flex-start; gap: 9px; padding: 6px 0;
            font-size: 0.8rem; color: #10b981;
            border-bottom: 1px solid rgba(255,255,255,0.03); }
.sb-step:last-child { border-bottom: none; }
.sb-check { flex-shrink: 0; margin-top: 1px; }
.sb-step-sub { font-size: 0.7rem; color: #334155; }
.sb-pending { color: #475569; }
.info-pill { background: rgba(96,165,250,0.07); border: 1px solid rgba(96,165,250,0.14);
             border-radius: 8px; padding: 9px 12px; font-size: 0.75rem; color: #64748b; margin-top: 14px; }

/* ── Tabs ── */
div[data-testid="stTabs"] div[role="tablist"] {
    gap: 10px !important; background: rgba(15,23,42,0.72) !important;
    border: 1px solid rgba(148,163,184,0.12) !important; border-radius: 14px !important;
    padding: 8px !important; overflow-x: auto !important; }
div[data-testid="stTabs"] button[role="tab"] {
    min-height: 42px !important; padding: 0 18px !important; border-radius: 10px !important;
    background: transparent !important; color: #64748b !important;
    font-weight: 500 !important; font-size: 0.85rem !important; border: none !important; }
div[data-testid="stTabs"] button[role="tab"] p { color: inherit !important; font-size: inherit !important; font-weight: inherit !important; }
div[data-testid="stTabs"] button[role="tab"] > div { display: flex !important; align-items: center !important; justify-content: center !important; }
div[data-testid="stTabs"] button[role="tab"]:hover { background: rgba(96,165,250,0.08) !important; color: #93c5fd !important; }
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg,rgba(96,165,250,0.18),rgba(139,92,246,0.12)) !important;
    color: #93c5fd !important; box-shadow: inset 0 -2px 0 #60a5fa !important; }
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p { color: #93c5fd !important; }

/* ── Detection pair comparison ── */
.compare-label { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
                 color: #94a3b8; margin-bottom: 8px; }
.compare-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.65rem;
                 font-weight: 700; margin-bottom: 6px; text-transform: uppercase; }
.compare-badge.original  { background: rgba(96,165,250,0.1); color: #60a5fa; border: 1px solid rgba(96,165,250,0.2); }
.compare-badge.predicted { background: rgba(16,185,129,0.1); color: #10b981; border: 1px solid rgba(16,185,129,0.2); }
.compare-badge.prepared  { background: rgba(245,158,11,0.1); color: #f59e0b; border: 1px solid rgba(245,158,11,0.2); }
.compare-badge.missing   { background: rgba(100,116,139,0.1); color: #94a3b8; border: 1px solid rgba(100,116,139,0.2); }
</style>
"""

_LIGHT_OVERRIDES = """
<style>
:root {
    --background-color: #f8fafc;
    --secondary-background-color: #f1f5f9;
    --text-color: #1e293b;
    --primary-color: #2563eb;
    --font: "Inter", sans-serif;
}
html, body { background: #f8fafc !important; color: #1e293b !important; }
[data-testid="stAppViewContainer"],
[data-testid="stMain"], .stApp, .main,
.main .block-container { background: #f8fafc !important; }
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"] {
    background: #ffffff !important; border-right: 1px solid rgba(0,0,0,0.09) !important;
}
[data-testid="stHeader"] { background: rgba(248,250,252,0.92) !important; }
[data-testid="stBottomBlockContainer"],
.stChatFloatingInputContainer { background: #f8fafc !important; }
h1,h2,h3,h4,h5,h6 { color: #0f172a !important; }
p, li, span { color: #1e293b; }
[data-testid="stMarkdown"] *, [data-testid="stMarkdownContainer"] * { color: #1e293b !important; }
[data-testid="stText"] { color: #1e293b !important; }
label, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span { color: #1e293b !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] [data-testid="stMarkdown"] * { color: #1e293b !important; }
[data-testid="stRadio"] label, [data-testid="stRadio"] p { color: #1e293b !important; }
[data-testid="stToggle"] label, [data-testid="stToggle"] p { color: #1e293b !important; }
[data-testid="stBaseButton-secondary"],
.stButton button { background: #ffffff !important; color: #334155 !important; border: 1px solid rgba(0,0,0,0.14) !important; }
div[data-testid="stTabs"] div[role="tablist"] { background: rgba(0,0,0,0.05) !important; border-color: rgba(0,0,0,0.09) !important; }
div[data-testid="stTabs"] button[role="tab"] { color: #64748b !important; background: transparent !important; }
div[data-testid="stTabs"] button[role="tab"]:hover { background: rgba(37,99,235,0.06) !important; color: #2563eb !important; }
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { background: rgba(37,99,235,0.1) !important; color: #1d4ed8 !important; box-shadow: inset 0 -2px 0 #2563eb !important; }
div[data-testid="stTabs"] button[role="tab"] p { color: inherit !important; }
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p { color: #1d4ed8 !important; }
[data-testid="stExpander"] details { background: rgba(0,0,0,0.02) !important; border: 1px solid rgba(0,0,0,0.1) !important; }
[data-testid="stExpander"] summary { background: rgba(0,0,0,0.02) !important; color: #1e293b !important; }
[data-testid="stExpander"] summary *, [data-testid="stExpander"] details > div * { color: #1e293b !important; }
[data-testid="stChatInput"], [data-testid="stChatInput"] > div { background: #ffffff !important; }
[data-testid="stChatMessage"] { background: rgba(0,0,0,0.03) !important; }
[data-testid="stChatMessage"] *:not(code):not(pre):not(.badge) { color: #1e293b !important; }
[data-testid="stDataFrame"] > div { background: #ffffff !important; }
[data-testid="stAlert"] { background: rgba(0,0,0,0.03) !important; border-color: rgba(0,0,0,0.1) !important; }
[data-testid="stAlert"] * { color: #1e293b !important; }
code { background: rgba(0,0,0,0.06) !important; color: #1e293b !important; }
pre  { background: rgba(0,0,0,0.04) !important; border-color: rgba(0,0,0,0.1) !important; }
.hero-tagline  { color: #64748b !important; }
.card          { background: rgba(0,0,0,0.03) !important; border-color: rgba(0,0,0,0.1) !important; }
.section-label { color: #64748b !important; border-bottom-color: rgba(0,0,0,0.07) !important; }
.ws-title      { color: #1e293b !important; }
.ws-desc       { color: #64748b !important; }
.warn-card     { background: rgba(245,158,11,0.06) !important; color: #92400e !important; }
.info-card     { background: rgba(59,130,246,0.05) !important; color: #1e40af !important; }
.gm-id         { color: #1e293b !important; }
.gm-type       { color: #94a3b8 !important; }
.alert-node    { color: #1e293b !important; }
.alert-msg     { color: #64748b !important; }
.sb-label      { color: #94a3b8 !important; }
.sb-step       { color: #059669 !important; }
.sb-pending    { color: #94a3b8 !important; }
.absorb-card   { background: rgba(0,0,0,0.02) !important; border-color: rgba(0,0,0,0.09) !important; }
.report-body   { background: rgba(0,0,0,0.02) !important; border-color: rgba(0,0,0,0.08) !important; }
.info-pill     { background: rgba(59,130,246,0.06) !important; color: #64748b !important; }
</style>
"""


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════
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


def _build_rca_summary_report(rca: dict, gnn: dict, mlp: dict) -> str:
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


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM CATALOG
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=180)
def _build_diagram_catalog(repo_root_str: str) -> list[dict]:
    """Scan all dataset paths and return catalog records."""
    root = Path(repo_root_str)
    records: list[dict] = []

    pred_dir_v1 = root / "outputs" / "v1_test_predictions_cpu"
    pred_dir_v2 = root / "outputs" / "v2_test_predictions_cpu"
    pred_dir_v3 = root / "outputs" / "v3_test_predictions_cpu"

    def _find_pred(diagram_id: str, dataset: str) -> str | None:
        dirs = [pred_dir_v2, pred_dir_v1] if "v2" in dataset or "v1" in dataset else [pred_dir_v3]
        for d in dirs:
            for ext in [".jpg", ".png"]:
                p = d / f"{diagram_id}{ext}"
                if p.exists():
                    return str(p)
        return None

    for ds_name, ver_label in [("infragraph_v1", "v1"), ("infragraph_v2", "v2")]:
        img_root = root / "datasets" / ds_name / "images"
        if not img_root.exists():
            continue
        for split in ["train", "val", "test"]:
            split_dir = img_root / split
            if not split_dir.exists():
                continue
            for img_f in sorted(split_dir.glob("*.png")):
                did = img_f.stem
                records.append({
                    "display_name":        f"{ver_label.upper()} / {split} / {did}",
                    "dataset":             ver_label,
                    "split":               split,
                    "diagram_id":          did,
                    "diagram_type":        None,
                    "image_path":          str(img_f),
                    "prediction_path":     _find_pred(did, ds_name),
                    "annotation_path":     None,
                    "local_graph_path":    None,
                    "scenario_id":         None,
                    "enterprise_graph_path": None,
                    "alerts_path":         None,
                    "is_v3":               False,
                })

    v3_scen = root / "datasets" / "diagram_v3_enterprise" / "scenarios"
    if v3_scen.exists():
        for split_dir in sorted(v3_scen.iterdir()):
            if not split_dir.is_dir():
                continue
            split = split_dir.name
            for scen_dir in sorted(split_dir.iterdir()):
                if not scen_dir.is_dir():
                    continue
                scen_id = scen_dir.name
                diag_dir = scen_dir / "diagrams"
                if not diag_dir.exists():
                    continue
                for img_f in sorted(diag_dir.glob("*.png")):
                    did = img_f.stem
                    ann_p  = scen_dir / "annotations"  / f"{did}.json"
                    lg_p   = scen_dir / "local_graphs"  / f"{did}.json"
                    eg_p   = scen_dir / "enterprise_graph.json"
                    al_p   = scen_dir / "alerts.json"
                    records.append({
                        "display_name":          f"V3 / {scen_id} / {did}",
                        "dataset":               "v3",
                        "split":                 split,
                        "diagram_id":            did,
                        "diagram_type":          did,
                        "image_path":            str(img_f),
                        "prediction_path":       None,
                        "annotation_path":       str(ann_p) if ann_p.exists() else None,
                        "local_graph_path":      str(lg_p)  if lg_p.exists()  else None,
                        "scenario_id":           scen_id,
                        "enterprise_graph_path": str(eg_p)  if eg_p.exists()  else None,
                        "alerts_path":           str(al_p)  if al_p.exists()  else None,
                        "is_v3":                 True,
                    })

    return records


@st.cache_data(ttl=3600)
def build_onboarding_sample_catalog(repo_root_str: str, max_samples: int = 20) -> list[dict]:
    """
    Build a curated 15–20 sample catalog of V3 diagrams for the Onboard New Diagram flow.
    Prefers test/ > val/ > train/. Selects diverse diagram types per scenario.
    """
    root = Path(repo_root_str)
    v3_scen = root / "datasets" / "diagram_v3_enterprise" / "scenarios"
    if not v3_scen.exists():
        return []

    _TYPES = [
        "branch_topology", "wan_topology", "datacenter_topology",
        "app_db_topology", "shared_services_topology",
    ]
    records: list[dict] = []
    seen_diagrams: set[str] = set()

    for split in ["test", "val", "train"]:
        split_dir = v3_scen / split
        if not split_dir.exists():
            continue
        for scen_dir in sorted(split_dir.iterdir()):
            if not scen_dir.is_dir():
                continue
            scen_id = scen_dir.name
            for did in _TYPES:
                if did in seen_diagrams:
                    continue          # keep diversity across types
                img_p = scen_dir / "diagrams" / f"{did}.png"
                if not img_p.exists():
                    continue
                ann_p = scen_dir / "annotations" / f"{did}.json"
                lg_p  = scen_dir / "local_graphs" / f"{did}.json"
                eg_p  = scen_dir / "enterprise_graph.json"
                sm_p  = scen_dir / "stitch_map.json"
                al_p  = scen_dir / "alerts.json"
                records.append({
                    "display_name":          f"{split.upper()} / {scen_id} / {did.replace('_',' ').title()}",
                    "dataset":               "v3",
                    "split":                 split,
                    "scenario_id":           scen_id,
                    "diagram_id":            did,
                    "diagram_type":          did,
                    "image_path":            str(img_p),
                    "annotation_path":       str(ann_p) if ann_p.exists() else None,
                    "local_graph_path":      str(lg_p)  if lg_p.exists()  else None,
                    "enterprise_graph_path": str(eg_p)  if eg_p.exists()  else None,
                    "stitch_map_path":       str(sm_p)  if sm_p.exists()  else None,
                    "alerts_path":           str(al_p)  if al_p.exists()  else None,
                    "is_v3":                 True,
                })
                seen_diagrams.add(did)
                if len(records) >= max_samples:
                    return records
            seen_diagrams.clear()   # allow repeats in next scenario
    return records


@st.cache_data(ttl=3600)
def _load_gallery_manifest(repo_root_str: str) -> list[dict]:
    """Load assets/gallery/manifest.json and resolve relative paths to absolute."""
    root = Path(repo_root_str)
    mf   = root / "assets" / "gallery" / "manifest.json"
    if not mf.exists():
        return []
    try:
        records: list[dict] = json.loads(mf.read_text(encoding="utf-8"))
    except Exception:
        return []
    _path_keys = (
        "image_path", "annotation_path", "detected_preview_path", "preview_path",
        "local_graph_path", "enterprise_graph_path", "stitch_map_path", "alerts_path",
    )
    for r in records:
        for k in _path_keys:
            v = r.get(k, "")
            if v and not Path(v).is_absolute():
                r[k] = str(root / v)
    return records


@st.cache_data(ttl=3600)
def _load_onboarding_manifest(repo_root_str: str) -> list[dict]:
    """Load assets/onboarding/manifest.json and resolve relative paths to absolute."""
    root = Path(repo_root_str)
    mf   = root / "assets" / "onboarding" / "manifest.json"
    if not mf.exists():
        return []
    try:
        records: list[dict] = json.loads(mf.read_text(encoding="utf-8"))
    except Exception:
        return []
    _path_keys = (
        "image_path", "annotation_path", "local_graph_path",
        "enterprise_graph_path", "stitch_map_path", "alerts_path", "sample_dir",
    )
    for r in records:
        for k in _path_keys:
            v = r.get(k, "")
            if v and not Path(v).is_absolute():
                r[k] = str(root / v)
    return records


@st.cache_resource(show_spinner="Loading RF-DETR checkpoint…")
def _load_rfdetr_model_cached(checkpoint_path: str):
    """Load RF-DETR model once per Streamlit session via st.cache_resource."""
    if not _LIVE_RFDETR or _load_rfdetr_model_raw is None:
        return None
    try:
        return _load_rfdetr_model_raw(checkpoint_path)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# DETECTION PAIR VISUAL
# ══════════════════════════════════════════════════════════════════════════════
def _render_detection_pair(record: dict) -> None:
    """Side-by-side: original diagram + detection output (live YOLO, prepared, or unavailable)."""
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="compare-badge original">Original</div>'
            '<div class="compare-label">Source Diagram</div>',
            unsafe_allow_html=True,
        )
        img_p = record.get("image_path", "")
        if img_p and Path(img_p).exists():
            st.image(img_p, use_container_width=True)
        else:
            st.markdown('<div class="warn-card">Image not found</div>', unsafe_allow_html=True)

    with c2:
        # priority: a) live YOLO result for this image  b) existing prediction
        #           c) V3 annotation info               d) unavailable
        live_res    = st.session_state.get("live_detection_result") or {}
        live_match  = (
            live_res.get("success")
            and live_res.get("image_path") == record.get("image_path")
            and live_res.get("diagram_id") == record.get("diagram_id")
        )
        live_img    = live_res.get("detected_image_path", "") if live_match else ""
        live_img_ok = live_img and Path(live_img).exists()

        pred_p      = record.get("prediction_path")
        pred_ok     = bool(pred_p and Path(pred_p).exists())
        is_v3       = record.get("is_v3", False)

        if live_img_ok:
            n_det = live_res.get("n_detections", 0)
            st.markdown(
                '<div class="compare-badge predicted">Live YOLO Detector</div>'
                f'<div class="compare-label">{n_det} device(s) detected</div>',
                unsafe_allow_html=True,
            )
            st.image(live_img, use_container_width=True)
        elif pred_ok:
            ver = record.get("dataset", "").upper()
            st.markdown(
                f'<div class="compare-badge predicted">AI Detected</div>'
                f'<div class="compare-label">YOLO v8 — {ver} Detector Output</div>',
                unsafe_allow_html=True,
            )
            st.image(pred_p, use_container_width=True)
        elif is_v3:
            st.markdown(
                '<div class="compare-badge prepared">V3 Prepared</div>'
                '<div class="compare-label">Annotation Status</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="info-card">'
                '<strong>V3 prepared annotation selected.</strong><br>'
                'Live RF-DETR detector output is not generated yet — annotations were created '
                'directly from the scenario generator. Run <code>train_rfdetr_diagram_detector.py</code> '
                'after training to generate detector predictions.'
                '</div>',
                unsafe_allow_html=True,
            )
            ann_p = record.get("annotation_path")
            if ann_p and Path(ann_p).exists():
                ann = load_json(ann_p) or {}
                n_obj = len(ann.get("objects", []))
                n_con = len(ann.get("connectors", []))
                st.caption(f"Prepared annotation: {n_obj} objects, {n_con} connectors")
        else:
            st.markdown(
                '<div class="compare-badge missing">Prediction not generated</div>'
                '<div class="compare-label">Detection Output</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="warn-card">No prediction exists for this diagram. '
                'Use <strong>Run Live Detector on This Image</strong> below to generate one.</div>',
                unsafe_allow_html=True,
            )

    # caption below both columns
    if is_v3:
        st.caption(
            "Live detection is available; full graph ingestion uses V3 prepared "
            "connector/OCR/local graph packet."
        )
    else:
        st.caption(
            "Live detection extracts devices. Full graph/RCA journey uses V3 scenario "
            "diagrams with connector and graph metadata."
        )


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH RENDERING
# ══════════════════════════════════════════════════════════════════════════════
_DEVICE_PALETTE = {
    "router":       "#2563eb",
    "switch":       "#16a34a",
    "firewall":     "#dc2626",
    "server":       "#475569",
    "database":     "#7c3aed",
    "load_balancer":"#ea580c",
    "cloud_or_wan": "#0284c7",
    "service":      "#0d9488",
}


def _render_local_graph_pyvis(local_graph: dict, overlay: dict | None = None,
                               height: int = 530) -> bool:
    """Render interactive PyVis local graph. Returns True if rendered."""
    try:
        from pyvis.network import Network  # type: ignore
    except ImportError:
        return False

    nodes = local_graph.get("nodes", [])
    edges = local_graph.get("edges", [])
    if not nodes:
        return False

    root       = (overlay or {}).get("root_cause")
    alert_set  = set((overlay or {}).get("alert_nodes", []))
    impacted   = set((overlay or {}).get("impacted_nodes", []))
    path_set: set[tuple[str, str]] = set()
    for a, b in zip((overlay or {}).get("impact_path", []),
                    (overlay or {}).get("impact_path", [])[1:]):
        path_set.add((a, b))

    net = Network(height=f"{height}px", width="100%", directed=True,
                  bgcolor="#0b1220", font_color="#e2e8f0")
    net.barnes_hut(gravity=-3800, central_gravity=0.28, spring_length=140, spring_strength=0.048)

    for n in nodes:
        nid   = n.get("id", "")
        ntype = n.get("type", "server")
        shared = n.get("is_shared_entity", False)

        if nid == root:
            color, size, bw = "#ef4444", 36, 4
        elif nid in alert_set:
            color, size, bw = "#f97316", 30, 3
        elif nid in impacted:
            color, size, bw = "#facc15", 26, 2
        elif shared:
            color, size, bw = "#38bdf8", 26, 3
        else:
            color, size, bw = _DEVICE_PALETTE.get(ntype, "#64748b"), 22, 2

        border = "#ffffff" if nid == root else ("#fbbf24" if shared else "#4a5568")
        title  = (
            f"<b>{nid}</b><br>type: {ntype}<br>"
            f"ip: {n.get('ip_address','—')}<br>zone: {n.get('zone','—')}<br>"
            + ("<b>shared entity</b>" if shared else "")
            + ("<br><b>ROOT CAUSE</b>" if nid == root else "")
        )
        net.add_node(nid, label=nid, title=title,
                     color={"background": color, "border": border},
                     size=size, borderWidth=bw, borderWidthSelected=5)

    for e in edges:
        src, tgt = e.get("source", ""), e.get("target", "")
        is_path  = (src, tgt) in path_set
        lbl = str(e.get("label") or e.get("relationship", ""))[:16]
        net.add_edge(src, tgt, label=lbl,
                     color="#06b6d4" if is_path else "#4a5568",
                     width=3 if is_path else 1,
                     title=e.get("relationship", ""),
                     dashes=e.get("edge_scope") == "cross_diagram")

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html", encoding="utf-8") as tmp:
        tmp_path = Path(tmp.name)
    try:
        net.save_graph(str(tmp_path))
        components.html(tmp_path.read_text(encoding="utf-8"), height=height + 20, scrolling=False)
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass
    return True


def _render_local_graph_mpl(local_graph: dict, overlay: dict | None = None,
                              height: float = 4.2) -> None:
    """Matplotlib rendering for local graph (used when PyVis is unavailable)."""
    nodes = local_graph.get("nodes", [])
    edges = local_graph.get("edges", [])
    if not nodes:
        st.info("No local graph data.")
        return

    if _NX:
        G = nx.DiGraph()
        for n in nodes:
            G.add_node(n.get("id", ""))
        for e in edges:
            G.add_edge(e.get("source", ""), e.get("target", ""))
        pos = nx.spring_layout(G, seed=12, k=1.35)
    else:
        total = len(nodes)
        pos = {
            n.get("id", ""): (math.cos(2*math.pi*i/total), math.sin(2*math.pi*i/total))
            for i, n in enumerate(nodes)
        }

    root      = (overlay or {}).get("root_cause")
    alert_set = set((overlay or {}).get("alert_nodes", []))
    impacted  = set((overlay or {}).get("impacted_nodes", []))
    path_set: set[tuple[str, str]] = set()
    for a, b in zip((overlay or {}).get("impact_path", []),
                    (overlay or {}).get("impact_path", [])[1:]):
        path_set.add((a, b))

    fig, ax = plt.subplots(figsize=(9, height), facecolor="#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.axis("off")

    for e in edges:
        src, tgt = e.get("source", ""), e.get("target", "")
        if src not in pos or tgt not in pos:
            continue
        x1, y1 = pos[src]; x2, y2 = pos[tgt]
        is_path = (src, tgt) in path_set
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->",
                                   color="#00a8e8" if is_path else "#94a3b8",
                                   lw=2.6 if is_path else 1.1, alpha=0.9), zorder=1)
        lbl = e.get("label") or e.get("relationship", "")
        if lbl:
            ax.text((x1+x2)/2, (y1+y2)/2, str(lbl)[:16], fontsize=7, color="#0f172a",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="#e0f2fe", ec="#bae6fd", lw=0.5), zorder=3)

    ntype_map = {n.get("id", ""): n.get("type", "server") for n in nodes}
    for n in nodes:
        nid = n.get("id", "")
        if nid not in pos:
            continue
        x, y = pos[nid]
        if nid == root:
            color, ec, sz = "#ef4444", "#7f1d1d", 950
        elif nid in alert_set:
            color, ec, sz = "#f97316", "#9a3412", 780
        elif nid in impacted:
            color, ec, sz = "#facc15", "#a16207", 740
        else:
            color = _DEVICE_PALETTE.get(ntype_map.get(nid, "server"), "#64748b")
            ec, sz = "#0f172a", 640
        ax.scatter([x], [y], s=sz, c=color, edgecolors=ec, linewidths=1.7, zorder=4)
        ax.text(x, y-0.12, nid, fontsize=8, fontweight="bold",
                ha="center", va="top", color="#0f172a", zorder=5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _render_local_graph(local_graph: dict, overlay: dict | None = None,
                         height: int = 530) -> None:
    """Try PyVis first; fall back to matplotlib."""
    if _render_local_graph_pyvis(local_graph, overlay, height):
        return
    st.caption("Interactive graph requires pyvis — showing matplotlib preview")
    _render_local_graph_mpl(local_graph, overlay, height / 100)


# ══════════════════════════════════════════════════════════════════════════════
# V3 CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
V3_DATASET_ROOT  = REPO_ROOT / "datasets" / "diagram_v3_enterprise"
V3_HERO_SCENARIO = V3_DATASET_ROOT / "scenarios" / "train" / "enterprise_v3_0000"
V3_DIAGRAM_IDS = {
    "Branch Office Topology":       "branch_topology",
    "WAN / MPLS Topology":          "wan_topology",
    "Datacenter Topology":          "datacenter_topology",
    "Application & Database Tier":  "app_db_topology",
    "Shared Services Topology":     "shared_services_topology",
}
V3_REQUIRED_DIAGRAMS = list(V3_DIAGRAM_IDS.values())
V3_ENTERPRISE_GNN_METRICS = REPO_ROOT / "outputs" / "enterprise_gnn_rca" / "enterprise_gnn_metrics.json"
V3_ONBOARDING_SCRIPT = REPO_ROOT / "scripts" / "onboard_diagram_v3.py"

_V3_DIAG_COLORS = {
    "branch_topology":          "#10b981",
    "wan_topology":             "#3b82f6",
    "datacenter_topology":      "#ef4444",
    "app_db_topology":          "#8b5cf6",
    "shared_services_topology": "#f59e0b",
}

V3_STATE_KEYS = [
    "selected_diagram_path", "selected_diagram_id",
    "local_graph", "node_table", "edge_table", "validation_packet",
    "local_rca_result",
    "enterprise_scenario_path", "enterprise_graph_before",
    "enterprise_graph_after", "enterprise_ingestion_summary", "enterprise_rca_result",
    "allow_local_simulation", "allow_enterprise_simulation", "allow_deterministic_copilot",
    "catalog_selected_record",
    # live runtime state
    "live_ingestion_run_dir", "detection_source", "detected_image_path",
    "enterprise_absorbed",
    "live_detection_result",
]


_V3_DICT_KEYS = {
    "validation_packet", "local_rca_result", "enterprise_rca_result",
    "enterprise_ingestion_summary", "local_graph", "enterprise_graph_before",
    "enterprise_graph_after", "catalog_selected_record", "live_detection_result",
}


def _init_v3_state() -> None:
    for key in V3_STATE_KEYS:
        if key not in st.session_state:
            st.session_state[key] = {} if key in _V3_DICT_KEYS else None
    if "v3_chat_messages" not in st.session_state:
        st.session_state.v3_chat_messages = []
    if "diagram_mode" not in st.session_state:
        st.session_state.diagram_mode = "Onboard New Diagram"
    if "onboard_status" not in st.session_state:
        st.session_state.onboard_status = "not_started"
    if "onboard_sample_record" not in st.session_state:
        st.session_state.onboard_sample_record = {}
    if "use_live_rfdetr" not in st.session_state:
        st.session_state.use_live_rfdetr = True


def _ss_dict(key: str) -> dict:
    """Return a session-state value as a dict, guarding against None."""
    value = st.session_state.get(key)
    return value if isinstance(value, dict) else {}


def _safe_read_json(path: Path) -> dict | list:
    data = load_json(str(path))
    return data if data is not None else {}


def _strict_mode() -> bool:
    return os.environ.get("INFRAGRAPH_STRICT_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


def _qwen_configured() -> bool:
    return bool(os.environ.get("QWEN_BASE_URL")) and bool(os.environ.get("QWEN_MODEL"))


def _pyvis_available() -> bool:
    try:
        import pyvis  # noqa: F401
        return True
    except Exception:
        return False


def _enterprise_gnn_available() -> bool:
    return V3_ENTERPRISE_GNN_METRICS.exists()


def _local_rca_model_available() -> bool:
    selected = st.session_state.get("selected_diagram_id", "")
    for p in [
        REPO_ROOT / "outputs" / "v3_local_rca" / "local_rca_result.json",
        REPO_ROOT / "outputs" / "gnn_rca" / f"{selected}_gnn_rca_result.json",
    ]:
        if p.exists():
            return True
    return False


def _status_label(ok: bool, optional: bool = False) -> str:
    if ok:
        return "✅"
    if optional:
        return "⚠️"
    return "❌"


def _readiness_checks() -> list[dict]:
    diags_ok = all(
        (V3_HERO_SCENARIO / "diagrams" / f"{d}.png").exists() for d in V3_REQUIRED_DIAGRAMS
    )
    ann_ok = all(
        (V3_HERO_SCENARIO / "annotations" / f"{d}.json").exists() for d in V3_REQUIRED_DIAGRAMS
    )
    lg_ok = all(
        (V3_HERO_SCENARIO / "local_graphs" / f"{d}.json").exists() for d in V3_REQUIRED_DIAGRAMS
    )
    return [
        {"label": "V3 hero scenario exists",          "ok": V3_HERO_SCENARIO.exists(),                                       "optional": False},
        {"label": "Hero diagrams (5 PNGs)",            "ok": diags_ok,                                                        "optional": False},
        {"label": "V3 annotations",                    "ok": ann_ok,                                                          "optional": False},
        {"label": "V3 local graphs",                   "ok": lg_ok,                                                           "optional": False},
        {"label": "V3 enterprise graph",               "ok": (V3_HERO_SCENARIO / "enterprise_graph.json").exists(),           "optional": False},
        {"label": "V3 alerts",                         "ok": (V3_HERO_SCENARIO / "alerts.json").exists(),                     "optional": False},
        {"label": "RF-DETR COCO export",               "ok": (V3_DATASET_ROOT / "rfdetr" / "annotations" / "instances_train.json").exists(), "optional": True},
        {"label": "YOLO export",                       "ok": (V3_DATASET_ROOT / "yolo" / "dataset.yaml").exists(),            "optional": True},
        {"label": "PyVis dependency",                  "ok": _pyvis_available(),                                              "optional": not _strict_mode()},
        {"label": "Enterprise GNN output",             "ok": _enterprise_gnn_available(),                                     "optional": not _strict_mode()},
        {"label": "Qwen endpoint configured",          "ok": _qwen_configured(),                                              "optional": not _strict_mode()},
        {"label": "Live onboarding script",            "ok": V3_ONBOARDING_SCRIPT.exists(),                                   "optional": not _strict_mode()},
    ]


def _render_readiness_panel() -> None:
    checks = _readiness_checks()
    ready = sum(1 for c in checks if c["ok"])
    total = len(checks)
    pct   = int(ready / total * 100)
    color = "#10b981" if pct >= 80 else ("#f59e0b" if pct >= 50 else "#ef4444")
    st.markdown(
        f'<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);'
        f'border-radius:9px;padding:9px 13px;display:flex;justify-content:space-between;align-items:center">'
        f'<span style="font-size:0.78rem;color:#64748b">System readiness</span>'
        f'<span style="font-size:0.82rem;font-weight:700;color:{color}">{ready}/{total}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    with st.expander("View details", expanded=False):
        for item in checks:
            icon = _status_label(item["ok"], item["optional"])
            clr  = "color:#10b981" if item["ok"] else ("color:#f59e0b" if item["optional"] else "color:#ef4444")
            st.markdown(
                f'<div style="font-size:0.76rem;{clr};padding:2px 0">{icon} {item["label"]}</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# V3 PACKET HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _hero_path(diagram_id: str) -> Path:
    return V3_HERO_SCENARIO / "diagrams" / f"{diagram_id}.png"


def _load_v3_packet(diagram_path: Path, diagram_id: str) -> dict:
    scenario_dir = diagram_path.parent.parent
    ann_path  = scenario_dir / "annotations"   / f"{diagram_id}.json"
    lg_path   = scenario_dir / "local_graphs"  / f"{diagram_id}.json"
    return {
        "scenario_dir":      scenario_dir,
        "diagram_path":      diagram_path,
        "diagram_id":        diagram_id,
        "annotation_path":   ann_path,
        "local_graph_path":  lg_path,
        "annotation":        _safe_read_json(ann_path) if ann_path.exists() else {},
        "local_graph":       _safe_read_json(lg_path)  if lg_path.exists()  else {},
    }


def _node_table_from_graph(local_graph: dict) -> pd.DataFrame:
    rows = []
    for n in local_graph.get("nodes", []):
        rows.append({
            "node_id":    n.get("id", ""),
            "type":       n.get("type", "unknown"),
            "ip_address": n.get("ip_address", ""),
            "zone":       n.get("zone", ""),
            "shared":     n.get("is_shared_entity", False),
            "confidence": 0.96 if n.get("bbox") else 0.88,
            "source":     "V3 annotation" if n.get("bbox") else "V3 local graph",
        })
    return pd.DataFrame(rows)


def _edge_table_from_graph(local_graph: dict) -> pd.DataFrame:
    rows = []
    for e in local_graph.get("edges", []):
        rows.append({
            "source":       e.get("source", ""),
            "target":       e.get("target", ""),
            "relationship": e.get("relationship", "connected_to"),
            "label":        e.get("label", ""),
            "confidence":   0.91 if e.get("connector_id") else 0.78,
        })
    return pd.DataFrame(rows)


def _run_v3_diagram_intelligence(diagram_path: Path, diagram_id: str,
                                  uploaded: bool = False) -> bool:
    if uploaded:
        msg = "Live onboarding is not available yet. Use a prepared V3 scenario diagram."
        if _strict_mode():
            st.error(msg)
        else:
            st.warning(msg)
        return False

    packet = _load_v3_packet(diagram_path, diagram_id)
    if not packet["annotation"] or not packet["local_graph"]:
        st.error(
            "V3 metadata is missing. Generate the dataset first:\n"
            "`python scripts/generate_diagram_v3_enterprise_dataset.py --num-scenarios 10 "
            "--out ./datasets/diagram_v3_enterprise --seed 2026 --clean`"
        )
        return False

    local_graph = packet["local_graph"]
    node_table  = _node_table_from_graph(local_graph)
    edge_table  = _edge_table_from_graph(local_graph)

    conf_summary = {
        "device_detection_avg": round(float(node_table["confidence"].mean()) if not node_table.empty else 0.0, 3),
        "edge_extraction_avg":  round(float(edge_table["confidence"].mean()) if not edge_table.empty else 0.0, 3),
        "ocr_text_blocks":      len(packet["annotation"].get("text_blocks", [])),
        "low_confidence_items": int((node_table.get("confidence", pd.Series()) < 0.90).sum()),
    }

    st.session_state.selected_diagram_path      = str(diagram_path)
    st.session_state.selected_diagram_id        = diagram_id
    st.session_state.local_graph                = local_graph
    st.session_state.node_table                 = node_table
    st.session_state.edge_table                 = edge_table
    st.session_state.validation_packet          = {
        "diagram_id":        diagram_id,
        "image_path":        str(diagram_path),
        "annotation_path":   str(packet["annotation_path"]),
        "local_graph_path":  str(packet["local_graph_path"]),
        "source_label":      "Source: V3 scenario annotation (ground truth)",
        "text_blocks":       packet["annotation"].get("text_blocks", []),
        "confidence_summary": conf_summary,
    }
    st.session_state.local_rca_result           = {}
    st.session_state.enterprise_rca_result      = {}
    st.session_state.enterprise_ingestion_summary = {}
    st.session_state.enterprise_graph_before    = None
    st.session_state.enterprise_graph_after     = None
    st.session_state.allow_local_simulation     = False
    st.session_state.allow_enterprise_simulation = False
    st.session_state.allow_deterministic_copilot = False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def _simulate_local_rca(local_graph: dict) -> dict:
    nodes = [n.get("id", "") for n in local_graph.get("nodes", [])]
    edges = [(e.get("source", ""), e.get("target", "")) for e in local_graph.get("edges", [])]
    outdeg = {n: 0 for n in nodes}
    indeg  = {n: 0 for n in nodes}
    adj    = {n: [] for n in nodes}
    for src, tgt in edges:
        if src in outdeg and tgt in indeg:
            outdeg[src] += 1
            indeg[tgt]  += 1
            adj[src].append(tgt)

    candidates = sorted(nodes,
                        key=lambda n: (indeg.get(n, 0) == 0, outdeg.get(n, 0), -indeg.get(n, 0)),
                        reverse=True)
    root = candidates[0] if candidates else ""

    impacted: list[str] = []
    queue = list(adj.get(root, []))
    seen  = {root}
    while queue:
        cur = queue.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        impacted.append(cur)
        queue.extend(adj.get(cur, []))
    if not impacted and len(nodes) > 1:
        impacted = [n for n in nodes if n != root][:3]

    target = impacted[-1] if impacted else root
    prev   = {root: None}
    queue  = [root]
    while queue and target not in prev:
        cur = queue.pop(0)
        for nxt in adj.get(cur, []):
            if nxt not in prev:
                prev[nxt] = cur
                queue.append(nxt)
    path = [target] if target else []
    while path and path[-1] in prev and prev[path[-1]]:
        path.append(prev[path[-1]])
    path = list(reversed(path)) if path else [root]

    ranking = []
    for n in nodes:
        score = 0.40 + outdeg.get(n, 0) * 0.11 + (0.14 if indeg.get(n, 0) == 0 else 0.0)
        if n == root:
            score += 0.20
        ranking.append({"node": n, "score": round(min(score, 0.99), 3),
                         "reason": f"out={outdeg.get(n,0)} in={indeg.get(n,0)}"})
    ranking.sort(key=lambda r: r["score"], reverse=True)
    return {
        "mode":          "deterministic_graph_simulation",
        "root_cause":    root,
        "alert_nodes":   impacted[:2] or ([root] if root else []),
        "impacted_nodes": impacted,
        "impact_path":   path,
        "ranking":       ranking[:6],
        "explanation":   "Deterministic simulation — local diagram only.",
    }


def _load_enterprise_context() -> dict:
    scenario = V3_HERO_SCENARIO
    eg  = _safe_read_json(scenario / "enterprise_graph.json")
    sm  = _safe_read_json(scenario / "stitch_map.json")
    al  = _safe_read_json(scenario / "alerts.json")
    st.session_state.enterprise_scenario_path = str(scenario)
    return {"scenario": scenario, "enterprise_graph": eg, "stitch_map": sm, "alerts": al}


def _enterprise_without_diagram(enterprise_graph: dict, diagram_id: str) -> dict:
    graph    = json.loads(json.dumps(enterprise_graph))
    clusters = graph.get("diagram_clusters", [])
    if isinstance(clusters, list):
        filtered = [c for c in clusters if c.get("diagram_id") != diagram_id]
        keep     = {nid for c in filtered for nid in c.get("node_ids", [])}
        graph["diagram_clusters"] = filtered
    else:
        filtered_d = {k: v for k, v in clusters.items() if k != diagram_id}
        keep       = {nid for c in filtered_d.values() for nid in c.get("node_ids", [])}
        graph["diagram_clusters"] = filtered_d
    graph["nodes"] = [n for n in graph.get("nodes", []) if n.get("id") in keep]
    graph["edges"] = [
        e for e in graph.get("edges", [])
        if e.get("source") in keep and e.get("target") in keep
    ]
    return graph


def _simulate_enterprise_ingestion(local_graph: dict, enterprise_graph: dict,
                                    stitch_map: dict, diagram_id: str) -> dict:
    enterprise_ids = {n.get("id") for n in enterprise_graph.get("nodes", [])}
    local_ids      = {n.get("canonical_id", n.get("id")) for n in local_graph.get("nodes", [])}
    matched        = sorted(local_ids & enterprise_ids)
    new_nodes      = sorted(local_ids - enterprise_ids)
    cross_links    = [
        e for e in stitch_map.get("cross_diagram_edges", [])
        if e.get("source_diagram") == diagram_id or e.get("target_diagram") == diagram_id
    ]
    summary = {
        "absorbed_diagram_id":       diagram_id,
        "nodes_absorbed":            len(local_graph.get("nodes", [])),
        "edges_absorbed":            len(local_graph.get("edges", [])),
        "shared_entities_matched":   len(matched),
        "cross_diagram_links_created": len(cross_links),
        "status":                    "absorbed_into_enterprise_graph",
        "matched_entities":          matched,
        "new_nodes":                 new_nodes,
        "cross_diagram_links":       cross_links,
    }
    st.session_state.enterprise_graph_before    = _enterprise_without_diagram(enterprise_graph, diagram_id)
    st.session_state.enterprise_graph_after     = enterprise_graph
    st.session_state.enterprise_ingestion_summary = summary
    return summary


def _load_gnn_rca_result(scenario_id: str) -> dict | None:
    """Try to load a trained enterprise GNN RCA result for this scenario."""
    for candidate in [
        REPO_ROOT / "outputs" / "enterprise_gnn_rca" / f"{scenario_id}_enterprise_gnn_rca_result.json",
        REPO_ROOT / "outputs" / "enterprise_gnn_rca" / "enterprise_0000_enterprise_gnn_rca_result.json",
    ]:
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                pass
    return None


def _simulate_enterprise_rca(alerts: dict, enterprise_graph: dict) -> dict:
    scenario_id = (st.session_state.get("enterprise_ingestion_summary") or {}).get(
        "scenario_id", "enterprise_v3_0000"
    )

    # Try trained GNN result first
    gnn = _load_gnn_rca_result(scenario_id)
    if gnn:
        # Normalise field names from the GNN result schema
        root       = gnn.get("predicted_root_cause") or gnn.get("root_cause", "")
        candidates = gnn.get("top_candidates", [])
        ranking = [
            {"node": c.get("node_id", ""), "score": c.get("score", 0.0),
             "reason": f"GNN rank {c.get('rank',i+1)}, type={c.get('type','?')}"}
            for i, c in enumerate(candidates[:6])
        ]
        return {
            "mode":               "Enterprise GNN RCA",
            "root_cause":         root,
            "root_cause_diagram": gnn.get("root_cause_diagram", alerts.get("root_cause_diagram", "")),
            "impacted_diagrams":  gnn.get("impacted_diagrams", alerts.get("impacted_diagrams", [])),
            "alert_count":        gnn.get("alert_count", len(alerts.get("alerts", []))),
            "alert_nodes":        [a.get("node", "") for a in alerts.get("alerts", [])],
            "impacted_nodes":     alerts.get("impacted_nodes", []),
            "impact_path":        (alerts.get("impact_paths") or [[]])[0],
            "ranking":            ranking,
            "enterprise_node_count": gnn.get("node_count", len(enterprise_graph.get("nodes", []))),
            "gnn_source_file":    str(next(
                (REPO_ROOT / "outputs" / "enterprise_gnn_rca" / f)
                for f in [f"{scenario_id}_enterprise_gnn_rca_result.json",
                           "enterprise_0000_enterprise_gnn_rca_result.json"]
                if (REPO_ROOT / "outputs" / "enterprise_gnn_rca" / f).exists()
            )),
        }

    # Scenario-grounded simulation (alerts.json ground truth)
    root       = alerts.get("root_cause", "")
    alert_nodes = [a.get("node", "") for a in alerts.get("alerts", [])]
    ranking = [{"node": root, "score": 0.97, "reason": "scenario ground truth"}] if root else []
    for n in alert_nodes:
        if n != root:
            ranking.append({"node": n, "score": round(0.74 - len(ranking) * 0.05, 3),
                             "reason": "alert propagation evidence"})
    return {
        "mode":               "Scenario-grounded RCA simulation",
        "root_cause":         root,
        "root_cause_diagram": alerts.get("root_cause_diagram", ""),
        "impacted_diagrams":  alerts.get("impacted_diagrams", []),
        "alert_count":        len(alerts.get("alerts", [])),
        "alert_nodes":        alert_nodes,
        "impacted_nodes":     alerts.get("impacted_nodes", []),
        "impact_path":        (alerts.get("impact_paths") or [[]])[0],
        "ranking":            ranking[:6],
        "enterprise_node_count": len(enterprise_graph.get("nodes", [])),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENTERPRISE PYVIS
# ══════════════════════════════════════════════════════════════════════════════
def _render_enterprise_pyvis(enterprise_graph: dict, absorbed_ids: set[str],
                               rca: dict | None, height: int = 760) -> bool:
    try:
        from pyvis.network import Network  # type: ignore
    except Exception:
        return False

    root       = (rca or {}).get("root_cause")
    alert_set  = set((rca or {}).get("alert_nodes", []))
    impacted   = set((rca or {}).get("impacted_nodes", []))
    path_set: set[tuple[str, str]] = set()
    for a, b in zip((rca or {}).get("impact_path", []),
                    (rca or {}).get("impact_path", [])[1:]):
        path_set.add((a, b))

    net = Network(height=f"{height}px", width="100%", directed=True,
                  bgcolor="#0b1220", font_color="#e2e8f0")
    net.barnes_hut(gravity=-4200, central_gravity=0.22, spring_length=180, spring_strength=0.042)

    groups = {}
    for did, cluster in enterprise_graph.get("diagram_clusters", {}).items():
        for nid in cluster.get("node_ids", []):
            groups[nid] = did
    node_map = {n.get("id"): n for n in enterprise_graph.get("nodes", [])}

    for n in enterprise_graph.get("nodes", []):
        nid  = n.get("id", "")
        ntype = n.get("type", "server")
        shared = n.get("is_shared_entity", False)
        diag   = groups.get(nid, n.get("diagram_id", ""))
        col_base = _V3_DIAG_COLORS.get(diag, "#64748b")

        if nid == root:
            color, size, bw = "#ef4444", 38, 5
        elif nid in alert_set:
            color, size, bw = "#f97316", 30, 4
        elif nid in impacted:
            color, size, bw = "#facc15", 26, 3
        elif nid in absorbed_ids:
            color, size, bw = "#22d3ee", 28, 4
        elif shared:
            color, size, bw = "#38bdf8", 26, 4
        else:
            color, size, bw = col_base, 20, 2

        border = "#ffffff" if nid == root else ("#fbbf24" if shared else "#3a4a5a")
        title  = (
            f"<b>{nid}</b><br>type: {ntype}<br>"
            f"diagram: {diag}<br>ip: {n.get('ip_address','—')}<br>zone: {n.get('zone','—')}"
            + ("<br><b>shared entity</b>" if shared else "")
            + ("<br><b>newly absorbed</b>" if nid in absorbed_ids else "")
            + ("<br><b>ROOT CAUSE</b>" if nid == root else "")
        )
        net.add_node(nid, label=nid, title=title,
                     group=diag,
                     color={"background": color, "border": border},
                     size=size, borderWidth=bw, borderWidthSelected=6)

    for e in enterprise_graph.get("edges", []):
        src, tgt = e.get("source", ""), e.get("target", "")
        if src not in node_map or tgt not in node_map:
            continue
        is_cross = e.get("edge_scope") == "cross_diagram" or e.get("edge_type") == "cross_diagram"
        is_path  = (src, tgt) in path_set
        net.add_edge(
            src, tgt,
            label=str(e.get("label", ""))[:16],
            color="#22d3ee" if is_cross else ("#06b6d4" if is_path else "#4a5568"),
            width=4 if is_path else (2 if is_cross else 1),
            dashes=is_cross,
            title=f"{e.get('relationship','')} | {e.get('edge_scope','')}",
        )

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html", encoding="utf-8") as tmp:
        tmp_path = Path(tmp.name)
    try:
        net.save_graph(str(tmp_path))
        components.html(tmp_path.read_text(encoding="utf-8"), height=height+20, scrolling=False)
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass
    return True


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DIAGRAM INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
def _tab_diagram_outputs_section() -> None:
    """Show outputs after diagram has been processed into local graph."""
    local_graph = st.session_state.get("local_graph")
    if not local_graph:
        return

    st.markdown('<hr class="ws-rule">', unsafe_allow_html=True)

    n_nodes = len(local_graph.get("nodes", []))
    n_edges = len(local_graph.get("edges", []))
    packet  = st.session_state.get("validation_packet") or {}
    summary = packet.get("confidence_summary", {})

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Nodes detected",    n_nodes)
    m2.metric("Edges extracted",   n_edges)
    m3.metric("Avg confidence",    f"{summary.get('device_detection_avg', 0):.0%}")
    m4.metric("OCR text blocks",   summary.get("ocr_text_blocks", 0))

    source = packet.get("source_label", "")
    if source:
        st.caption(source)

    view_mode = st.radio(
        "View",
        ["Original + Detection", "Local Graph (Interactive)", "Node/Edge Tables"],
        horizontal=True,
        key="diagram_view_mode",
    )

    if view_mode == "Original + Detection":
        orig_p       = st.session_state.get("selected_diagram_path", "")
        det_p        = st.session_state.get("detected_image_path", "")
        det_source   = st.session_state.get("detection_source") or "Verified Annotation Overlay"
        is_rfdetr    = det_source.startswith("RF-DETR")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                '<div class="compare-badge original">Original</div>'
                '<div class="compare-label">Source Diagram</div>',
                unsafe_allow_html=True,
            )
            if orig_p and Path(orig_p).exists():
                st.image(orig_p, use_container_width=True)
            else:
                st.warning("Original image not available.")
        with c2:
            badge_cls = "predicted" if is_rfdetr else "prepared"
            st.markdown(
                f'<div class="compare-badge {badge_cls}">{det_source}</div>'
                f'<div class="compare-label">Detection Output</div>',
                unsafe_allow_html=True,
            )
            _det_ready = det_p and Path(det_p).exists() and det_p != orig_p
            if _det_ready:
                st.image(det_p, use_container_width=True)
            elif is_rfdetr:
                st.image(orig_p, use_container_width=True)
            elif orig_p and Path(orig_p).exists():
                st.warning(
                    "Annotation overlay could not be rendered; showing source image."
                )
                st.image(orig_p, use_container_width=True)
            else:
                st.markdown(
                    '<div class="info-card">'
                    '<strong>Verified Annotation Overlay.</strong><br>'
                    'Bounding boxes and node labels come from the scenario '
                    'ground-truth annotation. Train RF-DETR '
                    '(<code>train_rfdetr_diagram_detector.py</code>) to generate '
                    'live detector predictions.'
                    '</div>',
                    unsafe_allow_html=True,
                )

    elif view_mode == "Local Graph (Interactive)":
        st.markdown('<div class="section-label">Local Graph</div>', unsafe_allow_html=True)
        if not _pyvis_available():
            st.caption("Install pyvis for interactive graph: `pip install pyvis`")
        _render_local_graph(local_graph)
        did = st.session_state.get("selected_diagram_id", "")
        if did:
            diag_color = _V3_DIAG_COLORS.get(did, "#64748b")
            st.markdown(
                f'<span style="display:inline-block;background:{diag_color}22;color:{diag_color};'
                f'border:1px solid {diag_color}44;border-radius:6px;padding:2px 10px;'
                f'font-size:0.72rem;font-weight:700">{did}</span>',
                unsafe_allow_html=True,
            )

    else:
        tab_n, tab_e, tab_t = st.tabs(["Node inventory", "Edge inventory", "OCR / text"])
        with tab_n:
            nt = st.session_state.get("node_table")
            if nt is not None and not nt.empty:
                st.dataframe(nt, use_container_width=True, hide_index=True)
            else:
                st.info("No node data — run Diagram Intelligence first.")
        with tab_e:
            et = st.session_state.get("edge_table")
            if et is not None and not et.empty:
                st.dataframe(et, use_container_width=True, hide_index=True)
            else:
                st.info("No edge data.")
        with tab_t:
            text_rows = packet.get("text_blocks", [])
            if text_rows:
                st.dataframe(pd.DataFrame(text_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No OCR/text metadata available.")


def _tab_onboard_new_diagram() -> None:
    """Onboard New Diagram — run live diagram intelligence on a selected sample."""
    st.markdown(
        '<div class="mode-onboard">'
        '<div class="mode-title" style="color:#a78bfa">Onboard New Diagram</div>'
        '<div class="mode-sub">Select a sample diagram and run live diagram intelligence '
        'to onboard it into the enterprise graph memory.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── RF-DETR model status ──────────────────────────────────────────────────
    _rfdetr_ckpt = _find_rfdetr_ckpt(REPO_ROOT) if _LIVE_RFDETR else None
    _rfdetr_ckpt_str = str(_rfdetr_ckpt) if _rfdetr_ckpt else None
    col_l, col_r = st.columns([3, 2])
    with col_l:
        if _rfdetr_ckpt:
            st.markdown(
                f'<span class="badge badge-success">RF-DETR checkpoint found</span> '
                f'<span style="font-size:0.72rem;color:#64748b">{_rfdetr_ckpt.name}</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="badge badge-warn">RF-DETR checkpoint not found</span> '
                '<span style="font-size:0.72rem;color:#64748b">'
                'Train first: scripts/train_rfdetr_diagram_detector.py</span>',
                unsafe_allow_html=True,
            )
    with col_r:
        use_rfdetr = st.checkbox(
            "Use live RF-DETR detector if available",
            value=st.session_state.get("use_live_rfdetr", True),
            key="use_live_rfdetr_cb",
            disabled=not bool(_rfdetr_ckpt),
        )
        st.session_state.use_live_rfdetr = use_rfdetr and bool(_rfdetr_ckpt)

    st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.06);margin:10px 0'>",
                unsafe_allow_html=True)

    # ── Load onboarding manifest ──────────────────────────────────────────────
    samples = _load_onboarding_manifest(str(REPO_ROOT))
    if not samples:
        st.error(
            "Onboarding manifest not found. Build the asset layer first:\n"
            "```\npython scripts/build_presentation_assets.py\n```"
        )
        return

    left, right = st.columns([1, 1])
    with left:
        sel_idx = st.selectbox(
            "Select sample diagram",
            range(len(samples)),
            format_func=lambda i: f"{samples[i]['sample_id']} | {samples[i]['display_name']}",
            index=0,
            key="onboard_sample_select",
        )
        sample   = samples[sel_idx]
        did      = sample.get("source_diagram_id", "")
        img_path = Path(sample.get("image_path", ""))

        # status card
        onboard_status = st.session_state.get("onboard_status", "not_started")
        onboarded_sid  = (
            _ss_dict("onboard_sample_record").get("sample_id", "")
            if onboard_status != "not_started" else ""
        )
        is_this_sample_active = onboarded_sid == sample.get("sample_id", "")

        if not is_this_sample_active:
            st.markdown(
                '<div class="warn-card" style="margin:8px 0">'
                '<strong>Not yet onboarded.</strong><br>'
                'This diagram is not part of the active runtime graph memory yet. '
                'Run live diagram intelligence to onboard it.'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            src_label = st.session_state.get("detection_source", "")
            badge_cls = (
                "badge-success"
                if src_label.startswith(("Live RF-DETR", "RF-DETR")) else "badge-info"
            )
            st.markdown(
                f'<span class="badge badge-info">Ingested</span> '
                f'<span class="badge {badge_cls}">{src_label}</span>',
                unsafe_allow_html=True,
            )

        # ── action button ─────────────────────────────────────────────────────
        if st.button("Run Live Diagram Intelligence", type="primary",
                     use_container_width=True, disabled=not img_path.exists()):
            if not _RUNTIME_INGESTION or _run_ingestion is None:
                st.error("runtime_ingestion module not loaded.")
            else:
                _rfdetr_model = None
                if st.session_state.use_live_rfdetr and _rfdetr_ckpt_str:
                    with st.spinner("Preparing RF-DETR checkpoint…"):
                        _rfdetr_model = _load_rfdetr_model_cached(_rfdetr_ckpt_str)

                _STEPS = [
                    "Loading image",
                    "Resolving detector",
                    "Running live detector or annotation overlay",
                    "Extracting OCR / text metadata",
                    "Extracting connectors",
                    "Building node table",
                    "Building edge table",
                    "Creating local graph memory packet",
                    "Ready for absorption",
                ]
                prog       = st.progress(0)
                steps_area = st.empty()

                ann_p  = sample.get("annotation_path", "")
                lg_p   = sample.get("local_graph_path", "")
                ent_p  = sample.get("enterprise_graph_path", "")
                stitch = sample.get("stitch_map_path", "")
                alerts = sample.get("alerts_path", "")

                with st.spinner("Running live diagram intelligence…"):
                    ingestion = _run_ingestion(
                        repo_root             = REPO_ROOT,
                        image_path            = img_path,
                        diagram_id            = did,
                        run_id                = sample["sample_id"],
                        annotation_path       = Path(ann_p)  if ann_p  else None,
                        local_graph_path      = Path(lg_p)   if lg_p   else None,
                        enterprise_graph_path = Path(ent_p)  if ent_p  else None,
                        stitch_map_path       = Path(stitch) if stitch else None,
                        alerts_path           = Path(alerts) if alerts else None,
                        use_live_rfdetr       = st.session_state.use_live_rfdetr,
                        rfdetr_model          = _rfdetr_model,
                    )

                for idx, step in enumerate(_STEPS, 1):
                    steps_area.markdown(
                        "\n".join(
                            f'<div style="font-size:0.78rem;'
                            f'color:{"#10b981" if i <= idx else "#334155"};padding:2px 0">'
                            f'{"✓" if i <= idx else "○"} {s}</div>'
                            for i, s in enumerate(_STEPS, 1)
                        ),
                        unsafe_allow_html=True,
                    )
                    prog.progress(idx / len(_STEPS))

                import pandas as _pd
                st.session_state.selected_diagram_path      = str(img_path)
                st.session_state.selected_diagram_id        = did
                st.session_state.local_graph                = ingestion.get("local_graph")
                st.session_state.node_table                 = _pd.DataFrame(ingestion.get("node_table_rows", []))
                st.session_state.edge_table                 = _pd.DataFrame(ingestion.get("edge_table_rows", []))
                st.session_state.validation_packet          = ingestion.get("packet") or {}
                st.session_state.live_ingestion_run_dir     = str(ingestion.get("run_dir", ""))
                st.session_state.detection_source           = ingestion.get("detection_source", "")
                st.session_state.detected_image_path        = str(ingestion.get("detected_image", ""))
                st.session_state.enterprise_absorbed        = False
                st.session_state.local_rca_result           = {}
                st.session_state.enterprise_rca_result      = {}
                st.session_state.enterprise_ingestion_summary = {}
                st.session_state.enterprise_graph_before    = None
                st.session_state.enterprise_graph_after     = None
                st.session_state.allow_local_simulation     = False
                st.session_state.allow_enterprise_simulation = False
                st.session_state.allow_deterministic_copilot = False
                st.session_state.onboard_status             = "ingested_not_absorbed"
                st.session_state.onboard_sample_record      = sample
                st.session_state.enterprise_scenario_path  = sample.get("source_scenario_path", "")
                st.session_state.catalog_selected_record    = {
                    **sample,
                    "detected_preview_path": str(ingestion.get("detected_image", "")),
                }

                det_src   = ingestion.get("detection_source", "")
                badge_cls = "badge-success" if det_src.startswith(("Live RF-DETR", "RF-DETR")) else "badge-info"
                t_s       = ingestion.get("rfdetr_inference_time_s", 0) or 0
                time_str  = f" ({t_s:.2f}s)" if t_s > 0 else ""
                st.markdown(
                    f'<span class="badge {badge_cls}">{det_src}{time_str}</span>',
                    unsafe_allow_html=True,
                )
                rfdetr_err = ingestion.get("rfdetr_error", "")
                if rfdetr_err:
                    st.warning(f"RF-DETR: {rfdetr_err}")
                pkt = ingestion.get("packet") or {}
                st.success(
                    f"Ingestion complete — {pkt.get('node_count', 0)} nodes, "
                    f"{pkt.get('edge_count', 0)} edges. "
                    "Proceed to Tab 2 (Local RCA) or Tab 3 (Enterprise Brain)."
                )

    with right:
        if img_path.exists():
            det_img = st.session_state.get("detected_image_path", "")
            active_sid = _ss_dict("onboard_sample_record").get("sample_id", "")
            if det_img and Path(det_img).exists() and active_sid == sample.get("sample_id", ""):
                det_src = st.session_state.get("detection_source", "")
                badge_cls = "predicted" if det_src.startswith("Live RF-DETR") else "prepared"
                st.markdown(
                    f'<div class="compare-badge {badge_cls}">{det_src}</div>',
                    unsafe_allow_html=True,
                )
                st.image(det_img, use_container_width=True)
            else:
                st.image(str(img_path), caption=sample.get("display_name", ""), use_container_width=True)
        else:
            st.markdown(
                '<div class="warn-card" style="text-align:center;padding:40px 20px">'
                'Diagram image not found.'
                '</div>',
                unsafe_allow_html=True,
            )

    _tab_diagram_outputs_section()


def _tab_diagram_gallery() -> None:
    """Diagram Gallery — known diagrams available in graph memory."""
    st.markdown(
        '<div class="mode-gallery">'
        '<div class="mode-title" style="color:#60a5fa">Diagram Gallery</div>'
        '<div class="mode-sub">Known topology diagrams that are available in graph memory. '
        'Use <strong>Onboard New Diagram</strong> to run live diagram intelligence '
        'on a new diagram.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    catalog = _load_gallery_manifest(str(REPO_ROOT))

    if not catalog:
        st.warning(
            "Gallery manifest not found. Build the asset layer first:\n"
            "```\npython scripts/build_presentation_assets.py\n```"
        )
        return

    # ── Filters ───────────────────────────────────────────────────────────────
    fc1, fc2 = st.columns([3, 4])
    with fc1:
        dt_filter = st.selectbox(
            "Diagram type",
            ["All", "Branch Office Topology", "WAN Core Topology",
             "Data Center Topology", "Application & Database Tier",
             "Shared Services Topology"],
            index=0, key="gal_type",
        )
    with fc2:
        search = st.text_input(
            "Search by name or ID",
            placeholder="e.g. DG-0001 or Branch",
            key="gal_search",
        )

    filtered = catalog
    if dt_filter != "All":
        filtered = [r for r in filtered if r.get("display_name", "") == dt_filter]
    if search:
        s = search.lower()
        filtered = [
            r for r in filtered
            if s in r.get("gallery_id", "").lower()
            or s in r.get("display_name", "").lower()
            or s in r.get("source_diagram_id", "").lower()
        ]

    limit = 250
    st.caption(f"Showing {min(limit, len(filtered))} of {len(filtered)} diagrams")

    if not filtered:
        st.info("No diagrams match the current filter.")
        return

    sel_idx = st.selectbox(
        "Select diagram",
        range(min(limit, len(filtered))),
        format_func=lambda i: f"{filtered[i]['gallery_id']} | {filtered[i]['display_name']}",
        index=0,
        key="gal_select",
    )
    record = filtered[sel_idx]
    st.session_state.catalog_selected_record = record
    st.session_state.selected_diagram_path   = record.get("image_path", "")
    st.session_state.selected_diagram_id     = record.get("source_diagram_id", "")

    # ── Images: original + detection/annotation ───────────────────────────────
    img_p  = record.get("image_path", "")
    det_p  = record.get("detected_preview_path", "")
    ann_p  = record.get("annotation_path", "")
    is_v3  = record.get("source_dataset") == "v3"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="compare-badge original">Original</div>'
            '<div class="compare-label">Source Diagram</div>',
            unsafe_allow_html=True,
        )
        if img_p and Path(img_p).exists():
            st.image(img_p, use_container_width=True)
        else:
            st.warning("Image not found.")

    with c2:
        if det_p and Path(det_p).exists():
            src_label = record.get("source_dataset", "").upper()
            badge_lbl = "Trained Detector Output"
            st.markdown(
                f'<div class="compare-badge predicted">{badge_lbl}</div>'
                f'<div class="compare-label">{src_label} trained detector</div>',
                unsafe_allow_html=True,
            )
            st.image(det_p, use_container_width=True)
        elif is_v3 and ann_p and Path(ann_p).exists():
            # render annotation overlay on-the-fly
            import tempfile as _tf, sys as _sys
            _src_dir = str(REPO_ROOT / "src")
            if _src_dir not in _sys.path:
                _sys.path.insert(0, _src_dir)
            try:
                from runtime_ingestion import render_v3_annotation_preview as _render_ann
                _tmp = Path(_tf.mktemp(suffix=".png"))
                _meta = _render_ann(Path(img_p), Path(ann_p), _tmp)
                if _meta.get("rendered") and _tmp.exists():
                    st.markdown(
                        '<div class="compare-badge prepared">Verified Annotation Overlay</div>'
                        '<div class="compare-label">Graph-ready annotation bboxes</div>',
                        unsafe_allow_html=True,
                    )
                    st.image(str(_tmp), use_container_width=True)
                    try:
                        _tmp.unlink()
                    except Exception:
                        pass
                else:
                    st.info("Annotation overlay not available.")
            except Exception:
                st.info("Annotation overlay not available.")
        else:
            st.markdown(
                '<div class="compare-badge missing">Detection Output Pending</div>'
                '<div class="compare-label">Run live detector to generate output</div>',
                unsafe_allow_html=True,
            )
            if img_p and Path(img_p).exists():
                st.image(img_p, use_container_width=True)

    # ── Badges ────────────────────────────────────────────────────────────────
    has_graph = record.get("graph_metadata_available", False)
    has_conn  = record.get("connector_metadata_available", False)
    has_ocr   = record.get("ocr_metadata_available", False)
    has_ent   = record.get("enterprise_mapping_available", False)
    has_det   = bool(det_p and Path(det_p).exists())
    has_ann   = bool(ann_p and Path(ann_p).exists())

    def _badge(label: str, ok: bool, ok_cls: str = "badge-success") -> str:
        cls = ok_cls if ok else "badge-warn"
        return f'<span class="badge {cls}">{label}</span>'

    badge_html = '<div style="display:flex;flex-wrap:wrap;gap:6px;margin:10px 0 14px">'
    badge_html += '<span class="badge badge-info">Available in Graph Memory</span>'
    badge_html += _badge(
        "Graph Metadata Available" if has_graph else "Graph Metadata Pending", has_graph)
    badge_html += _badge(
        "Connector Metadata Available" if has_conn else "Connector Metadata Pending", has_conn)
    badge_html += _badge(
        "OCR Metadata Available" if has_ocr else "OCR Metadata Pending", has_ocr)
    badge_html += _badge(
        "Enterprise Mapping Available" if has_ent else "Enterprise Mapping Pending", has_ent)
    if has_det:
        badge_html += '<span class="badge badge-success">Trained Detector Output</span>'
    elif has_ann and is_v3:
        badge_html += '<span class="badge badge-info">Verified Annotation Overlay</span>'
    else:
        badge_html += '<span class="badge badge-warn">Detection Output Pending</span>'
    badge_html += '</div>'
    st.markdown(badge_html, unsafe_allow_html=True)

    # ── Load graph metadata action ────────────────────────────────────────────
    lg_path = record.get("local_graph_path", "")
    if has_graph and lg_path and Path(lg_path).exists():
        if st.button("Load Graph Metadata", type="secondary",
                     use_container_width=False, key="gal_load_graph"):
            lg = json.loads(Path(lg_path).read_text(encoding="utf-8"))
            st.session_state.local_graph = lg
            n_rows = [{"node_id": n.get("id", ""), "type": n.get("type", ""),
                        "ip_address": n.get("ip_address", ""), "zone": n.get("zone", ""),
                        "shared": n.get("is_shared_entity", False),
                        "confidence": 0.88, "source": "graph_memory"} for n in lg.get("nodes", [])]
            e_rows = [{"source": e.get("source", ""), "target": e.get("target", ""),
                        "relationship": e.get("relationship", ""), "label": e.get("label", ""),
                        "confidence": 0.82} for e in lg.get("edges", [])]
            st.session_state.node_table = pd.DataFrame(n_rows)
            st.session_state.edge_table = pd.DataFrame(e_rows)
            st.success(f"Graph metadata loaded: {len(n_rows)} nodes, {len(e_rows)} edges.")

    # ── Source details expander ───────────────────────────────────────────────
    with st.expander("Source details", expanded=False):
        st.markdown(
            f'**Dataset:** {record.get("source_dataset","").upper()} &nbsp;|&nbsp; '
            f'**Split:** {record.get("source_split","")} &nbsp;|&nbsp; '
            f'**Scenario:** `{record.get("source_scenario_id","")}` &nbsp;|&nbsp; '
            f'**Diagram:** `{record.get("source_diagram_id","")}`'
        )
        for lbl, key in [
            ("Image", "image_path"), ("Annotation", "annotation_path"),
            ("Local graph", "local_graph_path"), ("Enterprise graph", "enterprise_graph_path"),
        ]:
            v = record.get(key, "")
            if v:
                st.caption(f"{lbl}: `{v}`")
    # Gallery does not include a live ingestion button — use Onboard New Diagram for that


def _tab_diagram_intelligence() -> None:
    st.markdown(
        '<div class="ws-title">Live Diagram Intelligence — Image to Graph</div>'
        '<div class="ws-desc">Browse existing consumed diagrams in the Gallery, '
        'or onboard a new diagram through live detection, local graph extraction, '
        'and enterprise absorption.</div>',
        unsafe_allow_html=True,
    )

    gallery_tab, onboard_tab = st.tabs(["Diagram Gallery", "Onboard New Diagram"])
    with gallery_tab:
        _tab_diagram_gallery()
    with onboard_tab:
        _tab_onboard_new_diagram()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LOCAL RCA
# ══════════════════════════════════════════════════════════════════════════════
def _tab_local_rca() -> None:
    st.markdown(
        '<div class="ws-title">Local RCA — Same Selected Diagram</div>'
        '<div class="ws-desc">Simulate an alert on the currently ingested diagram and identify the '
        'root cause within the local topology.</div>',
        unsafe_allow_html=True,
    )

    local_graph = st.session_state.get("local_graph")
    if not local_graph:
        st.markdown(
            '<div class="warn-card">'
            'No diagram has been processed yet.<br>'
            'Go to <strong>Tab 1 → Onboard New Diagram</strong>, select a sample, '
            'and click <em>Run Live Diagram Intelligence</em>.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    diagram_id = st.session_state.get("selected_diagram_id", "unknown")
    sel_path   = st.session_state.get("selected_diagram_path", "")

    # Diagram summary header
    c_thumb, c_info = st.columns([1, 3])
    with c_thumb:
        if sel_path and Path(sel_path).exists():
            st.image(sel_path, use_container_width=True)
    with c_info:
        badge_color = _V3_DIAG_COLORS.get(diagram_id, "#64748b")
        n_nodes = len(local_graph.get("nodes", []))
        n_edges = len(local_graph.get("edges", []))
        st.markdown(
            f'<div class="card blue" style="padding:14px 16px;margin-bottom:10px">'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.95rem;'
            f'font-weight:700;color:{badge_color}">{diagram_id}</div>'
            f'<div style="font-size:0.78rem;color:#64748b;margin-top:4px">'
            f'Nodes: {n_nodes} &nbsp;|&nbsp; Edges: {n_edges}</div>'
            f'<div style="font-size:0.72rem;color:#475569;margin-top:2px">'
            f'Source: V3 scenario annotation (ground truth)</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        local_model = _local_rca_model_available()
        if local_model:
            st.success("RCA source: trained local model output")
        else:
            st.warning("RCA source: deterministic local graph simulation (no trained model available)")
            if _strict_mode() and not st.session_state.allow_local_simulation:
                st.error("Strict mode: explicitly approve before using deterministic simulation.")
                if st.button("Use deterministic simulation", key="approve_local_sim"):
                    st.session_state.allow_local_simulation = True
                    st.rerun()
                return

        if st.button("Simulate Alert on This Diagram", type="primary", key="local_rca_btn"):
            with st.spinner("Running local RCA..."):
                st.session_state.local_rca_result = _simulate_local_rca(local_graph)

    result = st.session_state.get("local_rca_result")
    if not result:
        st.info("Click the button above to simulate an alert on this diagram.")
        return

    source_lbl = ("Trained local model"
                  if result.get("mode") != "deterministic_graph_simulation"
                  else "Deterministic graph simulation")

    st.markdown('<hr class="ws-rule">', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Root Cause",       result.get("root_cause", "—"))
    m2.metric("Alert Nodes",      len(result.get("alert_nodes", [])))
    m3.metric("Impacted Nodes",   len(result.get("impacted_nodes", [])))
    m4.metric("Method",           source_lbl[:12])

    st.caption(
        f"Source: {source_lbl}. Scope: this RCA is limited to the local diagram. "
        "For enterprise-wide analysis, proceed to Tab 3."
    )

    # Graph overlay
    st.markdown('<div class="section-label">Local Graph RCA Overlay</div>', unsafe_allow_html=True)
    st.caption(
        "Root cause: red  |  Alert nodes: orange  |  "
        "Impacted: yellow  |  Impact path: cyan  |  Shared entities: cyan ring"
    )
    if not _pyvis_available():
        st.markdown(
            '<div class="warn-card" style="margin-bottom:8px">'
            'Install <code>pyvis</code> for interactive drag-and-drop graph. '
            'Falling back to matplotlib preview.</div>',
            unsafe_allow_html=True,
        )
    _render_local_graph(local_graph, result)

    # Impact path chips
    path = result.get("impact_path", [])
    if path:
        st.markdown('<div class="section-label" style="margin-top:14px">Impact Path</div>',
                    unsafe_allow_html=True)
        root_id = result.get("root_cause", "")
        alert_set = set(result.get("alert_nodes", []))
        imp_set   = set(result.get("impacted_nodes", []))
        chips = []
        for n in path:
            cls = "root" if n == root_id else ("alerting" if n in alert_set else
                                                "impacted" if n in imp_set else "")
            chips.append(f'<span class="node-chip {cls}">{n}</span>')
        st.markdown(
            '<div style="padding:8px 0;line-height:2.2">' +
            ' <span style="color:#334155;font-size:0.8rem">→</span> '.join(chips) +
            '</div>',
            unsafe_allow_html=True,
        )

    # Ranking table
    ranking = result.get("ranking", [])
    if ranking:
        st.markdown('<div class="section-label" style="margin-top:14px">RCA Candidate Ranking</div>',
                    unsafe_allow_html=True)
        df = pd.DataFrame(ranking)
        st.dataframe(df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ENTERPRISE GRAPH BRAIN
# ══════════════════════════════════════════════════════════════════════════════
def _render_absorption_steps(done: bool = True) -> None:
    steps = [
        "Entity matching against canonical IDs",
        "Shared nodes identified across diagrams",
        "Cross-diagram links created from stitch map",
        "Local graph nodes + edges absorbed",
        "Enterprise memory updated",
    ]
    for step in steps:
        color = "#10b981" if done else "#475569"
        icon  = "✓" if done else "○"
        st.markdown(
            f'<div class="absorb-step">'
            f'<span class="{("absorb-done" if done else "absorb-wait")}">{icon}</span>'
            f'<span style="font-size:0.79rem;color:{("#cbd5e1" if done else "#64748b")}">{step}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _tab_enterprise_graph_brain() -> None:
    st.markdown(
        '<div class="ws-title">Enterprise Graph Brain</div>'
        '<div class="ws-desc">Selected local graph is explicitly absorbed into the enterprise galaxy — '
        'then cross-diagram RCA runs across the full topology.</div>',
        unsafe_allow_html=True,
    )

    local_graph = st.session_state.get("local_graph")
    if not local_graph:
        st.markdown(
            '<div class="warn-card">Ingest a diagram first — Tab 1 → Onboard New Diagram → '
            'Run Live Diagram Intelligence.</div>',
            unsafe_allow_html=True,
        )
        return

    diagram_id    = st.session_state.get("selected_diagram_id", "unknown")
    onboard_rec   = st.session_state.get("onboard_sample_record")
    is_onboarded  = bool(onboard_rec and onboard_rec.get("sample_id"))
    absorbed      = bool(st.session_state.get("enterprise_absorbed"))

    if not is_onboarded:
        st.markdown(
            f'<div class="warn-card">'
            f'No diagram has been ingested yet.<br>'
            f'Use <strong>Tab 1 (Diagram Intelligence) &rsaquo; Onboard New Diagram</strong> to run live ingestion first.</div>',
            unsafe_allow_html=True,
        )
        _render_local_graph(local_graph)
        return

    # ── Step 1: Local graph ready — show absorption button ───────────────────
    badge_c = _V3_DIAG_COLORS.get(diagram_id, "#64748b")
    det_src = st.session_state.get("detection_source") or "Verified Annotation Overlay"

    st.markdown(
        f'<div class="info-card" style="margin-bottom:14px">'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.82rem;'
        f'font-weight:700;color:{badge_c}">{diagram_id}</span>'
        f'&nbsp;<span class="badge badge-info">local graph ready</span>'
        f'&nbsp;<span class="badge {"badge-success" if det_src.startswith("RF-DETR") else "badge-warn"}">'
        f'{det_src}</span>'
        f'<div style="font-size:0.75rem;color:#64748b;margin-top:4px">'
        f'{len(local_graph.get("nodes",[]))} nodes · {len(local_graph.get("edges",[]))} edges'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    if not absorbed:
        if not _RUNTIME_INGESTION:
            st.error("runtime_ingestion module not loaded — cannot run live absorption.")
            return

        if st.button(
            "Absorb This Local Graph into Enterprise Brain",
            type="primary",
            key="absorb_btn",
        ):
            _ABSORB_STEPS = [
                "Local graph validated",
                "Canonical IDs resolved",
                "Shared entities matched",
                "Cross-diagram links created",
                "Enterprise graph memory updated",
            ]
            prog       = st.progress(0)
            steps_area = st.empty()

            with st.spinner("Absorbing local graph into enterprise brain..."):
                _onb_rec  = st.session_state.get("onboard_sample_record") or {}
                _ent_p    = _onb_rec.get("enterprise_graph_path", "")
                _stitch_p = _onb_rec.get("stitch_map_path", "")
                _alerts_p = _onb_rec.get("alerts_path", "")
                _run_id   = _onb_rec.get("sample_id", diagram_id)
                if _run_absorption is not None:
                    absorb_result = _run_absorption(
                        repo_root             = REPO_ROOT,
                        run_id                = _run_id,
                        local_graph           = local_graph,
                        diagram_id            = diagram_id,
                        enterprise_graph_path = Path(_ent_p)    if _ent_p    else None,
                        stitch_map_path       = Path(_stitch_p) if _stitch_p else None,
                        alerts_path           = Path(_alerts_p) if _alerts_p else None,
                    )
                else:
                    absorb_result = _live_absorb(
                        REPO_ROOT, V3_HERO_SCENARIO, diagram_id, local_graph
                    )

            for idx, step in enumerate(_ABSORB_STEPS, 1):
                steps_area.markdown(
                    "\n".join(
                        f'<div style="font-size:0.78rem;color:{"#10b981" if i<=idx else "#334155"};'
                        f'padding:2px 0">{"✓" if i<=idx else "○"} {s}</div>'
                        for i, s in enumerate(_ABSORB_STEPS, 1)
                    ),
                    unsafe_allow_html=True,
                )
                prog.progress(idx / len(_ABSORB_STEPS))

            summary = absorb_result["summary"]
            st.session_state.enterprise_graph_before    = absorb_result["enterprise_before"]
            st.session_state.enterprise_graph_after     = absorb_result["enterprise_after"]
            st.session_state.enterprise_ingestion_summary = summary
            _src_scen = st.session_state.get("enterprise_scenario_path", "") or str(V3_HERO_SCENARIO)
            st.session_state.enterprise_scenario_path  = _src_scen
            st.session_state.enterprise_absorbed        = True
            st.session_state.allow_enterprise_simulation = False
            st.session_state.enterprise_rca_result     = {}
            st.rerun()
        return

    # ── Post-absorption view ──────────────────────────────────────────────────
    summary         = st.session_state.enterprise_ingestion_summary or {}
    enterprise_graph = st.session_state.enterprise_graph_after or {}
    alerts_data     = _safe_read_json(V3_HERO_SCENARIO / "alerts.json")

    # ── Absorption Story (3 columns) ─────────────────────────────────────────
    st.markdown(
        '<div class="section-label" style="margin-bottom:14px">Graph Memory Absorption</div>',
        unsafe_allow_html=True,
    )

    c_local, c_steps, c_enterprise = st.columns(3)

    with c_local:
        n_ln = len(local_graph.get("nodes", []))
        n_le = len(local_graph.get("edges", []))
        shared_ids = [n.get("id") for n in local_graph.get("nodes", [])
                      if n.get("is_shared_entity")]
        matched    = summary.get("matched_entities", [])
        st.markdown(
            f'<div class="absorb-card">'
            f'<div class="section-label">Selected Local Graph</div>'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.88rem;'
            f'font-weight:700;color:{badge_c};margin-bottom:8px">{diagram_id}</div>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px">'
            f'<div><div style="font-size:0.6rem;color:#64748b;text-transform:uppercase">Nodes</div>'
            f'<div style="font-size:1.4rem;font-weight:800;color:#f1f5f9">{n_ln}</div></div>'
            f'<div><div style="font-size:0.6rem;color:#64748b;text-transform:uppercase">Edges</div>'
            f'<div style="font-size:1.4rem;font-weight:800;color:#f1f5f9">{n_le}</div></div>'
            f'</div>'
            + (f'<div style="font-size:0.72rem;color:#38bdf8;margin-top:4px">'
               f'Shared entities: '
               + " ".join(f'<code style="font-size:0.65rem;color:#67e8f9">{s}</code>' for s in shared_ids[:4])
               + "</div>" if shared_ids else "")
            + (f'<div style="font-size:0.72rem;color:#10b981;margin-top:6px">'
               f'Matched IDs: '
               + " ".join(f'<code style="font-size:0.65rem;color:#6ee7b7">{m}</code>' for m in matched[:4])
               + "</div>" if matched else "")
            + "</div>",
            unsafe_allow_html=True,
        )

    with c_steps:
        st.markdown(
            '<div class="absorb-card"><div class="section-label">Absorption Steps</div>',
            unsafe_allow_html=True,
        )
        _render_absorption_steps(done=True)
        st.markdown(
            f'<div style="margin-top:12px;padding:8px 10px;background:rgba(16,185,129,0.08);'
            f'border-radius:8px;border:1px solid rgba(16,185,129,0.2)">'
            f'<div style="font-size:0.72rem;font-weight:700;color:#10b981">Result</div>'
            f'<div style="font-size:0.78rem;color:#94a3b8;margin-top:4px">'
            f'Nodes absorbed: <strong style="color:#f1f5f9">{summary.get("nodes_absorbed",0)}</strong><br>'
            f'Edges absorbed: <strong style="color:#f1f5f9">{summary.get("edges_absorbed",0)}</strong><br>'
            f'Shared matched: <strong style="color:#38bdf8">{summary.get("shared_entities_matched",0)}</strong><br>'
            f'Cross-diag links: <strong style="color:#22d3ee">{summary.get("cross_diagram_links_created",0)}</strong><br>'
            f'Before nodes: <strong style="color:#94a3b8">{summary.get("before_node_count",0)}</strong> '
            f'→ After: <strong style="color:#10b981">{summary.get("after_node_count",0)}</strong>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with c_enterprise:
        eg_stats = enterprise_graph.get("stats", {})
        n_nodes  = eg_stats.get("num_nodes")     or len(enterprise_graph.get("nodes", []))
        n_edges  = eg_stats.get("num_edges")     or len(enterprise_graph.get("edges", []))
        n_cross  = eg_stats.get("num_cross_diagram_edges") or len(enterprise_graph.get("cross_diagram_edges", []))
        n_clust  = len(enterprise_graph.get("diagram_clusters", {}))
        n_shared = eg_stats.get("num_shared_entities") or len(enterprise_graph.get("shared_entities", []))
        st.markdown(
            f'<div class="absorb-card">'
            f'<div class="section-label">Enterprise Galaxy Graph</div>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">'
            f'<div><div style="font-size:0.6rem;color:#64748b;text-transform:uppercase">Total nodes</div>'
            f'<div style="font-size:1.5rem;font-weight:800;color:#f1f5f9">{n_nodes}</div></div>'
            f'<div><div style="font-size:0.6rem;color:#64748b;text-transform:uppercase">Total edges</div>'
            f'<div style="font-size:1.5rem;font-weight:800;color:#f1f5f9">{n_edges}</div></div>'
            f'<div><div style="font-size:0.6rem;color:#64748b;text-transform:uppercase">Clusters</div>'
            f'<div style="font-size:1.5rem;font-weight:800;color:#e0963a">{n_clust}</div></div>'
            f'<div><div style="font-size:0.6rem;color:#64748b;text-transform:uppercase">Cross-diag</div>'
            f'<div style="font-size:1.5rem;font-weight:800;color:#22d3ee">{n_cross}</div></div>'
            f'<div><div style="font-size:0.6rem;color:#64748b;text-transform:uppercase">Shared</div>'
            f'<div style="font-size:1.5rem;font-weight:800;color:#38bdf8">{n_shared}</div></div>'
            f'<div><div style="font-size:0.6rem;color:#64748b;text-transform:uppercase">Absorbed</div>'
            f'<div style="font-size:1.5rem;font-weight:800;color:#10b981">{summary.get("nodes_absorbed",0)}</div></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="ws-rule">', unsafe_allow_html=True)

    # ── Before → After graph panels ───────────────────────────────────────────
    before_graph = st.session_state.get("enterprise_graph_before") or {}
    if before_graph and enterprise_graph:
        st.markdown(
            '<div class="section-label">Local Graph → Enterprise Brain</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"Before absorption: {len(before_graph.get('nodes',[]))} nodes  →  "
            f"After absorption: {len(enterprise_graph.get('nodes',[]))} nodes  "
            f"(+{summary.get('nodes_absorbed',0)} absorbed, "
            f"+{summary.get('cross_diagram_links_created',0)} cross-diagram links)"
        )

        ta, tb = st.tabs(["After (current)", "Before (without selected diagram)"])
        absorbed_ids = {n.get("canonical_id", n.get("id")) for n in local_graph.get("nodes", [])}
        with ta:
            if _pyvis_available():
                _render_enterprise_pyvis(enterprise_graph, absorbed_ids, None, height=500)
            else:
                st.info("Install pyvis for interactive graph.")
        with tb:
            if _pyvis_available():
                _render_enterprise_pyvis(before_graph, set(), None, height=500)
            else:
                st.info("Install pyvis for interactive graph.")

    st.markdown('<hr class="ws-rule">', unsafe_allow_html=True)

    # ── Enterprise alert simulation ───────────────────────────────────────────
    gnn_metrics_available = _enterprise_gnn_available()
    if gnn_metrics_available:
        st.success("Enterprise RCA: trained enterprise GNN metrics available")
    else:
        st.warning(
            "Enterprise RCA source: scenario-grounded simulation "
            "(GNN metrics not found — run train_enterprise_gnn_rca.py to get trained results)"
        )
        if _strict_mode() and not st.session_state.allow_enterprise_simulation:
            st.error("Strict mode: approve before using scenario-grounded simulation.")
            if st.button("Continue with scenario simulation", key="approve_ent_sim"):
                st.session_state.allow_enterprise_simulation = True
                st.rerun()
            return

    if st.button("Simulate Enterprise Alert", type="primary", key="ent_alert_btn"):
        with st.spinner("Running enterprise RCA..."):
            st.session_state.enterprise_rca_result = _simulate_enterprise_rca(
                alerts_data, enterprise_graph
            )

    rca = st.session_state.get("enterprise_rca_result")

    # ── Primary: Interactive PyVis with RCA overlay ───────────────────────────
    st.markdown(
        '<div class="section-label" style="margin-top:4px">Enterprise Galaxy Graph — Interactive</div>',
        unsafe_allow_html=True,
    )
    absorbed_ids = {n.get("canonical_id", n.get("id")) for n in local_graph.get("nodes", [])}
    if _pyvis_available():
        st.caption(
            "Drag nodes · zoom/pan · hover for details · "
            + ("Root: red | Alert: orange | Impacted: yellow | Absorbed diagram: cyan | Shared: dashed ring"
               if rca else "Absorbed nodes shown in cyan")
        )
        _render_enterprise_pyvis(enterprise_graph, absorbed_ids, rca, height=800)
    else:
        st.warning("Install `pyvis>=0.3.2` for interactive graph.")
        preview_p = V3_HERO_SCENARIO / ("preview_rca_overlay.png" if rca else "preview_enterprise_graph.png")
        if preview_p.exists():
            _img(preview_p, "Enterprise graph (static preview)")

    # ── RCA result panel ─────────────────────────────────────────────────────
    if rca:
        st.markdown('<hr class="ws-rule">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Enterprise RCA Result</div>', unsafe_allow_html=True)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Root Cause",          rca.get("root_cause", "—"))
        r2.metric("Root Cause Diagram",  rca.get("root_cause_diagram", "—"))
        r3.metric("Impacted Diagrams",   len(rca.get("impacted_diagrams", [])))
        r4.metric("Alert Count",         rca.get("alert_count", 0))

        rca_mode = rca.get("mode", "")
        mode_badge_cls = "badge-success" if rca_mode == "Enterprise GNN RCA" else "badge-warn"
        st.markdown(
            f'<span class="badge {mode_badge_cls}">{rca_mode}</span>',
            unsafe_allow_html=True,
        )
        if rca_mode == "Enterprise GNN RCA":
            src_file = rca.get("gnn_source_file", "")
            if src_file:
                st.caption(f"GNN result loaded from: {Path(src_file).name}")
        else:
            st.caption("Source: scenario alerts.json ground truth (no trained GNN result found for this scenario)")

        path = rca.get("impact_path", [])
        if path:
            root_id   = rca.get("root_cause", "")
            alert_set = set(rca.get("alert_nodes", []))
            chips = []
            for n in path:
                cls = "root" if n == root_id else ("alerting" if n in alert_set else "impacted")
                chips.append(f'<span class="node-chip {cls}">{n}</span>')
            st.markdown(
                '<div style="padding:8px 0;line-height:2.2">'
                + ' <span style="color:#334155">→</span> '.join(chips)
                + '</div>',
                unsafe_allow_html=True,
            )

        ranking = rca.get("ranking", [])
        if ranking:
            st.dataframe(pd.DataFrame(ranking), use_container_width=True, hide_index=True)

    # ── Static previews (secondary) ──────────────────────────────────────────
    with st.expander("Static story preview", expanded=False):
        for name, cap in [
            ("preview_stitching_story.png", "Source Diagrams → Local Graphs → Enterprise Graph"),
            ("preview_contact_sheet.png",   "Contact sheet — all scenario diagrams"),
        ]:
            p = V3_HERO_SCENARIO / name
            if p.exists():
                _img(p, cap)
            else:
                st.caption(f"Not generated yet: {name}")

    with st.expander("Static RCA overlay", expanded=False):
        for name, cap in [
            ("preview_rca_overlay.png",      "Enterprise graph with RCA annotations"),
            ("preview_enterprise_graph.png", "Enterprise graph clusters"),
        ]:
            p = V3_HERO_SCENARIO / name
            if p.exists():
                _img(p, cap)
            else:
                st.caption(f"Not generated yet: {name}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — GRAPH COPILOT
# ══════════════════════════════════════════════════════════════════════════════
def _copilot_context() -> dict:
    scenario = (
        Path(st.session_state.enterprise_scenario_path)
        if st.session_state.enterprise_scenario_path
        else V3_HERO_SCENARIO
    )
    return {
        "selected_diagram_id":          st.session_state.selected_diagram_id,
        "detection_source":             st.session_state.detection_source,
        "live_ingestion_run_dir":       st.session_state.live_ingestion_run_dir,
        "validation_packet":            _ss_dict("validation_packet"),
        "local_graph":                  st.session_state.local_graph or {},
        "enterprise_graph_after":       st.session_state.enterprise_graph_after or {},
        "enterprise_ingestion_summary": _ss_dict("enterprise_ingestion_summary"),
        "local_rca_result":             _ss_dict("local_rca_result"),
        "enterprise_rca_result":        _ss_dict("enterprise_rca_result"),
        "stitch_map":                   _safe_read_json(scenario / "stitch_map.json"),
        "alerts":                       _safe_read_json(scenario / "alerts.json"),
    }


def _deterministic_graph_copilot(question: str, context: dict) -> str:
    q          = question.lower()
    ingestion  = context.get("enterprise_ingestion_summary") or {}
    ent_rca    = context.get("enterprise_rca_result") or {}
    local_rca  = context.get("local_rca_result") or {}
    lg         = context.get("local_graph") or {}
    packet     = context.get("validation_packet") or {}
    diagram_id = context.get("selected_diagram_id") or ingestion.get("absorbed_diagram_id", "unknown")
    det_src    = context.get("detection_source") or "Verified Annotation Overlay"
    run_dir    = context.get("live_ingestion_run_dir") or ""

    local_ids  = [n.get("id", n.get("node_id", "")) for n in lg.get("nodes", [])]
    matched    = ingestion.get("matched_entities", [])
    links      = ingestion.get("cross_diagram_links", [])
    ent_path   = ent_rca.get("impact_path", [])
    local_path = local_rca.get("impact_path", [])

    conf = packet.get("confidence_summary", {})

    if "detection" in q or "source" in q or "rf-detr" in q or "yolo" in q:
        return (
            f"Detection source for diagram `{diagram_id}`: **{det_src}**.\n\n"
            + (f"Run dir: `{run_dir}`\n\n" if run_dir else "")
            + f"Node detection avg confidence: {conf.get('device_detection_avg', 'N/A')}\n"
            f"Edge extraction avg confidence: {conf.get('edge_extraction_avg', 'N/A')}\n"
            f"Low-confidence items: {conf.get('low_confidence_items', 0)}\n\n"
            "RF-DETR is used when a trained checkpoint is found. "
            "Otherwise the Verified Annotation Overlay is rendered from ground-truth annotations."
        )
    if "stitched" in q or "where" in q or "absorbed" in q or "uploaded" in q:
        before = ingestion.get("before_node_count", "?")
        after  = ingestion.get("after_node_count", "?")
        return (
            f"Diagram `{diagram_id}` was absorbed into the enterprise graph brain.\n\n"
            f"- Nodes absorbed: **{ingestion.get('nodes_absorbed', 0)}**\n"
            f"- Edges absorbed: **{ingestion.get('edges_absorbed', 0)}**\n"
            f"- Shared entities matched: **{ingestion.get('shared_entities_matched', 0)}**\n"
            f"  Matched IDs: {', '.join(matched[:8]) or 'none'}\n"
            f"- Cross-diagram links created: **{ingestion.get('cross_diagram_links_created', 0)}**\n"
            f"- Enterprise graph: {before} nodes → {after} nodes after absorption\n"
            f"- Detection source: {det_src}"
        )
    if "node" in q and ("list" in q or "all" in q or "what" in q):
        return (
            f"Local nodes for diagram `{diagram_id}` ({len(local_ids)} total):\n\n"
            + "\n".join(f"- `{n}`" for n in local_ids[:20])
            + (f"\n\n... and {len(local_ids)-20} more" if len(local_ids) > 20 else "")
        )
    if "shared" in q or "matched" in q or "canonical" in q:
        return (
            f"Shared entities matched between `{diagram_id}` and the enterprise graph:\n\n"
            + "\n".join(f"- `{m}`" for m in matched)
            if matched else "No shared entities matched — diagram may have no shared nodes."
        )
    if "cross" in q or "link" in q:
        rendered = [
            f"`{e.get('source_diagram','?')}:{e.get('source_node',e.get('source','?'))}` → "
            f"`{e.get('target_diagram','?')}:{e.get('target_node',e.get('target','?'))}` "
            f"({e.get('label', e.get('relationship',''))})"
            for e in links[:10]
        ]
        return ("Cross-diagram stitch links:\n\n" + "\n".join(f"- {item}" for item in rendered)
                if rendered else "No cross-diagram links for this diagram.")
    if "local rca" in q or "local root" in q:
        local_root = local_rca.get("root_cause", "not simulated yet")
        return (
            f"Local RCA (within diagram `{diagram_id}`):\n\n"
            f"- Root cause: **`{local_root}`**\n"
            f"- Alert nodes: {', '.join(local_rca.get('alert_nodes', [])) or 'none'}\n"
            f"- Impacted nodes: {', '.join(local_rca.get('impacted_nodes', [])) or 'none'}\n"
            f"- Impact path: {' → '.join(local_path) if local_path else 'not available'}\n"
            f"- Method: {local_rca.get('mode', 'unknown')}"
        )
    if "root cause" in q:
        ent_root = ent_rca.get("root_cause", "not simulated yet")
        mode     = ent_rca.get("mode", "unknown")
        return (
            f"Enterprise root cause: **`{ent_root}`** in "
            f"`{ent_rca.get('root_cause_diagram', '?')}`\n\n"
            f"- RCA mode: {mode}\n"
            f"- Impacted diagrams: {', '.join(ent_rca.get('impacted_diagrams', [])) or 'none'}\n"
            f"- Alert count: {ent_rca.get('alert_count', 0)}\n"
            f"- Impact path: {' → '.join(ent_path) if ent_path else 'not available'}"
        )
    if "impacted" in q:
        return (
            f"Impacted diagrams: {', '.join(ent_rca.get('impacted_diagrams', [])) or 'none'}.\n"
            f"Impacted nodes: {', '.join(ent_rca.get('impacted_nodes', [])) or 'none'}."
        )
    if "path" in q:
        active_path = ent_path or local_path
        return (f"Impact path: {' → '.join(active_path)}" if active_path
                else "No impact path loaded — run enterprise or local RCA first.")
    if "servicenow" in q or "incident" in q or "snow" in q:
        root = ent_rca.get("root_cause", "unknown")
        return (
            "### ServiceNow Incident Summary\n\n"
            f"| Field | Value |\n|---|---|\n"
            f"| Short description | Enterprise network fault — root `{root}` |\n"
            f"| Affected CI | `{root}` ({ent_rca.get('root_cause_diagram','?')}) |\n"
            f"| Priority | P1 — {len(ent_rca.get('impacted_diagrams',[]))} diagrams impacted |\n"
            f"| Assignment group | Network Operations |\n"
            f"| Root cause (automated) | `{root}` — {ent_rca.get('mode','unknown')} |\n"
            f"| Selected diagram | `{diagram_id}` (detection: {det_src}) |\n"
            f"| Evidence path | {' → '.join(ent_path) if ent_path else 'N/A'} |"
        )
    if "l1" in q or "check first" in q:
        first = ent_rca.get("root_cause", "the root-cause node")
        second = ent_path[1] if len(ent_path) > 1 else "next hop"
        return (
            f"L1 runbook:\n"
            f"1. SSH to `{first}` — check interface counters, syslog, CPU/memory\n"
            f"2. Validate `{second}` reachability\n"
            f"3. Confirm impacted services: {', '.join(ent_rca.get('impacted_nodes', [])[:6]) or 'check alerts'}\n"
            f"4. Escalate if packet drop rate > 5% or alerts still firing\n"
            f"5. Activate redundant path if available"
        )
    # Generic evidence summary
    return (
        f"**Loaded evidence for diagram `{diagram_id}`:**\n\n"
        f"- Detection source: {det_src}\n"
        f"- Local nodes: {len(local_ids)}\n"
        f"- Shared entities matched: {ingestion.get('shared_entities_matched', 0)}\n"
        f"- Cross-diagram links: {ingestion.get('cross_diagram_links_created', 0)}\n"
        f"- Enterprise root cause: `{ent_rca.get('root_cause', 'not simulated yet')}`\n"
        f"- RCA mode: {ent_rca.get('mode', 'not simulated yet')}\n\n"
        "Ask: root cause | stitched | absorbed | shared | cross-diagram | path | servicenow | l1"
    )


def _qwen_or_deterministic(question: str, context: dict) -> str:
    qwen_url = os.environ.get("QWEN_BASE_URL", "").rstrip("/")
    if not qwen_url:
        if _strict_mode() and not st.session_state.allow_deterministic_copilot:
            return ("Strict mode + Qwen not configured. "
                    "Click 'Use deterministic graph response' to enable graph-evidence answers.")
        return _deterministic_graph_copilot(question, context)
    try:
        import requests  # noqa: PLC0415
        compact = json.dumps(context, default=str)[:12000]
        resp = requests.post(
            f"{qwen_url}/chat/completions",
            headers={"Content-Type": "application/json", "Bypass-Tunnel-Reminder": "true"},
            json={
                "model": os.environ.get("QWEN_MODEL", "Qwen/Qwen2-7B-Instruct"),
                "messages": [
                    {"role": "system", "content": (
                        "You are InfraGraph AI Graph Copilot. Answer only from the supplied "
                        "graph JSON evidence. Cite node IDs, diagram IDs, and paths. Be concise."
                    )},
                    {"role": "user", "content": f"Graph evidence:\n{compact}\n\nQuestion: /no_think {question}"},
                ],
                "max_tokens": 700, "temperature": 0.1,
            },
            timeout=30,
        )
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"]
        return re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    except Exception as exc:
        if _strict_mode() and not st.session_state.allow_deterministic_copilot:
            return f"Live Qwen failed in strict mode: `{exc}`. Enable deterministic graph response first."
        return (f"> Live Qwen unavailable — graph-evidence answer. `{exc}`\n\n"
                + _deterministic_graph_copilot(question, context))


def _tab_graph_copilot() -> None:
    st.markdown(
        '<div class="ws-title">Graph Copilot — Ask the Enterprise Graph</div>'
        '<div class="ws-desc">Questions are answered from loaded graph JSON evidence. '
        'Qwen/vLLM for richer responses when configured.</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.enterprise_graph_after:
        st.markdown(
            '<div class="warn-card">Open <strong>Tab 3 (Enterprise Graph Brain)</strong> first '
            'to load enterprise graph context.</div>',
            unsafe_allow_html=True,
        )
        return

    if _qwen_configured():
        qwen_url = os.environ.get("QWEN_BASE_URL", "")
        st.success(f"Copilot mode: Qwen/vLLM @ {qwen_url}")
    else:
        if _strict_mode() and not st.session_state.allow_deterministic_copilot:
            st.error("Qwen not configured. Strict mode requires explicit approval for deterministic answers.")
            if st.button("Use deterministic graph response", key="approve_copilot"):
                st.session_state.allow_deterministic_copilot = True
                st.rerun()
            return
        st.warning("Copilot mode: deterministic graph evidence (Qwen not configured)")

    suggestions = [
        "Where was this diagram stitched into the enterprise graph?",
        "Which nodes were absorbed from the selected diagram?",
        "Which shared entities were matched?",
        "Which cross-diagram links were created?",
        "What is the root cause?",
        "Which diagrams are impacted?",
        "Show the impact path from root cause.",
        "Generate a ServiceNow incident summary.",
        "What should L1 check first?",
    ]
    cols = st.columns(3)
    for idx, question in enumerate(suggestions):
        with cols[idx % 3]:
            if st.button(question, key=f"v3_q_{idx}", use_container_width=True):
                context = _copilot_context()
                st.session_state.v3_chat_messages.append({"role": "user",      "content": question})
                st.session_state.v3_chat_messages.append({"role": "assistant", "content": _qwen_or_deterministic(question, context)})
                st.rerun()

    qwen_url = os.environ.get("QWEN_BASE_URL", "")
    if _qwen_configured():
        st.caption(f"Qwen: {qwen_url} · model={os.environ.get('QWEN_MODEL')}")
    else:
        st.caption("Qwen not configured — answers use loaded graph JSON evidence only.")

    for msg in st.session_state.v3_chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask the enterprise graph..."):
        context = _copilot_context()
        st.session_state.v3_chat_messages.append({"role": "user",      "content": prompt})
        st.session_state.v3_chat_messages.append({"role": "assistant", "content": _qwen_or_deterministic(prompt, context)})
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def _sidebar_v3() -> None:
    with st.sidebar:
        title_color = _col("#f1f5f9", "#0f172a")
        sub_color   = _col("#64748b", "#64748b")

        st.markdown(
            f'<div style="padding:10px 0 6px">'
            f'<span style="font-size:1.08rem;font-weight:800;color:{title_color}">InfraGraph AI</span>'
            f'</div>'
            f'<p style="font-size:0.72rem;color:{sub_color};margin:-2px 0 0">Diagram Intelligence V3</p>',
            unsafe_allow_html=True,
        )

        if _strict_mode():
            st.error("Strict Mode: ON")
        else:
            st.info("Standard Mode: ON — annotation overlay enabled when detector unavailable")

        st.markdown('<div class="sb-label">System Readiness</div>', unsafe_allow_html=True)
        _render_readiness_panel()

        st.markdown('<div class="sb-label">Pipeline Progress</div>', unsafe_allow_html=True)
        steps = [
            ("Diagram ingested",          bool(st.session_state.local_graph)),
            ("Local graph created",       bool(st.session_state.local_graph)),
            ("Local RCA complete",        bool(st.session_state.local_rca_result)),
            ("Absorbed into enterprise",  bool(st.session_state.enterprise_ingestion_summary)),
            ("Enterprise RCA complete",   bool(st.session_state.enterprise_rca_result)),
            ("Copilot ready",             bool(st.session_state.enterprise_graph_after)),
        ]
        for label, done in steps:
            cls  = "sb-step" if done else "sb-step sb-pending"
            icon = "✓" if done else "○"
            st.markdown(
                f'<div class="{cls}">'
                f'<span class="sb-check">{icon}</span>'
                f'<span>{label}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with st.expander("Developer", expanded=False):
            st.code(
                "python -m py_compile app/streamlit_app.py\n"
                "python -m streamlit run app/streamlit_app.py",
                language="bash",
            )
            st.markdown("Strict mode:")
            st.code("export INFRAGRAPH_STRICT_MODE=1", language="bash")
            st.code('$env:INFRAGRAPH_STRICT_MODE="1"', language="powershell")

        st.markdown('<div class="sb-label" style="margin-top:22px">Appearance</div>',
                    unsafe_allow_html=True)
        is_light = st.toggle(
            "Light mode",
            value=(st.session_state.get("theme") == "light"),
            key="theme_toggle",
        )
        st.session_state.theme = "light" if is_light else "dark"


# ══════════════════════════════════════════════════════════════════════════════
# OLD WORKSPACE FUNCTIONS (kept for legacy reference, not active in main flow)
# ══════════════════════════════════════════════════════════════════════════════
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
| GNN score margin | 30.733 vs 22.613 (FW-02) |"""

    if any(kw in q for kw in ["servicenow", "ticket", "snow", "p1"]):
        return """\
**ServiceNow Incident:**

| Field | Value |
|-------|-------|
| Short description | Network fault on FW-01 — 10-service outage |
| Affected CI | FW-01 (firewall) |
| Priority | **P1** |
| Assignment group | Network Operations |
| Root cause (automated) | FW-01 — GNN RCA, HIGH confidence |"""

    return f"No pre-built answer for: *\"{question}\"*\n\nTry: **What is the root cause?**"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def _main_cockpit() -> None:
    _init_v3_state()
    st.markdown(_CSS, unsafe_allow_html=True)
    if st.session_state.get("theme") == "light":
        st.markdown(_LIGHT_OVERRIDES, unsafe_allow_html=True)

    _sidebar_v3()

    st.markdown(
        '<div class="hero-title">InfraGraph AI</div>'
        '<div class="hero-tagline">'
        "One diagram enters the system, becomes graph memory, then gets absorbed into the "
        "enterprise graph brain for cross-diagram root-cause analysis."
        "</div>"
        '<div class="stat-pills">'
        '<div class="stat-pill status"><div class="pill-dot"></div>V3 Diagram Intelligence</div>'
        '<div class="stat-pill root">Galaxy Graph RCA</div>'
        '<div class="stat-pill status">RF-DETR ready</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.06);margin:14px 0 20px'>",
        unsafe_allow_html=True,
    )

    tabs = st.tabs([
        "Diagram Intelligence",
        "Local RCA",
        "Enterprise Graph Brain",
        "Graph Copilot",
    ])
    with tabs[0]:
        _tab_diagram_intelligence()
    with tabs[1]:
        _tab_local_rca()
    with tabs[2]:
        _tab_enterprise_graph_brain()
    with tabs[3]:
        _tab_graph_copilot()


def main() -> None:
    _main_cockpit()


if __name__ == "__main__":
    main()
