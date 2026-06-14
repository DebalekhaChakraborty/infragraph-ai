"""
InfraGraph AI Command Center — Streamlit cockpit.

Two sections in Diagram Intelligence (Tab 1):
  A. Diagram Gallery     — Browse diagrams available in graph memory
  B. Onboard New Diagram — Live diagram intelligence: detect -> graph -> absorb -> RCA

Five workspaces (tabs):
  1. Diagram Intelligence   — image to local graph
  2. Topology RCA           — single-diagram graph reasoning
  3. Enterprise Graph Brain — local graph absorbed into enterprise graph
  4. Enterprise GNN RCA     — cross-diagram graph reasoning, alert simulation, interactive topology
  5. Graph Copilot          — ask the enterprise graph
"""
from __future__ import annotations

import json
import html
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
_GALLERY_REFERENCE_ID = "diagram_0373"
_OVERLAY_RENDERER_VERSION = "v3_clean_overlay_v1"


def get_infragraph_v3_root(repo_root: Path) -> Path:
    preferred = repo_root / "datasets" / "infragraph_v3"
    legacy = repo_root / "datasets" / "diagram_v3_enterprise"
    if preferred.exists():
        return preferred
    return legacy

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

# ── Centralized path helpers (src/paths.py) ───────────────────────────────────
try:
    from paths import (  # type: ignore
        demo_asset_path as _demo_asset_path,
        runtime_path as _runtime_path,
        model_artifact_path as _model_artifact_path,
        report_path as _report_path,
        RUNTIME_STATE_DIR as _RUNTIME_STATE_DIR,
        DEMO_ASSETS_DIR as _DEMO_ASSETS_DIR,
    )
except ImportError:
    # Fallback: route everything through legacy outputs/ so the app never crashes
    def _demo_asset_path(*parts):       # type: ignore[misc]
        return REPO_ROOT / "outputs" / Path(*parts)
    def _runtime_path(*parts):          # type: ignore[misc]
        return REPO_ROOT / "outputs" / Path(*parts)
    def _model_artifact_path(*parts):   # type: ignore[misc]
        return REPO_ROOT / "outputs" / Path(*parts)
    def _report_path(*parts):           # type: ignore[misc]
        return REPO_ROOT / "outputs" / Path(*parts)

# Force-evict any stale cached module so we always load from the current file on disk.
_sys.modules.pop("runtime_ingestion", None)
_RUNTIME_INGESTION_ERR = ""
try:
    import importlib as _importlib
    _ri_mod        = _importlib.import_module("runtime_ingestion")
    _live_ingest   = getattr(_ri_mod, "run_live_v3_ingestion")                             # type: ignore
    _live_absorb   = getattr(_ri_mod, "run_enterprise_absorption")                         # type: ignore
    _run_ingestion = getattr(_ri_mod, "run_ingestion")                                     # type: ignore
    _run_absorption= getattr(_ri_mod, "run_absorption")                                    # type: ignore
    _RUNTIME_INGESTION = True
except Exception as _e:
    _RUNTIME_INGESTION     = False
    _RUNTIME_INGESTION_ERR = f"{type(_e).__name__}: {_e}"
    _live_ingest           = None  # type: ignore
    _live_absorb           = None  # type: ignore
    _run_ingestion         = None  # type: ignore
    _run_absorption        = None  # type: ignore

# render_v3_annotation_preview is a lightweight drawing helper that only requires
# stdlib + Pillow — import it separately so a missing heavy dep never blocks it.
try:
    from runtime_ingestion import render_v3_annotation_preview as _render_ann_preview      # type: ignore
    _RENDER_ANN_IMPORT_ERR = ""
except Exception as _e:
    _render_ann_preview       = None  # type: ignore
    _RENDER_ANN_IMPORT_ERR    = str(_e)

try:
    from live_detector import find_best_yolo_checkpoint as _find_ckpt  # type: ignore
    from live_detector import run_live_yolo_detection as _run_yolo      # type: ignore
    _LIVE_DETECTOR = True
except Exception:
    _LIVE_DETECTOR = False
    _find_ckpt = None  # type: ignore
    _run_yolo  = None  # type: ignore

_app_dir_for_bridge = str(REPO_ROOT / "app")
if _app_dir_for_bridge not in _sys.path:
    _sys.path.insert(0, _app_dir_for_bridge)
try:
    from rfdetr_subprocess_bridge import (                                      # type: ignore
        resolve_rfdetr_python as _resolve_rfdetr_python,
        resolve_rfdetr_python_details as _resolve_rfdetr_python_details,
        find_best_rfdetr_checkpoint as _find_rfdetr_ckpt,
        check_rfdetr_runtime as _check_rfdetr_runtime,
        check_rfdetr_http_service as _check_rfdetr_http_service,
        rfdetr_service_base_url as _rfdetr_service_base_url,
        run_rfdetr_detection as _run_rfdetr_detection,
        run_rfdetr_subprocess as _run_rfdetr_subprocess,
    )
    _RFDETR_BRIDGE_OK = True
    _RFDETR_BRIDGE_ERR = ""
except Exception as _rfb_exc:
    _RFDETR_BRIDGE_OK = False
    _RFDETR_BRIDGE_ERR = str(_rfb_exc)
    _resolve_rfdetr_python = None  # type: ignore
    _resolve_rfdetr_python_details = None  # type: ignore
    _find_rfdetr_ckpt = None       # type: ignore
    _check_rfdetr_runtime = None   # type: ignore
    _check_rfdetr_http_service = None  # type: ignore
    _rfdetr_service_base_url = None    # type: ignore
    _run_rfdetr_detection = None       # type: ignore
    _run_rfdetr_subprocess = None  # type: ignore

# ── Incident simulation package ───────────────────────────────────────────────
# Evict any stale cached version so the current on-disk code is always loaded.
for _k in [k for k in _sys.modules if "incident_simulation" in k]:
    _sys.modules.pop(_k, None)
_INCIDENT_SIM_ERR: str = ""
try:
    from incident_simulation import (                                         # type: ignore
        build_local_incident              as _build_local_incident_pkg,
        build_enterprise_incident         as _build_enterprise_incident_pkg,
        build_cross_diagram_hero_incident as _build_cross_hero_incident_pkg,
    )
    _INCIDENT_SIM_OK = True
except Exception as _isim_exc:
    _INCIDENT_SIM_OK  = False
    _INCIDENT_SIM_ERR = str(_isim_exc)
    _build_local_incident_pkg      = None  # type: ignore
    _build_enterprise_incident_pkg = None  # type: ignore
    _build_cross_hero_incident_pkg = None  # type: ignore

# ── AI remediation package ───────────────────────────────────────────────────
import os as _os
def _resolve_qwen_runtime_config() -> dict:
    timeout_raw = (
        _os.environ.get("INFRAGRAPH_QWEN_TIMEOUT")
        or _os.environ.get("QWEN_TIMEOUT")
        or "60"
    )
    try:
        timeout = int(timeout_raw)
    except Exception:
        timeout = 60
    return {
        "base_url": (
            _os.environ.get("INFRAGRAPH_QWEN_BASE_URL")
            or _os.environ.get("QWEN_BASE_URL")
            or "http://localhost:8000/v1"
        ).rstrip("/"),
        "model": (
            _os.environ.get("INFRAGRAPH_QWEN_MODEL")
            or _os.environ.get("QWEN_MODEL")
            or "infragraph"
        ),
        "timeout": timeout,
    }


_QWEN_CONFIG    = _resolve_qwen_runtime_config()
_QWEN_BASE_URL  = _QWEN_CONFIG["base_url"]
_QWEN_MODEL     = _QWEN_CONFIG["model"]
_QWEN_TIMEOUT   = _QWEN_CONFIG["timeout"]
_LORA_ADAPTER   = _os.environ.get("INFRAGRAPH_LORA_ADAPTER_PATH", "")

for _k in [k for k in _sys.modules if "ai_remediation" in k]:
    _sys.modules.pop(_k, None)
_AI_REM_OK = False
try:
    from ai_remediation import (                                               # type: ignore
        make_remediation_input          as _make_remediation_input,
        generate_resolution_plan        as _generate_resolution_plan,
        generate_remediation_with_qwen  as _generate_qwen_remediation,
        check_vllm_available            as _check_vllm_available_fn,
        generate_template_remediation   as _generate_template_remediation,
        get_qwen_runtime_config         as _get_qwen_runtime_config,
    )
    _AI_REM_OK = True
except Exception:
    _make_remediation_input         = None  # type: ignore
    _generate_resolution_plan       = None  # type: ignore
    _generate_qwen_remediation      = None  # type: ignore
    _check_vllm_available_fn        = None  # type: ignore
    _generate_template_remediation  = None  # type: ignore
    _get_qwen_runtime_config        = None  # type: ignore

# ── FalconVue 3D WebGL renderer (optional; PyVis remains available) ──
_app_dir = str(Path(__file__).parent)
if _app_dir not in _sys.path:
    _sys.path.insert(0, _app_dir)
try:
    from falconvue_graph import render_falconvue_graph as _render_falconvue_graph  # type: ignore
    _FALCONVUE_OK = True
except Exception:
    _FALCONVUE_OK = False
    _render_falconvue_graph = None  # type: ignore

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

/* ── Incident timeline ── */
.timeline-section { margin: 16px 0 8px; }
.timeline-diag-hdr {
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase;
    color: #475569; padding: 8px 0 4px; margin-top: 10px;
    border-top: 1px solid rgba(255,255,255,0.05);
}
.diag-label {
    display: inline-block; padding: 2px 8px; border-radius: 5px;
    font-size: 0.65rem; font-weight: 600; letter-spacing: 0.04em;
    background: rgba(139,92,246,0.1); color: #a78bfa;
    border: 1px solid rgba(139,92,246,0.2); margin-left: 6px; vertical-align: middle;
}
.diag-label.cross { background: rgba(245,158,11,0.1); color: #f59e0b;
                    border-color: rgba(245,158,11,0.25); }
/* ── Traversal ── */
.traversal-card {
    background: rgba(96,165,250,0.04); border: 1px solid rgba(96,165,250,0.18);
    border-radius: 12px; padding: 16px 20px; margin: 8px 0 12px;
}
.traversal-step-lbl {
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
    color: #60a5fa; margin-bottom: 6px;
}
.traversal-node {
    font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; font-weight: 900;
    color: #67e8f9; line-height: 1.2;
}
.traversal-desc { font-size: 0.8rem; color: #94a3b8; margin-top: 6px; line-height: 1.55; }
.node-chip.visited { background: rgba(96,165,250,0.15); color: #93c5fd;
                     border-color: rgba(96,165,250,0.35); }
.node-chip.current { background: rgba(103,232,249,0.15); color: #67e8f9;
                     border: 2px solid #67e8f9; }
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

## ITSM Ticket

| Field | Value |
|-------|-------|
| Short description | Network fault on FW-01 — {n}-service outage |
| Affected CI | FW-01 (firewall) |
| Priority | P1 — {n} downstream nodes impacted |
| Assignment group | Network Operations |
| Root cause (automated) | FW-01 — Enterprise GNN RCA, confidence HIGH |

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

    v3_scen = get_infragraph_v3_root(root) / "scenarios"
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
    v3_scen = get_infragraph_v3_root(root) / "scenarios"
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


_MANIFEST_PATH_KEYS_GALLERY = (
    "image_path", "annotation_path", "detected_preview_path", "preview_path",
    "local_graph_path", "enterprise_graph_path", "stitch_map_path", "alerts_path",
)
_MANIFEST_PATH_KEYS_ONBOARD = (
    "image_path", "annotation_path", "local_graph_path",
    "enterprise_graph_path", "stitch_map_path", "alerts_path", "sample_dir",
)
_ROOTED_SEGS = ("datasets", "assets", "outputs", "scripts", "src", "app")


def _resolve_manifest_path(raw: str, root: Path) -> str:
    """
    Resolve a manifest path to an absolute string that exists on disk.

    Strategy (in order):
    1. Empty → return "".
    2. Already a non-absolute (relative) path → join with root.
    3. Absolute and exists → return as-is.
    4. Absolute but stale (built on another machine) → find the first known
       top-level segment and re-root under `root`.
    5. Fall through → return original string (caller checks existence).
    """
    if not raw:
        return ""
    p = Path(raw)
    if not p.is_absolute():
        return str(root / p)
    if p.exists():
        return str(p)
    # Attempt re-rooting for stale absolute paths
    parts = p.parts
    for i, part in enumerate(parts):
        if part in _ROOTED_SEGS:
            candidate = root.joinpath(*parts[i:])
            if candidate.exists():
                return str(candidate)
    return str(p)


@st.cache_data(ttl=3600)
def _load_gallery_manifest(repo_root_str: str) -> list[dict]:
    """Load assets/gallery/manifest.json and resolve/repair all paths."""
    root = Path(repo_root_str)
    mf   = root / "assets" / "gallery" / "manifest.json"
    if not mf.exists():
        return []
    try:
        records: list[dict] = json.loads(mf.read_text(encoding="utf-8"))
    except Exception:
        return []
    for r in records:
        for k in _MANIFEST_PATH_KEYS_GALLERY:
            v = r.get(k, "")
            if v:
                r[k] = _resolve_manifest_path(v, root)
    return records


@st.cache_data(ttl=3600)
def _load_onboarding_manifest(repo_root_str: str) -> list[dict]:
    """Load assets/onboarding/manifest.json and resolve/repair all paths."""
    root = Path(repo_root_str)
    mf   = root / "assets" / "onboarding" / "manifest.json"
    if not mf.exists():
        return []
    try:
        records: list[dict] = json.loads(mf.read_text(encoding="utf-8"))
    except Exception:
        return []
    for r in records:
        for k in _MANIFEST_PATH_KEYS_ONBOARD:
            v = r.get(k, "")
            if v:
                r[k] = _resolve_manifest_path(v, root)
    return records


@st.cache_data(ttl=60)
def _cached_rfdetr_runtime_check(python_executable: str) -> dict:
    if not _RFDETR_BRIDGE_OK or _check_rfdetr_runtime is None:
        return {"ok": False, "error": _RFDETR_BRIDGE_ERR, "python_executable": python_executable}
    return _check_rfdetr_runtime(python_executable)


@st.cache_data(ttl=60)
def _cached_rfdetr_python_resolution() -> dict:
    if not _RFDETR_BRIDGE_OK or _resolve_rfdetr_python_details is None:
        return {
            "requested_detector_python": os.environ.get("INFRAGRAPH_RFDETR_PYTHON", "").strip() or "python",
            "resolved_detector_python": "",
            "python_executable": "",
            "import_ok": False,
            "runtime": {"ok": False, "error": _RFDETR_BRIDGE_ERR},
            "fallback_reason": _RFDETR_BRIDGE_ERR,
        }
    return _resolve_rfdetr_python_details()


@st.cache_data(ttl=30)
def _cached_rfdetr_http_health() -> dict:
    if not _RFDETR_BRIDGE_OK or _rfdetr_service_base_url is None or _check_rfdetr_http_service is None:
        return {"ok": False, "service_url": "", "error": _RFDETR_BRIDGE_ERR}
    base_url = _rfdetr_service_base_url()
    if not base_url:
        return {"ok": False, "service_url": "", "error": "INFRAGRAPH_RFDETR_BASE_URL is not set"}
    return _check_rfdetr_http_service(base_url)


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
    """Try PyVis first; use matplotlib."""
    if _render_local_graph_pyvis(local_graph, overlay, height):
        return
    st.caption("Interactive graph requires pyvis — showing matplotlib preview")
    _render_local_graph_mpl(local_graph, overlay, height / 100)


# ══════════════════════════════════════════════════════════════════════════════
# V3 CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
V3_DATASET_ROOT  = get_infragraph_v3_root(REPO_ROOT)
V3_HERO_SELECTION = _demo_asset_path("demo_hero") / "hero_scenario.json"


def _resolve_v3_hero_scenario() -> Path:
    default_path = V3_DATASET_ROOT / "scenarios" / "train" / "enterprise_v3_0000"
    if V3_HERO_SELECTION.exists():
        try:
            data = json.loads(V3_HERO_SELECTION.read_text(encoding="utf-8"))
            raw_path = data.get("scenario_path") or data.get("source_scenario_path") or data.get("path")
            if raw_path:
                candidate = Path(raw_path)
                if not candidate.is_absolute():
                    candidate = REPO_ROOT / candidate
                if candidate.exists():
                    return candidate
        except Exception:
            pass
    return default_path


V3_HERO_SCENARIO = _resolve_v3_hero_scenario()
V3_DIAGRAM_IDS = {
    "Branch Office Topology":       "branch_topology",
    "WAN / MPLS Topology":          "wan_topology",
    "Datacenter Topology":          "datacenter_topology",
    "Application & Database Tier":  "app_db_topology",
    "Shared Services Topology":     "shared_services_topology",
}
V3_REQUIRED_DIAGRAMS = list(V3_DIAGRAM_IDS.values())
V3_ENTERPRISE_GNN_METRICS = _demo_asset_path("enterprise_gnn_rca") / "enterprise_gnn_metrics.json"
V3_ENTERPRISE_GNN_MODEL   = REPO_ROOT / "model_artifacts" / "enterprise_gnn_rca" / "enterprise_gnn_rca.pt"
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
    "local_rca_result", "local_incident",
    "enterprise_scenario_path", "enterprise_graph_before",
    "enterprise_graph_after", "enterprise_ingestion_summary",
    "enterprise_rca_result", "enterprise_incident",
    "allow_local_simulation", "allow_enterprise_simulation", "allow_deterministic_copilot",
    "catalog_selected_record",
    # live runtime state
    "live_ingestion_run_dir", "detection_source", "detected_image_path",
    "enterprise_absorbed",
    "live_detection_result",
]


_V3_DICT_KEYS = {
    "validation_packet", "local_rca_result", "local_incident",
    "enterprise_rca_result", "enterprise_incident",
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


# ── Path repair ───────────────────────────────────────────────────────────────
_ROOTED_SEGMENTS = ("datasets", "assets", "outputs", "scripts", "src", "app")


def _repair_path(raw: str, repo_root: Path) -> Path:
    """
    Return a resolved Path for `raw`.

    Priority:
    1. If the path exists as-is → return it.
    2. If it is absolute but stale (built on a different machine), try re-rooting:
       - Find the first occurrence of a known top-level segment in the path parts
         and reconstruct using repo_root.
    3. If it is relative, join with repo_root.
    4. Fall through: return the original Path unchanged (caller checks .exists()).
    """
    if not raw:
        return Path(raw)
    p = Path(raw)
    if p.exists():
        return p
    # try to re-root absolute stale paths
    if p.is_absolute():
        parts = p.parts
        for i, part in enumerate(parts):
            if part in _ROOTED_SEGMENTS:
                candidate = repo_root.joinpath(*parts[i:])
                if candidate.exists():
                    return candidate
        # last resort: join just the filename under the same relative structure
        return p
    # relative path
    candidate = repo_root / p
    if candidate.exists():
        return candidate
    return p


def _load_renderer(repo_root: Path) -> "tuple[object | None, str]":
    """
    Return (render_v3_annotation_preview_fn, error_str).

    Tries in order:
    1. Module-level _render_ann_preview (works when import succeeded at startup).
    2. importlib.util.spec_from_file_location — loads from the known absolute path,
       completely bypassing sys.modules.  This is immune to Streamlit's hot-reload
       leaving a stale cached module that predates the function being added.
    """
    if _render_ann_preview is not None:
        return _render_ann_preview, ""

    ri_path = repo_root / "src" / "runtime_ingestion.py"
    if not ri_path.exists():
        return None, f"runtime_ingestion.py not found at {ri_path}"

    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("_ri_fresh", str(ri_path))
        _mod  = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)           # type: ignore[union-attr]
        fn = getattr(_mod, "render_v3_annotation_preview", None)
        if fn is None:
            return None, "render_v3_annotation_preview not found in module"
        return fn, ""
    except Exception as exc:
        return None, str(exc)


def _ensure_annotation_overlay(
    record: dict,
    repo_root: Path,
    img_p: "Path | None" = None,
    ann_p: "Path | None" = None,
    draw_connectors: bool = False,
) -> tuple[str, dict]:
    """
    Render a Verified Annotation Overlay for a gallery record and cache to disk.

    Accepts optional pre-resolved img_p / ann_p so the caller's already-repaired
    paths are used directly (avoids double path-repair).

    Returns (overlay_path_str, render_meta).  overlay_path_str is "" on failure.
    Never raises.
    """
    renderer, load_err = _load_renderer(repo_root)
    if renderer is None:
        return "", {"error": load_err, "renderer_imported": False}

    # ── resolve paths ──────────────────────────────────────────────────────────
    if img_p is None:
        img_raw = record.get("image_path", "")
        if not img_raw:
            return "", {"error": "image_path missing from record"}
        img_p = _repair_path(img_raw, repo_root)

    if ann_p is None:
        ann_raw = record.get("annotation_path", "")
        if not ann_raw:
            return "", {"error": "annotation_path missing from record"}
        ann_p = _repair_path(ann_raw, repo_root)

    if not img_p.exists():
        return "", {"error": f"image not found: {img_p}", "renderer_imported": True}
    if not ann_p.exists():
        return "", {"error": f"annotation not found: {ann_p}", "renderer_imported": True}

    # ── stable output path (reused across renders) ─────────────────────────────
    gid = record.get("gallery_id", "")
    if not gid:
        scen = record.get("source_scenario_id", "unknown")
        did  = record.get("source_diagram_id",  "unknown")
        gid  = f"{scen}__{did}"
    out_dir  = repo_root / "outputs" / "annotation_overlays" / gid
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "detected_clean_connectors" if draw_connectors else "detected_clean"
    out_path = out_dir / f"{stem}.png"
    meta_path = out_dir / f"{stem}.meta.json"

    # ── skip re-render if already on disk ─────────────────────────────────────
    cached_meta = load_json(str(meta_path)) if meta_path.exists() else None
    if (
        out_path.exists()
        and isinstance(cached_meta, dict)
        and cached_meta.get("renderer_version") == _OVERLAY_RENDERER_VERSION
        and cached_meta.get("overlay_mode") == "clean"
        and bool(cached_meta.get("draw_connectors", False)) == bool(draw_connectors)
    ):
        return str(out_path), {"rendered": True, "cached": True,
                               "boxes_rendered": cached_meta.get("boxes_rendered", 0),
                               "boxes_skipped": cached_meta.get("boxes_skipped", 0),
                               "boxes_skipped_large": cached_meta.get("boxes_skipped_large", 0),
                               "connectors_rendered": cached_meta.get("connectors_rendered", 0),
                               "connectors_skipped": cached_meta.get("connectors_skipped", 0),
                               "connectors_skipped_long": cached_meta.get("connectors_skipped_long", 0),
                               "overlay_mode": cached_meta.get("overlay_mode", "clean"),
                               "draw_connectors": cached_meta.get("draw_connectors", False),
                               "renderer_version": cached_meta.get("renderer_version"),
                               "renderer_imported": True}

    try:
        meta = renderer(
            img_p,
            ann_p,
            out_path,
            overlay_mode="clean",
            draw_connectors=draw_connectors,
        )
    except Exception as exc:
        return "", {"error": str(exc), "renderer_imported": True}

    meta.setdefault("renderer_imported", True)
    if out_path.exists():
        return str(out_path), meta
    return "", meta


def _safe_read_json(path: Path) -> dict | list:
    data = load_json(str(path))
    return data if data is not None else {}


def _strict_mode() -> bool:
    return os.environ.get("INFRAGRAPH_STRICT_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


def _qwen_configured() -> bool:
    return bool(_QWEN_BASE_URL and _QWEN_MODEL)


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
        _demo_asset_path("v3_local_rca") / "local_rca_result.json",
        _demo_asset_path("gnn_rca") / f"{selected}_gnn_rca_result.json",
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
    rfdetr_resolution = _cached_rfdetr_python_resolution()
    rfdetr_python = str(rfdetr_resolution.get("python_executable") or rfdetr_resolution.get("resolved_detector_python") or "")
    rfdetr_runtime = rfdetr_resolution.get("runtime") or {"ok": bool(rfdetr_resolution.get("import_ok"))}
    rfdetr_http = _cached_rfdetr_http_health()
    rfdetr_ckpt = _find_rfdetr_ckpt(REPO_ROOT) if _RFDETR_BRIDGE_OK and _find_rfdetr_ckpt else None
    return [
        {"label": "V3 hero scenario exists",          "ok": V3_HERO_SCENARIO.exists(),                                       "optional": False},
        {"label": "Hero diagrams (5 PNGs)",            "ok": diags_ok,                                                        "optional": False},
        {"label": "V3 annotations",                    "ok": ann_ok,                                                          "optional": False},
        {"label": "V3 local graphs",                   "ok": lg_ok,                                                           "optional": False},
        {"label": "V3 enterprise graph",               "ok": (V3_HERO_SCENARIO / "enterprise_graph.json").exists(),           "optional": False},
        {"label": "V3 alerts",                         "ok": (V3_HERO_SCENARIO / "alerts.json").exists(),                     "optional": False},
        {"label": "RF-DETR COCO export",               "ok": (V3_DATASET_ROOT / "rfdetr" / "annotations" / "instances_train.json").exists(), "optional": True},
        {"label": "YOLO export",                       "ok": (V3_DATASET_ROOT / "yolo" / "dataset.yaml").exists(),            "optional": True},
        {"label": "RF-DETR checkpoint found",           "ok": bool(rfdetr_ckpt and Path(rfdetr_ckpt).exists()),                "optional": not _strict_mode()},
        {"label": f"RF-DETR HTTP service: {rfdetr_http.get('service_url') or 'not configured'}", "ok": bool(rfdetr_http.get("ok")), "optional": True},
        {"label": f"RF-DETR requested Python: {rfdetr_resolution.get('requested_detector_python') or 'python'}", "ok": True, "optional": not _strict_mode()},
        {"label": f"RF-DETR resolved Python: {rfdetr_python or 'not resolved'}", "ok": bool(rfdetr_python),                    "optional": not _strict_mode()},
        {"label": "RF-DETR import in external runtime", "ok": bool(rfdetr_runtime.get("ok")),                                "optional": not _strict_mode()},
        {"label": "Live onboarding bridge available",   "ok": _RFDETR_BRIDGE_OK,                                              "optional": not _strict_mode()},
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
            "`python scripts/generate_infragraph_v3_dataset.py --num-scenarios 10 "
            "--out ./datasets/infragraph_v3 --seed 2026 --clean`"
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
    st.session_state.local_incident             = {}
    st.session_state.enterprise_rca_result      = {}
    st.session_state.enterprise_incident        = {}
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
_DIAG_INCIDENT_TEMPLATES: dict[str, dict] = {
    "branch_topology": {
        "titles":           ["Branch application access degradation", "Branch WAN path instability"],
        "severity":         "High",
        "suspected_domain": "WAN / Branch Edge",
        "alert_summary":    "Branch endpoint reports application access failures and elevated packet loss.",
        "symptoms": [
            "Endpoint reports application access failures and packet loss",
            "Access switch sees downstream impact from multiple workstations",
            "Firewall / router is on the access path for all affected flows",
            "WAN handoff is the common upstream dependency",
        ],
        "recommended_actions": [
            "Check WAN / MPLS handoff status and verify circuit health",
            "Validate router uplink interface metrics (packet loss, latency, CRC errors)",
            "Review firewall tunnel and NAT policy logs for recent changes",
            "Notify network operations if upstream packet loss continues",
        ],
        "root_types":      ["cloud_or_wan", "router", "firewall", "switch"],
        "first_obs_types": ["endpoint", "workstation", "pc", "server"],
    },
    "app_db_topology": {
        "titles":           ["Application transaction failure", "Database connectivity degradation"],
        "severity":         "Critical",
        "suspected_domain": "Application / Database Tier",
        "alert_summary":    "Application layer reports transaction errors and elevated response latency.",
        "symptoms": [
            "Application layer reports transaction errors and elevated latency",
            "Database connections timing out or returning query errors",
            "Load balancer health checks show degraded backend pool",
            "Cache layer serving stale data or failing to respond",
        ],
        "recommended_actions": [
            "Check database connection pool utilization and query latency",
            "Verify load balancer backend health probe responses",
            "Review application error logs for connection exceptions",
            "Restart affected service instances if connection pool is exhausted",
        ],
        "root_types":      ["database", "load_balancer", "app_server", "switch"],
        "first_obs_types": ["app_server", "web_server", "client", "endpoint"],
    },
    "datacenter_topology": {
        "titles":           ["Data center service degradation", "East-west fabric disruption"],
        "severity":         "High",
        "suspected_domain": "Data Center Fabric",
        "alert_summary":    "Inter-service communication failures detected across the data center fabric.",
        "symptoms": [
            "Servers report inter-service communication failures",
            "Core switching fabric shows interface errors or packet drops",
            "Firewall policy may be blocking east-west traffic flows",
            "Load balancer backend pool reporting degraded health checks",
        ],
        "recommended_actions": [
            "Check core switch interface error counters and spanning tree state",
            "Review firewall policy change audit logs for recent modifications",
            "Verify load balancer health probe responses from backend servers",
            "Inspect physical link status on affected uplinks and trunk ports",
        ],
        "root_types":      ["core_switch", "firewall", "load_balancer", "router", "switch"],
        "first_obs_types": ["server", "database", "vm", "application"],
    },
    "shared_services_topology": {
        "titles":           ["Shared service dependency failure", "Infrastructure service outage"],
        "severity":         "High",
        "suspected_domain": "Shared Infrastructure",
        "alert_summary":    "Shared infrastructure service failure is cascading to dependent workloads.",
        "symptoms": [
            "DNS resolution failures affecting multiple dependent services",
            "Authentication / IAM service response times are elevated",
            "NTP synchronization failures causing log timestamp drift",
            "Monitoring heartbeats missing for downstream dependent services",
        ],
        "recommended_actions": [
            "Verify DNS server availability and zone file consistency",
            "Check IAM / identity service health and TLS certificate validity",
            "Validate NTP server reachability and stratum level",
            "Review monitoring system logs for cascading service failures",
        ],
        "root_types":      ["dns_server", "ntp_server", "iam_server", "monitoring", "switch"],
        "first_obs_types": ["endpoint", "server", "workstation", "client"],
    },
    "wan_topology": {
        "titles":           ["WAN backbone path degradation", "MPLS circuit instability"],
        "severity":         "High",
        "suspected_domain": "WAN Transport",
        "alert_summary":    "Branch sites reporting high latency and packet loss toward the datacenter.",
        "symptoms": [
            "Branch site reporting high latency or packet loss toward datacenter",
            "WAN edge router shows BGP session instability",
            "MPLS circuit health degraded on primary path",
            "Failover path not engaging or slow to converge",
        ],
        "recommended_actions": [
            "Check MPLS circuit utilization and carrier status page",
            "Verify BGP session state and routing table completeness",
            "Test failover path reachability from affected branch sites",
            "Contact carrier if physical layer errors are detected",
        ],
        "root_types":      ["cloud_or_wan", "router", "switch"],
        "first_obs_types": ["switch", "router", "endpoint", "server"],
    },
}
_DEFAULT_INCIDENT_TEMPLATE: dict = {
    "titles":           ["Network topology incident"],
    "severity":         "High",
    "suspected_domain": "Network",
    "alert_summary":    "Network topology incident detected.",
    "symptoms": [
        "Service connectivity degraded",
        "Upstream dependency showing errors",
        "Downstream nodes reporting failures",
    ],
    "recommended_actions": [
        "Investigate the root cause node connectivity",
        "Review recent configuration changes on the root node",
        "Check interface error counters and logs",
    ],
    "root_types":      ["router", "switch", "firewall"],
    "first_obs_types": ["endpoint", "server", "workstation"],
}


def _bfs_path(src: str, dst: str, adj: dict) -> list[str]:
    """BFS shortest path; returns [] if unreachable."""
    if src == dst:
        return [src]
    prev: dict[str, str | None] = {src: None}
    queue = [src]
    while queue:
        cur = queue.pop(0)
        for nxt in adj.get(cur, []):
            if nxt not in prev:
                prev[nxt] = cur
                queue.append(nxt)
                if nxt == dst:
                    queue.clear()
                    break
    if dst not in prev:
        return []
    path: list[str] = []
    cur = dst
    while cur is not None:
        path.append(cur)
        cur = prev[cur]  # type: ignore[assignment]
    return list(reversed(path))


def _local_rca_ranking_reason(
    n_id: str, root: str, first_obs: str,
    path_set: set, impacted_set: set,
    outdeg: dict, indeg: dict,
    diagram_id: str, nodes: dict,
) -> str:
    ntype = nodes.get(n_id, {}).get("type", "").lower()
    if n_id == root:
        if ntype in ("cloud_or_wan", "wan"):
            return "WAN handoff is common upstream dependency for all impacted nodes"
        if ntype == "router":
            return "Router aggregates all downstream traffic — on the path of every affected flow"
        if ntype == "firewall":
            return "Firewall is on all flows between impacted nodes and the upstream"
        if ntype in ("database", "db"):
            return "Database is the shared backend dependency for all application tiers"
        if ntype in ("load_balancer", "lb"):
            return "Load balancer distributes traffic; its failure impacts all backends"
        if ntype in ("dns_server", "dns"):
            return "DNS is a shared upstream dependency; its failure cascades to all consumers"
        return f"Common upstream dependency for impacted {diagram_id.replace('_', ' ')} nodes"
    if n_id == first_obs:
        return "First observed symptom node; downstream impact evidence points here"
    if n_id in path_set:
        return f"On the access path between {first_obs} and {root}"
    if n_id in impacted_set:
        if outdeg.get(n_id, 0) == 0:
            return "Downstream leaf node — likely impacted rather than causal"
        return "Downstream node reachable from root; impacted rather than causal"
    if outdeg.get(n_id, 0) > 2:
        return "Multiple downstream paths converge through this node"
    return "Peripheral node — no direct evidence of involvement in this incident"


# ══════════════════════════════════════════════════════════════════════════════
# INCIDENT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
# AI REMEDIATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _check_vllm_available() -> bool:
    """Return True if the vLLM server answers at _QWEN_BASE_URL/models."""
    if not _AI_REM_OK or _check_vllm_available_fn is None:
        return False
    return _check_vllm_available_fn(_QWEN_BASE_URL, timeout=4)


def _build_remediation_context(
    rca: dict,
    ent_incident: dict,
    enterprise_graph: dict,
    alerts_data: dict,
    diagram_id: str,
    gnn_result: "dict | None",
) -> dict:
    """Assemble an enterprise make_remediation_input() context from session state."""
    if not _AI_REM_OK or _make_remediation_input is None:
        return {}

    nodes = enterprise_graph.get("nodes", [])
    edges = enterprise_graph.get("edges", [])

    device_ctx = [
        {
            "node_id":    n.get("id", n.get("node_id", "")),
            "device_type": n.get("type", n.get("class_name", "")),
            "ip_address": n.get("ip_address", ""),
            "diagram_id": n.get("diagram_id", n.get("source_diagram", "")),
        }
        for n in nodes[:20]
    ]
    connector_ctx = [
        {
            "source":     e.get("source", ""),
            "target":     e.get("target", ""),
            "type":       e.get("label", e.get("relationship", "")),
            "diagram_id": e.get("diagram_id", ""),
        }
        for e in edges[:15]
    ]
    n_cross = len(enterprise_graph.get("cross_diagram_edges", []))
    clusters = enterprise_graph.get("diagram_clusters", {})
    n_domains = len(clusters) if isinstance(clusters, dict) else len(clusters)
    graph_summary = (
        f"{len(nodes)} nodes, {len(edges)} intra-diagram edges, "
        f"{n_cross} cross-diagram edges across {n_domains} topology domains."
    )

    root_cause   = rca.get("root_cause", "")            if rca else alerts_data.get("root_cause", "")
    rc_diagram   = rca.get("root_cause_diagram", "")    if rca else alerts_data.get("root_cause_diagram", "")
    imp_nodes    = rca.get("impacted_nodes", [])         if rca else ent_incident.get("impacted_nodes", [])
    imp_diagrams = rca.get("impacted_diagrams", [])      if rca else ent_incident.get("impacted_diagrams", [])
    impact_path  = rca.get("impact_path", [])            if rca else ent_incident.get("impact_path", [])
    ranking      = rca.get("ranking", [])                if rca else ent_incident.get("candidate_ranking", [])
    rca_source   = rca.get("mode", "Scenario-grounded RCA simulation") if rca else ""
    incident_id  = ent_incident.get("incident_id", "")
    scenario_id  = ent_incident.get("scenario_id", alerts_data.get("scenario_id", ""))

    # Load causal_evidence from the main enterprise GNN RCA JSON (plain {scenario_id}.json),
    # which contains CE-* evidence IDs, correlation_reasons, and cluster metadata.
    # The *_enterprise_gnn_rca_result.json loaded by _load_gnn_rca_result() has raw scores
    # but not causal_evidence — so we read the main JSON separately.
    _main_rca_json: dict = {}
    if scenario_id:
        _main_rca_p = _demo_asset_path("enterprise_gnn_rca") / f"{scenario_id}.json"
        if _main_rca_p.exists():
            try:
                _main_rca_json = json.loads(_main_rca_p.read_text(encoding="utf-8"))
            except Exception:
                pass
    causal_evidence     = (_main_rca_json.get("causal_evidence", [])
                           or (gnn_result or {}).get("causal_evidence", []))
    correlation_reasons = (_main_rca_json.get("correlation_reasons", [])
                           or (gnn_result or {}).get("correlation_reasons", []))
    cluster_id          = (_main_rca_json.get("cluster_id", "")
                           or (gnn_result or {}).get("cluster_id", ""))
    cluster_score       = (_main_rca_json.get("cluster_score")
                           or (gnn_result or {}).get("cluster_score"))

    return _make_remediation_input(
        incident_id=incident_id,
        scope="enterprise",
        selected_diagram_id=diagram_id,
        scenario_id=scenario_id,
        alert_timeline=ent_incident.get("alert_timeline", []),
        graph_memory_summary=graph_summary,
        root_cause=root_cause,
        root_cause_diagram=rc_diagram,
        impacted_nodes=list(imp_nodes),
        impacted_diagrams=list(imp_diagrams),
        impact_path=list(impact_path),
        candidate_ranking=list(ranking),
        gnn_result_available=bool(gnn_result),
        rca_source=rca_source,
        device_context=device_ctx,
        connector_context=connector_ctx,
        cluster_id=cluster_id,
        cluster_score=cluster_score,
        correlation_reasons=correlation_reasons,
        causal_evidence=causal_evidence,
    )


def _build_local_remediation_context(
    result: dict,
    incident: dict,
    local_graph: dict,
    diagram_id: str,
) -> dict:
    """Assemble a local make_remediation_input() context from Tab 2 session state."""
    if not _AI_REM_OK or _make_remediation_input is None:
        return {}

    nodes = local_graph.get("nodes", [])
    edges = local_graph.get("edges", [])

    device_ctx = [
        {
            "node_id":     n.get("id", n.get("node_id", "")),
            "device_type": n.get("type", n.get("class_name", "")),
            "ip_address":  n.get("ip_address", ""),
            "diagram_id":  diagram_id,
        }
        for n in nodes[:15]
    ]
    connector_ctx = [
        {
            "source":     e.get("source", ""),
            "target":     e.get("target", ""),
            "type":       e.get("label", e.get("relationship", "")),
            "diagram_id": diagram_id,
        }
        for e in edges[:12]
    ]
    graph_summary = f"{len(nodes)} nodes, {len(edges)} edges in {diagram_id}."

    root_cause    = result.get("root_cause", "")
    first_obs     = result.get("first_observed", incident.get("first_observed_node", ""))
    imp_nodes     = result.get("impacted_nodes", [])
    impact_path   = result.get("impact_path", [])
    ranking       = result.get("ranking", [])
    rca_source    = result.get("mode", result.get("rca_source", "Topology BFS RCA"))
    incident_id   = incident.get("incident_id", "")
    scenario_id   = incident.get("scenario_id", "")

    return _make_remediation_input(
        incident_id=incident_id,
        scope="local",
        selected_diagram_id=diagram_id,
        diagram_type=diagram_id,
        scenario_id=scenario_id,
        alert_timeline=incident.get("alert_timeline", []),
        graph_memory_summary=graph_summary,
        root_cause=root_cause,
        root_cause_diagram=diagram_id,
        first_observed_node=first_obs,
        impacted_nodes=list(imp_nodes),
        impacted_diagrams=[diagram_id],
        impact_path=list(impact_path),
        candidate_ranking=list(ranking),
        gnn_result_available=False,
        rca_source=rca_source,
        device_context=device_ctx,
        connector_context=connector_ctx,
    )


def _render_remediation_plan(plan: dict) -> None:
    """Render a remediation plan dict in the Streamlit UI."""
    resp = plan.get("response", {})
    if not resp:
        st.warning("Remediation plan is empty.")
        return

    def _esc(value: object) -> str:
        return html.escape(str(value))

    def _items(value: "str | list | None") -> list[str]:
        if isinstance(value, list):
            return [str(v) for v in value if str(v)]
        if value:
            return [str(value)]
        return []

    def _source_badge(result: dict) -> None:
        src = result.get("source", "")
        ok = bool(result.get("ok", False))
        if src == "qwen_vllm" and ok:
            text, color = "Live Qwen/vLLM", "#10b981"
        elif src in ("template", "template_fallback"):
            text, color = "Template/fallback response — deterministic, not model-generated", "#f59e0b"
        elif not ok:
            text, color = "Strict mode blocked", "#ef4444"
        else:
            text, color = str(src or "Unknown source"), "#64748b"
        st.markdown(
            f'<span style="display:inline-block;margin:4px 0 10px 0;padding:4px 10px;'
            f'border-radius:999px;border:1px solid {color};color:{color};'
            f'background:rgba(15,23,42,0.45);font-size:0.72rem;font-weight:800">{_esc(text)}</span>',
            unsafe_allow_html=True,
        )

    def _metric_cards() -> None:
        cards = [
            ("Risk Level", resp.get("risk_level", "unknown")),
            ("Blast Radius", resp.get("blast_radius", "unknown")),
            ("Automation Eligibility", resp.get("automation_eligibility", "unknown")),
            ("Evidence IDs", ", ".join(_items(resp.get("evidence_ids_used"))) or "none"),
        ]
        st.markdown(
            '<div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin:8px 0 12px">',
            unsafe_allow_html=True,
        )
        for label, value in cards:
            st.markdown(
                f'<div style="background:rgba(15,23,42,0.55);border:1px solid rgba(148,163,184,0.22);'
                f'border-radius:8px;padding:10px 12px;min-height:68px">'
                f'<div style="font-size:0.6rem;text-transform:uppercase;color:#94a3b8;margin-bottom:5px">{_esc(label)}</div>'
                f'<div style="font-size:0.84rem;color:#f8fafc;font-weight:800;word-break:break-word">{_esc(value)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    def _section_card(label: str, value: "str | list | None", *, accent: str = "#10b981") -> None:
        items = _items(value)
        if not items:
            return
        body = "".join(
            f'<li style="margin-bottom:5px;line-height:1.45">{_esc(item)}</li>'
            for item in items
        )
        st.markdown(
            f'<div style="background:rgba(15,23,42,0.45);border:1px solid rgba(148,163,184,0.18);'
            f'border-left:3px solid {accent};border-radius:8px;padding:12px 14px;margin:8px 0">'
            f'<div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.08em;'
            f'color:#94a3b8;margin-bottom:8px">{_esc(label)}</div>'
            f'<ol style="margin:0 0 0 18px;padding:0;color:#e2e8f0;font-size:0.84rem">{body}</ol>'
            f'</div>',
            unsafe_allow_html=True,
        )

    def _text_card(label: str, value: object, *, accent: str = "#38bdf8") -> None:
        if not value:
            return
        st.markdown(
            f'<div style="background:rgba(15,23,42,0.45);border:1px solid rgba(148,163,184,0.18);'
            f'border-left:3px solid {accent};border-radius:8px;padding:12px 14px;margin:8px 0">'
            f'<div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.08em;'
            f'color:#94a3b8;margin-bottom:8px">{_esc(label)}</div>'
            f'<div style="color:#e2e8f0;font-size:0.86rem;line-height:1.5">{_esc(value)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    def _runbook_chain_section() -> None:
        _chain = resp.get("runbook_chain", []) or []
        if not _chain:
            return
        with st.expander(
            f"Retrieved Approved Runbooks ({len(_chain)})",
            expanded=True,
        ):
            for _rb in _chain:
                _rb_id    = _rb.get("runbook_id", "?")
                _rb_title = _rb.get("title", "—")
                _rb_dom   = _rb.get("domain", "")
                _rb_mode  = _rb.get("execution_mode", "manual")
                _rb_approv = _rb.get("approval_required", True)
                _rb_auto  = _rb.get("automation_eligible", False)
                _rb_dry   = _rb.get("dry_run_supported", False)
                _rb_tool  = _rb.get("tool_name", "") or _rb.get("connector", "")
                _rb_eids  = ", ".join((_rb.get("evidence_ids") or [])[:6])
                _rb_secs  = ", ".join((_rb.get("sections_retrieved") or [])[:4])
                _approv_color = "#ef4444" if _rb_approv else "#10b981"
                _auto_color   = "#10b981" if _rb_auto else "#94a3b8"
                _dry_color    = "#38bdf8" if _rb_dry else "#94a3b8"
                st.markdown(
                    f'<div style="background:rgba(15,23,42,0.5);border:1px solid rgba(148,163,184,0.25);'
                    f'border-left:3px solid #a78bfa;border-radius:8px;padding:10px 14px;margin:6px 0">'
                    f'<div style="font-size:0.8rem;color:#a78bfa;font-weight:800;margin-bottom:4px">'
                    f'[{_esc(_rb_id)}] {_esc(_rb_title)}</div>'
                    f'<div style="font-size:0.7rem;color:#94a3b8;display:flex;gap:8px;flex-wrap:wrap;margin-bottom:4px">'
                    f'<span>domain: <b style="color:#e2e8f0">{_esc(_rb_dom)}</b></span>'
                    f'<span>mode: <b style="color:#e2e8f0">{_esc(_rb_mode)}</b></span>'
                    f'<span style="color:{_approv_color}">{"approval_required" if _rb_approv else "no_approval_required"}</span>'
                    f'<span style="color:{_auto_color}">{"automation_eligible" if _rb_auto else "manual_only"}</span>'
                    f'<span style="color:{_dry_color}">{"dry_run_ok" if _rb_dry else "no_dry_run"}</span>'
                    + (f'<span>tool: <b style="color:#e2e8f0">{_esc(_rb_tool)}</b></span>' if _rb_tool else "")
                    + f'</div>'
                    f'<div style="font-size:0.68rem;color:#64748b">'
                    f'evidence_ids: <span style="color:#c4b5fd">{_esc(_rb_eids) or "—"}</span>'
                    + (f' &nbsp;|&nbsp; sections: {_esc(_rb_secs)}' if _rb_secs else "")
                    + f'</div></div>',
                    unsafe_allow_html=True,
                )

        # Automation Plan
        _aplan = resp.get("automation_plan") or {}
        if _aplan:
            _ap_can     = _aplan.get("can_execute", False)
            _ap_approv  = _aplan.get("requires_approval", True)
            _ap_conn    = _aplan.get("connector", "")
            _ap_dry     = _aplan.get("dry_run_supported", False)
            _ap_note    = _aplan.get("execution_note", "")
            _ap_color   = "#10b981" if _ap_can else "#f59e0b"
            _ap_label   = "Automation Ready" if _ap_can else "Manual Execution Required"
            with st.expander("Automation Plan", expanded=False):
                st.markdown(
                    f'<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:8px">'
                    f'<span style="display:inline-block;padding:3px 10px;border-radius:999px;'
                    f'border:1px solid {_ap_color};color:{_ap_color};font-size:0.72rem;font-weight:800">'
                    f'{_esc(_ap_label)}</span>'
                    + (f'<span style="font-size:0.72rem;color:#ef4444">approval_required</span>' if _ap_approv else "")
                    + (f'<span style="font-size:0.72rem;color:#38bdf8">dry_run_supported</span>' if _ap_dry else "")
                    + (f'<span style="font-size:0.72rem;color:#94a3b8">connector: {_esc(_ap_conn)}</span>' if _ap_conn else "")
                    + f'</div>'
                    + (f'<div style="font-size:0.78rem;color:#cbd5e1">{_esc(_ap_note)}</div>' if _ap_note else ""),
                    unsafe_allow_html=True,
                )

    _source_badge(plan)
    _metric_cards()
    _runbook_chain_section()

    exec_sum = resp.get("executive_summary", "")
    if exec_sum:
        _text_card("Executive Summary", exec_sum, accent="#10b981")

    _text_card("Probable Root Cause", resp.get("probable_root_cause", ""))
    _section_card("Graph Evidence", resp.get("evidence_from_graph", []), accent="#22c55e")
    _section_card("Pre-checks", resp.get("pre_checks", []) or resp.get("triage_steps", []), accent="#38bdf8")
    _section_card("Validation Steps", resp.get("validation_steps", []), accent="#0ea5e9")
    _section_card("Remediation Steps", resp.get("remediation_steps", []), accent="#f59e0b")
    _section_card("Post-checks", resp.get("post_checks", []) or resp.get("validation_steps", []), accent="#2dd4bf")
    _section_card("Do Not Execute If", resp.get("do_not_execute_if", []), accent="#ef4444")
    _section_card("Rollback / Safety", resp.get("rollback_or_safety_notes", []), accent="#f97316")
    _text_card("Escalation Recommendation", resp.get("escalation_recommendation", ""), accent="#a78bfa")

    itsm = resp.get("itsm_ticket_summary")
    if itsm:
        if isinstance(itsm, dict):
            itsm_lines = []
            for key, label in [
                ("short_description", "Short description"),
                ("description", "Description"),
                ("affected_ci", "Affected CI"),
                ("priority", "Priority"),
                ("assignment_group", "Assignment group"),
            ]:
                if itsm.get(key):
                    itsm_lines.append(f"{label}: {itsm[key]}")
            _section_card("ITSM Ticket", itsm_lines, accent="#64748b")
        else:
            _text_card("ITSM Ticket", itsm, accent="#64748b")

    _text_card("Audit Summary", resp.get("audit_summary", ""), accent="#94a3b8")
    _text_card("Confidence Notes", resp.get("confidence_notes", ""), accent="#94a3b8")


def _render_ai_pipeline_trace(
    *,
    selected_diagram: str = "",
    selected_scenario: str = "",
    topology_rca_completed: bool = False,
    enterprise_gnn_available: bool = False,
    rca_source: str = "",
    root_cause: str = "",
    impacted_diagrams: list | None = None,
    vector_evidence_count: int = 0,
    response_source: str = "",
) -> None:
    impacted_diagrams = impacted_diagrams or []
    _sop_adapter_env  = _os.environ.get("INFRAGRAPH_LORA_ADAPTER_PATH") or ""
    _sop_adapter_path = _sop_adapter_env or str(REPO_ROOT / "model_artifacts" / "qwen_lora" / "infragraph_sop_grounded")
    _sop_adapter_ok   = Path(_sop_adapter_path).exists()
    _gnn_model_path   = str(_demo_asset_path("enterprise_gnn_rca") / "enterprise_gnn_model.pt")
    _gnn_metrics_path = str(_demo_asset_path("enterprise_gnn_rca") / "enterprise_gnn_metrics.json")
    _qwen_alias       = (_os.environ.get("INFRAGRAPH_QWEN_MODEL")
                         or _os.environ.get("QWEN_MODEL")
                         or _QWEN_MODEL)
    _qwen_base_url_v  = (_os.environ.get("INFRAGRAPH_QWEN_BASE_URL")
                         or _os.environ.get("QWEN_BASE_URL")
                         or _QWEN_BASE_URL)
    rows = {
        "Selected diagram": selected_diagram or "—",
        "Selected scenario": selected_scenario or "—",
        "Topology RCA completed": "yes" if topology_rca_completed else "no",
        "Enterprise GNN result available": "yes" if enterprise_gnn_available else "no",
        "RCA source": rca_source or "—",
        "Root cause": root_cause or "—",
        "Impacted diagram count": str(len(set(str(d) for d in impacted_diagrams if d))),
        "Vector evidence retrieved": str(vector_evidence_count),
        "GNN model path": _gnn_model_path,
        "GNN metrics path": _gnn_metrics_path,
        "GNN model available": "yes" if Path(_gnn_model_path).exists() else "no",
        "Qwen model alias": _qwen_alias,
        "Qwen base URL": _qwen_base_url_v,
        "Qwen configured": "yes" if _qwen_configured() else "no",
        "Response source": response_source or "—",
        "SOP-grounded adapter path": _sop_adapter_path,
        "Adapter available": "yes" if _sop_adapter_ok else "no — run scripts/train_qwen_sop_lora.py",
    }
    with st.expander("AI Pipeline Trace", expanded=False):
        st.table(pd.DataFrame([{"Signal": k, "Value": v} for k, v in rows.items()]))


def _render_qwen_runtime_proof(plan_result: dict) -> None:
    """Expandable Qwen Runtime Proof panel with source badge."""
    import datetime as _dt

    src = plan_result.get("source", "")
    ok  = bool(plan_result.get("ok", False))

    # ── Source badge (always visible, outside expander) ───────────────────────
    if src == "qwen_vllm" and ok:
        badge_text  = "Live Qwen/vLLM response"
        badge_color = "#10b981"
    else:
        badge_text  = "Template/fallback response"
        badge_color = "#f59e0b"

    st.markdown(
        f'<span style="display:inline-block;margin:8px 0 4px 0;padding:4px 12px;'
        f'border-radius:999px;border:1px solid {badge_color};color:{badge_color};'
        f'background:rgba(15,23,42,0.5);font-size:0.72rem;font-weight:800">'
        f'{html.escape(badge_text)}</span>',
        unsafe_allow_html=True,
    )

    # ── Expandable proof panel ────────────────────────────────────────────────
    with st.expander("Qwen Runtime Proof", expanded=False):
        base_url    = (_os.environ.get("INFRAGRAPH_QWEN_BASE_URL")
                       or _os.environ.get("QWEN_BASE_URL")
                       or _QWEN_BASE_URL)
        model_id    = (_os.environ.get("INFRAGRAPH_QWEN_MODEL")
                       or _os.environ.get("QWEN_MODEL")
                       or _QWEN_MODEL)
        timeout_s   = _os.environ.get("INFRAGRAPH_QWEN_TIMEOUT") or str(_QWEN_TIMEOUT)
        max_tokens  = _os.environ.get("INFRAGRAPH_QWEN_MAX_TOKENS") or "—"
        adapter_env = _os.environ.get("INFRAGRAPH_LORA_ADAPTER_PATH") or ""
        adapter_eff = adapter_env or "model_artifacts/qwen_lora/infragraph_sop_grounded"
        vllm_up     = _check_vllm_available()
        raw_output  = str(plan_result.get("raw") or "")
        raw_len     = len(raw_output)
        gen_ts      = plan_result.get("_generated_at") or "—"
        qwen_err    = plan_result.get("qwen_error") or ""

        rows: dict[str, str] = {
            "INFRAGRAPH_QWEN_BASE_URL":     base_url,
            "INFRAGRAPH_QWEN_MODEL":        model_id,
            "INFRAGRAPH_QWEN_TIMEOUT":      f"{timeout_s}s",
            "INFRAGRAPH_QWEN_MAX_TOKENS":   max_tokens,
            "INFRAGRAPH_LORA_ADAPTER_PATH": adapter_eff,
            "vLLM /models check":           "PASS" if vllm_up else "FAIL",
            "Response source":              src or "—",
            "Response ok":                  str(ok),
            "Raw output length (chars)":    str(raw_len),
            "Generated at":                 gen_ts,
        }
        if qwen_err:
            rows["Qwen error (caused fallback)"] = qwen_err[:300]

        st.table(pd.DataFrame([{"Field": k, "Value": v} for k, v in rows.items()]))

        if raw_len > 0:
            with st.expander("Raw model output (first 300 chars)", expanded=False):
                st.code(raw_output[:300], language="json")
        else:
            st.caption("No raw model output — template mode or plan not yet generated.")


def _render_gnn_rca_model_evidence(gnn_result: dict, metrics: dict) -> None:
    """Expandable Trained RCA Model Evidence panel for Enterprise GNN RCA."""
    if not gnn_result and not metrics:
        return

    inference_source = gnn_result.get("inference_source", "")
    inference_mode   = (
        "precomputed_gnn_inference_artifact"
        if inference_source in ("trained_enterprise_gnn", "precomputed")
        or gnn_result.get("rca_source") == "Enterprise GNN RCA"
        else "live_gnn_inference"
    )
    backend      = metrics.get("backend") or gnn_result.get("backend") or "torch"
    model_type   = metrics.get("model_type") or gnn_result.get("model_type") or "Enterprise GCN RCA"
    architecture = metrics.get("architecture", "—")
    epochs       = metrics.get("epochs_trained", "—")
    best_epoch   = metrics.get("best_val_epoch", "—")
    feature_dim  = metrics.get("feature_dim", "—")
    device       = metrics.get("device_name") or metrics.get("selected_device") or "—"
    train_m      = metrics.get("train_metrics", {})
    val_m        = metrics.get("val_metrics", {})
    test_m       = metrics.get("test_metrics", {})

    _gnn_model_p   = str(_demo_asset_path("enterprise_gnn_rca") / "enterprise_gnn_model.pt")
    _gnn_metrics_p = str(_demo_asset_path("enterprise_gnn_rca") / "enterprise_gnn_metrics.json")

    with st.expander("Trained RCA Model Evidence", expanded=False):
        # Inference mode provenance header
        _inf_color = "#38bdf8" if inference_mode == "precomputed_gnn_inference_artifact" else "#10b981"
        st.markdown(
            f'<div style="background:rgba(15,23,42,0.6);border:1px solid rgba(56,189,248,0.25);'
            f'border-left:3px solid {_inf_color};border-radius:8px;padding:10px 14px;margin-bottom:12px">'
            f'<div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.08em;'
            f'color:#64748b;margin-bottom:3px">Inference Mode</div>'
            f'<div style="font-family:monospace;font-size:0.82rem;font-weight:700;color:{_inf_color}">'
            f'{html.escape(inference_mode)}</div>'
            f'<div style="font-size:0.72rem;color:#94a3b8;margin-top:4px">'
            f'Generated by trained GNN RCA pipeline ({html.escape(model_type)}); '
            f'preloaded for stability.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Model metadata grid
        _meta_items = [
            ("Backend",      str(backend)),
            ("Architecture", str(architecture)),
            ("Epochs",       str(epochs)),
            ("Best Val Ep.", str(best_epoch)),
            ("Feature Dim",  str(feature_dim)),
            ("Device",       str(device)),
        ]
        st.markdown(
            '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:12px">'
            + "".join(
                f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);'
                f'border-radius:8px;padding:8px 10px">'
                f'<div style="font-size:0.58rem;color:#64748b;text-transform:uppercase;margin-bottom:3px">'
                f'{html.escape(lbl)}</div>'
                f'<div style="font-size:0.78rem;font-weight:700;color:#f1f5f9;font-family:monospace">'
                f'{html.escape(val)}</div>'
                f'</div>'
                for lbl, val in _meta_items
            )
            + '</div>',
            unsafe_allow_html=True,
        )

        # Metrics table (no ground-truth fields)
        if train_m or val_m or test_m:
            st.markdown(
                '<div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em;'
                'color:#94a3b8;margin-bottom:6px">Model Metrics</div>',
                unsafe_allow_html=True,
            )
            metric_rows = []
            for split, m in [("train", train_m), ("val", val_m), ("test", test_m)]:
                if m:
                    metric_rows.append({
                        "Split":     split,
                        "Top-1":     f"{m.get('top1', 0):.1%}",
                        "Top-3":     f"{m.get('top3', 0):.1%}",
                        "MRR":       f"{m.get('mrr', 0):.4f}",
                        "Scenarios": m.get("scenario_count", "—"),
                    })
            if metric_rows:
                st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

        # Top candidates (filter ground-truth fields)
        candidates = gnn_result.get("top_candidates", [])
        if candidates:
            st.markdown(
                '<div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em;'
                'color:#94a3b8;margin:10px 0 6px">GNN Top Candidates (Softmax-Normalised Scores)</div>',
                unsafe_allow_html=True,
            )
            _cand_rows = []
            for c in candidates:
                _cand_rows.append({
                    "Rank":    c.get("rank", "—"),
                    "Node":    c.get("node_id") or c.get("node", "—"),
                    "Diagram": c.get("diagram_id") or c.get("diagram_type", "—"),
                    "Type":    c.get("node_type") or c.get("type", "—"),
                    "Score":   f"{float(c.get('score', 0)):.4f}",
                })
            st.dataframe(pd.DataFrame(_cand_rows), use_container_width=True, hide_index=True)

        # Artifact paths
        _model_exists   = Path(_gnn_model_p).exists()
        _metrics_exists = Path(_gnn_metrics_p).exists()
        _model_suffix   = "" if _model_exists   else " <em style=\"color:#ef4444\">(not found)</em>"
        _metrics_suffix = "" if _metrics_exists else " <em style=\"color:#ef4444\">(not found)</em>"
        st.markdown(
            f'<div style="font-size:0.68rem;color:#475569;margin-top:10px;'
            f'border-top:1px solid rgba(255,255,255,0.06);padding-top:8px">'
            f'<strong style="color:#64748b">Model:</strong> '
            f'<code style="font-size:0.65rem">{html.escape(_gnn_model_p)}</code>'
            f'{_model_suffix}<br>'
            f'<strong style="color:#64748b">Metrics:</strong> '
            f'<code style="font-size:0.65rem">{html.escape(_gnn_metrics_p)}</code>'
            f'{_metrics_suffix}'
            f'</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════

def _render_alert_timeline(timeline: list[dict], show_diagram_col: bool = False) -> None:
    """Render an alert timeline as styled event cards."""
    if not timeline:
        st.caption("No alert events in timeline.")
        return

    prev_diag = None
    for ev in timeline:
        sev      = ev.get("severity", "major")
        sev_css  = "critical" if sev == "critical" else "major"
        is_first = ev.get("is_first_observed", False)
        is_root  = ev.get("is_root_signal", False)
        diag     = ev.get("diagram_id", "")
        node     = ev.get("node", "")
        dtype    = ev.get("device_type", "")
        msg      = ev.get("message", "")
        atype    = ev.get("alert_type", "")
        tlbl     = ev.get("time_label", "")

        # Cross-diagram section header
        if show_diagram_col and diag and diag != prev_diag:
            prev_diag = diag
            st.markdown(
                f'<div class="timeline-diag-hdr">'
                f'Diagram <span class="diag-label">{diag}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        badges = ""
        if is_first:
            badges += '<span class="badge badge-warn" style="margin-left:6px">FIRST OBSERVED</span>'
        if is_root:
            badges += '<span class="badge badge-critical" style="margin-left:6px">ROOT SIGNAL</span>'

        st.markdown(
            f'<div class="alert-item {sev_css}">'
            f'<div class="alert-dot {sev_css}"></div>'
            f'<div style="flex:1;min-width:0">'
            f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">'
            f'<span class="alert-node">{node}</span>'
            f'<span style="font-size:0.68rem;color:#64748b">{dtype}</span>'
            f'{badges}'
            f'</div>'
            f'<div class="alert-msg">{msg}</div>'
            f'<div style="font-size:0.69rem;color:#475569;margin-top:2px">'
            f'Type: {atype}'
            f'</div>'
            f'</div>'
            f'<span class="alert-time">{tlbl}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_traversal_steps(impact_path: list[str], result: dict,
                              slider_key: str = "traversal_slider") -> None:
    """Render a step-by-step traversal control for an impact path."""
    if len(impact_path) < 2:
        return

    st.markdown(
        '<div class="section-label" style="margin-top:14px">'
        'Step-by-step Traversal</div>',
        unsafe_allow_html=True,
    )

    root      = result.get("root_cause", "")
    first_obs = (result.get("first_observed_node") or result.get("first_observed", ""))
    n_steps   = len(impact_path)

    step_idx = st.slider(
        "Traversal step",
        min_value=1, max_value=n_steps, value=1, step=1,
        key=slider_key,
        help="Walk through the impact path one node at a time",
    )
    current = impact_path[step_idx - 1]

    if current == root:
        step_label = "Root Cause Confirmed"
        step_color = "#10b981"
        step_desc  = (
            f"**{current}** identified as root cause — "
            f"common upstream dependency for all impacted nodes."
        )
    elif current == first_obs:
        step_label = "Initial Alert"
        step_color = "#ef4444"
        step_desc  = (
            f"**{current}** reports the first symptom — "
            f"first observed signal that triggers the incident."
        )
    else:
        step_label = f"Propagation Step {step_idx}"
        step_color = "#f59e0b"
        step_desc  = (
            f"Impact propagates through **{current}** — "
            f"upstream dependency on the path to root cause."
        )

    st.markdown(
        f'<div class="traversal-card">'
        f'<div class="traversal-step-lbl" style="color:{step_color}">{step_label}</div>'
        f'<div class="traversal-node">{current}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(step_desc)

    # Path chips with visited/current/unvisited states
    chips = []
    for i, n in enumerate(impact_path):
        if i < step_idx - 1:
            cls = "visited"
        elif i == step_idx - 1:
            cls = "current"
        else:
            base = "root" if n == root else ("alerting" if n == first_obs else "")
            cls = base
        chips.append(f'<span class="node-chip {cls}">{n}</span>')
    st.markdown(
        '<div style="padding:6px 0;line-height:2.4">'
        + ' <span style="color:#334155;font-size:0.75rem">→</span> '.join(chips)
        + '</div>',
        unsafe_allow_html=True,
    )


# ── Propagation visual context helpers ───────────────────────────────────────

_STEP_ROLE_TITLES: dict[str, str] = {
    "first_observed": "First observed symptom",
    "alert":          "Alert propagation",
    "bridge":         "Cross-diagram bridge",
    "root_candidate": "Root cause ranked",
    "impacted":       "Downstream enterprise impact",
}


def _build_node_aliases(
    enterprise_graph: dict,
) -> "tuple[dict[str, str], dict[str, dict]]":
    """Build node alias map from enterprise_graph.

    Returns (node_aliases, id_to_node) where node_aliases maps any ID variant
    (canonical_id, node_id, display_label, global_node_id, last colon-segment,
    short segment without prefix) to the canonical graph node ID.
    """
    id_to_node: dict[str, dict] = {}
    aliases:    dict[str, str]  = {}

    for n in enterprise_graph.get("nodes", []):
        canonical = str(n.get("id") or n.get("node_id") or "")
        if not canonical:
            continue
        id_to_node[canonical] = n
        for key in ("id", "node_id", "canonical_id", "display_label", "global_node_id"):
            val = str(n.get(key) or "")
            if val and val not in aliases:
                aliases[val] = canonical
        for sep in (":", "/", "."):
            if sep in canonical:
                parts = canonical.split(sep)
                last = parts[-1]
                if last and last not in aliases:
                    aliases[last] = canonical
                if len(parts) > 2:
                    semi = sep.join(parts[-2:])
                    if semi not in aliases:
                        aliases[semi] = canonical

    return aliases, id_to_node


def _build_propagation_visual_context(
    enterprise_graph: dict,
    ent_incident: "dict | None",
    rca: "dict | None",
) -> dict:
    """Build unified propagation visual context with normalized node IDs.

    Derives propagation steps from alert_timeline when propagation_steps is
    empty, assigns roles (first_observed/alert/bridge/root_candidate/impacted),
    and resolves all node IDs across namespace variants.

    Reads ``st.session_state["ent_prop_slider"]`` to determine the current
    step index (set by the slider widget on the previous Streamlit frame).
    """
    node_aliases, id_to_node = _build_node_aliases(enterprise_graph)

    def _resolve(raw_id: str) -> str:
        return node_aliases.get(raw_id, raw_id) if raw_id else raw_id

    def _resolve_list(ids: "list") -> "list[str]":
        return [_resolve(str(i)) for i in ids if i]

    inc   = ent_incident or {}
    rca_d = rca or {}

    alert_timeline: list[dict] = list(inc.get("alert_timeline", []) or [])
    prop_steps:     list[dict] = list(inc.get("propagation_steps", []) or [])

    root_cause_raw  = str(rca_d.get("root_cause") or inc.get("root_cause", "") or "")
    root_cause_norm = _resolve(root_cause_raw)

    if not prop_steps and alert_timeline:
        _sorted_tl = sorted(
            alert_timeline,
            key=lambda x: str(x.get("timestamp", x.get("time", ""))),
        )
        impact_path_norm = set(_resolve_list(rca_d.get("impact_path", [])))
        bridge_nodes = {
            _resolve(str(e.get("node", "")))
            for e in alert_timeline
            if e.get("correlation_role") == "bridge"
        }
        path_so_far: list[str] = []
        for i, ev in enumerate(_sorted_tl):
            raw_n  = str(ev.get("node", "") or "")
            norm_n = _resolve(raw_n)
            if not norm_n:
                continue
            if i == 0 or ev.get("is_first_observed"):
                role = "first_observed"
            elif norm_n in bridge_nodes or ev.get("correlation_role") == "bridge":
                role = "bridge"
            elif norm_n == root_cause_norm:
                role = "root_candidate"
            elif norm_n in impact_path_norm:
                role = "impacted"
            else:
                role = "alert"
            path_so_far = list(dict.fromkeys(path_so_far + [norm_n]))
            prop_steps.append({
                "node":        norm_n,
                "raw_node":    raw_n,
                "diagram_id":  ev.get("diagram_id", ""),
                "timestamp":   ev.get("timestamp", ev.get("time", "")),
                "title":       _STEP_ROLE_TITLES.get(role, role),
                "description": ev.get("message", ev.get("description", "")),
                "role":        role,
                "path_so_far": list(path_so_far),
                "evidence":    str(ev.get("alert_id", ev.get("rule", "")) or ""),
            })
        if root_cause_norm and rca_d and prop_steps:
            if prop_steps[-1]["node"] != root_cause_norm:
                full_path = list(dict.fromkeys(path_so_far + [root_cause_norm]))
                prop_steps.append({
                    "node":        root_cause_norm,
                    "raw_node":    root_cause_raw,
                    "diagram_id":  rca_d.get("root_cause_diagram", ""),
                    "timestamp":   "",
                    "title":       _STEP_ROLE_TITLES["root_candidate"],
                    "description": "GNN-ranked root cause node",
                    "role":        "root_candidate",
                    "path_so_far": full_path,
                    "evidence":    "GNN RCA",
                })
    else:
        _normalized: list[dict] = []
        _path_so_far: list[str] = []
        for step in prop_steps:
            raw_n  = str(step.get("node", "") or "")
            norm_n = _resolve(raw_n) or raw_n
            _path_so_far = list(dict.fromkeys(_path_so_far + ([norm_n] if norm_n else [])))
            _normalized.append({
                **step,
                "node":        norm_n,
                "raw_node":    raw_n,
                "path_so_far": list(_path_so_far),
            })
        prop_steps = _normalized

    n_steps   = len(prop_steps)
    raw_idx   = st.session_state.get("ent_prop_slider", 1)
    step_idx  = max(0, min(int(raw_idx) - 1, n_steps - 1)) if n_steps else 0
    current   = prop_steps[step_idx] if prop_steps else {}

    return {
        "alert_timeline":      alert_timeline,
        "propagation_steps":   prop_steps,
        "current_step_index":  step_idx,
        "current_step_node":   current.get("node") or None,
        "current_path_so_far": current.get("path_so_far", []),
        "root_cause":          root_cause_norm or None,
        "first_observed_node": _resolve(str(
            inc.get("first_observed_node") or
            next((e.get("node", "") for e in alert_timeline if e.get("is_first_observed")), "")
        )) or None,
        "alert_nodes":         _resolve_list(
            rca_d.get("alert_nodes") or [e.get("node", "") for e in alert_timeline]
        ),
        "impacted_nodes":      _resolve_list(
            rca_d.get("impacted_nodes") or inc.get("impacted_nodes", [])
        ),
        "impact_path":         _resolve_list(
            rca_d.get("impact_path") or inc.get("impact_path", [])
        ),
        "impacted_diagrams":   list(
            rca_d.get("impacted_diagrams") or inc.get("impacted_diagrams", [])
        ),
        "cross_diagram_edges": enterprise_graph.get("cross_diagram_edges", []),
        "node_aliases":        node_aliases,
        "id_to_node":          id_to_node,
    }


def _render_propagation_journey_panel(
    pvc: dict,
    slider_key: str = "ent_prop_slider",
) -> None:
    """Render the explicit Propagation Journey step timeline panel above the graph."""
    prop_steps = pvc.get("propagation_steps", [])
    if not prop_steps:
        return

    n_steps = len(prop_steps)
    st.markdown(
        '<div class="section-label" style="margin-top:14px">Propagation Journey</div>',
        unsafe_allow_html=True,
    )

    step_idx = st.slider(
        "Propagation journey step",
        min_value=1, max_value=n_steps, value=1, step=1,
        key=slider_key,
        help="Step through the cross-diagram incident propagation journey",
    )
    current = prop_steps[step_idx - 1]

    _ROLE_COLORS_J: dict[str, str] = {
        "first_observed": "#ef4444",
        "alert":          "#f97316",
        "bridge":         "#22d3ee",
        "root_candidate": "#10b981",
        "impacted":       "#a78bfa",
    }

    rows_html = []
    for i, step in enumerate(prop_steps):
        is_cur  = (i == step_idx - 1)
        is_past = (i < step_idx - 1)
        role    = step.get("role", "alert")
        rc      = _ROLE_COLORS_J.get(role, "#67e8f9")
        row_bg  = "rgba(255,255,255,0.08)" if is_cur else ("rgba(255,255,255,0.03)" if is_past else "transparent")
        opacity = "1" if is_cur else ("0.7" if is_past else "0.35")
        ind     = "▶" if is_cur else ("✓" if is_past else "·")
        ind_c   = "#38bdf8" if is_cur else ("#4ade80" if is_past else "#334155")
        ev_txt  = (step.get("evidence") or "")[:40]
        rows_html.append(
            f'<tr style="background:{row_bg};opacity:{opacity}">'
            f'<td style="padding:4px 8px;color:{ind_c};font-weight:700;white-space:nowrap">{ind} {i+1}</td>'
            f'<td style="padding:4px 8px"><code style="font-size:0.7rem;color:#a78bfa">'
            f'{step.get("diagram_id", "—")}</code></td>'
            f'<td style="padding:4px 8px"><code style="font-size:0.72rem;color:#67e8f9">'
            f'{step.get("node", "—")}</code></td>'
            f'<td style="padding:4px 8px"><span style="color:{rc};font-size:0.72rem">'
            f'{step.get("title", role)}</span></td>'
            f'<td style="padding:4px 8px;font-size:0.68rem;color:#64748b">{ev_txt}</td>'
            f'</tr>'
        )

    st.markdown(
        '<div style="overflow-x:auto;margin-bottom:6px">'
        '<table style="width:100%;border-collapse:collapse;font-size:0.78rem">'
        '<thead><tr style="border-bottom:1px solid #1e293b">'
        '<th style="padding:4px 8px;text-align:left;color:#64748b">#</th>'
        '<th style="padding:4px 8px;text-align:left;color:#64748b">Diagram</th>'
        '<th style="padding:4px 8px;text-align:left;color:#64748b">Node</th>'
        '<th style="padding:4px 8px;text-align:left;color:#64748b">Role</th>'
        '<th style="padding:4px 8px;text-align:left;color:#64748b">Evidence</th>'
        '</tr></thead>'
        '<tbody>' + ''.join(rows_html) + '</tbody>'
        '</table></div>',
        unsafe_allow_html=True,
    )

    role      = current.get("role", "alert")
    rc        = _ROLE_COLORS_J.get(role, "#67e8f9")
    cur_title = current.get("title", _STEP_ROLE_TITLES.get(role, role))
    cur_node  = current.get("node", "")
    cur_diag  = current.get("diagram_id", "")
    cur_desc  = current.get("description", "")

    st.markdown(
        f'<div class="traversal-card">'
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">'
        f'<div class="traversal-step-lbl" style="color:{rc}">Step {step_idx} — {cur_title}</div>'
        f'<span class="diag-label">{cur_diag}</span>'
        f'</div>'
        f'<div class="traversal-node">{cur_node}</div>'
        f'<div style="font-size:0.76rem;color:#94a3b8;margin-top:4px">{cur_desc}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    chips = []
    for i, step in enumerate(prop_steps):
        nd  = step.get("node", "")
        cls = "current" if i == step_idx - 1 else ("visited" if i < step_idx - 1 else "")
        chips.append(f'<span class="node-chip {cls}">{nd}</span>')
    st.markdown(
        '<div style="padding:6px 0;line-height:2.4">'
        + ' <span style="color:#334155;font-size:0.75rem">→</span> '.join(chips)
        + '</div>',
        unsafe_allow_html=True,
    )


def _render_propagation_steps_slider(
    prop_steps: list[dict],
    incident: dict | None = None,
    slider_key: str = "ent_prop_slider",
) -> tuple[str, list[str]]:
    """Render a cross-diagram propagation step slider.

    Returns (current_step_node, traversal_path_so_far) for graph rendering.
    Returns ("", []) if prop_steps is empty.
    """
    if not prop_steps:
        return "", []

    st.markdown(
        '<div class="section-label" style="margin-top:14px">'
        'Propagation Steps</div>',
        unsafe_allow_html=True,
    )

    n_steps = len(prop_steps)
    step_idx = st.slider(
        "Propagation step",
        min_value=1, max_value=n_steps, value=1, step=1,
        key=slider_key,
        help="Step through the cross-diagram incident propagation",
    )
    current = prop_steps[step_idx - 1]
    traversal_so_far = [s["node"] for s in prop_steps[:step_idx] if s.get("node")]
    current_node     = current.get("node", "")
    current_diag     = current.get("diagram_id", "")
    current_title    = current.get("title", f"Step {step_idx}")
    current_desc     = current.get("description", "")

    # Step role colour
    _role_colors = {
        "First observed symptom":    "#ef4444",
        "Local dependency expansion": "#f59e0b",
        "Cross-diagram bridge":      "#22d3ee",
        "Downstream enterprise impact": "#a78bfa",
        "Root cause ranked":         "#10b981",
    }
    step_color = _role_colors.get(current_title, "#67e8f9")

    st.markdown(
        f'<div class="traversal-card">'
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">'
        f'<div class="traversal-step-lbl" style="color:{step_color}">{current_title}</div>'
        f'<span class="diag-label">{current_diag}</span>'
        f'</div>'
        f'<div class="traversal-node">{current_node}</div>'
        f'<div style="font-size:0.76rem;color:#94a3b8;margin-top:4px">{current_desc}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Path chips
    chips = []
    for i, s in enumerate(prop_steps):
        n = s.get("node", "")
        if i < step_idx - 1:
            cls = "visited"
        elif i == step_idx - 1:
            cls = "current"
        else:
            cls = ""
        chips.append(f'<span class="node-chip {cls}">{n}</span>')
    st.markdown(
        '<div style="padding:6px 0;line-height:2.4">'
        + ' <span style="color:#334155;font-size:0.75rem">→</span> '.join(chips)
        + '</div>',
        unsafe_allow_html=True,
    )

    return current_node, traversal_so_far


def _incident_to_local_rca(incident: dict) -> dict:
    """Convert an IncidentScenario dict to the local_rca_result format."""
    timeline   = incident.get("alert_timeline", [])
    first_obs  = incident.get("first_observed_node", "")
    alert_nodes = [e["node"] for e in timeline if e.get("node")]
    steps       = incident.get("reasoning_steps", [])
    why_root    = steps[2] if len(steps) > 2 else " ".join(steps)
    return {
        "mode":               "scenario_guided_graph_rca",
        "root_cause":         incident.get("root_cause", ""),
        "first_observed":     first_obs,
        "first_observed_node": first_obs,
        "alert_nodes":        alert_nodes,
        "impacted_nodes":     incident.get("impacted_nodes", []),
        "impact_path":        incident.get("impact_path", []),
        "ranking":            incident.get("candidate_ranking", []),
        "incident_title":     incident.get("incident_title", ""),
        "severity":           incident.get("severity", "High"),
        "alert_summary":      incident.get("alert_summary", ""),
        "suspected_domain":   incident.get("suspected_domain", ""),
        "symptoms":           [e.get("message", "") for e in timeline[:3]],
        "why_root":           why_root,
        "reasoning_steps":    steps,
        "recommended_actions": incident.get("recommended_actions", []),
        "path_note":          "",
        "rca_source":         incident.get("rca_source", "Scenario-guided graph RCA"),
    }


def _incident_to_enterprise_rca(incident: dict) -> dict:
    """Convert an enterprise IncidentScenario dict to the enterprise_rca_result format."""
    mode = incident.get("rca_source", "Scenario-grounded RCA simulation")
    gnn_src = ""
    if mode == "Enterprise GNN RCA":
        scen = incident.get("scenario_id", "")
        gnn_src = str(
            _demo_asset_path("enterprise_gnn_rca") /
            f"{scen}_enterprise_gnn_rca_result.json"
        ) if scen else ""
    return {
        "mode":               mode,
        "root_cause":         incident.get("root_cause", ""),
        "root_cause_diagram": incident.get("root_cause_diagram", ""),
        "impacted_diagrams":  incident.get("impacted_diagrams", []),
        "alert_count":        len(incident.get("alert_timeline", [])),
        "alert_nodes":        [e["node"] for e in incident.get("alert_timeline", []) if e.get("node")],
        "impacted_nodes":     incident.get("impacted_nodes", []),
        "impact_path":        incident.get("impact_path", []),
        "ranking":            incident.get("candidate_ranking", []),
        "enterprise_node_count": 0,
        "gnn_source_file":    gnn_src,
    }


def _persist_incident(incident: dict, kind: str) -> Path | None:
    """Write incident JSON to runtime_state/incident_runs/<hash>/."""
    try:
        raw     = json.dumps(incident, sort_keys=True)
        run_id  = __import__("hashlib").sha1(raw.encode()).hexdigest()[:12]
        out_dir = _runtime_path("incident_runs") / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{kind}_incident.json"
        out_path.write_text(raw, encoding="utf-8")
        return out_path
    except Exception:
        return None


def _build_local_incident_story(
    local_graph: dict,
    diagram_id: str,
    selected_record: dict | None = None,
) -> dict:
    """Build a topology-aware incident story for Topology RCA."""
    nodes_raw = local_graph.get("nodes", [])
    edges_raw = local_graph.get("edges", [])

    nodes: dict[str, dict] = {n["id"]: n for n in nodes_raw if n.get("id")}
    node_ids = list(nodes.keys())
    if not node_ids:
        return {"mode": "scenario_guided_graph_rca", "root_cause": "", "impact_path": []}

    adj_out: dict[str, list] = {n: [] for n in node_ids}
    adj_in:  dict[str, list] = {n: [] for n in node_ids}
    adj_un:  dict[str, list] = {n: [] for n in node_ids}
    for e in edges_raw:
        s, t = e.get("source", ""), e.get("target", "")
        if s in adj_out and t in adj_in:
            adj_out[s].append(t)
            adj_in[t].append(s)
            if t not in adj_un[s]:
                adj_un[s].append(t)
            if s not in adj_un[t]:
                adj_un[t].append(s)

    tmpl = _DIAG_INCIDENT_TEMPLATES.get(diagram_id, _DEFAULT_INCIDENT_TEMPLATE)

    def _first_matching(type_priority: list) -> str | None:
        for ptype in type_priority:
            for n_id, n_obj in nodes.items():
                if n_obj.get("type", "").lower() == ptype:
                    return n_id
        return None

    root = _first_matching(tmpl["root_types"])
    if not root:
        root = max(node_ids, key=lambda n: (len(adj_out[n]), -len(adj_in[n])))

    first_obs = _first_matching(tmpl["first_obs_types"])
    if not first_obs or first_obs == root:
        leaves = [n for n in node_ids if n != root and not adj_out.get(n) and adj_in.get(n)]
        first_obs = leaves[0] if leaves else next((n for n in node_ids if n != root), root)

    path = _bfs_path(root, first_obs, adj_out)
    path_note = ""
    if not path:
        path = _bfs_path(root, first_obs, adj_un)
    if not path:
        path = [root, first_obs] if root != first_obs else [root]
        path_note = "Path inferred from nearest topology dependency."

    seen: set = {root}
    queue = list(adj_out[root])
    impacted: list[str] = []
    while queue:
        cur = queue.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        impacted.append(cur)
        queue.extend(adj_out[cur])
    if not impacted:
        impacted = [n for n in node_ids if n != root][:4]

    path_str = " → ".join(path) if path else first_obs
    why_root = (
        f"{root} is the common upstream dependency for the impacted "
        f"{diagram_id.replace('_', ' ')} path. "
        f"The affected node {first_obs} depends on: {path_str}."
    )
    if path_note:
        why_root += f" ({path_note})"

    path_set = set(path)
    outdeg = {n: len(adj_out[n]) for n in node_ids}
    indeg  = {n: len(adj_in[n])  for n in node_ids}
    ranking = []
    for n_id in node_ids:
        score = 0.40 + outdeg[n_id] * 0.11 + (0.14 if indeg[n_id] == 0 else 0.0)
        if n_id == root:
            score += 0.20
        reason = _local_rca_ranking_reason(
            n_id, root, first_obs, path_set, set(impacted),
            outdeg, indeg, diagram_id, nodes,
        )
        ranking.append({"node": n_id, "score": round(min(score, 0.99), 3), "reason": reason})
    ranking.sort(key=lambda r: r["score"], reverse=True)

    impacted_preview = ", ".join(impacted[:3]) + ("…" if len(impacted) > 3 else "")

    return {
        "mode":               "scenario_guided_graph_rca",
        "root_cause":         root,
        "first_observed":     first_obs,
        "alert_nodes":        [first_obs] if first_obs else [],
        "impacted_nodes":     impacted,
        "impact_path":        path,
        "ranking":            ranking[:6],
        "incident_title":     tmpl["titles"][0],
        "severity":           tmpl["severity"],
        "alert_summary":      tmpl["alert_summary"],
        "suspected_domain":   tmpl["suspected_domain"],
        "symptoms":           tmpl["symptoms"],
        "why_root":           why_root,
        "reasoning_steps": [
            f"1. First symptom observed at {first_obs}.",
            f"2. Traced upstream path: {path_str}.",
            f"3. {root} has highest topological reach — zero or minimal in-degree, maximum out-degree.",
            f"4. All impacted nodes ({impacted_preview}) are reachable from {root}.",
        ],
        "recommended_actions": tmpl["recommended_actions"],
        "path_note":           path_note,
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
    """Load a trained enterprise GNN RCA result for the exact scenario only.

    Returns None if no result exists for this scenario — does NOT silently
    return a result from a different scenario.
    """
    if not scenario_id or scenario_id == "—":
        return None
    candidate = _demo_asset_path("enterprise_gnn_rca") / f"{scenario_id}_enterprise_gnn_rca_result.json"
    if candidate.exists():
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


def _simulate_enterprise_rca(alerts: dict, enterprise_graph: dict) -> dict:
    _ent_summary = st.session_state.get("enterprise_ingestion_summary") or {}
    _sel_rec     = _selected_enterprise_record()
    scenario_id  = (
        _ent_summary.get("scenario_id")
        or alerts.get("scenario_id")
        or _sel_rec.get("source_scenario_id")
        or "enterprise_v3_0000"
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
            "gnn_source_file":    str(
                _demo_asset_path("enterprise_gnn_rca") /
                f"{scenario_id}_enterprise_gnn_rca_result.json"
            ),
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
def _render_enterprise_pyvis(
    enterprise_graph: dict,
    absorbed_ids: "set[str]",
    rca: "dict | None",
    height: int = 760,
    current_step_node: "str | None" = None,
    traversal_path: "list[str] | None" = None,
) -> bool:
    try:
        from pyvis.network import Network  # type: ignore
    except Exception:
        return False

    root        = (rca or {}).get("root_cause")
    alert_set   = set((rca or {}).get("alert_nodes", []))
    impacted    = set((rca or {}).get("impacted_nodes", []))
    trav_set    = set(traversal_path or []) - ({current_step_node} if current_step_node else set())
    path_set: set[tuple[str, str]] = set()
    for a, b in zip((rca or {}).get("impact_path", []),
                    (rca or {}).get("impact_path", [])[1:]):
        path_set.add((a, b))

    net = Network(height=f"{height}px", width="100%", directed=True,
                  bgcolor="#0b1220", font_color="#e2e8f0")
    net.barnes_hut(gravity=-4200, central_gravity=0.22, spring_length=180, spring_strength=0.042)

    # Build node → diagram group map (list or dict cluster format)
    groups: dict[str, str] = {}
    _clusters = enterprise_graph.get("diagram_clusters", {})
    if isinstance(_clusters, dict):
        for did, cluster in _clusters.items():
            for nid in (cluster.get("node_ids", []) if isinstance(cluster, dict) else cluster if isinstance(cluster, list) else []):
                groups[nid] = did
    elif isinstance(_clusters, list):
        for cluster in _clusters:
            if isinstance(cluster, dict):
                for nid in cluster.get("node_ids", []):
                    groups[nid] = cluster.get("diagram_id", "")

    node_map = {n.get("id"): n for n in enterprise_graph.get("nodes", [])}

    for n in enterprise_graph.get("nodes", []):
        nid    = n.get("id", "")
        ntype  = n.get("type", "server")
        shared = n.get("is_shared_entity", False)
        diag   = groups.get(nid, n.get("diagram_id", ""))
        col_base = _V3_DIAG_COLORS.get(diag, "#64748b")

        # Priority: step node > root > alert > impacted > traversal > absorbed > shared > default
        if nid == current_step_node and current_step_node:
            color, size, bw = "#ffffff", 40, 6
        elif nid == root and root:
            color, size, bw = "#ef4444", 38, 5
        elif nid in alert_set:
            color, size, bw = "#f97316", 30, 4
        elif nid in impacted:
            color, size, bw = "#facc15", 26, 3
        elif nid in trav_set:
            color, size, bw = "#a855f7", 24, 4
        elif nid in absorbed_ids:
            color, size, bw = "#22d3ee", 28, 4
        elif shared:
            color, size, bw = "#38bdf8", 26, 4
        else:
            color, size, bw = col_base, 20, 2

        border = "#ffffff" if (nid == root or nid == current_step_node) else ("#fbbf24" if shared else "#3a4a5a")
        title  = (
            f"<b>{nid}</b><br>type: {ntype}<br>"
            f"diagram: {diag}<br>ip: {n.get('ip_address','—')}<br>zone: {n.get('zone','—')}"
            + ("<br><b>⚡ CURRENT STEP</b>" if nid == current_step_node else "")
            + ("<br><b>🔴 ROOT CAUSE</b>" if nid == root else "")
            + ("<br><b>shared entity</b>" if shared else "")
            + ("<br><b>newly absorbed</b>" if nid in absorbed_ids else "")
            + ("<br><b>traversal path</b>" if nid in trav_set else "")
        )
        net.add_node(nid, label=nid, title=title,
                     group=diag,
                     color={"background": color, "border": border},
                     size=size, borderWidth=bw, borderWidthSelected=6)

    def _add_edges(edge_list: list, force_cross: bool = False) -> None:
        for e in edge_list:
            src, tgt = e.get("source", ""), e.get("target", "")
            if src not in node_map or tgt not in node_map:
                continue
            is_cross = (
                force_cross
                or e.get("edge_scope") == "cross_diagram"
                or e.get("edge_type") == "cross_diagram"
            )
            is_path = (src, tgt) in path_set
            net.add_edge(
                src, tgt,
                label=str(e.get("label", ""))[:16],
                color="#ffffff" if is_path else ("#22d3ee" if is_cross else "#4a5568"),
                width=5 if is_path else (2.5 if is_cross else 1),
                dashes=is_cross and not is_path,
                title=f"{e.get('relationship','')} | {e.get('edge_scope','')}",
            )

    _add_edges(enterprise_graph.get("edges", []))
    _add_edges(enterprise_graph.get("cross_diagram_edges", []), force_cross=True)

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


# ── RCA Journey role constants (shared across journey functions) ───────────────
_JOURNEY_ROLE_COLORS: dict[str, str] = {
    "first_observed_alert": "#f97316",
    "alert_node":           "#fb923c",
    "dependency_hop":       "#60a5fa",
    "cross_diagram_bridge": "#22d3ee",
    "gnn_root_cause":       "#ef4444",
    "impacted_service":     "#facc15",
}
_JOURNEY_ROLE_LABELS: dict[str, str] = {
    "first_observed_alert": "First alert",
    "alert_node":           "Alert node",
    "dependency_hop":       "Dependency hop",
    "cross_diagram_bridge": "Cross-diagram bridge",
    "gnn_root_cause":       "GNN root cause",
    "impacted_service":     "Impacted service",
}
_JOURNEY_ROLE_ICONS: dict[str, str] = {
    "first_observed_alert": "⚡",
    "alert_node":           "🔔",
    "dependency_hop":       "→",
    "cross_diagram_bridge": "⇒",
    "gnn_root_cause":       "🎯",
    "impacted_service":     "💥",
}


def _build_rca_journey_context(
    enterprise_graph: dict,
    ent_incident: "dict | None",
    rca: "dict | None",
    gnn_result: "dict | None" = None,
) -> dict:
    """Build RCA journey steps with normalized node IDs and role assignments."""
    node_aliases, id_to_node = _build_node_aliases(enterprise_graph)

    def _res(raw: str) -> str:
        return node_aliases.get(str(raw), str(raw)) if raw else ""

    def _res_list(lst) -> "list[str]":
        return [_res(str(x)) for x in (lst or []) if x]

    inc   = ent_incident or {}
    rca_d = rca or {}
    gnn_d = gnn_result or {}

    alert_timeline     = list(inc.get("alert_timeline", []) or [])
    impact_path_raw    = list(rca_d.get("impact_path") or inc.get("impact_path", []) or [])
    root_cause_raw     = str(rca_d.get("root_cause") or inc.get("root_cause", "") or "")
    root_cause_norm    = _res(root_cause_raw)
    root_cause_diag    = str(rca_d.get("root_cause_diagram", "") or "")
    impacted_raw       = list(rca_d.get("impacted_nodes") or inc.get("impacted_nodes", []) or [])
    impacted_diags     = list(rca_d.get("impacted_diagrams") or inc.get("impacted_diagrams", []) or [])

    # Build lookup structures
    cross_edges  = enterprise_graph.get("cross_diagram_edges", [])
    cross_nodes  = set()
    for e in cross_edges:
        cross_nodes.add(str(e.get("source", e.get("source_node", "")) or ""))
        cross_nodes.add(str(e.get("target", e.get("target_node", "")) or ""))

    tl_by_node: dict[str, dict] = {}
    for ev in alert_timeline:
        nk = _res(str(ev.get("node", "") or ""))
        if nk and nk not in tl_by_node:
            tl_by_node[nk] = ev

    bridge_nodes = {
        _res(str(e.get("node", "")))
        for e in alert_timeline if e.get("correlation_role") == "bridge"
    } | {n for n in cross_nodes if n in id_to_node}

    # first observed node
    _first_raw = str(
        inc.get("first_observed_node") or
        next((e.get("node", "") for e in alert_timeline if e.get("is_first_observed")),
             alert_timeline[0].get("node", "") if alert_timeline else "")
    )
    first_obs_norm = _res(_first_raw)

    # edge index for path edge lookups
    all_edges = enterprise_graph.get("edges", []) + cross_edges
    edge_idx: dict[tuple[str, str], dict] = {}
    for e in all_edges:
        src = str(e.get("source", e.get("source_node", "")) or "")
        tgt = str(e.get("target", e.get("target_node", "")) or "")
        if src and tgt:
            edge_idx.setdefault((src, tgt), e)

    def _find_edge(a: str, b: str) -> "dict | None":
        return edge_idx.get((a, b)) or edge_idx.get((b, a))

    def _get_diag(nid: str) -> str:
        return str((id_to_node.get(nid) or {}).get("diagram_id", "") or "")

    def _assign_role(norm_id: str) -> str:
        if norm_id == first_obs_norm:
            return "first_observed_alert"
        if norm_id == root_cause_norm:
            return "gnn_root_cause"
        if norm_id in bridge_nodes:
            return "cross_diagram_bridge"
        if norm_id in tl_by_node:
            return "alert_node"
        return "dependency_hop"

    def _make_reason(norm_id: str, role: str, ev: "dict | None") -> str:
        if role == "first_observed_alert":
            return (ev or {}).get("message", (ev or {}).get("description", "First alert observed")) or "First alert observed"
        if role == "gnn_root_cause":
            score = None
            for cand in (gnn_d.get("top_candidates") or rca_d.get("top_candidates") or []):
                c_norm = _res(str(cand.get("node_id", "") or ""))
                if c_norm == norm_id:
                    score = cand.get("score")
                    break
            if score is None:
                first_c = ((gnn_d.get("top_candidates") or rca_d.get("top_candidates") or [{}])[0])
                score = first_c.get("score") if first_c else None
            score_str = f", GNN score {score:.3f}" if score is not None else ""
            return f"GNN rank #1{score_str}"
        if role == "cross_diagram_bridge":
            return (ev or {}).get("message", "Cross-diagram dependency bridge") or "Cross-diagram dependency bridge"
        if role == "alert_node":
            return (ev or {}).get("message", "Alert converging on this node") or "Alert converging on this node"
        return "Dependency hop on impact path"

    # Build journey steps
    steps: list[dict] = []
    seen: set[str]    = set()
    path_so_far: list[str] = []

    def _add(raw_id: str, role: str, reason: str, ev: "dict | None" = None) -> None:
        nonlocal path_so_far
        norm_id = _res(raw_id) or raw_id
        if norm_id in seen:
            return
        seen.add(norm_id)
        prev = path_so_far[-1] if path_so_far else None
        path_so_far = path_so_far + [norm_id]
        edge_from_prev = None
        if prev:
            e = _find_edge(prev, norm_id)
            if e:
                is_cross = (e.get("edge_scope") == "cross_diagram" or e.get("edge_type") == "cross_diagram")
                edge_from_prev = {
                    "source":   prev,
                    "target":   norm_id,
                    "label":    str(e.get("label", e.get("relationship", "")) or ""),
                    "is_cross": is_cross,
                }
        diag = str((ev or {}).get("diagram_id", "") or "") or _get_diag(norm_id)
        steps.append({
            "step":              len(steps) + 1,
            "node_id":           raw_id,
            "normalized_node_id": norm_id,
            "diagram_id":        diag,
            "role":              role,
            "reason":            reason,
            "edge_from_previous": edge_from_prev,
            "path_so_far":       list(path_so_far),
        })

    # Strategy: use impact_path as backbone if available, else alert timeline
    impact_path_norm = _res_list(impact_path_raw)
    if impact_path_norm:
        for i, norm_id in enumerate(impact_path_norm):
            raw_id = str(impact_path_raw[i]) if i < len(impact_path_raw) else norm_id
            ev     = tl_by_node.get(norm_id)
            role   = _assign_role(norm_id)
            _add(raw_id, role, _make_reason(norm_id, role, ev), ev)
    else:
        sorted_tl = sorted(alert_timeline, key=lambda x: str(x.get("timestamp", x.get("time", ""))))
        for ev in sorted_tl:
            raw_id = str(ev.get("node", "") or "")
            if not raw_id:
                continue
            norm_id = _res(raw_id)
            role    = _assign_role(norm_id)
            _add(raw_id, role, _make_reason(norm_id, role, ev), ev)

    # Ensure root cause appears
    if root_cause_norm and root_cause_norm not in seen:
        ev = tl_by_node.get(root_cause_norm)
        _add(root_cause_raw, "gnn_root_cause", _make_reason(root_cause_norm, "gnn_root_cause", ev), ev)

    # Append impacted services
    impacted_norm = _res_list(impacted_raw)
    for i, norm_id in enumerate(impacted_norm):
        if norm_id not in seen and norm_id != root_cause_norm:
            raw_id = str(impacted_raw[i]) if i < len(impacted_raw) else norm_id
            ev     = tl_by_node.get(norm_id)
            _add(raw_id, "impacted_service", "Impacted by root cause propagation", ev)

    # Unmatched
    all_raw = (
        {str(ev.get("node", "")) for ev in alert_timeline if ev.get("node")} |
        {str(n) for n in impact_path_raw if n} |
        ({root_cause_raw} if root_cause_raw else set()) |
        {str(n) for n in impacted_raw if n}
    )
    unmatched = sorted(r for r in all_raw if r and r not in node_aliases and r not in id_to_node)

    # Current step from session state slider
    n_steps  = len(steps)
    raw_idx  = st.session_state.get("rca_journey_slider", 1)
    step_idx = max(0, min(int(raw_idx) - 1, n_steps - 1)) if n_steps else 0

    return {
        "steps":             steps,
        "root_cause":        root_cause_norm or None,
        "root_cause_diagram": root_cause_diag,
        "impact_path":       impact_path_norm,
        "impacted_nodes":    impacted_norm,
        "impacted_diagrams": impacted_diags,
        "unmatched_nodes":   unmatched,
        "current_step_index": step_idx,
        "current_step":      steps[step_idx] if steps else {},
        "node_aliases":      node_aliases,
        "id_to_node":        id_to_node,
        "alert_timeline":    alert_timeline,
    }


def _render_rca_journey_stepper(jctx: dict, slider_key: str = "rca_journey_slider") -> None:
    """Render the RCA investigation journey stepper panel above the graph."""
    steps = jctx.get("steps", [])
    if not steps:
        return

    n_steps = len(steps)
    st.markdown(
        '<div class="section-label" style="margin-top:14px">RCA Investigation Journey</div>'
        '<div style="font-size:0.73rem;color:#64748b;margin-bottom:8px">'
        'First observed alert → dependency path → cross-diagram bridge → GNN root cause → impacted services</div>',
        unsafe_allow_html=True,
    )

    step_idx = st.slider(
        "RCA journey step", min_value=1, max_value=n_steps, value=1, step=1,
        key=slider_key,
        help="Step through the RCA investigation journey",
    )
    current = steps[step_idx - 1]

    # Timeline table
    rows = []
    for i, step in enumerate(steps):
        is_cur  = (i == step_idx - 1)
        is_past = (i < step_idx - 1)
        role    = step["role"]
        rc      = _JOURNEY_ROLE_COLORS.get(role, "#94a3b8")
        icon    = _JOURNEY_ROLE_ICONS.get(role, "·")
        rl      = _JOURNEY_ROLE_LABELS.get(role, role)
        row_bg  = "rgba(255,255,255,0.08)" if is_cur else ("rgba(255,255,255,0.03)" if is_past else "transparent")
        opacity = "1" if is_cur else ("0.7" if is_past else "0.35")
        ind     = "▶" if is_cur else ("✓" if is_past else str(i + 1))
        ind_c   = "#38bdf8" if is_cur else ("#4ade80" if is_past else "#475569")
        reason_txt = (step.get("reason") or "")[:60]
        rows.append(
            f'<tr style="background:{row_bg};opacity:{opacity}">'
            f'<td style="padding:5px 8px;color:{ind_c};font-weight:700;white-space:nowrap">{ind}</td>'
            f'<td style="padding:5px 8px"><span style="color:{rc}">{icon}</span>'
            f' <span style="color:{rc};font-size:0.72rem">{rl}</span></td>'
            f'<td style="padding:5px 8px"><code style="font-size:0.73rem;color:#67e8f9">'
            f'{step["normalized_node_id"]}</code></td>'
            f'<td style="padding:5px 8px"><code style="font-size:0.7rem;color:#a78bfa">'
            f'{step.get("diagram_id", "—")}</code></td>'
            f'<td style="padding:5px 8px;font-size:0.68rem;color:#94a3b8">{reason_txt}</td>'
            f'</tr>'
        )

    st.markdown(
        '<div style="overflow-x:auto;margin-bottom:8px">'
        '<table style="width:100%;border-collapse:collapse;font-size:0.78rem">'
        '<thead><tr style="border-bottom:1px solid #1e293b">'
        '<th style="padding:5px 8px;color:#64748b;text-align:left">#</th>'
        '<th style="padding:5px 8px;color:#64748b;text-align:left">Role</th>'
        '<th style="padding:5px 8px;color:#64748b;text-align:left">Node</th>'
        '<th style="padding:5px 8px;color:#64748b;text-align:left">Diagram</th>'
        '<th style="padding:5px 8px;color:#64748b;text-align:left">Evidence</th>'
        '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>',
        unsafe_allow_html=True,
    )

    role     = current["role"]
    rc       = _JOURNEY_ROLE_COLORS.get(role, "#94a3b8")
    icon     = _JOURNEY_ROLE_ICONS.get(role, "·")
    rl       = _JOURNEY_ROLE_LABELS.get(role, role)
    ef       = current.get("edge_from_previous")
    edge_info = ""
    if ef:
        bridge_lbl = " (cross-diagram)" if ef.get("is_cross") else ""
        edge_info  = f' via <code style="font-size:0.7rem">{ef.get("label","link")}{bridge_lbl}</code>'
    st.markdown(
        f'<div class="traversal-card">'
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">'
        f'<span style="font-size:1.2rem">{icon}</span>'
        f'<div class="traversal-step-lbl" style="color:{rc}">Step {step_idx} — {rl}</div>'
        f'<span class="diag-label">{current.get("diagram_id","")}</span>'
        f'</div>'
        f'<div class="traversal-node">{current["normalized_node_id"]}</div>'
        f'<div style="font-size:0.76rem;color:#94a3b8;margin-top:4px">'
        f'{current.get("reason","")}{edge_info}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    chips = []
    for i, step in enumerate(steps):
        nid  = step["normalized_node_id"]
        role = step["role"]
        rc2  = _JOURNEY_ROLE_COLORS.get(role, "#94a3b8")
        cls  = "current" if i == step_idx - 1 else ("visited" if i < step_idx - 1 else "")
        label = f'{i+1}:{nid}'
        chips.append(f'<span class="node-chip {cls}" style="color:{rc2 if cls else "#475569"}">{label}</span>')
    st.markdown(
        '<div style="padding:6px 0;line-height:2.6">'
        + ' <span style="color:#1e293b;font-size:0.75rem">→</span> '.join(chips)
        + '</div>',
        unsafe_allow_html=True,
    )


def _render_enterprise_pyvis_rca_journey(
    enterprise_graph: dict,
    jctx: dict,
    absorbed_ids: "set[str]",
    height: int = 800,
) -> bool:
    """PyVis renderer that visualizes the RCA journey with role-colored nodes and dimmed background topology."""
    try:
        from pyvis.network import Network  # type: ignore
    except Exception:
        return False

    steps          = jctx.get("steps", [])
    step_idx       = jctx.get("current_step_index", 0)
    current_step   = jctx.get("current_step", {})
    current_node   = current_step.get("normalized_node_id", "")
    root_cause     = jctx.get("root_cause", "")

    # Journey lookup
    journey_map: dict[str, dict] = {s["normalized_node_id"]: s for s in steps}
    # path so far = nodes in steps[0..step_idx]
    path_so_far = [s["normalized_node_id"] for s in steps[: step_idx + 1]]
    path_set    = set(path_so_far)
    path_pairs: set[tuple[str, str]] = set()
    for a, b in zip(path_so_far, path_so_far[1:]):
        path_pairs.add((a, b))
        path_pairs.add((b, a))  # undirected match

    # Edge step label: for each consecutive pair in path_so_far, label = "Step N"
    edge_step_lbl: dict[tuple[str, str], str] = {}
    for i, (a, b) in enumerate(zip(path_so_far, path_so_far[1:])):
        lbl = f"Step {i+1}"
        edge_step_lbl[(a, b)] = lbl
        edge_step_lbl[(b, a)] = lbl

    _RCOL: dict[str, str] = {
        "first_observed_alert": "#f97316",
        "alert_node":           "#fb923c",
        "dependency_hop":       "#60a5fa",
        "cross_diagram_bridge": "#22d3ee",
        "gnn_root_cause":       "#ef4444",
        "impacted_service":     "#facc15",
    }
    _RSIZ: dict[str, int] = {
        "first_observed_alert": 34,
        "alert_node":           28,
        "dependency_hop":       26,
        "cross_diagram_bridge": 30,
        "gnn_root_cause":       44,
        "impacted_service":     26,
    }

    net = Network(height=f"{height}px", width="100%", directed=True,
                  bgcolor="#0b1220", font_color="#e2e8f0")
    net.barnes_hut(gravity=-4200, central_gravity=0.22, spring_length=180, spring_strength=0.042)

    # group map
    groups: dict[str, str] = {}
    _cls = enterprise_graph.get("diagram_clusters", {})
    if isinstance(_cls, dict):
        for did, cl in _cls.items():
            for nid in (cl.get("node_ids", []) if isinstance(cl, dict) else cl if isinstance(cl, list) else []):
                groups[nid] = did
    elif isinstance(_cls, list):
        for cl in _cls:
            if isinstance(cl, dict):
                for nid in cl.get("node_ids", []):
                    groups[nid] = cl.get("diagram_id", "")

    node_map = {n.get("id"): n for n in enterprise_graph.get("nodes", [])}

    for n in enterprise_graph.get("nodes", []):
        nid    = n.get("id", "")
        ntype  = n.get("type", "server")
        diag   = groups.get(nid, n.get("diagram_id", ""))
        step_d = journey_map.get(nid)
        in_path = nid in path_set
        is_cur  = (nid == current_node and current_node)
        is_root = (nid == root_cause and root_cause)

        if is_cur:
            color, size, bw, border = "#ffffff", 46, 7, "#ffffff"
            font = {"size": 16, "color": "#ffffff", "bold": True}
            node_label = str(step_d["step"]) if step_d else nid
        elif is_root and in_path:
            color, size, bw, border = "#ef4444", 44, 6, "#ffffff"
            font = {"size": 14, "color": "#ffffff"}
            node_label = str(step_d["step"]) if step_d else nid
        elif step_d and in_path:
            role   = step_d["role"]
            color  = _RCOL.get(role, "#60a5fa")
            size   = _RSIZ.get(role, 28)
            bw     = 4
            border = color
            font   = {"size": 12, "color": "#e2e8f0"}
            node_label = str(step_d["step"])
        elif step_d:
            # Future journey step (not yet revealed)
            color, size, bw, border = "#1e293b", 16, 2, "#334155"
            font = {"size": 9, "color": "#334155"}
            node_label = nid
        else:
            # Non-journey background topology — dim
            color, size, bw, border = "#172033", 12, 1, "#1e3a5f"
            font = {"size": 9, "color": "#1e3a5f"}
            node_label = nid

        step_info = ""
        if step_d:
            role_lbl = _JOURNEY_ROLE_LABELS.get(step_d["role"], step_d["role"])
            step_info = f"<br><b>Journey step {step_d['step']} — {role_lbl}</b><br><i>{step_d.get('reason','')}</i>"

        title = (
            f"<b>{nid}</b><br>type: {ntype}<br>diagram: {diag}"
            + step_info
            + ("<br><b>⚡ CURRENT STEP</b>" if is_cur else "")
            + ("<br><b>🎯 GNN ROOT CAUSE</b>" if is_root else "")
        )
        net.add_node(nid, label=node_label, title=title, group=diag,
                     color={"background": color, "border": border},
                     size=size, borderWidth=bw, borderWidthSelected=8,
                     font=font)

    def _add_edges_rca(edge_list: list, force_cross: bool = False) -> None:
        for e in edge_list:
            src = str(e.get("source", e.get("source_node", "")) or "")
            tgt = str(e.get("target", e.get("target_node", "")) or "")
            if src not in node_map or tgt not in node_map:
                continue
            is_cross   = force_cross or e.get("edge_scope") == "cross_diagram" or e.get("edge_type") == "cross_diagram"
            is_journey = (src, tgt) in path_pairs
            step_lbl   = edge_step_lbl.get((src, tgt), "")
            if is_journey:
                net.add_edge(src, tgt,
                             label=step_lbl,
                             color="#22d3ee", width=6, dashes=False,
                             title=f"Journey path — {e.get('relationship','')}")
            elif is_cross:
                net.add_edge(src, tgt,
                             label=str(e.get("label", ""))[:10],
                             color="#a855f7", width=2.5, dashes=True,
                             title=f"Cross-diagram — {e.get('relationship','')}")
            else:
                net.add_edge(src, tgt,
                             label="",
                             color="#172033", width=0.7, dashes=False,
                             title=e.get("relationship", ""))

    _add_edges_rca(enterprise_graph.get("edges", []))
    _add_edges_rca(enterprise_graph.get("cross_diagram_edges", []), force_cross=True)

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


def _render_rca_explanation_card(
    jctx: dict,
    rca: "dict | None",
    gnn_result: "dict | None" = None,
) -> None:
    """Render the RCA explanation and blast radius card."""
    rca_d  = rca or {}
    gnn_d  = gnn_result or {}
    steps  = jctx.get("steps", [])

    root_cause   = jctx.get("root_cause") or rca_d.get("root_cause", "—")
    root_diag    = jctx.get("root_cause_diagram") or rca_d.get("root_cause_diagram", "—")
    mode         = rca_d.get("mode", "—")
    alert_cnt    = rca_d.get("alert_count", len(jctx.get("alert_timeline", [])))
    imp_nodes    = jctx.get("impacted_nodes", [])
    imp_diags    = jctx.get("impacted_diagrams", [])
    impact_path  = jctx.get("impact_path", [])

    # Find top GNN score
    top_score = None
    for cand in (gnn_d.get("top_candidates") or rca_d.get("top_candidates") or []):
        c_norm = jctx.get("node_aliases", {}).get(str(cand.get("node_id", "") or ""), str(cand.get("node_id", "") or ""))
        if c_norm == root_cause or str(cand.get("node_id", "")) == root_cause:
            top_score = cand.get("score")
            break
    if top_score is None:
        first_c = ((gnn_d.get("top_candidates") or rca_d.get("top_candidates") or [{}])[0])
        top_score = first_c.get("score") if isinstance(first_c, dict) else None

    converging = [s for s in steps if s["role"] in ("alert_node", "first_observed_alert")]
    bridges    = [s for s in steps if s["role"] == "cross_diagram_bridge"]

    score_str = f"GNN score: **{top_score:.3f}**" if top_score is not None else ""

    st.markdown('<hr class="ws-rule" style="margin:10px 0">', unsafe_allow_html=True)
    st.markdown('<div class="section-label" style="margin-top:4px">RCA Explanation</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown(
            f'<div class="info-card" style="margin-bottom:8px">'
            f'<div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.08em;color:#64748b;margin-bottom:4px">Root Cause</div>'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:1.05rem;font-weight:700;color:#ef4444;margin-bottom:2px">{root_cause}</div>'
            f'<div style="font-size:0.73rem;color:#94a3b8;margin-bottom:4px">Diagram: <code style="color:#a78bfa">{root_diag}</code>'
            + (f'&nbsp;·&nbsp;{score_str}' if score_str else '') +
            f'</div>'
            f'<div style="font-size:0.72rem;color:#64748b">RCA mode: {mode}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        evidence = []
        if converging:
            diag_set = set(s.get("diagram_id", "?") for s in converging)
            evidence.append(f"**{len(converging)} converging alerts** across {', '.join(diag_set)}")
        if bridges:
            evidence.append(f"**{len(bridges)} cross-diagram bridge(s)**: {', '.join(s['normalized_node_id'] for s in bridges[:3])}")
        if impact_path:
            path_str = " → ".join(f"`{n}`" for n in impact_path[:5]) + (" …" if len(impact_path) > 5 else "")
            evidence.append(f"**Impact path**: {path_str}")
        if imp_diags:
            evidence.append(f"**Impacted diagrams**: {', '.join(f'`{d}`' for d in imp_diags[:5])}")
        if evidence:
            st.markdown("**Evidence:**")
            for item in evidence:
                st.markdown(f"- {item}")

    with col_b:
        st.markdown(
            f'<div class="info-card">'
            f'<div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.08em;color:#64748b;margin-bottom:8px">Blast Radius</div>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">'
            f'<div><div style="font-size:0.6rem;color:#64748b">Impacted nodes</div>'
            f'<div style="font-size:1.4rem;font-weight:700;color:#facc15">{len(imp_nodes)}</div></div>'
            f'<div><div style="font-size:0.6rem;color:#64748b">Impacted diagrams</div>'
            f'<div style="font-size:1.4rem;font-weight:700;color:#a78bfa">{len(imp_diags)}</div></div>'
            f'<div><div style="font-size:0.6rem;color:#64748b">Alert count</div>'
            f'<div style="font-size:1.4rem;font-weight:700;color:#f97316">{alert_cnt}</div></div>'
            f'<div><div style="font-size:0.6rem;color:#64748b">Journey steps</div>'
            f'<div style="font-size:1.4rem;font-weight:700;color:#38bdf8">{len(steps)}</div></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DIAGRAM INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
def _tab_diagram_outputs_section() -> None:
    """Show outputs after diagram has been processed into local graph."""
    if st.session_state.get("onboard_status", "not_started") == "not_started":
        return
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
    _avg_c = summary.get("device_detection_avg", 0)
    m3.metric("Avg confidence", f"{_avg_c:.0%}" if isinstance(_avg_c, (int, float)) else "—")
    m4.metric("OCR text blocks",   summary.get("ocr_text_blocks", 0))

    source = packet.get("source_label", "")
    if source:
        st.caption(source)

    view_mode = st.radio(
        "View",
        ["Original + Detection", "Local Graph (Interactive)"],
        horizontal=True,
        key="diagram_view_mode",
    )

    if view_mode == "Original + Detection":
        orig_p       = st.session_state.get("selected_diagram_path", "")
        det_p        = st.session_state.get("detected_image_path", "")
        det_source   = st.session_state.get("detection_source") or "Verified Annotation Overlay"
        is_rfdetr    = (
            det_source.startswith("RF-DETR")
            or det_source == "LIVE_RFDETR_INFERENCE"
            or "RFDETR" in det_source.upper()
        )

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
                st.caption("Detection overlay image not available for this run — showing source diagram.")
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

    # ── Evidence tables (full graph-memory view) ──────────────────────────────
    _render_evidence_tables(_selected_diagram_record())


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

    # ── RF-DETR runtime status ────────────────────────────────────────────────
    _rfdetr_resolution = _cached_rfdetr_python_resolution()
    _rfdetr_python     = str(_rfdetr_resolution.get("python_executable") or _rfdetr_resolution.get("resolved_detector_python") or "")
    _rfdetr_runtime    = _rfdetr_resolution.get("runtime") or {"ok": bool(_rfdetr_resolution.get("import_ok"))}
    _rfdetr_http       = _cached_rfdetr_http_health()
    _rfdetr_ckpt       = _find_rfdetr_ckpt(REPO_ROOT) if _RFDETR_BRIDGE_OK and _find_rfdetr_ckpt else None

    _rf_stat_col, _rf_cb_col = st.columns([4, 1])
    with _rf_stat_col:
        _ckpt_ok  = bool(_rfdetr_ckpt)
        _http_ok  = bool(_rfdetr_http.get("ok"))
        _imp_ok   = bool(_rfdetr_runtime.get("ok"))
        _dot_clr  = "#10b981" if (_ckpt_ok and _imp_ok) else "#f59e0b"
        _parts    = []
        if _rfdetr_ckpt:
            _parts.append(_rfdetr_ckpt.name)
        if _rfdetr_http.get("service_url"):
            _svc = "HTTP live" if _http_ok else "HTTP unavailable"
            _parts.append(f"{_svc} · {_rfdetr_http.get('service_url','')}")
        if _rfdetr_python:
            _parts.append(_rfdetr_python)
        _summary = " · ".join(_parts) if _parts else "RF-DETR not configured"
        st.markdown(
            f'<div style="font-size:0.74rem;color:#64748b;margin:6px 0 10px">'
            f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;'
            f'background:{_dot_clr};margin-right:6px;vertical-align:middle"></span>'
            f'RF-DETR · {_summary}</div>',
            unsafe_allow_html=True,
        )
    with _rf_cb_col:
        use_rfdetr = st.checkbox(
            "Use live RF-DETR",
            value=st.session_state.get("use_live_rfdetr", True),
            key="use_live_rfdetr_cb",
            disabled=not bool(_rfdetr_ckpt),
        )
        st.session_state.use_live_rfdetr = use_rfdetr and bool(_rfdetr_ckpt)

    # ── Load onboarding manifest ──────────────────────────────────────────────
    samples = _load_onboarding_manifest(str(REPO_ROOT))
    if not samples:
        st.error(
            "Onboarding manifest not found. Build the asset layer first:\n"
            "```\npython scripts/build_presentation_assets.py\n```"
        )
        return

    _s_col, _b_col = st.columns([3, 2])
    with _s_col:
        sel_idx = st.selectbox(
            "Select sample diagram",
            range(len(samples)),
            format_func=lambda i: f"{samples[i]['sample_id']} | {samples[i]['display_name']}",
            index=None,
            placeholder="Select a diagram from the pool…",
            key="onboard_sample_select",
        )

    if sel_idx is None:
        is_this_sample_active = False
        img_path = Path("")
    else:
        sample   = samples[sel_idx]
        did      = sample.get("source_diagram_id", "")
        img_path = Path(sample.get("image_path", ""))

        onboard_status = st.session_state.get("onboard_status", "not_started")
        onboarded_sid  = (
            _ss_dict("onboard_sample_record").get("sample_id", "")
            if onboard_status != "not_started" else ""
        )
        is_this_sample_active = onboarded_sid == sample.get("sample_id", "")

    if is_this_sample_active:
        src_label = st.session_state.get("detection_source", "")
        badge_cls = (
            "badge-success"
            if src_label == "LIVE_RFDETR_INFERENCE" or src_label.startswith(("Live RF-DETR", "RF-DETR"))
            else "badge-info"
        )
        st.markdown(
            f'<span class="badge badge-info">Ingested</span> '
            f'<span class="badge {badge_cls}">{src_label}</span>',
            unsafe_allow_html=True,
        )

    # ── action button (side-by-side with selectbox) ───────────────────────
    _btn_disabled = (sel_idx is None) or (not img_path.exists())
    with _b_col:
        _run_clicked = st.button("Run Live Diagram Intelligence", type="primary",
                                 use_container_width=True, disabled=_btn_disabled)
    if _run_clicked:
        if not _RUNTIME_INGESTION or _run_ingestion is None:
            _err_d = f"  \n`{_RUNTIME_INGESTION_ERR}`" if _RUNTIME_INGESTION_ERR else ""
            st.error(f"runtime_ingestion failed to load at startup — restart the app to retry.{_err_d}")
        else:
            _external_rfdetr_result = {}
            if st.session_state.use_live_rfdetr and _rfdetr_ckpt and _run_rfdetr_detection is not None:
                _conf = float(os.environ.get("INFRAGRAPH_RFDETR_CONFIDENCE", "0.25"))
                _timeout = int(os.environ.get("INFRAGRAPH_RFDETR_TIMEOUT", "180"))
                with st.spinner("Running external RF-DETR detector runtime…"):
                    _external_rfdetr_result = _run_rfdetr_detection(
                        img_path,
                        Path(_rfdetr_ckpt),
                        confidence=_conf,
                        timeout=_timeout,
                    )
            elif st.session_state.use_live_rfdetr:
                _fallback_err = _RFDETR_BRIDGE_ERR or "RF-DETR bridge unavailable"
                if not _rfdetr_ckpt:
                    _fallback_err = "RF-DETR checkpoint not found"
                _external_rfdetr_result = {
                    "ok": False,
                    "source": "verified_annotation_fallback",
                    "detector_runtime_mode": "verified_annotation_fallback",
                    "error": _fallback_err,
                    "fallback_reason": _fallback_err,
                    "checkpoint_path": "",
                }

            _STEPS = [
                "Loading image",
                "Resolving external RF-DETR runtime",
                "Running live detector or verified annotation fallback",
                "Extracting OCR / text metadata",
                "Extracting connectors",
                "Building node table",
                "Building edge table",
                "Creating local graph memory packet",
                "Ready for absorption",
            ]
            _res_c1, _res_c2, _res_c3 = st.columns(3)
            prog       = _res_c1.progress(0)
            steps_area = _res_c1.empty()

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
                    use_live_rfdetr       = False,
                    rfdetr_model          = None,
                    external_rfdetr_result= _external_rfdetr_result,
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
            st.session_state.live_detection_result       = _external_rfdetr_result
            st.session_state.detected_image_path        = str(ingestion.get("detected_image", ""))
            st.session_state.enterprise_absorbed        = False
            st.session_state.local_rca_result           = {}
            st.session_state.local_incident             = {}
            st.session_state.enterprise_rca_result      = {}
            st.session_state.enterprise_incident        = {}
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
            badge_cls = "badge-success" if det_src == "LIVE_RFDETR_INFERENCE" else "badge-warn"
            t_s       = ingestion.get("rfdetr_inference_time_s", 0) or 0
            time_str  = f" ({t_s:.2f}s)" if t_s > 0 else ""
            with _res_c1:
                st.markdown(
                    f'<span class="badge {badge_cls}">{det_src}{time_str}</span>',
                    unsafe_allow_html=True,
                )
            with _res_c2:
                if _external_rfdetr_result.get("ok"):
                    _det_runtime_mode = (
                        _external_rfdetr_result.get("detector_runtime_mode")
                        or _external_rfdetr_result.get("source")
                        or "live_rfdetr_subprocess"
                    )
                    _runtime_lines = [
                        '<div class="info-card" style="border-color:rgba(59,130,246,0.45)">',
                        f'<strong>Detector runtime mode: {_det_runtime_mode}</strong><br>',
                        'Detector: RF-DETR<br>',
                    ]
                    if _det_runtime_mode == "live_rfdetr_http_service":
                        _runtime_lines.append(f'Service URL: <code>{_external_rfdetr_result.get("service_url","")}</code><br>')
                    else:
                        _runtime_lines.append(f'Python executable: <code>{_external_rfdetr_result.get("python_executable","")}</code><br>')
                    _runtime_lines.extend([
                        f'Checkpoint path: <code>{_external_rfdetr_result.get("checkpoint_path","")}</code><br>',
                        f'Inference runtime: {_external_rfdetr_result.get("inference_runtime_ms",0)} ms<br>',
                        f'Detection count: {len(_external_rfdetr_result.get("detections", []))}<br>',
                        f'Source: {_external_rfdetr_result.get("source","live_rfdetr_subprocess")}',
                    ])
                    if _external_rfdetr_result.get("fallback_reason"):
                        _runtime_lines.append(f'<br>Fallback reason: {_external_rfdetr_result.get("fallback_reason")}')
                    _runtime_lines.append('</div>')
                    st.markdown(
                        "".join(_runtime_lines),
                        unsafe_allow_html=True,
                    )
                    # Detection rows shown in Graph Memory Extracted → Devices tab below
                elif _external_rfdetr_result:
                    st.warning("Live RF-DETR unavailable — using verified annotation fallback")
                    st.caption(f"RF-DETR error: {_external_rfdetr_result.get('error', 'unknown error')}")
                    if _external_rfdetr_result.get("fallback_reason"):
                        st.caption(f"Fallback reason: {_external_rfdetr_result.get('fallback_reason')}")
                    st.markdown(
                        '<div class="warn-card">'
                        '<strong>Detector runtime mode: verified_annotation_fallback</strong><br>'
                        'Source: verified training annotation / safe curated sample'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                rfdetr_err = ingestion.get("rfdetr_error", "")
                if rfdetr_err:
                    st.warning(f"RF-DETR: {rfdetr_err}")
            pkt = ingestion.get("packet") or {}
            with _res_c3:
                st.markdown(
                    '<div class="info-card">'
                    '<strong>Live Diagram Intelligence Output</strong><br>'
                    f'source: <code>{pkt.get("source", pkt.get("detection_source", ""))}</code><br>'
                    f'runtime mode: <code>{pkt.get("runtime_mode", "")}</code><br>'
                    f'nodes detected: {pkt.get("node_count", 0)}<br>'
                    f'edges inferred: {pkt.get("edge_count", 0)}<br>'
                    f'graph packet ID: <code>{sample["sample_id"]}</code><br>'
                    f'absorption mode: <code>{pkt.get("absorption_mode", "SESSION_MEMORY_ABSORPTION")}</code><br>'
                    'refresh behavior: refresh resets onboarded graph'
                    '</div>',
                    unsafe_allow_html=True,
                )
                st.success(
                    f"Ingestion complete — {pkt.get('node_count', 0)} nodes, "
                    f"{pkt.get('edge_count', 0)} edges. "
                    "Proceed to Tab 2 (Topology RCA) or Tab 3 (Enterprise Brain)."
                )
            # Ingestion just completed — the selected sample is now active.
            # is_this_sample_active was computed before the button ran (stale),
            # so force it to True so _tab_diagram_outputs_section() renders below.
            is_this_sample_active = True

    # Only show graph outputs if LDI was actually run for the currently selected sample.
    # Stale local_graph from a previous run or catalog load must not bleed through.
    if is_this_sample_active:
        _tab_diagram_outputs_section()


def _selected_diagram_record() -> dict:
    """Return the best available selected diagram record."""
    onboard = st.session_state.get("onboard_sample_record") or {}
    if onboard.get("sample_id"):
        return onboard
    for key in ("catalog_selected_record", "selected_gallery_record", "active_gallery_record"):
        rec = st.session_state.get(key) or {}
        if rec.get("gallery_id") or rec.get("source_diagram_id"):
            return rec
    return {}


def _load_json_if_exists(path_str: str) -> dict:
    """Safely load a JSON file; returns {} on any failure."""
    if not path_str:
        return {}
    try:
        p = Path(path_str)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _render_evidence_tables(record: dict) -> None:
    """Render Graph Memory Extracted tabs for the given record."""
    # ── 1. Resolve graph_memory_packet (best case: written by ingestion run) ──
    packet: dict = {}
    run_dir_str = st.session_state.get("live_ingestion_run_dir", "")
    if run_dir_str:
        packet = _load_json_if_exists(str(Path(run_dir_str) / "graph_memory_packet.json"))
    if not packet:
        packet = st.session_state.get("validation_packet") or {}
    if not packet and record:
        for _pkey in ("local_graph_path", "enterprise_graph_path"):
            _lp = _resolve_manifest_path(record.get(_pkey, ""), REPO_ROOT)
            if _lp:
                _cand = Path(_lp).parent / "graph_memory_packet.json"
                if _cand.exists():
                    packet = _load_json_if_exists(str(_cand))
                    if packet:
                        break

    # ── 2. Get local_graph — session state first, then record path ────────────
    local_graph: dict = st.session_state.get("local_graph") or {}
    if not local_graph and record:
        _lp2 = _resolve_manifest_path(record.get("local_graph_path", ""), REPO_ROOT)
        if _lp2:
            local_graph = _load_json_if_exists(_lp2)

    # ── 3. Get annotation ─────────────────────────────────────────────────────
    _ann_p = _resolve_manifest_path(record.get("annotation_path", ""), REPO_ROOT) if record else ""
    annotation: dict = _load_json_if_exists(_ann_p)

    det_src = (
        packet.get("detection_source")
        or st.session_state.get("detection_source")
        or "Verified Annotation Overlay"
    )
    _is_overlay = "Verified Annotation" in det_src
    _ev_src     = "verified_metadata" if _is_overlay else "live_detector"

    # ── 4. Build rows — inline first (always works), enrich via helpers ───────
    if packet.get("devices"):
        device_rows    = packet["devices"]
        connector_rows = packet.get("connectors", [])
        interface_rows = packet.get("interfaces", [])
        ocr_rows       = packet.get("ocr_text", [])
    else:
        # Inline build — no external import needed, works immediately
        device_rows = [
            {
                "node_id":         n.get("id") or n.get("node_id") or "",
                "device_type":     n.get("type", ""),
                "display_label":   n.get("label") or n.get("id") or "",
                "canonical_id":    n.get("canonical_id") or n.get("id") or "",
                "ip_address":      n.get("ip_address", ""),
                "zone":            n.get("zone", ""),
                "interface":       n.get("interface", ""),
                "vlan":            n.get("vlan", ""),
                "is_shared_entity": n.get("is_shared_entity", False),
                "evidence_source": _ev_src,
                "confidence":      "" if _is_overlay else str(round(float(n["confidence"]), 3))
                                   if n.get("confidence") else "",
            }
            for n in (local_graph.get("nodes") or annotation.get("objects") or [])
            if n.get("id") or n.get("node_id") or n.get("object_id")
        ]
        connector_rows = [
            {
                "source":       e.get("source", ""),
                "target":       e.get("target", ""),
                "relationship": e.get("relationship", "connected_to"),
                "label":        e.get("label", ""),
                "protocol":     e.get("protocol", ""),
                "scope":        e.get("edge_scope", e.get("scope", "")),
                "evidence_source": _ev_src,
                "confidence":   "" if _is_overlay else str(round(float(e["confidence"]), 3))
                                if e.get("confidence") else "",
            }
            for e in (local_graph.get("edges") or annotation.get("connectors") or [])
            if e.get("source") and e.get("target")
        ]
        interface_rows = [
            {
                "node_id":    n.get("id") or n.get("node_id") or "",
                "device_type": n.get("type", ""),
                "ip_address": n.get("ip_address", ""),
                "interface":  n.get("interface", ""),
                "vlan":       n.get("vlan", ""),
                "zone":       n.get("zone", ""),
            }
            for n in (local_graph.get("nodes") or [])
            if n.get("id") or n.get("node_id")
        ]
        ocr_rows = [
            {
                "text":        blk.get("text", ""),
                "text_type":   blk.get("type", blk.get("role", "")),
                "linked_node": blk.get("linked_node", blk.get("node_id", "")),
            }
            for blk in (annotation.get("text_blocks") or [])
        ]
        # Optionally enrich with richer columns from runtime_ingestion helpers
        try:
            from runtime_ingestion import (  # type: ignore
                build_device_rows as _bdr, build_connector_rows as _bcr,
                build_interface_rows as _bir, build_ocr_rows as _bor,
            )
            device_rows    = _bdr(local_graph, annotation, det_src)
            connector_rows = _bcr(local_graph, annotation, det_src)
            interface_rows = _bir(local_graph, annotation)
            ocr_rows       = _bor(annotation)
        except (ImportError, AttributeError):
            pass  # inline rows above are already good enough

    if not device_rows and not connector_rows and not local_graph:
        return  # nothing to show — silent, no blocking message

    ev_src_label = {
        "verified_metadata": "Verified metadata view",
        "live_detector":     "Live RF-DETR Detector",
        "trained_detector":  "RF-DETR Trained Prediction",
        "inferred":          "Inferred from graph",
    }.get(packet.get("evidence_source", _ev_src), det_src)

    st.markdown(
        '<div class="section-label" style="margin-top:18px">Graph Memory Extracted</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Structured device, connector, interface and OCR evidence generated from the "
        "processed image — the graph-memory layer used by RCA, enterprise absorption, "
        f"and Copilot.  Detection source: **{det_src}**"
    )

    tab_dev, tab_conn, tab_iface, tab_ocr, tab_pkt = st.tabs(
        ["Devices", "Connectors", "Interfaces & IPs", "OCR / Text", "Graph Memory Packet"]
    )

    with tab_dev:
        if device_rows:
            df = pd.DataFrame(device_rows)
            # Drop empty/all-blank columns for cleaner display
            df = df.loc[:, (df != "").any(axis=0)]
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(
                f"{len(device_rows)} devices · evidence source: {ev_src_label} · "
                "confidence shown only for live detector outputs"
            )
        else:
            st.info("No device records available for this diagram.")

    with tab_conn:
        if connector_rows:
            df = pd.DataFrame(connector_rows)
            df = df.loc[:, (df != "").any(axis=0)]
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(f"{len(connector_rows)} connectors")
        else:
            st.info("No connector records available for this diagram.")

    with tab_iface:
        if interface_rows:
            df = pd.DataFrame(interface_rows)
            df = df.loc[:, (df != "").any(axis=0)]
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(f"{len(interface_rows)} interface / IP records")
        else:
            st.info("No interface / IP records available for this diagram.")

    with tab_ocr:
        if ocr_rows:
            df = pd.DataFrame(ocr_rows)
            df = df.loc[:, (df != "").any(axis=0)]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No OCR / text evidence available for this record.")

    with tab_pkt:
        counts = {
            "Devices":        len(device_rows),
            "Connectors":     len(connector_rows),
            "Interfaces":     len(interface_rows),
            "OCR blocks":     len(ocr_rows),
        }
        cA, cB, cC, cD = st.columns(4)
        for col, (lbl, val) in zip([cA, cB, cC, cD], counts.items()):
            col.metric(lbl, val)

        extra_c1, extra_c2 = st.columns(2)
        extra_c1.metric("Detection source",  det_src[:28])
        extra_c2.metric("Evidence source",   ev_src_label[:28])

        if packet:
            with st.expander("Full graph_memory_packet.json", expanded=False):
                st.json(packet)
        elif local_graph:
            with st.expander("Local graph summary", expanded=False):
                st.json({
                    "nodes": len(local_graph.get("nodes", [])),
                    "edges": len(local_graph.get("edges", [])),
                    "diagram_id": local_graph.get("diagram_id", ""),
                })


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

    _ctrl_l, _ctrl_r = st.columns(2)
    with _ctrl_l:
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

    # Auto-load local_graph on record selection so Topology RCA and Enterprise Brain
    # work immediately without requiring the "Load Graph Metadata" button click.
    _auto_lg_path = _resolve_manifest_path(record.get("local_graph_path", ""), REPO_ROOT)
    if _auto_lg_path and Path(_auto_lg_path).exists():
        try:
            _auto_lg = json.loads(Path(_auto_lg_path).read_text(encoding="utf-8"))
            st.session_state.local_graph = _auto_lg
            _auto_gmp = Path(_auto_lg_path).parent / "graph_memory_packet.json"
            if _auto_gmp.exists():
                st.session_state.validation_packet      = json.loads(
                    _auto_gmp.read_text(encoding="utf-8")
                )
                st.session_state.live_ingestion_run_dir = str(Path(_auto_lg_path).parent)
        except Exception:
            pass

    # ── Images: original + detection/annotation ───────────────────────────────
    img_p  = record.get("image_path", "")
    det_p  = record.get("detected_preview_path", "")
    ann_p  = record.get("annotation_path", "")
    is_v3  = record.get("source_dataset") == "v3"

    # Repair stale paths in case the record came from a manifest built on another machine
    img_p = _resolve_manifest_path(img_p, REPO_ROOT)
    det_p = _resolve_manifest_path(det_p, REPO_ROOT)
    ann_p = _resolve_manifest_path(ann_p, REPO_ROOT)

    img_exists = bool(img_p and Path(img_p).exists())
    det_exists = bool(det_p and Path(det_p).exists())
    ann_exists = bool(ann_p and Path(ann_p).exists())

    with _ctrl_r:
        if is_v3 and img_exists and ann_exists and not det_exists:
            show_overlay_connectors = st.checkbox(
                "Show connectors",
                value=False,
                key=f"show_connectors_{record.get('gallery_id', record.get('source_diagram_id', 'v3'))}",
                help="Draw subtle metadata connector lines on the verified overlay.",
            )
            st.caption(
                "Verified metadata view: node identity + device type. "
                "Confidence scores appear only on live detector outputs."
            )
        else:
            show_overlay_connectors = False

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="compare-badge original">Original</div>'
            '<div class="compare-label">Source Diagram</div>',
            unsafe_allow_html=True,
        )
        if img_exists:
            st.image(img_p, use_container_width=True)
        else:
            st.warning(f"Image not found: `{img_p}`")

    with c2:
        # Priority: a) trained detector preview  b) Verified Annotation Overlay  c) pending
        _overlay_path = ""
        _overlay_meta: dict = {}
        _render_err   = ""

        if det_exists:
            src_label = record.get("source_dataset", "").upper()
            st.markdown(
                f'<div class="compare-badge predicted">Trained Detector Output</div>'
                f'<div class="compare-label">{src_label} trained detector</div>',
                unsafe_allow_html=True,
            )
            st.image(det_p, use_container_width=True)
        elif is_v3 and img_exists and ann_exists:
            with st.spinner("Rendering annotation overlay…"):
                _overlay_path, _overlay_meta = _ensure_annotation_overlay(
                    record, REPO_ROOT,
                    img_p=Path(img_p),
                    ann_p=Path(ann_p),
                    draw_connectors=show_overlay_connectors,
                )
            _render_err = _overlay_meta.get("error", "")
            if _overlay_path and Path(_overlay_path).exists():
                n_obj = _overlay_meta.get("boxes_rendered", 0)
                n_con = _overlay_meta.get("connectors_rendered", 0)
                _cached = _overlay_meta.get("cached", False)
                _sub = f"{n_obj} objects"
                if show_overlay_connectors:
                    _sub += f", {n_con} connectors"
                if _cached:
                    _sub += " · cached"
                st.markdown(
                    '<div class="compare-badge prepared">Verified Annotation Overlay</div>'
                    f'<div class="compare-label">{_sub}</div>',
                    unsafe_allow_html=True,
                )
                st.image(_overlay_path, use_container_width=True)
            else:
                st.markdown(
                    '<div class="compare-badge missing">Detection overlay pending.</div>'
                    '<div class="compare-label">Load graph metadata or run live intelligence '
                    'from Onboard New Diagram.</div>',
                    unsafe_allow_html=True,
                )
                if img_exists:
                    st.image(img_p, use_container_width=True)
        else:
            st.markdown(
                '<div class="compare-badge missing">Detection overlay pending.</div>'
                '<div class="compare-label">Load graph metadata or run live intelligence '
                'from Onboard New Diagram.</div>',
                unsafe_allow_html=True,
            )
            if img_exists:
                st.image(img_p, use_container_width=True)

    # ── Overlay diagnostics (under Source details expander further below) ─────
    # Store for use in the expander; keyed by gallery_id so it survives re-render
    st.session_state[f"_diag_{record.get('gallery_id','')}"] = {
        "image_path":       img_p,
        "img_exists":       img_exists,
        "annotation_path":  ann_p,
        "ann_exists":       ann_exists,
        "overlay_path":     _overlay_path,
        "render_error":     _render_err,
        "renderer_imported": _render_ann_preview is not None,
        "render_meta":      _overlay_meta,
    }

    # ── Badges ────────────────────────────────────────────────────────────────
    has_graph = record.get("graph_metadata_available", False)
    has_conn  = record.get("connector_metadata_available", False)
    has_ocr   = record.get("ocr_metadata_available", False)
    has_ent   = record.get("enterprise_mapping_available", False)
    has_det   = det_exists
    has_ann   = ann_exists

    def _badge(label: str, ok: bool, ok_cls: str = "badge-success") -> str:
        cls = ok_cls if ok else "badge-warn"
        return f'<span class="badge {cls}">{label}</span>'

    badge_html = '<div style="display:flex;flex-wrap:wrap;gap:6px;margin:10px 0 14px">'
    if has_det:
        badge_html += '<span class="badge badge-success">RF-DETR Live Detection</span>'
    elif has_ann and is_v3:
        badge_html += '<span class="badge badge-info">Verified Annotation</span>'
    else:
        badge_html += '<span class="badge badge-warn">Detection Pending</span>'
    badge_html += _badge("Local Graph Ready" if (has_graph or has_conn) else "Local Graph Pending", has_graph or has_conn)
    badge_html += _badge("Enterprise Mapped" if has_ent else "Enterprise Pending", has_ent)
    badge_html += _badge("OCR Available" if has_ocr else "OCR Pending", has_ocr)
    badge_html += _badge("Annotated" if has_ann else "Annotation Missing", has_ann)
    badge_html += '</div>'
    st.markdown(badge_html, unsafe_allow_html=True)

    # ── Source details + overlay diagnostics expander ────────────────────────
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

        # Overlay diagnostics
        _diag = st.session_state.get(f"_diag_{record.get('gallery_id','')}", {})
        st.markdown("**Overlay diagnostics**")
        st.caption(f"image_path exists: `{_diag.get('img_exists', img_exists)}`")
        st.caption(f"annotation_path exists: `{_diag.get('ann_exists', ann_exists)}`")
        _diag_rend, _diag_rend_err = _load_renderer(REPO_ROOT)
        st.caption(f"renderer available: `{_diag_rend is not None}`")
        if _diag_rend_err:
            st.caption(f"renderer load error: `{_diag_rend_err}`")
        elif _RENDER_ANN_IMPORT_ERR:
            st.caption(f"module-level import note: `{_RENDER_ANN_IMPORT_ERR}`")
        if _diag.get("overlay_path"):
            st.caption(f"overlay_path: `{_diag['overlay_path']}`")
        elif _overlay_path:
            st.caption(f"overlay_path: `{_overlay_path}`")
        if _diag.get("render_error") or _render_err:
            st.caption(f"render error: `{_diag.get('render_error') or _render_err}`")

    # ── Graph Memory Evidence Tables ──────────────────────────────────────────
    _render_evidence_tables(record)
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
# TAB 2 — TOPOLOGY RCA
# ══════════════════════════════════════════════════════════════════════════════
def _tab_local_rca() -> None:
    st.markdown(
        '<div class="ws-title">Topology RCA — single-diagram graph reasoning</div>'
        '<div class="ws-desc">Simulate a realistic alert stream, then trace the root cause '
            'through one selected diagram topology. Enterprise GNN RCA handles cross-diagram graph reasoning.</div>',
        unsafe_allow_html=True,
    )

    local_graph = st.session_state.get("local_graph")
    if not local_graph:
        st.markdown(
            '<div class="warn-card">'
            'No diagram loaded yet.<br>'
            'Select a diagram from <strong>Tab 1 → Diagram Gallery</strong>, '
            'or go to <strong>Tab 1 → Onboard New Diagram</strong> and click '
            '<em>Run Live Diagram Intelligence</em>.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    diagram_id = st.session_state.get("selected_diagram_id", "unknown")
    sel_path   = st.session_state.get("selected_diagram_path", "")

    # ── Diagram summary header ─────────────────────────────────────────────────
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
            st.info("RCA source: Scenario-guided graph RCA")
            if _strict_mode() and not st.session_state.allow_local_simulation:
                st.error("Strict mode: explicitly approve before using scenario-guided RCA.")
                if st.button("Approve scenario-guided RCA", key="approve_local_sim"):
                    st.session_state.allow_local_simulation = True
                    st.rerun()
                return

    st.markdown('<hr class="ws-rule" style="margin:12px 0">', unsafe_allow_html=True)

    # ── Step 1: Generate Local Alert Stream ───────────────────────────────────
    incident = st.session_state.get("local_incident") or {}

    # Auto-clear stale incident stored by an older code path (empty timeline).
    if incident and not incident.get("alert_timeline") and _INCIDENT_SIM_OK:
        st.session_state.pop("local_incident", None)
        incident = {}

    col_b1, col_b2 = st.columns([1, 1])
    with col_b1:
        if st.button(
            "Generate Topology Alert Stream",
            type="primary" if not incident else "secondary",
            key="gen_local_alerts_btn",
            help="Analyse the diagram topology and produce a realistic alert timeline",
        ):
            with st.spinner("Analysing topology and generating alert stream…"):
                if _INCIDENT_SIM_OK and _build_local_incident_pkg:
                    inc = _build_local_incident_pkg(local_graph, diagram_id)
                else:
                    # Package not importable — derive from existing story builder
                    story = _build_local_incident_story(local_graph, diagram_id, None)
                    inc   = story
                    inc["alert_timeline"] = []
                st.session_state.local_incident             = inc
                st.session_state.local_rca_result           = {}
                st.session_state.pop("local_ai_resolution_plan", None)
            st.rerun()

    if incident:
        with col_b2:
            if st.button(
                "Find Topology Root Cause",
                type="primary",
                key="local_rca_btn",
                help="Run graph-traversal RCA and reveal the root cause node",
            ):
                with st.spinner("Running Topology RCA…"):
                    rca = _incident_to_local_rca(incident)
                    st.session_state.local_rca_result = rca
                    st.session_state.pop("local_ai_resolution_plan", None)
                    _persist_incident(incident, "local")
                st.rerun()

    # ── Alert timeline display ────────────────────────────────────────────────
    if incident:
        timeline = incident.get("alert_timeline", [])
        severity_inc = incident.get("severity", "High")
        sev_color = {"Critical": "#ef4444", "High": "#f59e0b",
                     "Medium": "#3b82f6", "Low": "#10b981"}.get(severity_inc, "#f59e0b")

        st.markdown(
            f'<div class="info-card" style="margin:12px 0 10px;border-left:4px solid {sev_color}">'
            f'<div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;'
            f'color:#94a3b8;margin-bottom:4px">Topology Incident</div>'
            f'<div style="font-size:1.05rem;font-weight:700;color:#f1f5f9;margin-bottom:8px">'
            f'{incident.get("incident_title","Topology Incident")}</div>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px 20px;font-size:0.78rem">'
            f'<div><span style="color:#64748b">Severity</span>&nbsp;'
            f'<span style="font-weight:700;color:{sev_color}">{severity_inc}</span></div>'
            f'<div><span style="color:#64748b">Suspected domain</span>&nbsp;'
            f'<span style="color:#e2e8f0">{incident.get("suspected_domain","—")}</span></div>'
            f'<div><span style="color:#64748b">First observed</span>&nbsp;'
            f'<code style="font-size:0.75rem;color:#67e8f9">'
            f'{incident.get("first_observed_node","—")}</code></div>'
            f'<div><span style="color:#64748b">Alert summary</span>&nbsp;'
            f'<span style="color:#94a3b8">{incident.get("alert_summary","")[:80]}</span></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        if timeline:
            st.markdown(
                '<div class="section-label" style="margin-top:10px">Alert Stream</div>',
                unsafe_allow_html=True,
            )
            _render_alert_timeline(timeline, show_diagram_col=False)
        else:
            st.caption("Alert timeline will appear here once generated.")

    # ── Step 2 results: RCA output ─────────────────────────────────────────────
    result = st.session_state.get("local_rca_result")
    if not result:
        if not incident:
            st.info("Click **Generate Topology Alert Stream** to begin the incident simulation.")
        else:
            st.info("Click **Find Topology Root Cause** to run RCA on this alert stream.")
        return

    st.markdown('<hr class="ws-rule">', unsafe_allow_html=True)

    severity  = result.get("severity", "High")
    sev_color = {"Critical": "#ef4444", "High": "#f59e0b",
                 "Medium": "#3b82f6", "Low": "#10b981"}.get(severity, "#f59e0b")

    # ── Root cause ─────────────────────────────────────────────────────────────
    root = result.get("root_cause", "—")
    st.markdown(
        f'<div class="rca-winner">'
        f'<div class="rca-winner-title">Root Cause Identified</div>'
        f'<div class="rca-winner-sub">'
        f'RCA source: {result.get("rca_source","Scenario-guided graph RCA")}</div>'
        f'<div class="rca-winner-node">{root}</div>'
        f'<div class="rca-winner-meta">'
        f'Severity: {severity} &nbsp;·&nbsp; '
        f'First observed: {result.get("first_observed","—")} &nbsp;·&nbsp; '
        f'Impacted: {len(result.get("impacted_nodes",[]))}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    why = result.get("why_root", "")
    if why:
        st.markdown(
            f'<div style="font-size:0.8rem;color:#94a3b8;margin-bottom:10px">'
            f'<strong style="color:#cbd5e1">Why this root:</strong> {why}</div>',
            unsafe_allow_html=True,
        )

    # Reasoning steps
    with st.expander("Reasoning steps", expanded=True):
        for step in result.get("reasoning_steps", []):
            st.markdown(
                f'<div style="font-size:0.8rem;color:#cbd5e1;padding:3px 0 3px 8px;'
                f'border-left:2px solid #3b82f6;margin-bottom:4px">{step}</div>',
                unsafe_allow_html=True,
            )

    # Do not render RCA recommended_actions here.
    # Remediation must come from the AI Resolution Agent to avoid hardcoded-looking output.

    st.markdown('<hr class="ws-rule" style="margin:16px 0">', unsafe_allow_html=True)

    # ── Metrics ────────────────────────────────────────────────────────────────
    _first_obs  = result.get("first_observed", "—")
    _n_impacted = len(result.get("impacted_nodes", []))
    _sev_color  = {"Critical": "#f87171", "High": "#fb923c", "Medium": "#fbbf24"}.get(severity, "#94a3b8")
    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:4px">'
        f'<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);'
        f'border-radius:8px;padding:10px 14px">'
        f'<div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.08em;'
        f'color:#64748b;margin-bottom:4px">Root Cause</div>'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.95rem;font-weight:700;'
        f'color:#f1f5f9;word-break:break-all;overflow-wrap:anywhere;line-height:1.3">{root}</div>'
        f'</div>'
        f'<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);'
        f'border-radius:8px;padding:10px 14px">'
        f'<div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.08em;'
        f'color:#64748b;margin-bottom:4px">First Observed</div>'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.95rem;font-weight:700;'
        f'color:#f1f5f9;word-break:break-all;overflow-wrap:anywhere;line-height:1.3">{_first_obs}</div>'
        f'</div>'
        f'<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);'
        f'border-radius:8px;padding:10px 14px">'
        f'<div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.08em;'
        f'color:#64748b;margin-bottom:4px">Impacted Nodes</div>'
        f'<div style="font-size:1.5rem;font-weight:700;color:#f1f5f9;line-height:1.3">{_n_impacted}</div>'
        f'</div>'
        f'<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);'
        f'border-radius:8px;padding:10px 14px">'
        f'<div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.08em;'
        f'color:#64748b;margin-bottom:4px">Severity</div>'
        f'<div style="font-size:1.1rem;font-weight:700;color:{_sev_color};line-height:1.3">{severity}</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Graph overlay ──────────────────────────────────────────────────────────
    path = result.get("impact_path", [])
    path_disp = " → ".join(path[:6]) + ("…" if len(path) > 6 else "")
    st.markdown('<div class="section-label">Local Graph RCA Overlay</div>', unsafe_allow_html=True)
    if path_disp:
        st.markdown(
            f'<div style="font-size:0.78rem;color:#94a3b8;margin-bottom:6px">'
            f'RCA path: <code style="font-size:0.75rem;color:#67e8f9">{path_disp}</code></div>',
            unsafe_allow_html=True,
        )
    st.caption(
        "Root cause: red  |  First observed: orange  |  "
        "Impacted: yellow  |  Impact path: cyan  |  Shared entities: cyan ring"
    )
    if not _pyvis_available():
        st.markdown(
            '<div class="warn-card" style="margin-bottom:8px">'
            'Install <code>pyvis</code> for interactive drag-and-drop graph.</div>',
            unsafe_allow_html=True,
        )
    _render_local_graph(local_graph, result)

    # ── Traversal ──────────────────────────────────────────────────────────────
    _render_traversal_steps(path, result, slider_key="local_traversal_slider")

    # ── Impact path chips ──────────────────────────────────────────────────────
    if path:
        st.markdown(
            '<div class="section-label" style="margin-top:14px">Impact Path</div>',
            unsafe_allow_html=True,
        )
        root_id   = result.get("root_cause", "")
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

    # ── Ranking table ──────────────────────────────────────────────────────────
    ranking = result.get("ranking", [])
    if ranking:
        st.markdown(
            '<div class="section-label" style="margin-top:14px">RCA Candidate Ranking</div>',
            unsafe_allow_html=True,
        )
        df = pd.DataFrame(ranking)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Developer details ──────────────────────────────────────────────────────
    with st.expander("Developer details", expanded=False):
        st.caption(f"RCA mode: {result.get('mode', '—')}")
        st.caption("Graph traversal: BFS undirected (first_obs→root) + directed downstream BFS")
        st.caption("Scoring: out-degree × 0.11 + zero-in-degree +0.14 + root-match +0.20")
        if result.get("path_note"):
            st.caption(f"Path note: {result['path_note']}")

    # ── AI Enterprise Expansion ────────────────────────────────────────────────
    if st.session_state.get("enterprise_absorbed"):
        st.markdown('<hr class="ws-rule" style="margin:12px 0">', unsafe_allow_html=True)
        _ent_inc_exp  = st.session_state.get("enterprise_incident") or {}
        _ent_rca_exp  = st.session_state.get("enterprise_rca_result") or {}
        _ent_summ_exp = st.session_state.get("enterprise_ingestion_summary") or {}
        _ent_scen_exp = (
            _ent_inc_exp.get("scenario_id")
            or _ent_rca_exp.get("scenario_id")
            or _ent_summ_exp.get("scenario_id")
            or ""
        )
        _gnn_exp = _load_gnn_rca_result(_ent_scen_exp) if _ent_scen_exp else None
        _ent_rc_gnn    = (_ent_rca_exp.get("root_cause")
                          or (_gnn_exp or {}).get("predicted_root_cause", ""))
        _ent_rc_diag   = (_ent_rca_exp.get("root_cause_diagram")
                          or (_gnn_exp or {}).get("root_cause_diagram", ""))
        _ent_imp_diags = (_ent_rca_exp.get("impacted_diagrams")
                          or (_gnn_exp or {}).get("impacted_diagrams", []))
        _local_rc_exp  = result.get("root_cause", "")
        _outside       = bool(
            _ent_rc_gnn
            and _ent_rc_gnn != _local_rc_exp
            and _ent_rc_diag
            and _ent_rc_diag != diagram_id
        )
        _ent_exp_hdr = (
            "AI-expanded Enterprise RCA — Root cause outside this diagram"
            if _outside
            else "AI Enterprise Expansion"
        )
        st.markdown(
            f'<div class="section-label" style="margin-bottom:8px">{_ent_exp_hdr}</div>',
            unsafe_allow_html=True,
        )
        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            st.markdown(
                f'<div style="background:rgba(15,23,42,0.5);border:1px solid rgba(148,163,184,0.15);'
                f'border-left:3px solid #22c55e;border-radius:8px;padding:10px 14px">'
                f'<div style="font-size:0.6rem;text-transform:uppercase;color:#64748b;margin-bottom:3px">'
                f'Topology root cause</div>'
                f'<div style="font-family:monospace;font-size:0.9rem;font-weight:700;color:#22c55e">'
                f'{html.escape(_local_rc_exp or "—")}</div>'
                f'<div style="font-size:0.7rem;color:#475569;margin-top:2px">Diagram: {html.escape(str(diagram_id))}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_ex2:
            _ent_rc_color = "#f59e0b" if _outside else "#10b981"
            st.markdown(
                f'<div style="background:rgba(15,23,42,0.5);border:1px solid rgba(148,163,184,0.15);'
                f'border-left:3px solid {_ent_rc_color};border-radius:8px;padding:10px 14px">'
                f'<div style="font-size:0.6rem;text-transform:uppercase;color:#64748b;margin-bottom:3px">'
                f'Enterprise root cause</div>'
                f'<div style="font-family:monospace;font-size:0.9rem;font-weight:700;color:{_ent_rc_color}">'
                f'{html.escape(_ent_rc_gnn or "—")}</div>'
                f'<div style="font-size:0.7rem;color:#475569;margin-top:2px">'
                f'Diagram: {html.escape(_ent_rc_diag or "—")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        if _outside:
            st.warning(
                f"Root cause outside starting diagram: Enterprise GNN RCA identified "
                f"**{_ent_rc_gnn}** (in **{_ent_rc_diag}**) as the cross-diagram root cause. "
                f"Proceed to **Enterprise GNN RCA** tab for full cross-diagram investigation.",
            )
        if _ent_imp_diags:
            _diag_chips = " ".join(
                f'<span class="diag-label">{html.escape(str(d))}</span>'
                for d in _ent_imp_diags[:6]
            )
            st.markdown(
                f'<div style="font-size:0.75rem;color:#94a3b8;margin-top:8px">'
                f'<span style="color:#64748b">Impacted diagrams (enterprise):</span> {_diag_chips}</div>',
                unsafe_allow_html=True,
            )
        elif not _ent_rc_gnn:
            st.caption(
                "Enterprise graph absorbed. Run Enterprise GNN RCA in the Enterprise GNN RCA tab "
                "to see the cross-diagram root cause analysis."
            )

    # ── AI Resolution Agent (Local) ────────────────────────────────────────────
    st.markdown('<hr class="ws-rule" style="margin:22px 0 14px 0">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label" style="margin-bottom:10px">AI Resolution Agent — Topology Remediation</div>',
        unsafe_allow_html=True,
    )

    if not _AI_REM_OK:
        st.warning("AI remediation package unavailable — check src/ai_remediation/.")
    else:
        _loc_vllm_ok     = _check_vllm_available()
        _loc_lora_exists = bool(_LORA_ADAPTER and Path(_LORA_ADAPTER).exists())
        _loc_rem_src     = "InfraGraph LoRA via vLLM" if _loc_vllm_ok else "Template fallback"
        _loc_plan        = st.session_state.get("local_ai_resolution_plan")

        # Status grid
        st.markdown(
            f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:14px">'
            f'<div style="background:rgba(30,41,59,0.7);border:1px solid rgba(100,116,139,0.3);'
            f'border-radius:8px;padding:10px 12px">'
            f'<div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.08em;'
            f'color:#64748b;margin-bottom:4px">RCA Scope</div>'
            f'<div style="font-size:0.8rem;font-weight:600;color:#7dd3fc">Topology (single-diagram)</div></div>'
            f'<div style="background:rgba(30,41,59,0.7);border:1px solid rgba(100,116,139,0.3);'
            f'border-radius:8px;padding:10px 12px">'
            f'<div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.08em;'
            f'color:#64748b;margin-bottom:4px">RCA Source</div>'
            f'<div style="font-size:0.8rem;font-weight:600;color:#94a3b8">'
            f'{result.get("rca_source", result.get("mode","Topology BFS RCA"))}</div></div>'
            f'<div style="background:rgba(30,41,59,0.7);border:1px solid rgba(100,116,139,0.3);'
            f'border-radius:8px;padding:10px 12px">'
            f'<div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.08em;'
            f'color:#64748b;margin-bottom:4px">Qwen / vLLM</div>'
            f'<div style="font-size:0.8rem;font-weight:600;'
            f'color:{"#34d399" if _loc_vllm_ok else "#f87171"}">'
            f'{"Available" if _loc_vllm_ok else "Unavailable"}</div></div>'
            f'<div style="background:rgba(30,41,59,0.7);border:1px solid rgba(100,116,139,0.3);'
            f'border-radius:8px;padding:10px 12px">'
            f'<div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.08em;'
            f'color:#64748b;margin-bottom:4px">Fine-tuned Adapter</div>'
            f'<div style="font-size:0.8rem;font-weight:600;'
            f'color:{"#34d399" if _loc_lora_exists else "#f87171"}">'
            f'{"Loaded" if _loc_lora_exists else "Not loaded"}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if not _loc_vllm_ok:
            st.info(
                "Qwen3 vLLM server is not reachable — generating from **Template fallback**. "
                "Start a local vLLM server and set `INFRAGRAPH_QWEN_BASE_URL` to enable model inference.",
                icon="ℹ️",
            )
        if not _loc_lora_exists:
            st.caption(
                "SOP-grounded LoRA adapter not detected. "
                "Train with `scripts/train_qwen_sop_lora.py` or set `INFRAGRAPH_LORA_ADAPTER_PATH`."
            )

        # Action buttons
        _col_la, _col_lb = st.columns([1, 1])
        with _col_la:
            _loc_btn_lbl = (
                "Generate Topology AI Resolution Plan"
                if _loc_vllm_ok
                else "Generate Topology Template Resolution Plan"
            )
            if st.button(_loc_btn_lbl, key="local_ai_plan_btn", type="primary"):
                with st.spinner("Building resolution plan…"):
                    _loc_ctx = _build_local_remediation_context(result, incident, local_graph, diagram_id)
                    if _loc_ctx and _generate_resolution_plan is not None:
                        _root = (result or {}).get("root_cause", "")
                        _query = f"root cause {_root} diagram {diagram_id} impacted nodes remediation validation"
                        _vec_evidence, _vec_err = _retrieve_vector_evidence(_query, k=6)
                        st.session_state.last_local_ai_vector_evidence_count = len(_vec_evidence)
                        if _vec_evidence:
                            _loc_ctx["retrieved_graph_memory_evidence"] = _vec_evidence
                            _loc_ctx["retrieved_graph_memory_label"] = "Retrieved graph memory evidence"
                        _loc_plan_result = _generate_resolution_plan(
                            _loc_ctx,
                            scope="local",
                            prefer_qwen=_loc_vllm_ok,
                            base_url=_QWEN_BASE_URL,
                            model=_QWEN_MODEL,
                            timeout=_QWEN_TIMEOUT,
                        )
                        import datetime as _dt
                        _loc_plan_result["_generated_at"] = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.local_ai_resolution_plan = _loc_plan_result
                    else:
                        st.error("Could not build remediation context — RCA result may be incomplete.")
                st.rerun()

        with _col_lb:
            if _loc_plan and st.button(
                "View ITSM Ticket",
                key="local_itsm_btn",
            ):
                _loc_resp = _loc_plan.get("response", {})
                _loc_itsm = _loc_resp.get("itsm_ticket_summary", {})
                if isinstance(_loc_itsm, dict) and _loc_itsm.get("short_description"):
                    st.info(
                        f"**{_loc_itsm.get('short_description','')}**\n\n"
                        f"{_loc_itsm.get('description','')}\n\n"
                        f"Priority: {_loc_itsm.get('priority','')} | "
                        f"Assignment: {_loc_itsm.get('assignment_group','')}"
                    )
                elif _loc_itsm:
                    st.code(str(_loc_itsm), language="text")

        # Honesty banner + provenance proof
        if _loc_plan:
            _loc_src = _loc_plan.get("source", "")
            if _loc_src in ("template", "template_fallback"):
                _fb_detail = (
                    " Qwen returned an error — see Qwen Runtime Proof for details."
                    if _loc_src == "template_fallback"
                    else " Connect a vLLM server to enable Qwen3 AI inference."
                )
                st.markdown(
                    '<div style="background:rgba(251,191,36,0.07);border:1px solid rgba(251,191,36,0.3);'
                    'border-radius:8px;padding:10px 14px;margin:10px 0;font-size:0.8rem;color:#fbbf24">'
                    f'Template output — deterministic, not AI-generated.{html.escape(_fb_detail)}'
                    '</div>',
                    unsafe_allow_html=True,
                )
            if _loc_plan.get("qwen_error"):
                st.warning(f"Qwen error (caused fallback): {_loc_plan['qwen_error']}")
            elif _loc_plan.get("error"):
                st.warning(f"Inference error: {_loc_plan['error']}")
            _render_remediation_plan(_loc_plan)
            _render_qwen_runtime_proof(_loc_plan)
            _render_ai_pipeline_trace(
                selected_diagram=str(diagram_id),
                selected_scenario=str((incident or {}).get("scenario_id", "")),
                topology_rca_completed=True,
                enterprise_gnn_available=False,
                rca_source=str((result or {}).get("mode") or (result or {}).get("rca_source") or "Topology BFS RCA"),
                root_cause=str((result or {}).get("root_cause", "")),
                impacted_diagrams=[diagram_id],
                vector_evidence_count=int(st.session_state.get("last_local_ai_vector_evidence_count", 0)),
                response_source=str(_loc_plan.get("source") or "—"),
            )


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


def _selected_enterprise_record() -> dict:
    """Return the best available enterprise context record.

    Priority:
    1. onboard_sample_record (set by live ingestion — has sample_id)
    2. catalog_selected_record (set by Gallery selection — has gallery_id)
    """
    onboard_rec = st.session_state.get("onboard_sample_record") or {}
    if onboard_rec.get("sample_id"):
        return onboard_rec
    catalog_rec = st.session_state.get("catalog_selected_record") or {}
    if catalog_rec.get("gallery_id"):
        return catalog_rec
    return {}


def _selected_scenario_path() -> "Path | None":
    """Return the Path of the selected scenario directory.

    Priority:
    1. enterprise_scenario_path in session state (set on absorption or onboarding)
    2. source_scenario_path from the selected enterprise record
    3. None — caller must handle, do NOT silently substitute V3_HERO_SCENARIO
    """
    ss_path = st.session_state.get("enterprise_scenario_path", "")
    if ss_path:
        p = Path(ss_path)
        if p.exists():
            return p
    rec = _selected_enterprise_record()
    sp = rec.get("source_scenario_path", "")
    if sp:
        rp = _resolve_manifest_path(sp, REPO_ROOT)
        if rp and Path(rp).exists():
            return Path(rp)
    return None


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
            '<div class="warn-card">No diagram loaded yet — select one from '
            '<strong>Tab 1 → Diagram Gallery</strong>, or go to '
            '<strong>Tab 1 → Onboard New Diagram</strong> and click '
            '<em>Run Live Diagram Intelligence</em>.</div>',
            unsafe_allow_html=True,
        )
        return

    diagram_id    = st.session_state.get("selected_diagram_id", "unknown")
    ent_rec       = _selected_enterprise_record()
    absorbed      = bool(st.session_state.get("enterprise_absorbed"))

    _ent_p    = _resolve_manifest_path(ent_rec.get("enterprise_graph_path", ""), REPO_ROOT)
    _stitch_p = _resolve_manifest_path(ent_rec.get("stitch_map_path", ""), REPO_ROOT)
    _alerts_p = _resolve_manifest_path(ent_rec.get("alerts_path", ""), REPO_ROOT)
    _run_id   = ent_rec.get("sample_id") or ent_rec.get("gallery_id") or diagram_id

    if not ent_rec:
        st.markdown(
            '<div class="warn-card">'
            'No enterprise record selected. Select a record from the '
            '<strong>Diagram Gallery</strong> or use '
            '<strong>Tab 1 (Diagram Intelligence) → Onboard New Diagram</strong> '
            'to run live ingestion.</div>',
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
        f'&nbsp;<span class="badge {"badge-success" if det_src == "LIVE_RFDETR_INFERENCE" else "badge-warn"}">'
        f'{det_src}</span>'
        f'<div style="font-size:0.75rem;color:#64748b;margin-top:4px">'
        f'{len(local_graph.get("nodes",[]))} nodes · {len(local_graph.get("edges",[]))} edges'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    if not absorbed:
        _src_diag = ent_rec.get("source_diagram_id") or diagram_id
        st.info(
            f"This graph is ready for enterprise absorption — enterprise paths found for: **{_src_diag}**"
        )
        if not _RUNTIME_INGESTION:
            _err_detail = f"  \n`{_RUNTIME_INGESTION_ERR}`" if _RUNTIME_INGESTION_ERR else ""
            st.error(f"runtime_ingestion failed to load at startup — restart the app to retry.{_err_detail}")
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
            st.session_state.enterprise_graph_before      = absorb_result["enterprise_before"]
            st.session_state.enterprise_graph_after       = absorb_result["enterprise_after"]
            st.session_state.enterprise_ingestion_summary = summary
            _src_scen = st.session_state.get("enterprise_scenario_path", "") or str(V3_HERO_SCENARIO)
            st.session_state.enterprise_scenario_path     = _src_scen
            st.session_state.enterprise_absorbed          = True
            st.session_state.enterprise_alerts_path       = _alerts_p
            st.session_state.enterprise_graph_path        = _ent_p
            st.session_state.enterprise_stitch_map_path   = _stitch_p
            st.session_state.allow_enterprise_simulation  = False
            st.session_state.enterprise_rca_result        = {}
            st.session_state.enterprise_incident          = {}
            st.rerun()
        return

    # ── Post-absorption view ──────────────────────────────────────────────────
    summary          = st.session_state.enterprise_ingestion_summary or {}
    enterprise_graph = st.session_state.enterprise_graph_after or {}
    _sel_scen_p   = _selected_scenario_path()
    _alerts_src = (
        st.session_state.get("enterprise_alerts_path")
        or _alerts_p
        or (str(_sel_scen_p / "alerts.json") if _sel_scen_p else None)
    )
    alerts_data = _safe_read_json(Path(_alerts_src)) if _alerts_src else {}

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
        _steps_html = [
            f'<div class="absorb-step">'
            f'<span class="absorb-done">✓</span>'
            f'<span style="font-size:0.79rem;color:#cbd5e1">{s}</span>'
            f'</div>'
            for s in [
                "Entity matching against canonical IDs",
                "Shared nodes identified across diagrams",
                "Cross-diagram links created from stitch map",
                "Local graph nodes + edges absorbed",
                "Enterprise memory updated",
            ]
        ]
        st.markdown(
            '<div class="absorb-card">'
            '<div class="section-label">Absorption Steps</div>'
            + "".join(_steps_html)
            + f'<div style="margin-top:12px;padding:8px 10px;background:rgba(16,185,129,0.08);'
            f'border-radius:8px;border:1px solid rgba(16,185,129,0.2)">'
            f'<div style="font-size:0.72rem;font-weight:700;color:#10b981">Result</div>'
            f'<div style="font-size:0.78rem;color:#94a3b8;margin-top:4px">'
            f'Nodes absorbed: <strong style="color:#f1f5f9">{summary.get("nodes_absorbed",0)}</strong><br>'
            f'Edges absorbed: <strong style="color:#f1f5f9">{summary.get("edges_absorbed",0)}</strong><br>'
            f'Shared matched: <strong style="color:#38bdf8">{summary.get("shared_entities_matched",0)}</strong><br>'
            f'Cross-diag links: <strong style="color:#22d3ee">{summary.get("cross_diagram_links_created",0)}</strong><br>'
            f'Before nodes: <strong style="color:#94a3b8">{summary.get("before_node_count",0)}</strong> '
            f'→ After: <strong style="color:#10b981">{summary.get("after_node_count",0)}</strong>'
            f'</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with c_enterprise:
        eg_stats = enterprise_graph.get("stats", {})
        n_nodes  = eg_stats.get("num_nodes")     or len(enterprise_graph.get("nodes", []))
        n_edges  = eg_stats.get("num_edges")     or len(enterprise_graph.get("edges", []))
        n_cross  = eg_stats.get("num_cross_diagram_edges") or len(enterprise_graph.get("cross_diagram_edges", []))
        n_clust  = len(enterprise_graph.get("diagram_clusters", {}))
        n_shared = eg_stats.get("num_shared_entities") or len(enterprise_graph.get("shared_entities", []))
        st.markdown(
            f'<div class="absorb-card">'
            f'<div class="section-label">Scenario Enterprise Graph</div>'
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

    # ── Global InfraGraph Galaxy ──────────────────────────────────────────────
    _GLOBAL_GRAPH_PATH   = _runtime_path("global_graph_memory") / "infragraph_global_graph.json"
    _GLOBAL_SUMMARY_PATH = _runtime_path("global_graph_memory") / "summary.json"
    _GLOBAL_EDGES_CSV    = _runtime_path("global_graph_memory") / "edges.csv"
    _GLOBAL_NODES_CSV    = _runtime_path("global_graph_memory") / "nodes.csv"
    with st.expander("Global InfraGraph Galaxy", expanded=False):
        st.markdown(
            '<div style="font-size:0.75rem;color:#64748b;margin-bottom:10px">'
            'Global graph-memory index across all V3 scenarios. '
            'Used for exploration and cross-scenario evidence — not for per-scenario GNN inference.'
            '</div>',
            unsafe_allow_html=True,
        )
        if not _GLOBAL_GRAPH_PATH.exists():
            st.info(
                "Global graph not built yet. Run:\n\n"
                "```\npython scripts/build_global_infragraph_galaxy.py "
                "--dataset-root ./datasets/infragraph_v3 "
                "--out ./runtime_state/global_graph_memory\n```"
            )
        else:
            _gsum = _safe_read_json(_GLOBAL_SUMMARY_PATH) if _GLOBAL_SUMMARY_PATH.exists() else {}
            _gm1, _gm2, _gm3, _gm4 = st.columns(4)
            _gm1.metric("Scenarios",       _gsum.get("total_scenarios", "—"))
            _gm2.metric("Total nodes",     _gsum.get("total_nodes", "—"))
            _gm3.metric("Total edges",     _gsum.get("total_edges", "—"))
            _gm4.metric("Cross-diag edges", _gsum.get("total_cross_diagram_edges", "—"))

            _nt = _gsum.get("node_type_counts", {})
            _dt = _gsum.get("diagram_type_counts", {})
            if _nt or _dt:
                _gc1, _gc2 = st.columns(2)
                with _gc1:
                    st.caption("Node types")
                    st.dataframe(
                        pd.DataFrame(list(_nt.items()), columns=["type", "count"]),
                        use_container_width=True, hide_index=True,
                    )
                with _gc2:
                    st.caption("Diagram types")
                    st.dataframe(
                        pd.DataFrame(list(_dt.items()), columns=["diagram_type", "count"]),
                        use_container_width=True, hide_index=True,
                    )

            # ── Load graph data ───────────────────────────────────────────
            _galaxy_data = _safe_read_json(_GLOBAL_GRAPH_PATH)
            _all_gnodes  = _galaxy_data.get("nodes", [])
            _all_gedges  = _galaxy_data.get("edges", [])
            _global_n    = len(_all_gnodes)
            _global_e    = len(_all_gedges)

            # ── Graph controls ────────────────────────────────────────────
            if _pyvis_available():
                _gf1, _gf2, _gf3, _gf4, _gf5 = st.columns([2, 2, 2, 2, 2])
                _splits    = ["all"] + sorted({n.get("split", "")        for n in _all_gnodes if n.get("split")})
                _ntypes    = ["all"] + sorted({n.get("type", "")         for n in _all_gnodes if n.get("type")})
                _dtypes    = ["all"] + sorted({n.get("diagram_type", "") for n in _all_gnodes if n.get("diagram_type")})
                _filt_split = _gf1.selectbox("Filter split",        _splits, key="galaxy_split")
                _filt_ntype = _gf2.selectbox("Filter node type",    _ntypes, key="galaxy_ntype")
                _filt_dtype = _gf3.selectbox("Filter diagram type", _dtypes, key="galaxy_dtype")
                _cross_only = _gf4.checkbox("Cross-diagram edges only", value=False, key="galaxy_cross_only")
                _max_nodes  = _gf5.number_input(
                    "Max nodes", min_value=50, max_value=2500,
                    value=500, step=50, key="galaxy_max_nodes",
                )

                # ── Filter nodes ──────────────────────────────────────────
                _filtered_nodes = [
                    n for n in _all_gnodes
                    if (_filt_split == "all" or n.get("split")        == _filt_split)
                    and (_filt_ntype == "all" or n.get("type")         == _filt_ntype)
                    and (_filt_dtype == "all" or n.get("diagram_type") == _filt_dtype)
                ]
                _filt_node_ids = {n["global_node_id"] for n in _filtered_nodes}

                # ── Filter edges ──────────────────────────────────────────
                _filtered_edges = [
                    e for e in _all_gedges
                    if e.get("source_global_id") in _filt_node_ids
                    and e.get("target_global_id") in _filt_node_ids
                    and (not _cross_only or e.get("edge_scope") == "cross_diagram")
                ]

                # If cross-diagram only, restrict nodes to those with such edges
                if _cross_only:
                    _cross_nids = set()
                    for _ce in _filtered_edges:
                        _cross_nids.add(_ce.get("source_global_id", ""))
                        _cross_nids.add(_ce.get("target_global_id", ""))
                    _filtered_nodes = [n for n in _filtered_nodes if n["global_node_id"] in _cross_nids]
                    _filt_node_ids  = {n["global_node_id"] for n in _filtered_nodes}

                # ── Rendered set (capped by Max nodes) ───────────────────
                _rendered_nodes = _filtered_nodes[:int(_max_nodes)]
                _rendered_ids   = {n["global_node_id"] for n in _rendered_nodes}
                _rendered_edges = [
                    e for e in _filtered_edges
                    if e.get("source_global_id") in _rendered_ids
                    and e.get("target_global_id") in _rendered_ids
                ]

                # ── Count display ─────────────────────────────────────────
                _cnt1, _cnt2, _cnt3, _cnt4, _cnt5, _cnt6 = st.columns(6)
                _cnt1.metric("Global nodes",   f"{_global_n:,}")
                _cnt2.metric("Global edges",   f"{_global_e:,}")
                _cnt3.metric("Filtered nodes", f"{len(_filtered_nodes):,}")
                _cnt4.metric("Filtered edges", f"{len(_filtered_edges):,}")
                _cnt5.metric("Rendered nodes", f"{len(_rendered_nodes):,}")
                _cnt6.metric("Rendered edges", f"{len(_rendered_edges):,}")

                if len(_rendered_nodes) > 1000:
                    st.caption(
                        "Large graph: labels appear when zoomed in. "
                        "Reduce Max nodes for a lighter view."
                    )

                # ── Build PyVis graph ─────────────────────────────────────
                if _rendered_nodes:
                    try:
                        from pyvis.network import Network as _GalNet  # type: ignore
                        _is_large = len(_rendered_nodes) > 500
                        _gnet = _GalNet(
                            height="620px", width="100%", directed=True,
                            bgcolor="#0b1220", font_color="#e2e8f0",
                        )
                        if _is_large:
                            _gnet.set_options("""{
                                "physics": {
                                    "enabled": true,
                                    "barnesHut": {
                                        "gravitationalConstant": -4000,
                                        "centralGravity": 0.3,
                                        "springLength": 80,
                                        "springConstant": 0.04,
                                        "damping": 0.2,
                                        "avoidOverlap": 0
                                    },
                                    "stabilization": {
                                        "enabled": true,
                                        "iterations": 100,
                                        "updateInterval": 10,
                                        "fit": true
                                    },
                                    "maxVelocity": 50,
                                    "minVelocity": 2,
                                    "timestep": 0.5
                                },
                                "nodes": {
                                    "size": 5,
                                    "font": {"size": 14, "color": "#e2e8f0"}
                                },
                                "edges": {"width": 0.5, "smooth": {"enabled": false}},
                                "interaction": {
                                    "tooltipDelay": 200,
                                    "hideEdgesOnDrag": true,
                                    "navigationButtons": false
                                }
                            }""")
                        else:
                            _gnet.set_options("""{
                                "physics": {
                                    "enabled": true,
                                    "barnesHut": {
                                        "gravitationalConstant": -3000,
                                        "centralGravity": 0.15,
                                        "springLength": 160,
                                        "springConstant": 0.04
                                    }
                                },
                                "nodes": {
                                    "size": 8,
                                    "font": {"size": 14, "color": "#e2e8f0"}
                                }
                            }""")

                        _SCEN_COLORS = [
                            "#38bdf8", "#10b981", "#f59e0b", "#a78bfa", "#f472b6",
                            "#34d399", "#fb923c", "#60a5fa", "#e879f9", "#4ade80",
                        ]
                        _scen_list  = sorted({n.get("scenario_id", "") for n in _rendered_nodes})
                        _scen_color = {s: _SCEN_COLORS[i % len(_SCEN_COLORS)]
                                       for i, s in enumerate(_scen_list)}

                        for _gn in _rendered_nodes:
                            _gcol  = _scen_color.get(_gn.get("scenario_id", ""), "#64748b")
                            _gsz   = 5 if _is_large else 8
                            _ip    = _gn.get("ip_address") or _gn.get("ip", "")
                            _ghovr = (
                                f"global_node_id: {_gn.get('global_node_id','')}\n"
                                f"scenario_id: {_gn.get('scenario_id','')}\n"
                                f"node_id: {_gn.get('node_id','')}\n"
                                f"type: {_gn.get('type','')}\n"
                                f"diagram_id: {_gn.get('diagram_id','')}\n"
                                f"diagram_type: {_gn.get('diagram_type','')}\n"
                                f"canonical_id: {_gn.get('canonical_id','')}\n"
                                f"shared entity: {'Yes' if _gn.get('is_shared_entity') else 'No'}\n"
                                f"split: {_gn.get('split','')}"
                                + (f"\nIP: {_ip}" if _ip else "")
                            )
                            _gnet.add_node(
                                _gn["global_node_id"],
                                label=_gn.get("node_id", ""),
                                color=_gcol, title=_ghovr, size=_gsz,
                            )

                        for _ge in _rendered_edges:
                            try:
                                _esc   = _ge.get("edge_scope", "local")
                                _ecol  = "#7c3aed" if _esc == "cross_diagram" else "#334155"
                                _ew    = 1.5 if _esc == "cross_diagram" else 0.5
                                _ehovr = (
                                    f"{_ge.get('source_global_id','')} → "
                                    f"{_ge.get('target_global_id','')}\n"
                                    f"relationship: {_ge.get('relationship','')}\n"
                                    f"label/protocol: {_ge.get('label','')}\n"
                                    f"edge_scope: {_esc}\n"
                                    f"scenario_id: {_ge.get('scenario_id','')}\n"
                                    f"source_diagram → target_diagram: "
                                    f"{_ge.get('source_diagram','')} → "
                                    f"{_ge.get('target_diagram','')}"
                                )
                                _gnet.add_edge(
                                    _ge["source_global_id"],
                                    _ge["target_global_id"],
                                    color=_ecol, width=_ew, title=_ehovr,
                                )
                            except Exception:
                                pass

                        with tempfile.NamedTemporaryFile(
                            "w", delete=False, suffix=".html", encoding="utf-8"
                        ) as _tf:
                            _gal_tmp = Path(_tf.name)
                        try:
                            _gnet.save_graph(str(_gal_tmp))
                            _gal_html = _gal_tmp.read_text(encoding="utf-8")
                        finally:
                            try:
                                _gal_tmp.unlink()
                            except Exception:
                                pass

                        # For large graphs inject a zoom listener so labels are
                        # hidden at overview zoom and appear when zoomed in (≥2×),
                        # matching small-graph label behaviour exactly.
                        if _is_large:
                            _zoom_js = (
                                "<script>"
                                "(function waitNet(){"
                                "var n=window.network;"
                                "if(!n){setTimeout(waitNet,200);return;}"
                                "var ds=n.body.data.nodes;"
                                "var lb={};"
                                "ds.get().forEach(function(x){lb[x.id]=x.label;});"
                                "ds.update(ds.get().map(function(x){"
                                "return{id:x.id,label:''};"
                                "}));"
                                "n.on('zoom',function(e){"
                                "var show=e.scale>=0.5;"
                                "ds.update(ds.get().map(function(x){"
                                "return{id:x.id,label:show?(lb[x.id]||''):''};"
                                "}));"
                                "});"
                                "})();"
                                "</script>"
                            )
                            _gal_html = _gal_html.replace("</body>", _zoom_js + "</body>")

                        components.html(_gal_html, height=640, scrolling=False)

                        st.caption(
                            f"Showing {len(_rendered_nodes):,} of {len(_filtered_nodes):,} "
                            f"filtered nodes · {len(_rendered_edges):,} edges · "
                            "color = scenario · cross-diagram edges in purple"
                        )

                    except Exception as _gal_err:
                        st.error(
                            f"Galaxy render error: {_gal_err}. "
                            "Try applying more filters or reducing Max nodes."
                        )
                else:
                    st.info("No nodes match the current filter.")

            else:
                st.info("Install `pyvis>=0.3.2` for interactive galaxy graph.")

    # ── Global Edge Memory (parallel expander) ────────────────────────────────
    with st.expander("Global Edge Memory", expanded=False):
        st.markdown(
            '<div style="font-size:0.75rem;color:#64748b;margin-bottom:10px">'
            'Cross-scenario edge index — all connectors, relationships, and protocols '
            'across every V3 scenario graph.'
            '</div>',
            unsafe_allow_html=True,
        )
        if not _GLOBAL_EDGES_CSV.exists() and not _GLOBAL_GRAPH_PATH.exists():
            st.info(
                "Edge data not available. Build the global graph first:\n\n"
                "```\npython scripts/build_global_infragraph_galaxy.py "
                "--dataset-root ./datasets/infragraph_v3 "
                "--out ./runtime_state/global_graph_memory\n```"
            )
        else:
            # Load edges (prefer CSV; synthesise from JSON if CSV missing)
            _edf: "pd.DataFrame | None" = None
            _ndf: "pd.DataFrame | None" = None
            if _GLOBAL_EDGES_CSV.exists():
                try:
                    _edf = pd.read_csv(_GLOBAL_EDGES_CSV)
                except Exception:
                    pass
            if _edf is None and _GLOBAL_GRAPH_PATH.exists():
                try:
                    _jdata = _safe_read_json(_GLOBAL_GRAPH_PATH)
                    _edf = pd.DataFrame([
                        {
                            "source":         e.get("source_global_id", ""),
                            "target":         e.get("target_global_id", ""),
                            "scenario_id":    e.get("scenario_id", ""),
                            "edge_scope":     e.get("edge_scope", ""),
                            "relationship":   e.get("relationship", ""),
                            "label":          e.get("label", ""),
                            "source_diagram": e.get("source_diagram", ""),
                            "target_diagram": e.get("target_diagram", ""),
                        }
                        for e in _jdata.get("edges", [])
                    ])
                except Exception:
                    pass
            if _GLOBAL_NODES_CSV.exists():
                try:
                    _ndf = pd.read_csv(_GLOBAL_NODES_CSV)
                except Exception:
                    pass

            if _edf is not None and not _edf.empty:
                # Normalise column names from CSV
                if "source_global_id" in _edf.columns:
                    _edf = _edf.rename(columns={
                        "source_global_id": "source",
                        "target_global_id": "target",
                    })

                # Enrich with split / source_type / target_type from nodes CSV
                if _ndf is not None and not _ndf.empty:
                    _nmap_type  = _ndf.set_index("global_node_id")["type"].to_dict()
                    _nmap_split = _ndf.set_index("global_node_id")["split"].to_dict()
                    _edf["source_type"] = _edf["source"].map(lambda x: _nmap_type.get(x, ""))
                    _edf["target_type"] = _edf["target"].map(lambda x: _nmap_type.get(x, ""))
                    if "split" not in _edf.columns:
                        _edf["split"] = _edf["source"].map(lambda x: _nmap_split.get(x, ""))

                # ── Summary metrics ───────────────────────────────────────────
                _em1, _em2, _em3, _em4, _em5 = st.columns(5)
                _em1.metric("Total edges",         f"{len(_edf):,}")
                _em2.metric("Local edges",          f"{(_edf['edge_scope'] == 'local').sum():,}"         if "edge_scope"   in _edf.columns else "—")
                _em3.metric("Cross-diagram edges",  f"{(_edf['edge_scope'] == 'cross_diagram').sum():,}" if "edge_scope"   in _edf.columns else "—")
                _em4.metric("Unique relationships", f"{_edf['relationship'].nunique()}"                  if "relationship" in _edf.columns else "—")
                _em5.metric("Scenarios",            f"{_edf['scenario_id'].nunique()}"                   if "scenario_id"  in _edf.columns else "—")

                # ── Edge filters ──────────────────────────────────────────────
                _ef1, _ef2, _ef3 = st.columns(3)
                _ef4, _ef5, _ef6 = st.columns(3)
                _ef7, _ef8       = st.columns([3, 1])

                _e_scen_opts  = ["all"] + sorted(_edf["scenario_id"].dropna().unique().tolist())    if "scenario_id"    in _edf.columns else ["all"]
                _e_scope_opts = ["all", "local", "cross_diagram"]
                _e_rel_opts   = ["all"] + sorted(_edf["relationship"].dropna().unique().tolist())   if "relationship"   in _edf.columns else ["all"]
                _e_sdiag_opts = ["all"] + sorted(_edf["source_diagram"].dropna().unique().tolist()) if "source_diagram" in _edf.columns else ["all"]
                _e_tdiag_opts = ["all"] + sorted(_edf["target_diagram"].dropna().unique().tolist()) if "target_diagram" in _edf.columns else ["all"]

                _ef_scen    = _ef1.selectbox("Scenario",              _e_scen_opts,  key="em_scen")
                _ef_scope   = _ef2.selectbox("Edge scope",            _e_scope_opts, key="em_scope")
                _ef_rel     = _ef3.selectbox("Relationship",          _e_rel_opts,   key="em_rel")
                _ef_sdiag   = _ef4.selectbox("Source diagram",        _e_sdiag_opts, key="em_sdiag")
                _ef_tdiag   = _ef5.selectbox("Target diagram",        _e_tdiag_opts, key="em_tdiag")
                _ef_src_str = _ef6.text_input("Source node contains", key="em_src_str")
                _ef_tgt_str = _ef7.text_input("Target node contains", key="em_tgt_str")
                _ef_xdiag   = _ef8.checkbox("Cross-diagram only",     value=False,   key="em_cross_only")

                # Apply filters
                _eview = _edf.copy()
                if _ef_scen  != "all" and "scenario_id"    in _eview.columns:
                    _eview = _eview[_eview["scenario_id"]    == _ef_scen]
                if _ef_scope != "all" and "edge_scope"      in _eview.columns:
                    _eview = _eview[_eview["edge_scope"]     == _ef_scope]
                if _ef_rel   != "all" and "relationship"    in _eview.columns:
                    _eview = _eview[_eview["relationship"]   == _ef_rel]
                if _ef_sdiag != "all" and "source_diagram"  in _eview.columns:
                    _eview = _eview[_eview["source_diagram"] == _ef_sdiag]
                if _ef_tdiag != "all" and "target_diagram"  in _eview.columns:
                    _eview = _eview[_eview["target_diagram"] == _ef_tdiag]
                if _ef_src_str:
                    _eview = _eview[_eview["source"].str.contains(_ef_src_str, case=False, na=False)]
                if _ef_tgt_str:
                    _eview = _eview[_eview["target"].str.contains(_ef_tgt_str, case=False, na=False)]
                if _ef_xdiag and "edge_scope" in _eview.columns:
                    _eview = _eview[_eview["edge_scope"] == "cross_diagram"]

                _edge_show_cols = [c for c in [
                    "scenario_id", "split", "source", "target",
                    "source_type", "target_type",
                    "relationship", "label", "edge_scope",
                    "source_diagram", "target_diagram",
                ] if c in _eview.columns]

                st.caption(f"{len(_eview):,} edges matching current filters")
                st.dataframe(_eview[_edge_show_cols], use_container_width=True, hide_index=True)

            else:
                st.info(
                    "Edge data not available. Rebuild the global graph:\n\n"
                    "```\npython scripts/build_global_infragraph_galaxy.py "
                    "--dataset-root ./datasets/infragraph_v3 "
                    "--out ./runtime_state/global_graph_memory\n```"
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ENTERPRISE GNN RCA
# ══════════════════════════════════════════════════════════════════════════════
def _tab_gnn_rca() -> None:
    st.markdown(
        '<div class="ws-title">Enterprise GNN RCA — cross-diagram graph reasoning</div>'
        '<div class="ws-desc">Simulate cross-diagram alert propagation, then run enterprise '
        'root cause analysis — GNN-powered when a trained result exists, '
        'scenario-grounded otherwise.</div>',
        unsafe_allow_html=True,
    )

    local_graph = st.session_state.get("local_graph")
    if not local_graph:
        st.markdown(
            '<div class="warn-card">No diagram loaded yet — select one from '
            '<strong>Diagram Intelligence</strong> first.</div>',
            unsafe_allow_html=True,
        )
        return

    if not st.session_state.get("enterprise_absorbed"):
        st.markdown(
            '<div class="warn-card">Enterprise graph not absorbed yet — '
            'complete absorption in <strong>Enterprise Graph Brain</strong> first.</div>',
            unsafe_allow_html=True,
        )
        return

    enterprise_graph = st.session_state.get("enterprise_graph_after") or {}
    ent_rec          = _selected_enterprise_record()
    _alerts_p_gnn    = _resolve_manifest_path(ent_rec.get("alerts_path", ""), REPO_ROOT)
    _sel_scen_p_gnn  = _selected_scenario_path()
    _alerts_src_gnn  = (
        st.session_state.get("enterprise_alerts_path")
        or _alerts_p_gnn
        or (str(_sel_scen_p_gnn / "alerts.json") if _sel_scen_p_gnn else None)
    )
    alerts_data = _safe_read_json(Path(_alerts_src_gnn)) if _alerts_src_gnn else {}

    # ── RCA Engine panel ──────────────────────────────────────────────────────
    _ent_summary_pre = st.session_state.get("enterprise_ingestion_summary") or {}
    _sel_rec_pre     = _selected_enterprise_record()
    _rca_scenario_id = (
        _ent_summary_pre.get("scenario_id")
        or alerts_data.get("scenario_id")
        or _sel_rec_pre.get("source_scenario_id")
        or "—"
    )
    _gnn_result_pre   = _load_gnn_rca_result(_rca_scenario_id) if _rca_scenario_id != "—" else None
    _gnn_model_exists = V3_ENTERPRISE_GNN_MODEL.exists()
    _gnn_avail_str    = "Yes" if _gnn_result_pre else "No"
    _model_avail_str  = "Yes" if _gnn_model_exists else "No"
    _rca_src_str      = "Enterprise GNN RCA" if _gnn_result_pre else "Scenario-grounded RCA simulation"
    _n_nodes_pre      = len(enterprise_graph.get("nodes", []))
    _n_edges_pre      = len(enterprise_graph.get("edges", []))
    st.markdown(
        '<div style="background:rgba(15,23,42,0.7);border:1px solid rgba(51,65,85,0.6);'
        'border-radius:10px;padding:14px 16px;margin-bottom:14px">'
        '<div style="font-size:0.68rem;font-weight:700;color:#64748b;text-transform:uppercase;'
        'letter-spacing:0.08em;margin-bottom:10px">RCA Engine</div>'
        '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:10px">'
        f'<div><div style="font-size:0.6rem;color:#64748b;text-transform:uppercase">Selected scenario</div>'
        f'<div style="font-size:0.82rem;font-weight:700;color:#38bdf8;font-family:monospace">{_rca_scenario_id}</div></div>'
        f'<div><div style="font-size:0.6rem;color:#64748b;text-transform:uppercase">Scenario nodes</div>'
        f'<div style="font-size:0.82rem;font-weight:700;color:#f1f5f9">{_n_nodes_pre}</div></div>'
        f'<div><div style="font-size:0.6rem;color:#64748b;text-transform:uppercase">Scenario edges</div>'
        f'<div style="font-size:0.82rem;font-weight:700;color:#f1f5f9">{_n_edges_pre}</div></div>'
        f'<div><div style="font-size:0.6rem;color:#64748b;text-transform:uppercase">GNN model available</div>'
        f'<div style="font-size:0.82rem;font-weight:700;color:{"#10b981" if _gnn_model_exists else "#f59e0b"}">{_model_avail_str}</div></div>'
        f'<div><div style="font-size:0.6rem;color:#64748b;text-transform:uppercase">GNN result for scenario</div>'
        f'<div style="font-size:0.82rem;font-weight:700;color:{"#10b981" if _gnn_result_pre else "#f59e0b"}">{_gnn_avail_str}</div></div>'
        f'<div><div style="font-size:0.6rem;color:#64748b;text-transform:uppercase">RCA source</div>'
        f'<div style="font-size:0.82rem;font-weight:700;color:{"#10b981" if _gnn_result_pre else "#f59e0b"}">{_rca_src_str}</div></div>'
        '</div>'
        '<div style="font-size:0.72rem;color:#475569;border-top:1px solid rgba(51,65,85,0.4);padding-top:8px">'
        'Enterprise GNN is trained across many scenario graphs. Inference is displayed on the selected '
        'scenario graph. The Global InfraGraph Galaxy is graph-memory exploration, not the RCA inference graph.'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # ── Case A: model not trained yet ────────────────────────────────────────
    if not _gnn_model_exists:
        st.info(
            "Enterprise GNN model not trained yet. "
            "Current RCA uses scenario-grounded evidence."
        )
        st.code(
            "python scripts/train_enterprise_gnn_rca.py "
            "--dataset-root ./datasets/infragraph_v3 "
            "--out ./assets/preloaded/enterprise_gnn_rca --epochs 80",
            language="bash",
        )

    # ── Case B: model exists but no result for this scenario ─────────────────
    elif not _gnn_result_pre and _rca_scenario_id != "—":
        st.info("Enterprise GNN model is available. Generate RCA result for this selected scenario.")
        if st.button(
            "Generate Enterprise GNN RCA for Selected Scenario",
            type="primary",
            key="gen_gnn_rca_btn",
        ):
            _inf_cmd = [
                "python",
                str(REPO_ROOT / "scripts" / "run_enterprise_gnn_inference.py"),
                "--dataset-root", str(REPO_ROOT / "datasets" / "infragraph_v3"),
                "--scenario-id",  _rca_scenario_id,
                "--model-path",   str(REPO_ROOT / "model_artifacts" / "enterprise_gnn_rca" / "enterprise_gnn_rca.pt"),
                "--out",          str(REPO_ROOT / "outputs" / "enterprise_gnn_rca"),
            ]
            with st.spinner(f"Running GNN inference for {_rca_scenario_id}..."):
                import subprocess as _sp
                _inf_result = _sp.run(
                    _inf_cmd, capture_output=True, text=True, cwd=str(REPO_ROOT),
                )
            if _inf_result.returncode == 0:
                st.success("GNN inference complete. Reloading result...")
                st.rerun()
            else:
                st.error("GNN inference failed.")
                st.code(_inf_result.stderr or _inf_result.stdout, language="text")

    # ── Case C: result exists ─────────────────────────────────────────────────
    elif _gnn_result_pre:
        st.success(f"Enterprise GNN RCA result loaded for **{_rca_scenario_id}**.", icon=None)

    st.markdown('<hr class="ws-rule">', unsafe_allow_html=True)

    # ── Strict mode gate ──────────────────────────────────────────────────────
    gnn_metrics_available = _enterprise_gnn_available()
    if not gnn_metrics_available and not _gnn_model_exists:
        if _strict_mode() and not st.session_state.allow_enterprise_simulation:
            st.error("Strict mode: approve before using scenario-grounded simulation.")
            if st.button("Continue with scenario simulation", key="approve_ent_sim"):
                st.session_state.allow_enterprise_simulation = True
                st.rerun()
            return

    # ── Step 1: Generate Cross-Diagram Alert Stream ───────────────────────────
    ent_incident = st.session_state.get("enterprise_incident") or {}
    diagram_id   = st.session_state.get("selected_diagram_id", "")

    # Auto-clear stale incident that was stored by an older code path (empty timeline).
    # Only clear when the module is importable and no generation error is pending.
    if (ent_incident
            and not ent_incident.get("alert_timeline")
            and _INCIDENT_SIM_OK
            and not st.session_state.get("_ent_gen_error")):
        st.session_state.pop("enterprise_incident", None)
        ent_incident = {}

    col_eb1, col_eb2 = st.columns([1, 1])
    with col_eb1:
        if st.button(
            "Generate Cross-Diagram Alert Stream",
            type="primary" if not ent_incident else "secondary",
            key="gen_ent_alerts_btn",
            help="Build a cross-diagram hero incident from scenario evidence (3+ diagrams)",
        ):
            with st.spinner("Building cross-diagram alert stream…"):
                if _INCIDENT_SIM_OK and _build_cross_hero_incident_pkg:
                    try:
                        ent_inc = _build_cross_hero_incident_pkg(
                            enterprise_graph,
                            alerts_data,
                            diagram_id,
                            gnn_result=_gnn_result_pre,
                        )
                        st.session_state.pop("_ent_gen_error", None)
                    except Exception as _gen_exc:
                        st.session_state["_ent_gen_error"] = str(_gen_exc)
                        ent_inc = _simulate_enterprise_rca(alerts_data, enterprise_graph)
                        ent_inc["alert_timeline"]    = []
                        ent_inc["propagation_steps"] = []
                else:
                    ent_inc = _simulate_enterprise_rca(alerts_data, enterprise_graph)
                    ent_inc["alert_timeline"]    = []
                    ent_inc["propagation_steps"] = []
                st.session_state.enterprise_incident          = ent_inc
                st.session_state.enterprise_rca_result        = {}
                st.session_state.pop("enterprise_ai_resolution_plan", None)
            st.rerun()

    # Surface any generation or import error
    if _INCIDENT_SIM_ERR and not _INCIDENT_SIM_OK:
        st.error(f"Incident simulation module unavailable: {_INCIDENT_SIM_ERR}")
    elif st.session_state.get("_ent_gen_error"):
        st.error(f"Alert stream error: {st.session_state['_ent_gen_error']}")

    if ent_incident:
        with col_eb2:
            rca_btn_lbl = ("Run Enterprise RCA"
                           if not _gnn_result_pre
                           else "Run Enterprise GNN RCA")
            if st.button(
                rca_btn_lbl,
                type="primary",
                key="ent_rca_run_btn",
                help="Derive root cause from the alert stream",
            ):
                with st.spinner("Running enterprise RCA…"):
                    if (_INCIDENT_SIM_OK
                            and isinstance(ent_incident, dict)
                            and ent_incident.get("root_cause") is not None):
                        rca_result = _incident_to_enterprise_rca(ent_incident)
                    else:
                        rca_result = _simulate_enterprise_rca(alerts_data, enterprise_graph)
                    st.session_state.enterprise_rca_result = rca_result
                    _persist_incident(ent_incident, "enterprise")
                st.rerun()

    # ── Cross-diagram incident card + timeline ────────────────────────────────
    if ent_incident:
        ent_timeline = ent_incident.get("alert_timeline", [])
        if ent_timeline:
            _diag_set = list(dict.fromkeys(
                e.get("diagram_id", "") for e in ent_timeline if e.get("diagram_id")
            ))
            _n_diags    = len(_diag_set)
            _bridge_cnt = sum(
                1 for e in ent_timeline if e.get("correlation_role") == "bridge"
            )
            _fst_ev_inc    = next((e for e in ent_timeline if e.get("is_first_observed")), ent_timeline[0])
            _first_obs_diag = _fst_ev_inc.get("diagram_id", "—")

            st.markdown(
                f'<div class="info-card" style="margin:10px 0">'
                f'<div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;'
                f'color:#94a3b8;margin-bottom:4px">Cross-Diagram Incident</div>'
                f'<div style="font-size:1.0rem;font-weight:700;color:#f1f5f9;margin-bottom:8px">'
                f'{ent_incident.get("incident_title","Enterprise Incident")}</div>'
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px 20px;'
                f'font-size:0.77rem;margin-bottom:8px">'
                f'<div><span style="color:#64748b">First observed diagram</span>&nbsp;'
                f'<code style="font-size:0.72rem;color:#a78bfa">{_first_obs_diag}</code></div>'
                f'<div><span style="color:#64748b">First observed node</span>&nbsp;'
                f'<code style="font-size:0.73rem;color:#67e8f9">'
                f'{ent_incident.get("first_observed_node","—")}</code></div>'
                f'<div><span style="color:#64748b">Suspected root domain</span>&nbsp;'
                f'<span style="color:#e2e8f0">{ent_incident.get("suspected_domain","—")}</span></div>'
                f'<div><span style="color:#64748b">Alert count</span>&nbsp;'
                f'<strong style="color:#f1f5f9">{len(ent_timeline)}</strong></div>'
                f'<div><span style="color:#64748b">Diagrams affected</span>&nbsp;'
                f'<strong style="color:#f1f5f9">{_n_diags}</strong></div>'
                f'<div><span style="color:#64748b">Cross-diagram bridges</span>&nbsp;'
                f'<strong style="color:#22d3ee">{_bridge_cnt}</strong></div>'
                f'</div>'
                f'<div style="font-size:0.73rem">'
                f'<span style="color:#64748b">Affected diagrams:</span>&nbsp;'
                + " ".join(f'<span class="diag-label">{d}</span>' for d in _diag_set[:6])
                + f'</div></div>',
                unsafe_allow_html=True,
            )

            if _n_diags < 2:
                st.markdown(
                    '<div class="warn-card" style="margin-top:8px">'
                    'Selected scenario contains limited cross-diagram alert evidence. '
                    'The simulation will automatically expand to cross-diagram mode on the '
                    'next generation run.'
                    '</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<div class="section-label" style="margin-top:10px">'
                'Cross-Diagram Alert Stream</div>',
                unsafe_allow_html=True,
            )
            _render_alert_timeline(ent_timeline, show_diagram_col=True)

            # (propagation journey panel rendered below via _render_propagation_journey_panel)
        else:
            if _INCIDENT_SIM_OK:
                st.info(
                    "Click **Generate Cross-Diagram Alert Stream** to build the "
                    "cross-diagram incident story for this scenario.",
                    icon="ℹ️",
                )
            else:
                st.caption("Alert timeline will appear here once generated.")

    rca = st.session_state.get("enterprise_rca_result")

    # ── Scenario Enterprise Graph — Alert Propagation ─────────────────────────
    st.markdown('<hr class="ws-rule" style="margin:14px 0">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label" style="margin-top:4px">'
        'Scenario Enterprise Graph &#8212; Alert Propagation</div>'
        '<div style="font-size:0.75rem;color:#64748b;margin-bottom:8px">'
        'Cross-diagram incident propagation across the selected scenario graph.</div>',
        unsafe_allow_html=True,
    )
    absorbed_ids = {n.get("canonical_id", n.get("id")) for n in local_graph.get("nodes", [])}

    # ── Build RCA journey context (normalized node IDs + step roles) ───────────
    jctx = _build_rca_journey_context(enterprise_graph, ent_incident, rca, _gnn_result_pre)

    # ── RCA Investigation Journey stepper panel ───────────────────────────────
    _render_rca_journey_stepper(jctx)

    # ── Render mode selector ──────────────────────────────────────────────────
    _render_mode = st.radio(
        "Graph render mode",
        ["Stable 2D RCA Journey Graph",
         "Experimental 3D FalconVue"],
        horizontal=True,
        key="gnn_render_mode",
        label_visibility="collapsed",
    )

    # ── Developer details ─────────────────────────────────────────────────────
    with st.expander("Developer details", expanded=False):
        st.caption(f"Renderer: {_render_mode}")
        st.caption(f"FalconVue loaded: {'Yes' if _FALCONVUE_OK else 'No'} | PyVis: {'Yes' if _pyvis_available() else 'No'}")
        st.caption(f"Journey steps: {len(jctx['steps'])}")
        st.caption(f"Current step index: {jctx['current_step_index']}")
        st.caption(f"Current step node (raw): {jctx.get('current_step',{}).get('node_id','—')}")
        st.caption(f"Current step node (normalized): {jctx.get('current_step',{}).get('normalized_node_id','—')}")
        st.caption(f"Root cause (raw): {(rca or {}).get('root_cause','—')}")
        st.caption(f"Root cause (normalized): {jctx.get('root_cause','—')}")
        st.caption(f"Normalized path length: {len(jctx.get('current_step',{}).get('path_so_far',[]))}")
        st.caption(f"Alert timeline: {len(jctx['alert_timeline'])} events")
        st.caption(f"Node aliases: {len(jctx['node_aliases'])} → {len(jctx['id_to_node'])} unique nodes")
        st.caption(f"Nodes passed to graph: {len(enterprise_graph.get('nodes',[]))}")
        st.caption(f"Edges: {len(enterprise_graph.get('edges',[]))} intra + {len(enterprise_graph.get('cross_diagram_edges',[]))} cross")
        if jctx["unmatched_nodes"]:
            st.caption(f"⚠ Unmatched alert/path nodes: {', '.join(jctx['unmatched_nodes'][:10])}")
        else:
            st.caption("✓ All alert/path node IDs resolve to graph nodes")
        for s in jctx["steps"][:4]:
            st.caption(f"  Step {s['step']}: {s['node_id']!r} → {s['normalized_node_id']!r} [{s['role']}]")

    # ── Render the graph ──────────────────────────────────────────────────────
    if "FalconVue" in _render_mode:
        if _FALCONVUE_OK and _render_falconvue_graph is not None:
            _cur_step = jctx.get("current_step", {})
            try:
                _render_falconvue_graph(
                    enterprise_graph, absorbed_ids, rca,
                    incident=ent_incident or {},
                    height=800,
                    mode="scenario",
                    current_step_node=_cur_step.get("normalized_node_id"),
                    traversal_path=_cur_step.get("path_so_far", []),
                    alert_timeline=jctx["alert_timeline"],
                )
            except Exception as _fv_exc:
                st.error(f"FalconVue renderer error: {_fv_exc}")
                st.info("Switch to **Stable 2D RCA Journey Graph** for a reliable view.")
        else:
            st.warning("FalconVue module not loaded.")
            st.info("Switch to **Stable 2D RCA Journey Graph** above.")

    else:  # Stable 2D RCA Journey Graph — default
        if _pyvis_available():
            _legend_j = (
                "⚡ First alert: orange | 🎯 Root cause: red/large | "
                "⇒ Cross-diagram bridge: cyan | Step labels on nodes · Dim = background topology"
            )
            st.caption(f"Drag · zoom · hover for details · {_legend_j}")
            _render_enterprise_pyvis_rca_journey(enterprise_graph, jctx, absorbed_ids, height=800)
        else:
            st.warning("Install `pyvis>=0.3.2` for the RCA journey graph.")

    # ── RCA Explanation card ──────────────────────────────────────────────────
    if rca:
        _render_rca_explanation_card(jctx, rca, _gnn_result_pre)

    # ── Enterprise RCA Result ─────────────────────────────────────────────────
    if rca:
        st.markdown('<hr class="ws-rule">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Enterprise RCA Result</div>', unsafe_allow_html=True)

        _ent_root  = rca.get("root_cause", "—")
        _ent_diag  = rca.get("root_cause_diagram", "—")
        _ent_nimp  = len(rca.get("impacted_diagrams", []))
        _ent_alrt  = rca.get("alert_count", 0)
        st.markdown(
            f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:8px">'
            f'<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);'
            f'border-radius:8px;padding:10px 14px">'
            f'<div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.08em;'
            f'color:#64748b;margin-bottom:4px">Root Cause</div>'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.95rem;font-weight:700;'
            f'color:#f1f5f9;word-break:break-all;overflow-wrap:anywhere;line-height:1.3">{_ent_root}</div>'
            f'</div>'
            f'<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);'
            f'border-radius:8px;padding:10px 14px">'
            f'<div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.08em;'
            f'color:#64748b;margin-bottom:4px">Root Cause Diagram</div>'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.95rem;font-weight:700;'
            f'color:#f1f5f9;word-break:break-all;overflow-wrap:anywhere;line-height:1.3">{_ent_diag}</div>'
            f'</div>'
            f'<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);'
            f'border-radius:8px;padding:10px 14px">'
            f'<div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.08em;'
            f'color:#64748b;margin-bottom:4px">Impacted Diagrams</div>'
            f'<div style="font-size:1.5rem;font-weight:700;color:#f1f5f9;line-height:1.3">{_ent_nimp}</div>'
            f'</div>'
            f'<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);'
            f'border-radius:8px;padding:10px 14px">'
            f'<div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.08em;'
            f'color:#64748b;margin-bottom:4px">Alert Count</div>'
            f'<div style="font-size:1.5rem;font-weight:700;color:#f1f5f9;line-height:1.3">{_ent_alrt}</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        rca_mode = rca.get("mode", "")
        mode_badge_cls = "badge-success" if rca_mode == "Enterprise GNN RCA" else "badge-warn"
        st.markdown(
            f'<span class="badge {mode_badge_cls}">{rca_mode}</span>',
            unsafe_allow_html=True,
        )
        if rca_mode == "Enterprise GNN RCA":
            _inf_mode_lbl = "precomputed_gnn_inference_artifact"
            src_file = rca.get("gnn_source_file", "")
            _caption_parts = [
                f"Generated by trained GNN RCA pipeline; inference_mode={_inf_mode_lbl}"
            ]
            if src_file:
                _caption_parts.append(f"loaded from: {Path(src_file).name}")
            st.caption(" · ".join(_caption_parts))
        else:
            st.caption("Source: scenario-grounded evidence — no trained GNN result for this scenario")

        # Reasoning steps (from enterprise incident if available)
        _ent_steps = (ent_incident or {}).get("reasoning_steps", [])
        if _ent_steps:
            with st.expander("Reasoning steps", expanded=True):
                for _rs in _ent_steps:
                    st.markdown(
                        f'<div style="font-size:0.8rem;color:#cbd5e1;padding:3px 0 3px 8px;'
                        f'border-left:2px solid #3b82f6;margin-bottom:4px">{_rs}</div>',
                        unsafe_allow_html=True,
                    )

        # Impact path
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

            # Traversal
            _render_traversal_steps(path, rca, slider_key="ent_traversal_slider")

        ranking = rca.get("ranking", [])
        if ranking:
            st.markdown(
                '<div class="section-label" style="margin-top:10px">RCA Candidate Ranking</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(pd.DataFrame(ranking), use_container_width=True, hide_index=True)

        # Trained RCA Model Evidence — provenance expander for GNN model + metrics
        _gnn_metrics_data = _safe_read_json(
            REPO_ROOT / "assets" / "preloaded" / "enterprise_gnn_rca" / "enterprise_gnn_metrics.json"
        )
        _render_gnn_rca_model_evidence(_gnn_result_pre or {}, _gnn_metrics_data)

        # Do not render RCA recommended_actions here.
        # Remediation must come from the AI Resolution Agent to avoid hardcoded-looking output.

    elif ent_incident and not rca:
        st.info("Click **Run Enterprise RCA** to identify the root cause from the alert stream.")

    # ── AI Resolution Agent ───────────────────────────────────────────────────
    st.markdown('<hr class="ws-rule" style="margin:18px 0">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label">AI Resolution Agent — Enterprise Remediation</div>'
        '<div style="font-size:0.75rem;color:#64748b;margin-bottom:10px">'
        'Qwen3 remediation agent — grounded in graph memory, alert timeline, '
        'RCA path, and GNN ranking.</div>',
        unsafe_allow_html=True,
    )

    # Status cards
    _vllm_ok      = _check_vllm_available()
    _lora_path    = Path(_LORA_ADAPTER) if _LORA_ADAPTER else None
    _lora_exists  = bool(_lora_path and _lora_path.exists())
    _rem_src_lbl  = "Qwen3 via vLLM" if _vllm_ok else "Template fallback"
    _rca_src_disp = (rca or {}).get("mode", "—") if rca else "—"

    def _sbadge(ok: bool, yes_text: str, no_text: str) -> str:
        c = "#10b981" if ok else "#f59e0b"
        t = yes_text if ok else no_text
        return (
            f'<span style="background:rgba({("16,185,129" if ok else "245,158,11")},0.15);'
            f'color:{c};border:1px solid {c};border-radius:6px;'
            f'padding:2px 10px;font-size:0.7rem;font-weight:700">{t}</span>'
        )

    st.markdown(
        '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:12px">'
        f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);'
        f'border-radius:8px;padding:10px 12px">'
        f'<div style="font-size:0.6rem;color:#64748b;text-transform:uppercase;margin-bottom:5px">RCA Source</div>'
        f'<div style="font-size:0.78rem;color:#f1f5f9;font-weight:600">{_rca_src_disp}</div></div>'
        f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);'
        f'border-radius:8px;padding:10px 12px">'
        f'<div style="font-size:0.6rem;color:#64748b;text-transform:uppercase;margin-bottom:5px">Qwen / vLLM</div>'
        + _sbadge(_vllm_ok, "Available", "Not running")
        + '</div>'
        f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);'
        f'border-radius:8px;padding:10px 12px">'
        f'<div style="font-size:0.6rem;color:#64748b;text-transform:uppercase;margin-bottom:5px">Remediation Source</div>'
        f'<div style="font-size:0.78rem;color:#f1f5f9;font-weight:600">{_rem_src_lbl}</div></div>'
        f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);'
        f'border-radius:8px;padding:10px 12px">'
        f'<div style="font-size:0.6rem;color:#64748b;text-transform:uppercase;margin-bottom:5px">Fine-tuned Adapter</div>'
        + _sbadge(_lora_exists, "Detected", "Not loaded")
        + '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Honesty notices
    if not _vllm_ok:
        st.info(
            f"Qwen/vLLM is not running. Start the local model server to generate AI remediation.\n\n"
            f"```\npython -m vllm.entrypoints.openai.api_server "
            f"--model {_QWEN_MODEL} --host 0.0.0.0 --port 8000\n```",
        )
    if not _lora_exists:
        st.caption(
            "SOP-grounded LoRA adapter not detected. "
            "Train with `scripts/train_qwen_sop_lora.py` or set `INFRAGRAPH_LORA_ADAPTER_PATH`."
        )

    # Only show generation buttons when we have something to remediate
    _rem_plan    = st.session_state.get("enterprise_ai_resolution_plan") or {}
    _has_context = bool(ent_incident and (rca or ent_incident.get("root_cause")))

    if _has_context and _AI_REM_OK:
        _col_r1, _col_r2 = st.columns([1, 1])
        with _col_r1:
            _ai_btn_lbl = (
                "Generate Enterprise AI Resolution Plan"
                if _vllm_ok
                else "Generate Enterprise Template Resolution Plan"
            )
            if st.button(_ai_btn_lbl, type="primary", key="gen_ai_plan_btn"):
                with st.spinner(
                    "Calling Qwen3 via vLLM…" if _vllm_ok else "Generating template plan…"
                ):
                    _ctx = _build_remediation_context(
                        rca or {}, ent_incident or {}, enterprise_graph,
                        alerts_data, diagram_id, _gnn_result_pre,
                    )
                    if _ctx and _generate_resolution_plan is not None:
                        _root = (rca or {}).get("root_cause", "")
                        _impacted = " ".join((rca or {}).get("impacted_diagrams", []) or [])
                        _query = f"root cause {_root} impacted diagrams {_impacted} validation remediation"
                        _vec_evidence, _vec_err = _retrieve_vector_evidence(_query, k=6)
                        st.session_state.last_enterprise_ai_vector_evidence_count = len(_vec_evidence)
                        if _vec_evidence:
                            _ctx["retrieved_graph_memory_evidence"] = _vec_evidence
                            _ctx["retrieved_graph_memory_label"] = "Retrieved graph memory evidence"
                        _plan = _generate_resolution_plan(
                            _ctx,
                            scope="enterprise",
                            prefer_qwen=_vllm_ok,
                            base_url=_QWEN_BASE_URL,
                            model=_QWEN_MODEL,
                            timeout=_QWEN_TIMEOUT,
                        )
                    else:
                        _plan = {}
                    if _plan:
                        import datetime as _dt
                        _plan["_generated_at"] = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.enterprise_ai_resolution_plan = _plan
                st.rerun()

        with _col_r2:
            if _rem_plan and st.button(
                "View ITSM Ticket", key="gen_itsm_btn", type="secondary"
            ):
                _itsm = (_rem_plan.get("response") or {}).get("itsm_ticket_summary")
                if isinstance(_itsm, dict) and _itsm.get("short_description"):
                    st.info(
                        f"**{_itsm.get('short_description','')}**\n\n"
                        f"{_itsm.get('description','')}\n\n"
                        f"Priority: {_itsm.get('priority','')} | "
                        f"Assignment: {_itsm.get('assignment_group','')}"
                    )
                elif _itsm:
                    st.code(str(_itsm), language="text")

    elif not _has_context:
        st.caption(
            "Generate a Cross-Diagram Alert Stream and run Enterprise RCA first "
            "to enable the AI Resolution Agent."
        )
    elif not _AI_REM_OK:
        st.caption("AI remediation package failed to load — check the logs.")

    # Render existing plan
    if _rem_plan:
        _plan_src = _rem_plan.get("source", "")
        _plan_ok  = _rem_plan.get("ok", False)

        if _plan_src in ("template", "template_fallback"):
            _fb_detail = (
                " Qwen returned an error — see Qwen Runtime Proof for details."
                if _plan_src == "template_fallback"
                else " Connect a vLLM server to enable Qwen3 AI inference."
            )
            st.markdown(
                '<div style="background:rgba(251,191,36,0.07);border:1px solid rgba(251,191,36,0.3);'
                'border-radius:8px;padding:10px 14px;margin:10px 0;font-size:0.8rem;color:#fbbf24">'
                f'Template output — deterministic, not AI-generated.{html.escape(_fb_detail)}'
                '</div>',
                unsafe_allow_html=True,
            )
        if _rem_plan.get("qwen_error"):
            st.warning(f"Qwen error (caused fallback): {_rem_plan['qwen_error']}")
        elif not _plan_ok and _rem_plan.get("error"):
            st.error(f"Resolution plan error: {_rem_plan['error']}")

        if _plan_ok or _plan_src in ("template", "template_fallback"):
            _render_remediation_plan(_rem_plan)
            _render_qwen_runtime_proof(_rem_plan)
            _render_ai_pipeline_trace(
                selected_diagram=str(diagram_id),
                selected_scenario=str((ent_incident or {}).get("scenario_id") or (rca or {}).get("scenario_id") or ""),
                topology_rca_completed=bool(st.session_state.get("local_rca_result")),
                enterprise_gnn_available=bool(_gnn_result_pre),
                rca_source=str((rca or {}).get("mode") or (rca or {}).get("rca_source") or _rca_src_disp),
                root_cause=str((rca or {}).get("root_cause") or (ent_incident or {}).get("root_cause") or ""),
                impacted_diagrams=list((rca or {}).get("impacted_diagrams") or (ent_incident or {}).get("impacted_diagrams") or []),
                vector_evidence_count=int(st.session_state.get("last_enterprise_ai_vector_evidence_count", 0)),
                response_source=str(_rem_plan.get("source") or "—"),
            )



# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — GRAPH COPILOT
# ══════════════════════════════════════════════════════════════════════════════
def _copilot_context() -> dict:
    scenario = (
        Path(st.session_state.enterprise_scenario_path)
        if st.session_state.enterprise_scenario_path
        else V3_HERO_SCENARIO
    )
    loc_inc  = _ss_dict("local_incident")
    ent_inc  = _ss_dict("enterprise_incident")
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
        "local_ai_resolution_plan":     _ss_dict("local_ai_resolution_plan"),
        "enterprise_ai_resolution_plan": _ss_dict("enterprise_ai_resolution_plan"),
        "local_incident":               loc_inc,
        "enterprise_incident":          ent_inc,
        "alert_timeline":               (
            ent_inc.get("alert_timeline") or loc_inc.get("alert_timeline") or []
        ),
        "traversal_steps":              (
            _ss_dict("enterprise_rca_result").get("impact_path")
            or _ss_dict("local_rca_result").get("impact_path")
            or []
        ),
        "stitch_map":                   _safe_read_json(scenario / "stitch_map.json"),
        "alerts":                       _safe_read_json(scenario / "alerts.json"),
    }


_VECTOR_SETUP_MESSAGE = "Vector memory is not installed. Run pip install chromadb sentence-transformers."


def _vector_modules() -> tuple[dict, str]:
    try:
        import importlib.util
        if importlib.util.find_spec("chromadb") is None:
            return {}, _VECTOR_SETUP_MESSAGE
        if importlib.util.find_spec("sentence_transformers") is None:
            return {}, _VECTOR_SETUP_MESSAGE
        from vector_memory.chroma_store import get_or_create_collection, upsert_documents  # type: ignore
        from vector_memory.index_builder import build_vector_docs_from_graph_memory  # type: ignore
        from vector_memory.retriever import retrieve_graph_memory  # type: ignore
        return {
            "get_or_create_collection": get_or_create_collection,
            "upsert_documents": upsert_documents,
            "build_docs": build_vector_docs_from_graph_memory,
            "retrieve": retrieve_graph_memory,
        }, ""
    except Exception as exc:
        return {}, f"{_VECTOR_SETUP_MESSAGE} ({exc})"


def _current_ai_resolution_plan() -> dict:
    return _ss_dict("enterprise_ai_resolution_plan") or _ss_dict("local_ai_resolution_plan")


def _build_current_vector_docs() -> tuple[list[dict], str]:
    mods, err = _vector_modules()
    if err:
        return [], err
    build_docs = mods["build_docs"]
    context = _copilot_context()
    docs = build_docs(
        context.get("validation_packet") or {},
        local_graph=context.get("local_graph") or {},
        enterprise_graph=context.get("enterprise_graph_after") or {},
        local_incident=context.get("local_incident") or {},
        enterprise_incident=context.get("enterprise_incident") or {},
        local_rca_result=context.get("local_rca_result") or {},
        enterprise_rca_result=context.get("enterprise_rca_result") or {},
        ai_resolution_plan=_current_ai_resolution_plan(),
    )
    return docs, ""


def _index_current_context_to_vector_memory() -> tuple[int, str]:
    mods, err = _vector_modules()
    if err:
        return 0, err
    docs, err = _build_current_vector_docs()
    if err:
        return 0, err
    if not docs:
        return 0, "No graph memory evidence is loaded for indexing."
    try:
        collection = mods["get_or_create_collection"]("infragraph_memory", str(_runtime_path("vector_memory", "chroma")))
        count = mods["upsert_documents"](collection, docs)
        return count, ""
    except Exception as exc:
        return 0, f"{_VECTOR_SETUP_MESSAGE} ({exc})"


def _retrieve_vector_evidence(query: str, k: int = 8) -> tuple[list[dict], str]:
    mods, err = _vector_modules()
    if err:
        return [], err
    try:
        return mods["retrieve"](
            query,
            k=k,
            collection_name="infragraph_memory",
            persist_dir=str(_runtime_path("vector_memory", "chroma")),
        ), ""
    except Exception as exc:
        return [], f"{_VECTOR_SETUP_MESSAGE} ({exc})"


def _retrieve_vector_evidence_global(query: str, k: int = 8) -> tuple[list[dict], str]:
    """Retrieve from the global copilot memory (built by build_graph_copilot_memory.py)."""
    mods, err = _vector_modules()
    if err:
        return [], err
    try:
        return mods["retrieve"](
            query,
            k=k,
            collection_name="infragraph_global_memory",
            persist_dir=str(REPO_ROOT / "vector_store"),
        ), ""
    except Exception:
        return [], ""


def _build_global_graph_copilot_ctx() -> "dict | None":
    """Load a normalized graph copilot context from global memory + session state."""
    try:
        from graph_copilot.graph_context import load_global_graph_context  # type: ignore
        ctx = _copilot_context()
        return load_global_graph_context(
            scenario_graph=ctx.get("enterprise_graph_after"),
            enterprise_rca=ctx.get("enterprise_rca_result"),
            incident=ctx.get("enterprise_incident"),
        )
    except Exception:
        return None


def _format_retrieved_evidence(evidence: list[dict]) -> str:
    lines = []
    for idx, item in enumerate(evidence[:8], 1):
        meta = item.get("metadata") or {}
        evidence_id = meta.get("evidence_id") or f"E{idx:03d}"
        lines.append(
            f"{idx}. {evidence_id} [{meta.get('source_type','unknown')}] "
            f"scenario={meta.get('scenario_id','')} diagram={meta.get('diagram_id','')} "
            f"node={meta.get('node_id','')} "
            f"text={item.get('text','')}"
        )
    return "\n".join(lines)


def _answer_with_vector_context(question: str) -> str:
    context = _copilot_context()

    # Step 1 — deterministic graph query (no LLM; exact lookup from graph memory)
    graph_answer = ""
    try:
        from graph_copilot.query_engine import run_query, format_query_result  # type: ignore
        gctx = _build_global_graph_copilot_ctx()
        if gctx:
            result = run_query(gctx, question)
            if result:
                graph_answer = format_query_result(result)
                context["deterministic_graph_answer"] = graph_answer
    except Exception:
        pass

    # Step 2 — vector retrieval: try global memory first, fall back to session memory
    evidence, err = _retrieve_vector_evidence_global(question, k=8)
    if not evidence:
        evidence, err = _retrieve_vector_evidence(question, k=8)

    st.session_state.last_vector_evidence = evidence
    st.session_state.last_vector_error    = err
    if evidence:
        context["retrieved_graph_memory_evidence"] = evidence

    # Step 3 — LLM / deterministic answer, grounded in graph facts + vector evidence
    return _qwen_or_deterministic(question, context)


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
    retrieved = context.get("retrieved_graph_memory_evidence") or []
    retrieved_text = _format_retrieved_evidence(retrieved)

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
    if "topology rca" in q or "local rca" in q or "local root" in q or "topology root" in q:
        local_root = local_rca.get("root_cause", "not simulated yet")
        return (
            f"Topology RCA (within diagram `{diagram_id}`):\n\n"
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
            + (f"\n\nRetrieved graph memory evidence:\n{retrieved_text}" if retrieved_text else "")
        )
    if "impacted" in q:
        return (
            f"Impacted diagrams: {', '.join(ent_rca.get('impacted_diagrams', [])) or 'none'}.\n"
            f"Impacted nodes: {', '.join(ent_rca.get('impacted_nodes', [])) or 'none'}."
        )
    if "path" in q:
        active_path = ent_path or local_path
        return (f"Impact path: {' → '.join(active_path)}" if active_path
                else "No impact path loaded — run Enterprise GNN RCA or Topology RCA first.")
    if "itsm" in q or "incident" in q or "ticket" in q:
        root = ent_rca.get("root_cause", "unknown")
        return (
            "### ITSM Ticket\n\n"
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
        + (f"Retrieved graph memory evidence:\n{retrieved_text}\n\n" if retrieved_text else "")
        + "Ask: root cause | stitched | absorbed | shared | cross-diagram | path | itsm | l1"
    )


def _qwen_or_deterministic(question: str, context: dict) -> str:
    qwen_url = _QWEN_BASE_URL.rstrip("/")
    if not qwen_url:
        if _strict_mode() and not st.session_state.allow_deterministic_copilot:
            st.session_state.last_copilot_response_source = "strict_mode_blocked"
            return ("Strict mode + Qwen not configured. "
                    "Click 'Use deterministic graph response' to enable graph-evidence answers.")
        st.session_state.last_copilot_response_source = "deterministic_fallback"
        return _deterministic_graph_copilot(question, context)
    try:
        import requests  # noqa: PLC0415
        compact = json.dumps(context, default=str)[:12000]
        retrieved = context.get("retrieved_graph_memory_evidence") or []
        retrieved_text = _format_retrieved_evidence(retrieved)
        resp = requests.post(
            f"{qwen_url}/chat/completions",
            headers={"Content-Type": "application/json", "Bypass-Tunnel-Reminder": "true"},
            json={
                "model": _QWEN_MODEL,
                "messages": [
                    {"role": "system", "content": (
                        "You are InfraGraph AI Graph Copilot. Answer only from the supplied "
                        "graph JSON evidence. Cite node IDs, diagram IDs, paths, and retrieved evidence IDs when present. Be concise."
                    )},
                    {"role": "user", "content": (
                        f"Graph evidence:\n{compact}\n\n"
                        f"Retrieved graph memory evidence:\n{retrieved_text or 'none'}\n\n"
                        f"Question: /no_think {question}"
                    )},
                ],
                "max_tokens": 700, "temperature": 0.1,
            },
            timeout=min(_QWEN_TIMEOUT, 60),
        )
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"]
        st.session_state.last_copilot_response_source = "qwen_vllm"
        return re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    except Exception as exc:
        if _strict_mode() and not st.session_state.allow_deterministic_copilot:
            st.session_state.last_copilot_response_source = "strict_mode_blocked"
            return f"Live Qwen failed in strict mode: `{exc}`. Enable deterministic graph response first."
        st.session_state.last_copilot_response_source = "deterministic_fallback"
        return (f"> Live Qwen unavailable — graph-evidence answer. `{exc}`\n\n"
                + _deterministic_graph_copilot(question, context))


def _tab_graph_copilot() -> None:
    st.markdown(
        '<div class="ws-title">Graph Copilot — Ask the InfraGraph Galaxy</div>'
        '<div class="ws-desc">Deterministic graph lookup + vector evidence + Qwen enrichment. '
        'Answers cite node IDs, edge IDs, diagram IDs and evidence IDs.</div>',
        unsafe_allow_html=True,
    )

    # ── Status card ───────────────────────────────────────────────────────────
    _gctx = _build_global_graph_copilot_ctx()
    mods, vector_err = _vector_modules()
    _vector_ok = not bool(vector_err)

    _status_cols = st.columns(5)
    _status_vals = [
        ("Graph nodes",    str(_gctx["total_nodes"])     if _gctx else "—"),
        ("Graph edges",    str(_gctx["total_edges"])     if _gctx else "—"),
        ("Diagrams",       str(_gctx["total_diagrams"])  if _gctx else "—"),
        ("Scenarios",      str(_gctx["total_scenarios"]) if _gctx else "—"),
        ("Vector memory",  "enabled" if _vector_ok else "not installed"),
    ]
    for col, (label, val) in zip(_status_cols, _status_vals):
        with col:
            color = "#22d3ee" if (val not in ("—", "not installed")) else "#64748b"
            st.markdown(
                f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);'
                f'border-radius:8px;padding:8px 12px;text-align:center">'
                f'<div style="font-size:0.62rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em">{label}</div>'
                f'<div style="font-size:1.0rem;font-weight:700;color:{color}">{val}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Mode / tool status row ────────────────────────────────────────────────
    _tool_cols = st.columns(3)
    with _tool_cols[0]:
        _qe_ok = _gctx is not None
        st.markdown(
            f'<span class="badge {"badge-success" if _qe_ok else "badge-warn"}">'
            f'Exact graph tools {"enabled" if _qe_ok else "unavailable"}</span>',
            unsafe_allow_html=True,
        )
    with _tool_cols[1]:
        st.markdown(
            f'<span class="badge {"badge-success" if _vector_ok else "badge-warn"}">'
            f'Vector memory {"enabled" if _vector_ok else "not installed"}</span>',
            unsafe_allow_html=True,
        )
    with _tool_cols[2]:
        _qwen_ok = _qwen_configured()
        st.markdown(
            f'<span class="badge {"badge-success" if _qwen_ok else "badge-warn"}">'
            f'Qwen {"enabled" if _qwen_ok else "not configured (deterministic mode)"}</span>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── Strict-mode gate ──────────────────────────────────────────────────────
    if not _qwen_ok:
        if _strict_mode() and not st.session_state.allow_deterministic_copilot:
            st.error("Qwen not configured. Strict mode requires explicit approval for deterministic answers.")
            if st.button("Use deterministic graph response", key="approve_copilot"):
                st.session_state.allow_deterministic_copilot = True
                st.rerun()
            return

    # ── Memory action buttons ─────────────────────────────────────────────────
    _mem_cols = st.columns(2)
    with _mem_cols[0]:
        if _vector_ok:
            if st.button("Build / Refresh Global Copilot Memory", key="build_global_memory",
                         use_container_width=True):
                with st.spinner("Indexing global graph memory into ChromaDB…"):
                    count, err = _index_current_context_to_vector_memory()
                if err:
                    st.error(err)
                else:
                    st.success(f"Indexed {count} document(s) into graph memory.")
        else:
            st.info(_VECTOR_SETUP_MESSAGE)
    with _mem_cols[1]:
        if not st.session_state.enterprise_graph_after:
            st.warning("Load Tab 3 (Enterprise Graph Brain) to enable full graph context.")

    st.markdown('<hr style="border:none;border-top:1px solid rgba(255,255,255,0.06);margin:10px 0">',
                unsafe_allow_html=True)

    # ── Suggested investigation questions ─────────────────────────────────────
    suggestions = [
        "What is connected to the root cause node?",
        "Which diagrams are impacted?",
        "Show the impact path from root cause.",
        "What is the root cause?",
        "Show blast radius if FW-01 fails.",
        "Which cross-diagram edges connect the diagrams?",
        "Show upstream dependency path.",
        "What alerts are in the timeline?",
        "What should L1 check first?",
        "Which nodes were absorbed from the selected diagram?",
        "Which shared entities were matched?",
        "Generate an ITSM ticket summary.",
    ]
    cols = st.columns(3)
    for idx, question in enumerate(suggestions):
        with cols[idx % 3]:
            if st.button(question, key=f"v3_q_{idx}", use_container_width=True):
                st.session_state.v3_chat_messages.append({"role": "user", "content": question})
                st.session_state.v3_chat_messages.append({"role": "assistant", "content": _answer_with_vector_context(question)})
                st.rerun()

    if _qwen_ok:
        st.caption(f"Qwen: {_QWEN_BASE_URL} · model={_QWEN_MODEL}")
    else:
        st.caption("Deterministic mode: answers from exact graph queries and retrieved evidence only.")

    # ── Chat history ──────────────────────────────────────────────────────────
    for msg in st.session_state.v3_chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Retrieved evidence expander ───────────────────────────────────────────
    evidence   = st.session_state.get("last_vector_evidence") or []
    _vec_err   = st.session_state.get("last_vector_error", "")
    if evidence:
        with st.expander(f"Retrieved Graph Evidence ({len(evidence)} items)", expanded=False):
            for idx, item in enumerate(evidence, 1):
                meta = item.get("metadata") or {}
                evidence_id = meta.get("evidence_id") or f"E{idx:03d}"
                st.markdown(
                    f"<div style='border:1px solid rgba(148,163,184,0.25);border-radius:8px;"
                    f"padding:10px 12px;margin:8px 0;background:rgba(15,23,42,0.4)'>"
                    f"<div style='font-weight:800;color:#e2e8f0'>{idx}. {html.escape(str(evidence_id))}</div>"
                    f"<div style='font-size:0.75rem;color:#94a3b8;margin:4px 0'>"
                    f"type={html.escape(str(meta.get('source_type','?')))} · "
                    f"scenario={html.escape(str(meta.get('scenario_id','')))} · "
                    f"diagram={html.escape(str(meta.get('diagram_id','')))} · "
                    f"node={html.escape(str(meta.get('node_id','')))}</div>"
                    f"<div style='font-size:0.82rem;color:#cbd5e1;line-height:1.45'>"
                    f"{html.escape(str(item.get('text',''))[:900])}</div></div>",
                    unsafe_allow_html=True,
                )

    _trace_ctx = _copilot_context()
    _trace_rca = _trace_ctx.get("enterprise_rca_result") or {}
    _trace_local = _trace_ctx.get("local_rca_result") or {}
    _trace_ing = _trace_ctx.get("enterprise_ingestion_summary") or {}
    _render_ai_pipeline_trace(
        selected_diagram=str(_trace_ctx.get("selected_diagram_id") or ""),
        selected_scenario=str(_trace_ing.get("scenario_id") or _trace_rca.get("scenario_id") or ""),
        topology_rca_completed=bool(_trace_local),
        enterprise_gnn_available=bool((_trace_rca.get("mode") or "") == "Enterprise GNN RCA"),
        rca_source=str(_trace_rca.get("mode") or _trace_rca.get("rca_source") or _trace_local.get("mode") or ""),
        root_cause=str(_trace_rca.get("root_cause") or _trace_local.get("root_cause") or ""),
        impacted_diagrams=list(_trace_rca.get("impacted_diagrams") or []),
        vector_evidence_count=len(evidence),
        response_source=str(st.session_state.get("last_copilot_response_source") or "—"),
    )

    if prompt := st.chat_input("Ask the enterprise graph..."):
        st.session_state.v3_chat_messages.append({"role": "user",      "content": prompt})
        st.session_state.v3_chat_messages.append({"role": "assistant", "content": _answer_with_vector_context(prompt)})
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

        st.markdown('<div class="sb-label">Navigate</div>', unsafe_allow_html=True)
        st.radio(
            "Navigate",
            [
                "Diagram Intelligence",
                "Topology RCA",
                "Enterprise Graph Brain",
                "Enterprise GNN RCA",
                "Graph Copilot",
            ],
            key="main_nav",
            label_visibility="collapsed",
        )

        st.markdown('<div class="sb-label">Pipeline Progress</div>', unsafe_allow_html=True)
        steps = [
            ("Diagram ingested",          bool(st.session_state.local_graph)),
            ("Topology graph created",     bool(st.session_state.local_graph)),
            ("Topology RCA complete",      bool(st.session_state.local_rca_result)),
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

    if any(kw in q for kw in ["itsm", "ticket", "p1"]):
        return """\
**ITSM Ticket:**

| Field | Value |
|-------|-------|
| Short description | Network fault on FW-01 — 10-service outage |
| Affected CI | FW-01 (firewall) |
| Priority | **P1** |
| Assignment group | Network Operations |
| Root cause (automated) | FW-01 — Enterprise GNN RCA, HIGH confidence |"""

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

    _nav = st.session_state.get("main_nav", "Diagram Intelligence")
    if _nav == "Diagram Intelligence":
        _tab_diagram_intelligence()
    elif _nav == "Topology RCA":
        _tab_local_rca()
    elif _nav == "Enterprise Graph Brain":
        _tab_enterprise_graph_brain()
    elif _nav == "Enterprise GNN RCA":
        _tab_gnn_rca()
    else:
        _tab_graph_copilot()


def main() -> None:
    _main_cockpit()


if __name__ == "__main__":
    main()
