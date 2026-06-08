#!/usr/bin/env python3
"""
InfraGraph AI — Synthetic Network Diagram Dataset Generator

Generates enterprise-style network architecture diagrams with:
  • PNG images  (1400 × 900)
  • YOLO object-detection labels  (.txt)
  • Topology graph JSON  (.json)
  • Synthetic alert / RCA scenario JSON  (.json)
  • Dataset YAML for YOLO training
  • Preview contact sheet

Usage:
    python generate_infragraph_dataset.py --num 20  --out ./infragraph_dataset --seed 42
    python generate_infragraph_dataset.py --num 300 --out ./infragraph_dataset --seed 42
"""

# ─── std-lib ──────────────────────────────────────────────────────────────────
import os
import sys
import json
import math
import random
import argparse
from pathlib import Path
from collections import defaultdict

# ─── Pillow ───────────────────────────────────────────────────────────────────
try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
except ImportError:
    print("ERROR: Pillow not found.  Run: pip install Pillow")
    sys.exit(1)

# ═════════════════════════════════════════════════════════════════════════════
# GLOBAL CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

IMG_W, IMG_H = 1400, 900

CLASSES = [
    "router",        # 0
    "switch",        # 1
    "firewall",      # 2
    "server",        # 3
    "database",      # 4
    "load_balancer", # 5
    "cloud_or_wan",  # 6
]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}

TEMPLATES = [
    "simple_branch",
    "dual_router_branch",
    "load_balanced_app",
    "dmz_internal",
    "cloud_extension",
    "multi_zone_enterprise",
    "backup_link",
    "dense_enterprise",
]

# ─── Curriculum difficulty profiles ───────────────────────────────────────────

_EASY_TEMPLATES = [
    "simple_branch", "load_balanced_app", "dmz_internal",
    "cloud_extension", "backup_link",
]
_HARD_TEMPLATES = [
    "dense_enterprise", "multi_zone_enterprise",
    "dual_router_branch", "load_balanced_app", "dmz_internal",
]

_DIFF_PROFILE = {
    "easy": {
        "templates":         _EASY_TEMPLATES,
        "n_srv_range":       (1, 2),
        "n_db_range":        (1, 1),
        "n_app_range":       (1, 2),
        "icon_scale_range":  (1.05, 1.20),
        "label_font_size":   12,
        "connector_elbow_p": 0.15,
        "has_meta_p":        0.30,
        "has_legend_p":      0.20,
        "callout_p":         0.20,
        "has_footer_p":      0.20,
        "has_left_panel_p":  0.00,
        "has_watermark_p":   0.00,
        "extra_edges":       0,
        "jitter":            8,
        # LB presence — variable templates only (load_balanced_app always has LB)
        "lb_probs": {
            "simple_branch":        0.50,
            "cloud_extension":      0.50,
            "dual_router_branch":   0.45,
            "multi_zone_enterprise":0.45,
            "dmz_internal":         0.00,
            "backup_link":          0.12,
        },
        "second_lb_p": 0.00,
    },
    "medium": {
        "templates":         TEMPLATES,
        "n_srv_range":       (2, 4),
        "n_db_range":        (1, 2),
        "n_app_range":       (2, 4),
        "icon_scale_range":  (0.88, 1.12),
        "label_font_size":   11,
        "connector_elbow_p": 0.45,
        "has_meta_p":        0.60,
        "has_legend_p":      0.40,
        "callout_p":         0.50,
        "has_footer_p":      0.50,
        "has_left_panel_p":  0.00,
        "has_watermark_p":   0.00,
        "extra_edges":       0,
        "jitter":            14,
        # Higher LB presence when app servers are present
        "lb_probs": {
            "simple_branch":        0.38,
            "dual_router_branch":   0.38,
            "cloud_extension":      0.38,
            "multi_zone_enterprise":0.38,
            "dmz_internal":         0.10,
            "backup_link":          0.10,
        },
        "second_lb_p": 0.35,  # HA pair across LB-bearing templates
    },
    "hard": {
        "templates":         _HARD_TEMPLATES,
        "n_srv_range":       (4, 6),
        "n_db_range":        (2, 3),
        "n_app_range":       (3, 5),
        "icon_scale_range":  (0.78, 0.95),
        "label_font_size":   9,
        "connector_elbow_p": 0.70,
        "has_meta_p":        0.80,
        "has_legend_p":      0.70,
        "callout_p":         0.70,
        "has_footer_p":      0.75,
        "has_left_panel_p":  0.40,
        "has_watermark_p":   0.35,
        "extra_edges":       3,
        "jitter":            18,
        # Strongly prefer LB when servers are present
        "lb_probs": {
            "simple_branch":        0.60,
            "dual_router_branch":   0.60,
            "cloud_extension":      0.60,
            "multi_zone_enterprise":0.60,
            "dmz_internal":         0.30,
            "backup_link":          0.30,
        },
        "second_lb_p": 0.80,  # HA pair across all LB-bearing templates
    },
}


def _pick_difficulty(mode: str, rng: random.Random) -> str:
    """Return a concrete difficulty level for one diagram."""
    if mode == "mixed":
        return rng.choices(["easy", "medium", "hard"], weights=[30, 50, 20])[0]
    return mode

# ─── Label pools ──────────────────────────────────────────────────────────────
_RTR  = ["RTR-EDGE-01","RTR-BR-02","RTR-WAN-01","RTR-CORE-01",
          "RTR-EDGE-02","RTR-BR-01","RTR-PRI-01","RTR-SEC-01"]
_SW   = ["SW-CORE-01","SW-ACCESS-01","SW-ACCESS-02","SW-DIST-01",
          "SW-CORE-02","SW-AGG-01","SW-ACCESS-03","SW-DIST-02"]
_FW   = ["FW-EDGE-01","FW-DMZ-01","FW-INT-01","FW-EDGE-02",
          "FW-PERIMETER-01","FW-CORE-01","FW-DMZ-02"]
_SRV  = ["APP-01","WEB-01","API-01","MGMT-01","APP-02",
          "WEB-02","API-02","PROXY-01","APP-03","MAIL-01"]
_DB   = ["DB-01","PAYDB-01","CMDB-01","DB-02","AUTHDB-01","LOGDB-01"]
_LB   = ["LB-01","F5-LB-01","LB-02","HA-PROXY-01","F5-LB-02"]
_WAN  = ["WAN/MPLS","INTERNET","CLOUD/VPC","ISP-A",
          "ISP-B","MPLS-PE","INTERNET-GW","ISP-CORE"]
_LINK = ["10G","1G","MPLS","BGP","OSPF","VLAN-100","VLAN-200",
          "HTTPS/443","SQL/1521","Backup Link","Heartbeat",
          "API","VPN Tunnel","100M","VLAN-300"]

# ─── Zone metadata ────────────────────────────────────────────────────────────
ZONE_FILL = {
    "isp":    "#EBF5FB", "edge":   "#FEF9E7",
    "branch": "#EAFAF1", "dmz":    "#FDEDEC",
    "app":    "#EBF5FB", "db":     "#F5EEF8",
    "cloud":  "#EAF2F8", "mgmt":   "#FDF2E9",
}
ZONE_LABEL = {
    "isp":    "ISP / MPLS Backbone",  "edge":   "Internet Edge",
    "branch": "Branch Core Zone",     "dmz":    "DMZ",
    "app":    "Application Zone",     "db":     "Database Zone",
    "cloud":  "Cloud Zone",           "mgmt":   "Management Zone",
}

# ─── Per-device colours: (fill, outline, highlight) ───────────────────────────
DEV_COL = {
    "router":        ("#1565C0", "#0D47A1", "#BBDEFB"),
    "switch":        ("#2E7D32", "#1B5E20", "#C8E6C9"),
    "firewall":      ("#C62828", "#B71C1C", "#FFCDD2"),
    "server":        ("#455A64", "#263238", "#90A4AE"),
    "database":      ("#6A1B9A", "#4A148C", "#CE93D8"),
    "load_balancer": ("#E65100", "#BF360C", "#FFCCBC"),
    "cloud_or_wan":  ("#0277BD", "#01579B", "#B3E5FC"),
}
_CONN_COLS  = ["#212121", "#1565C0", "#2E7D32", "#E65100", "#7B1FA2"]
_SITE_NAMES = ["NYC-BRANCH-01","LON-HQ-01","SGP-DC-01","CHI-BRANCH-02",
               "SYD-BRANCH-01","FRA-DC-01","DXB-BRANCH-01","LAX-BRANCH-02",
               "TOR-BRANCH-01","MUM-DC-01","BOS-BRANCH-01","AMS-DC-01"]
_REGIONS    = ["EAST","WEST","CENTRAL","APAC","EMEA","LATAM"]
_SITE_TYPES = ["Small Branch","Medium Branch","Large Branch","Regional DC","Hub Site"]

# ═════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load a TrueType font with graceful fallback."""
    candidates = (
        ["C:/Windows/Fonts/calibrib.ttf",
         "C:/Windows/Fonts/arialbd.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]
        if bold else
        ["C:/Windows/Fonts/calibri.ttf",
         "C:/Windows/Fonts/arial.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]
    )
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _tsz(draw, text, font):
    """Return (width, height) of rendered text."""
    try:
        bb = draw.textbbox((0, 0), text, font=font)
        return bb[2] - bb[0], bb[3] - bb[1]
    except AttributeError:
        return draw.textsize(text, font=font)  # Pillow < 9


# ═════════════════════════════════════════════════════════════════════════════
# DIRECTORY SETUP
# ═════════════════════════════════════════════════════════════════════════════

def ensure_dirs(out: Path) -> None:
    for split in ("train", "val", "test"):
        for sub in ("images", "labels", "graphs", "alerts"):
            (out / sub / split).mkdir(parents=True, exist_ok=True)
    (out / "previews").mkdir(parents=True, exist_ok=True)


def get_split(idx: int, total: int) -> str:
    n_train = max(1, round(total * 0.80))
    n_val   = max(1, round(total * 0.13))
    if idx < n_train:           return "train"
    if idx < n_train + n_val:   return "val"
    return "test"


# ═════════════════════════════════════════════════════════════════════════════
# TOPOLOGY GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def _nd(nid: str, ntype: str, zone: str, label: str = None) -> dict:
    return {"id": nid, "type": ntype, "zone": zone,
            "label": label or nid, "cx": 0, "cy": 0, "bbox": None}


def _ed(src: str, dst: str, label=None, style="solid",
        color="#212121", width=2, arrow=False) -> dict:
    return {"source": src, "target": dst, "label": label,
            "style": style, "color": color, "width": width, "arrow": arrow}


def generate_topology(template: str, rng: random.Random, diff: dict = None):
    """Return (nodes, edges) for a named topology template."""
    dp = diff or _DIFF_PROFILE["medium"]
    _srv_lo, _srv_hi = dp["n_srv_range"]
    _db_lo,  _db_hi  = dp["n_db_range"]
    _app_lo, _app_hi = dp["n_app_range"]
    _lb_probs        = dp.get("lb_probs", {})
    _second_lb_p     = dp.get("second_lb_p", 0.0)

    def lb_p(tmpl):  return _lb_probs.get(tmpl, 0.40)
    def rc(lst):   return rng.choice(lst)
    def rlink():   return rc(_LINK)
    def rcolor():  return rc(_CONN_COLS)
    def rstyle():  return rng.choices(["solid","dashed"], weights=[3,1])[0]
    def rwidth():  return rc([2,2,2,3,1])
    def rarrow():  return rng.random() > 0.65
    def conn(s,d): return _ed(s,d, rlink(), rstyle(), rcolor(), rwidth(), rarrow())

    nodes, edges = [], []

    # ── Template 1 ─────────────────────────────────────────────────────────
    if template == "simple_branch":
        n_srv  = rng.randint(_srv_lo, min(_srv_hi, 3))
        n_db   = rng.randint(_db_lo,  min(_db_hi,  2))
        use_lb = rng.random() < lb_p(template)
        nodes  = [
            _nd("WAN-01","cloud_or_wan","isp",   rc(_WAN)),
            _nd("RTR-01","router",      "edge",  _RTR[0]),
            _nd("FW-01", "firewall",    "edge",  _FW[0]),
            _nd("SW-01", "switch",      "branch",_SW[0]),
        ]
        if use_lb:
            nodes.append(_nd("LB-01","load_balancer","branch", rc(_LB)))
        edges = [conn("WAN-01","RTR-01"), conn("RTR-01","FW-01"), conn("FW-01","SW-01")]
        if use_lb:
            edges.append(conn("SW-01","LB-01"))
        _srv_hub = "LB-01" if use_lb else "SW-01"
        for i in range(n_srv):
            nid = f"APP-0{i+1}"
            nodes.append(_nd(nid,"server","app", _SRV[i % len(_SRV)]))
            edges.append(conn(_srv_hub, nid))
        for i in range(n_db):
            nid = f"DB-0{i+1}"
            nodes.append(_nd(nid,"database","db", _DB[i % len(_DB)]))
            edges.append(conn("APP-01" if n_srv else _srv_hub, nid))

    # ── Template 2 ─────────────────────────────────────────────────────────
    elif template == "dual_router_branch":
        n_srv  = rng.randint(max(2, _srv_lo), min(_srv_hi, 6))
        use_lb = rng.random() < lb_p(template)
        nodes  = [
            _nd("WAN-01","cloud_or_wan","isp",   _WAN[0]),
            _nd("WAN-02","cloud_or_wan","isp",   _WAN[3]),
            _nd("RTR-01","router",      "edge",  _RTR[0]),
            _nd("RTR-02","router",      "edge",  _RTR[1]),
            _nd("FW-01", "firewall",    "edge",  _FW[0]),
            _nd("SW-01", "switch",      "branch",_SW[0]),
            _nd("DB-01", "database",    "db",    rc(_DB)),
        ]
        if use_lb:
            nodes.append(_nd("LB-01","load_balancer","branch", rc(_LB)))
        edges = [
            conn("WAN-01","RTR-01"), conn("WAN-02","RTR-02"),
            conn("RTR-01","FW-01"),  conn("RTR-02","FW-01"),
            conn("FW-01","SW-01"),   conn("SW-01","DB-01"),
        ]
        if use_lb:
            edges.append(conn("SW-01","LB-01"))
        # HA second LB when servers are present (medium/hard)
        use_lb2 = use_lb and _second_lb_p > 0 and rng.random() < _second_lb_p
        if use_lb2:
            nodes.append(_nd("LB-02","load_balancer","branch",
                             _LB[1] if len(_LB) > 1 else rc(_LB)))
            edges.append(conn("SW-01","LB-02"))
        _srv_hub = "LB-01" if use_lb else "SW-01"
        for i in range(n_srv):
            nid = f"APP-0{i+1}"
            nodes.append(_nd(nid,"server","app", _SRV[i % len(_SRV)]))
            hub = "LB-02" if (use_lb2 and i % 2 == 1) else _srv_hub
            edges.append(conn(hub, nid))

    # ── Template 3 ─────────────────────────────────────────────────────────
    elif template == "load_balanced_app":
        n_srv   = rng.randint(max(2, _srv_lo), min(_srv_hi, 6))
        use_lb2 = _second_lb_p > 0 and rng.random() < _second_lb_p
        nodes = [
            _nd("WAN-01","cloud_or_wan","isp",   rc(_WAN)),
            _nd("RTR-01","router",      "edge",  rc(_RTR)),
            _nd("FW-01", "firewall",    "edge",  rc(_FW)),
            _nd("SW-01", "switch",      "branch",rc(_SW)),
            _nd("LB-01", "load_balancer","branch",rc(_LB)),
            _nd("DB-01", "database",    "db",    rc(_DB)),
        ]
        if use_lb2:
            nodes.append(_nd("LB-02","load_balancer","branch",
                             _LB[1] if len(_LB) > 1 else rc(_LB)))
        edges = [
            conn("WAN-01","RTR-01"), conn("RTR-01","FW-01"),
            conn("FW-01","SW-01"),   conn("SW-01","LB-01"),
            conn("LB-01","DB-01"),
        ]
        if use_lb2:
            edges.append(conn("SW-01","LB-02"))
        for i in range(n_srv):
            nid = f"APP-0{i+1}"
            nodes.append(_nd(nid,"server","app", _SRV[i % len(_SRV)]))
            # alternate servers between LB-01 and LB-02 when present
            hub = "LB-02" if (use_lb2 and i % 2 == 1) else "LB-01"
            edges.append(conn(hub, nid))

    # ── Template 4 ─────────────────────────────────────────────────────────
    elif template == "dmz_internal":
        use_lb = rng.random() < lb_p(template)
        nodes = [
            _nd("WAN-01", "cloud_or_wan","isp",   rc(_WAN)),
            _nd("FW-EXT", "firewall",    "edge",  _FW[0]),
            _nd("SW-DMZ", "switch",      "dmz",   _SW[0]),
            _nd("WEB-01", "server",      "dmz",   "WEB-01"),
            _nd("FW-INT", "firewall",    "branch",_FW[1]),
            _nd("SW-INT", "switch",      "branch",_SW[1] if len(_SW)>1 else "SW-INT"),
            _nd("APP-01", "server",      "app",   "APP-01"),
            _nd("DB-01",  "database",    "db",    rc(_DB)),
        ]
        if use_lb:
            nodes.append(_nd("LB-01","load_balancer","branch",rc(_LB)))
        edges = [
            conn("WAN-01","FW-EXT"),  conn("FW-EXT","SW-DMZ"),
            conn("SW-DMZ","WEB-01"),  conn("SW-DMZ","FW-INT"),
            conn("FW-INT","SW-INT"),
        ]
        if use_lb:
            edges.extend([conn("SW-INT","LB-01"), conn("LB-01","APP-01")])
        else:
            edges.append(conn("SW-INT","APP-01"))
        edges.append(conn("APP-01","DB-01"))
        if rng.random() > 0.5:
            nodes.append(_nd("WEB-02","server","dmz","WEB-02"))
            edges.append(conn("SW-DMZ","WEB-02"))

    # ── Template 5 ─────────────────────────────────────────────────────────
    elif template == "cloud_extension":
        use_lb = rng.random() < lb_p(template)
        nodes  = [
            _nd("WAN-01",   "cloud_or_wan","isp",   "WAN/MPLS"),
            _nd("RTR-01",   "router",      "edge",  rc(_RTR)),
            _nd("FW-01",    "firewall",    "edge",  rc(_FW)),
            _nd("SW-01",    "switch",      "branch",rc(_SW)),
            _nd("APP-01",   "server",      "app",   "APP-01"),
            _nd("CLOUD-01", "cloud_or_wan","cloud", rc(["CLOUD/VPC","AWS-VPC","AZURE-VNET"])),
            _nd("DB-01",    "database",    "cloud", rc(_DB)),
        ]
        if use_lb:
            nodes.insert(4, _nd("LB-01","load_balancer","branch", rc(_LB)))
        edges = [
            conn("WAN-01","RTR-01"), conn("RTR-01","FW-01"), conn("FW-01","SW-01"),
        ]
        if use_lb:
            edges.extend([conn("SW-01","LB-01"), conn("LB-01","APP-01")])
        else:
            edges.append(conn("SW-01","APP-01"))
        edges.extend([conn("SW-01","CLOUD-01"), conn("CLOUD-01","DB-01")])
        if rng.random() > 0.5:
            nodes.append(_nd("APP-02","server","app","APP-02"))
            edges.append(conn("LB-01" if use_lb else "SW-01","APP-02"))

    # ── Template 6 ─────────────────────────────────────────────────────────
    elif template == "multi_zone_enterprise":
        n_app  = rng.randint(max(2, _app_lo), min(_app_hi, 6))
        use_lb = rng.random() < lb_p(template)
        nodes  = [
            _nd("WAN-01",  "cloud_or_wan","isp",   rc(_WAN)),
            _nd("RTR-01",  "router",      "edge",  rc(_RTR)),
            _nd("FW-01",   "firewall",    "edge",  rc(_FW)),
            _nd("SW-CORE", "switch",      "branch",_SW[0]),
            _nd("SW-APP",  "switch",      "app",   _SW[1] if len(_SW)>1 else "SW-APP"),
            _nd("SW-DB",   "switch",      "db",    _SW[2] if len(_SW)>2 else "SW-DB"),
            _nd("MGMT-01", "server",      "mgmt",  "MGMT-01"),
            _nd("DB-01",   "database",    "db",    rc(_DB)),
        ]
        if use_lb:
            nodes.append(_nd("LB-01","load_balancer","app", rc(_LB)))
        edges = [
            conn("WAN-01","RTR-01"),  conn("RTR-01","FW-01"),
            conn("FW-01","SW-CORE"),  conn("SW-CORE","SW-APP"),
            conn("SW-CORE","SW-DB"),  conn("SW-CORE","MGMT-01"),
            conn("SW-DB","DB-01"),
        ]
        if use_lb:
            edges.append(conn("SW-APP","LB-01"))
        # HA second LB when app servers are present (medium/hard)
        use_lb2 = use_lb and _second_lb_p > 0 and rng.random() < _second_lb_p
        if use_lb2:
            nodes.append(_nd("LB-02","load_balancer","app",
                             _LB[1] if len(_LB) > 1 else rc(_LB)))
            edges.append(conn("SW-APP","LB-02"))
        _app_hub = "LB-01" if use_lb else "SW-APP"
        for i in range(n_app):
            nid = f"APP-0{i+1}"
            nodes.append(_nd(nid,"server","app", _SRV[i % len(_SRV)]))
            hub = "LB-02" if (use_lb2 and i % 2 == 1) else _app_hub
            edges.append(conn(hub, nid))

    # ── Template 7 ─────────────────────────────────────────────────────────
    elif template == "backup_link":
        use_lb = rng.random() < lb_p(template)
        nodes = [
            _nd("WAN-PRI","cloud_or_wan","isp",   "WAN/MPLS"),
            _nd("WAN-BAK","cloud_or_wan","isp",   "INTERNET"),
            _nd("RTR-PRI","router",      "edge",  "RTR-PRI-01"),
            _nd("RTR-BAK","router",      "edge",  "RTR-BAK-01"),
            _nd("FW-01",  "firewall",    "edge",  rc(_FW)),
            _nd("SW-01",  "switch",      "branch",rc(_SW)),
            _nd("APP-01", "server",      "app",   rc(_SRV)),
            _nd("DB-01",  "database",    "db",    rc(_DB)),
        ]
        if use_lb:
            nodes.append(_nd("LB-01","load_balancer","branch",rc(_LB)))
        edges = [
            _ed("WAN-PRI","RTR-PRI", "MPLS",       "solid", "#1565C0", 3),
            _ed("WAN-BAK","RTR-BAK", "Backup Link", "dashed","#E65100", 2),
            conn("RTR-PRI","FW-01"), conn("RTR-BAK","FW-01"),
            conn("FW-01","SW-01"),
        ]
        if use_lb:
            edges.extend([conn("SW-01","LB-01"), conn("LB-01","APP-01")])
        else:
            edges.append(conn("SW-01","APP-01"))
        edges.append(conn("APP-01","DB-01"))

    # ── Template 8 ─────────────────────────────────────────────────────────
    else:  # dense_enterprise
        n_srv = rng.randint(max(3, _srv_lo), min(_srv_hi, 7))
        n_db  = rng.randint(max(1, _db_lo),  min(_db_hi,  3))
        nodes = [
            _nd("WAN-01",  "cloud_or_wan","isp",   "WAN/MPLS"),
            _nd("WAN-02",  "cloud_or_wan","isp",   "INTERNET"),
            _nd("RTR-01",  "router",      "edge",  _RTR[0]),
            _nd("RTR-02",  "router",      "edge",  _RTR[1]),
            _nd("FW-01",   "firewall",    "edge",  _FW[0]),
            _nd("FW-02",   "firewall",    "dmz",   _FW[1]),
            _nd("SW-CORE", "switch",      "branch",_SW[0]),
            _nd("SW-APP",  "switch",      "app",   _SW[1] if len(_SW)>1 else "SW-APP"),
            _nd("LB-01",   "load_balancer","app",  rc(_LB)),
            _nd("MGMT-01", "server",      "mgmt",  "MGMT-01"),
        ]
        edges = [
            conn("WAN-01","RTR-01"),  conn("WAN-02","RTR-02"),
            conn("RTR-01","FW-01"),   conn("RTR-02","FW-01"),
            conn("FW-01","SW-CORE"),  conn("FW-01","FW-02"),
            conn("FW-02","SW-APP"),   conn("SW-CORE","SW-APP"),
            conn("SW-APP","LB-01"),   conn("SW-CORE","MGMT-01"),
        ]
        for i in range(n_srv):
            nid = f"APP-0{i+1}"
            nodes.append(_nd(nid,"server","app", _SRV[i % len(_SRV)]))
            edges.append(conn("LB-01", nid))
        for i in range(n_db):
            nid = f"DB-0{i+1}"
            nodes.append(_nd(nid,"database","db", _DB[i % len(_DB)]))
            edges.append(conn(f"APP-0{min(i+1,n_srv)}", nid))
        if rng.random() > 0.45:
            nodes.append(_nd("CLOUD-01","cloud_or_wan","cloud","CLOUD/VPC"))
            edges.append(conn("SW-CORE","CLOUD-01"))
        # HA load-balancer pair in medium/hard
        if _second_lb_p > 0 and rng.random() < _second_lb_p:
            nodes.append(_nd("LB-02","load_balancer","app",
                             _LB[1] if len(_LB) > 1 else rc(_LB)))
            edges.append(conn("SW-APP","LB-02"))
            for i in range(1, n_srv, 2):   # alternate servers share LB-02
                edges.append(conn("LB-02", f"APP-0{i+1}"))

    # Hard mode: redundant cross-links for visual density
    if dp.get("extra_edges", 0) > 0 and len(nodes) >= 3:
        seen  = {(e["source"], e["target"]) for e in edges}
        seen |= {(e["target"], e["source"]) for e in edges}
        nids  = [n["id"] for n in nodes]
        added = 0
        for _ in range(dp["extra_edges"] * 4):
            if added >= dp["extra_edges"]: break
            s = rng.choice(nids); d = rng.choice(nids)
            if s != d and (s, d) not in seen:
                edges.append(conn(s, d))
                seen.add((s, d)); seen.add((d, s))
                added += 1

    return nodes, edges


# ═════════════════════════════════════════════════════════════════════════════
# LAYOUT ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def layout_topology(nodes, edges, template, rng, diagram_area, jitter=14):
    """Assign pixel (cx,cy) to every node.  Returns {node_id: (cx,cy)}."""
    ax1, ay1, ax2, ay2 = diagram_area
    aw, ah = ax2 - ax1, ay2 - ay1

    def p(fx, fy):
        return (int(ax1 + clamp(fx, 0.02, 0.98) * aw),
                int(ay1 + clamp(fy, 0.05, 0.95) * ah))

    pos = {}
    by_type = defaultdict(list)
    for n in nodes:
        by_type[n["type"]].append(n["id"])

    def tier(type_x_map):
        """Distribute each type's nodes evenly along y."""
        for ntype, xf in type_x_map.items():
            nids = by_type.get(ntype, [])
            cnt = len(nids)
            if not cnt: continue
            for j, nid in enumerate(nids):
                pos[nid] = p(xf, (j + 1) / (cnt + 1))

    # ── per-template assignments ──────────────────────────────────────────────
    if template == "simple_branch":
        tier({"cloud_or_wan": 0.07, "router": 0.20, "firewall": 0.35,
              "switch": 0.50, "load_balancer": 0.62, "server": 0.76, "database": 0.90})

    elif template == "dual_router_branch":
        for j, nid in enumerate(by_type["cloud_or_wan"]): pos[nid] = p(0.07, 0.30+j*0.40)
        for j, nid in enumerate(by_type["router"]):       pos[nid] = p(0.22, 0.30+j*0.40)
        for nid in by_type["firewall"]:      pos[nid] = p(0.38, 0.50)
        for nid in by_type["switch"]:        pos[nid] = p(0.55, 0.50)
        for nid in by_type["load_balancer"]: pos[nid] = p(0.65, 0.50)
        tier({"server": 0.78, "database": 0.91})

    elif template == "load_balanced_app":
        for nid in by_type["cloud_or_wan"]:  pos[nid] = p(0.07, 0.50)
        for nid in by_type["router"]:        pos[nid] = p(0.20, 0.50)
        for nid in by_type["firewall"]:      pos[nid] = p(0.34, 0.50)
        for nid in by_type["switch"]:        pos[nid] = p(0.48, 0.50)
        for nid in by_type["load_balancer"]: pos[nid] = p(0.62, 0.50)
        tier({"server": 0.78, "database": 0.92})

    elif template == "dmz_internal":
        explicit = {
            "WAN-01": p(0.06,0.50), "FW-EXT": p(0.19,0.50),
            "SW-DMZ": p(0.33,0.38), "WEB-01": p(0.47,0.22),
            "WEB-02": p(0.47,0.58), "FW-INT": p(0.33,0.72),
            "SW-INT": p(0.52,0.72), "LB-01":  p(0.65,0.72),
            "APP-01": p(0.70,0.50), "DB-01":  p(0.87,0.50),
        }
        for n in nodes:
            if n["id"] in explicit: pos[n["id"]] = explicit[n["id"]]

    elif template == "cloud_extension":
        wan = [n for n in nodes if n["type"] == "cloud_or_wan"]
        if wan:               pos[wan[0]["id"]] = p(0.07, 0.30)
        if len(wan) > 1:      pos[wan[1]["id"]] = p(0.72, 0.75)
        for nid in by_type["router"]:        pos[nid] = p(0.22, 0.30)
        for nid in by_type["firewall"]:      pos[nid] = p(0.37, 0.30)
        for nid in by_type["switch"]:        pos[nid] = p(0.52, 0.30)
        for nid in by_type["load_balancer"]: pos[nid] = p(0.63, 0.30)
        srvs = by_type.get("server",[])
        for j, nid in enumerate(srvs):  pos[nid] = p(0.76, 0.15+j*0.22)
        dbs  = [n for n in nodes if n["type"]=="database"]
        for j, n  in enumerate(dbs):    pos[n["id"]] = p(0.88, 0.65+j*0.15)

    elif template == "multi_zone_enterprise":
        explicit = {
            "WAN-01": p(0.07,0.50), "RTR-01": p(0.20,0.50),
            "FW-01":  p(0.34,0.50), "SW-CORE":p(0.48,0.50),
            "SW-APP": p(0.60,0.30), "SW-DB":  p(0.60,0.72),
            "MGMT-01":p(0.48,0.84), "DB-01":  p(0.82,0.72),
            "LB-01":  p(0.72,0.30),
        }
        for n in nodes:
            if n["id"] in explicit: pos[n["id"]] = explicit[n["id"]]
        srvs = [n for n in nodes if n["type"]=="server" and n["id"]!="MGMT-01" and n["id"] not in pos]
        for j, n in enumerate(srvs):  pos[n["id"]] = p(0.84, 0.10+j*0.20)
        dbs  = [n for n in nodes if n["type"]=="database" and n["id"] not in pos]
        for j, n in enumerate(dbs):   pos[n["id"]] = p(0.88, 0.50+j*0.20)

    elif template == "backup_link":
        explicit = {
            "WAN-PRI":p(0.07,0.27), "WAN-BAK":p(0.07,0.73),
            "RTR-PRI":p(0.23,0.27), "RTR-BAK":p(0.23,0.73),
            "FW-01":  p(0.40,0.50), "SW-01":  p(0.57,0.50),
            "APP-01": p(0.73,0.50), "DB-01":  p(0.89,0.50),
        }
        for n in nodes:
            if n["id"] in explicit: pos[n["id"]] = explicit[n["id"]]

    else:  # dense_enterprise
        explicit = {
            "WAN-01":  p(0.06,0.22), "WAN-02":  p(0.06,0.66),
            "RTR-01":  p(0.18,0.22), "RTR-02":  p(0.18,0.66),
            "FW-01":   p(0.30,0.38), "FW-02":   p(0.30,0.70),
            "SW-CORE": p(0.42,0.38), "SW-APP":  p(0.42,0.70),
            "LB-01":   p(0.56,0.55), "MGMT-01": p(0.56,0.20),
            "CLOUD-01":p(0.72,0.20),
        }
        for n in nodes:
            if n["id"] in explicit: pos[n["id"]] = explicit[n["id"]]
        srvs = [n for n in nodes if n["type"]=="server" and n["id"]!="MGMT-01" and n["id"] not in pos]
        for j, n in enumerate(srvs): pos[n["id"]] = p(0.72, 0.38+j*0.17)
        dbs  = [n for n in nodes if n["type"]=="database" and n["id"] not in pos]
        for j, n in enumerate(dbs):  pos[n["id"]] = p(0.88, 0.48+j*0.20)

    # ── fill any nodes still unpositioned ────────────────────────────────────
    _TX = {"cloud_or_wan":0.07,"router":0.20,"firewall":0.34,
           "switch":0.48,"load_balancer":0.62,"server":0.76,"database":0.90}
    miss = defaultdict(list)
    for n in nodes:
        if n["id"] not in pos: miss[n["type"]].append(n["id"])
    for ntype, nids in miss.items():
        xf = _TX.get(ntype, 0.5)
        for j, nid in enumerate(nids):
            pos[nid] = p(xf + rng.uniform(-0.02,0.02), (j+1)/(len(nids)+1))

    # ── small position jitter so diagrams aren't perfectly uniform ───────────
    JIT = jitter
    for nid in pos:
        cx, cy = pos[nid]
        pos[nid] = (clamp(cx + rng.randint(-JIT,JIT), ax1+52, ax2-52),
                    clamp(cy + rng.randint(-JIT,JIT), ay1+30, ay2-30))
    return pos


# ═════════════════════════════════════════════════════════════════════════════
# ICON DRAWING  — each returns [x1, y1, x2, y2] bounding box (icon only)
# ═════════════════════════════════════════════════════════════════════════════

def draw_router(draw, cx, cy, scale=1.0):
    fill, out, hi = DEV_COL["router"]
    r  = int(26 * scale)
    lw = max(1, int(2*scale))
    ar = max(4, int(6*scale))

    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=fill, outline=out, width=lw)
    draw.line([(cx-r+5, cy), (cx+r-5, cy)], fill=hi, width=lw)
    draw.line([(cx, cy-r+5), (cx, cy+r-5)], fill=hi, width=lw)
    # Cardinal arrowheads
    for dx, dy in [(r,0),(-r,0),(0,-r),(0,r)]:
        tx, ty = cx+dx, cy+dy
        bx, by = cx+dx//2, cy+dy//2
        if dy == 0:
            pts = [(tx,ty),(bx,by-ar//2),(bx,by+ar//2)]
        else:
            pts = [(tx,ty),(bx-ar//2,by),(bx+ar//2,by)]
        draw.polygon(pts, fill=hi)
    return [cx-r, cy-r, cx+r, cy+r]


def draw_switch(draw, cx, cy, scale=1.0):
    fill, out, hi = DEV_COL["switch"]
    hw = int(36*scale); hh = int(20*scale)
    lw = max(1, int(2*scale))
    draw.rectangle([cx-hw, cy-hh, cx+hw, cy+hh], fill=fill, outline=out, width=lw)
    # Port bank
    n_p = 6; pw = max(3,int(4*scale)); ph = max(4,int(7*scale))
    total_pw = n_p*pw + (n_p-1)*2
    sx = cx - total_pw//2
    for i in range(n_p):
        px = sx + i*(pw+2); py = cy-ph//2
        draw.rectangle([px,py,px+pw,py+ph],
                        fill=("#90EE90" if i<n_p-1 else "#FF6347"),
                        outline="#1B5E20", width=1)
    draw.line([(cx-hw+4,cy),(cx+hw-4,cy)], fill="#FFFFFF", width=1)
    return [cx-hw, cy-hh, cx+hw, cy+hh]


def draw_firewall(draw, cx, cy, scale=1.0):
    fill, out, hi = DEV_COL["firewall"]
    r  = int(28*scale)
    lw = max(1, int(2*scale))
    # Hexagon body
    pts = [(int(cx + r*math.cos(math.radians(i*60-30))),
            int(cy + r*math.sin(math.radians(i*60-30)))) for i in range(6)]
    draw.polygon(pts, fill=fill, outline=out)
    # Inner vertical bars (flame motif)
    n_b = 4; bw = max(2,int(3*scale)); bh = max(10,int(16*scale))
    tot = n_b*bw + (n_b-1)*int(3*scale); sx = cx - tot//2
    for i in range(n_b):
        bx = sx + i*(bw+int(3*scale)); by = cy - bh//2 + i
        draw.rectangle([bx,by,bx+bw,by+bh-i*2], fill=hi, outline=hi)
    return [cx-r, cy-r, cx+r, cy+r]


def draw_server(draw, cx, cy, scale=1.0):
    fill, out, hi = DEV_COL["server"]
    hw = int(32*scale); hh = int(26*scale)
    lw = max(1, int(2*scale))
    draw.rectangle([cx-hw, cy-hh, cx+hw, cy+hh], fill=fill, outline=out, width=lw)
    n_u = 3; uh = max(4,(2*hh-4)//n_u)
    shades = ["#607D8B","#546E7A","#607D8B"]
    for i in range(n_u):
        uy = cy-hh+2+i*uh
        draw.rectangle([cx-hw+2,uy,cx+hw-2,uy+uh-1], fill=shades[i%3])
        lx=cx+hw-9; ly=uy+(uh-1)//2
        draw.ellipse([lx-3,ly-3,lx+3,ly+3],
                      fill=("#00E676" if i<n_u-1 else "#FF5252"))
    return [cx-hw, cy-hh, cx+hw, cy+hh]


def draw_database(draw, cx, cy, scale=1.0):
    fill, out, hi = DEV_COL["database"]
    rw  = int(24*scale); reh = int(9*scale); bh = int(32*scale)
    lw  = max(1, int(2*scale))
    cy_top = cy - bh//2; cy_bot = cy + bh//2

    # Body
    draw.rectangle([cx-rw, cy_top, cx+rw, cy_bot], fill=fill)
    draw.line([(cx-rw,cy_top),(cx-rw,cy_bot)], fill=out, width=lw)
    draw.line([(cx+rw,cy_top),(cx+rw,cy_bot)], fill=out, width=lw)
    # Data-layer arcs in body
    for k in range(1, 3):
        ly = cy_top + k*bh//3
        try:
            draw.arc([cx-rw,ly-reh,cx+rw,ly+reh], start=180, end=0, fill=out, width=1)
        except TypeError:
            draw.arc([cx-rw,ly-reh,cx+rw,ly+reh], start=180, end=0, fill=out)
    # Top ellipse
    draw.ellipse([cx-rw,cy_top-reh,cx+rw,cy_top+reh], fill=fill, outline=out, width=lw)
    # Bottom ellipse
    draw.ellipse([cx-rw,cy_bot-reh,cx+rw,cy_bot+reh], fill=fill, outline=out, width=lw)
    return [cx-rw, cy_top-reh, cx+rw, cy_bot+reh]


def draw_load_balancer(draw, cx, cy, scale=1.0):
    fill, out, hi = DEV_COL["load_balancer"]
    hw = int(32*scale); hh = int(22*scale)
    pts = [(cx,cy-hh),(cx+hw,cy),(cx,cy+hh),(cx-hw,cy)]
    draw.polygon(pts, fill=fill, outline=out)
    n = 3; lh = max(5, int(7*scale))
    for i in range(n):
        ly = cy-(n-1)*lh//2+i*lh
        lx1=cx-hw//3; lx2=cx+hw//2
        draw.line([(lx1,ly),(lx2,ly)], fill=hi, width=max(1,int(1.5*scale)))
        aw=max(3,int(5*scale))
        draw.polygon([(lx2,ly),(lx2-aw,ly-aw//2),(lx2-aw,ly+aw//2)], fill=hi)
    return [cx-hw, cy-hh, cx+hw, cy+hh]


def draw_cloud_or_wan(draw, cx, cy, scale=1.0):
    fill, out, hi = DEV_COL["cloud_or_wan"]
    lw = max(1, int(2*scale))
    r0=int(26*scale); r1=int(16*scale); r2=int(13*scale)
    bumps = [
        (cx,            cy,          r0),
        (cx-int(22*scale), cy+int(4*scale), r1),
        (cx+int(22*scale), cy+int(4*scale), r1),
        (cx-int(11*scale), cy-int(12*scale),r2),
        (cx+int(11*scale), cy-int(12*scale),r2),
    ]
    for bx,by,br in bumps:
        draw.ellipse([bx-br,by-br,bx+br,by+br], fill=fill)
    for bx,by,br in bumps:
        draw.ellipse([bx-br,by-br,bx+br,by+br], outline=out, width=lw)
    # Flat base
    base = cy + int(6*scale)
    draw.rectangle([cx-r0-r1//2, cy, cx+r0+r1//2, base+lw], fill=fill, outline=fill)
    draw.line([(cx-r0-r1//2+lw, base+lw),(cx+r0+r1//2-lw, base+lw)], fill=out, width=lw)
    # Bbox
    hw = r0+r1+lw
    return [cx-hw, cy-r0, cx+hw, cy+r1+lw+2]


_DRAW = {
    "router":        draw_router,
    "switch":        draw_switch,
    "firewall":      draw_firewall,
    "server":        draw_server,
    "database":      draw_database,
    "load_balancer": draw_load_balancer,
    "cloud_or_wan":  draw_cloud_or_wan,
}


# ═════════════════════════════════════════════════════════════════════════════
# CONNECTOR DRAWING
# ═════════════════════════════════════════════════════════════════════════════

def _dashy(draw, pts, color, width, dash=10, gap=5):
    """Draw a dashed polyline."""
    for k in range(len(pts)-1):
        x1,y1=pts[k]; x2,y2=pts[k+1]
        dx,dy=x2-x1,y2-y1
        L=math.hypot(dx,dy)
        if L<1: continue
        ux,uy=dx/L,dy/L
        t,on=0.0,True
        while t<L:
            t2=min(t+(dash if on else gap),L)
            if on:
                draw.line([(int(x1+ux*t),int(y1+uy*t)),
                            (int(x1+ux*t2),int(y1+uy*t2))], fill=color, width=width)
            t=t2; on=not on


def _arrow(draw, p1, p2, color, size=8):
    x1,y1=p1; x2,y2=p2
    dx,dy=x2-x1,y2-y1; L=math.hypot(dx,dy)
    if L<1: return
    ux,uy=dx/L,dy/L; px,py=-uy,ux
    bx=x2-int(ux*size); by=y2-int(uy*size)
    h=size//2
    draw.polygon([(int(x2),int(y2)),
                   (int(bx+px*h),int(by+py*h)),
                   (int(bx-px*h),int(by-py*h))], fill=color)


def _dashed_rect(draw, x1,y1,x2,y2, color, dash=8, gap=4):
    for seg in [((x1,y1),(x2,y1)),((x2,y1),(x2,y2)),
                ((x2,y2),(x1,y2)),((x1,y2),(x1,y1))]:
        _dashy(draw, list(seg), color, 1, dash=dash, gap=gap)


def draw_connector(draw, p1, p2, edge, font=None, rng=None, elbow_prob=0.45):
    style = edge.get("style","solid")
    color = edge.get("color","#212121")
    width = edge.get("width",2)
    arrow = edge.get("arrow",False)
    label = edge.get("label")
    x1,y1=p1; x2,y2=p2

    use_elbow = rng and rng.random() < elbow_prob
    if use_elbow:
        mx=(x1+x2)//2
        pts=[(x1,y1),(mx,y1),(mx,y2),(x2,y2)]
    else:
        pts=[(x1,y1),(x2,y2)]

    (_dashy if style=="dashed" else draw.line)(draw, pts, color, width) if style=="dashed" \
        else draw.line(pts, fill=color, width=width)

    if arrow and len(pts) >= 2:
        _arrow(draw, pts[-2], pts[-1], color, size=max(6,width*3))

    if label and font:
        mid_x=(x1+x2)//2; mid_y=(y1+y2)//2
        tw,th=_tsz(draw,label,font)
        pad=2
        draw.rectangle([mid_x-tw//2-pad, mid_y-th//2-pad,
                         mid_x+tw//2+pad, mid_y+th//2+pad],
                        fill="#FFFFFF", outline="#DDDDDD", width=1)
        draw.text((mid_x-tw//2, mid_y-th//2), label, fill=color, font=font)


# ═════════════════════════════════════════════════════════════════════════════
# FULL DIAGRAM RENDERER
# ═════════════════════════════════════════════════════════════════════════════

def render_diagram(nodes, edges, pos, style, img_path, diagram_id, rng):
    """
    Draw the complete diagram to img_path.
    Mutates nodes in-place (sets cx, cy, bbox).  Returns nodes.
    """
    # ── apply positions ───────────────────────────────────────────────────────
    for n in nodes:
        n["cx"], n["cy"] = pos.get(n["id"], (IMG_W//2, IMG_H//2))

    bg         = style.get("bg",           "#FFFFFF")
    has_meta   = style.get("has_metadata", True)
    has_legend = style.get("has_legend",   True)
    has_footer = style.get("has_footer",   True)
    iscale     = style.get("icon_scale",   1.0)
    site       = style.get("site_name",    "SITE-01")
    region     = style.get("region",       "EAST")
    stype      = style.get("site_type",    "Branch")
    asn        = style.get("as_number",    65001)
    tpl_lbl    = style.get("template_label", diagram_id)
    elbow_p    = style.get("connector_elbow_p", 0.45)
    callout_p  = style.get("callout_p", 0.32)
    lbl_fsz    = style.get("label_font_size", 11)
    left_w     = style.get("left_panel_w", 0)
    apply_ns   = style.get("apply_noise", False)

    meta_w = 195 if has_meta else 0
    hdr_h  = 70
    ftr_h  = 38 if has_footer else 0

    img  = Image.new("RGB", (IMG_W, IMG_H), bg)
    draw = ImageDraw.Draw(img)

    # ── fonts ─────────────────────────────────────────────────────────────────
    f_title = _font(20, bold=True)
    f_sub   = _font(9)
    f_label = _font(lbl_fsz)
    f_small = _font(9)
    f_zone  = _font(9)
    f_meta  = _font(10)
    f_legb  = _font(9, bold=True)
    f_leg   = _font(9)

    # ── header ────────────────────────────────────────────────────────────────
    draw.rectangle([0, 0, IMG_W, hdr_h], fill="#1A237E")
    draw.text((22,12), f"Network Topology — {site}", fill="#FFFFFF", font=f_title)
    draw.text((22,44), f"{tpl_lbl}  |  {region}  |  {stype}  |  AS {asn}",
              fill="#90CAF9", font=f_sub)
    dw,_ = _tsz(draw, diagram_id, f_sub)
    draw.text((IMG_W-meta_w-dw-15, 52), diagram_id, fill="#90CAF9", font=f_sub)

    # ── page border ───────────────────────────────────────────────────────────
    draw.rectangle([1,1,IMG_W-2,IMG_H-2], outline="#9E9E9E", width=1)

    # ── metadata panel ────────────────────────────────────────────────────────
    if has_meta:
        mx1 = IMG_W - meta_w
        draw.rectangle([mx1, hdr_h, IMG_W, IMG_H], fill="#F8F9FA", outline="#E0E0E0", width=1)
        draw.text((mx1+8, hdr_h+8), "SITE INFORMATION", fill="#1A237E", font=_font(10,bold=True))
        draw.line([(mx1+8, hdr_h+26),(IMG_W-8, hdr_h+26)], fill="#BBBBBB", width=1)
        rows = [("Branch",site),("Region",region),("Type",stype),
                ("AS",str(asn)),("Nodes",str(len(nodes))),
                ("Links",str(len(edges))),("Version","v0.1")]
        for i,(k,v) in enumerate(rows):
            y = hdr_h+34+i*22
            draw.text((mx1+8,  y), k, fill="#777777", font=f_meta)
            draw.text((mx1+70, y), v, fill="#212121", font=f_meta)

    # ── footer ────────────────────────────────────────────────────────────────
    if has_footer:
        fy = IMG_H - ftr_h
        draw.line([(0,fy),(IMG_W,fy)], fill="#E0E0E0", width=1)
        draw.text((16, fy+11),
                  f"InfraGraph AI Synthetic Dataset  |  {diagram_id}  |  {tpl_lbl}",
                  fill="#AAAAAA", font=f_small)

    # ── diagram boundary ──────────────────────────────────────────────────────
    dx1=18+left_w; dy1=hdr_h+8; dx2=IMG_W-meta_w-12; dy2=IMG_H-ftr_h-8

    # ── zones (background rectangles) ─────────────────────────────────────────
    by_zone = defaultdict(list)
    for n in nodes: by_zone[n["zone"]].append(n)

    for zone_key, zone_nodes in by_zone.items():
        xs=[n["cx"] for n in zone_nodes]; ys=[n["cy"] for n in zone_nodes]
        if not xs: continue
        pad=30
        zx1=clamp(min(xs)-55-pad, dx1, dx2-30)
        zy1=clamp(min(ys)-50-pad, dy1, dy2-30)
        zx2=clamp(max(xs)+55+pad, dx1+30, dx2)
        zy2=clamp(max(ys)+55+pad, dy1+30, dy2)
        if zx2-zx1<60 or zy2-zy1<40: continue

        draw.rectangle([zx1,zy1,zx2,zy2],
                        fill=ZONE_FILL.get(zone_key,"#F5F5F5"),
                        outline="#CCCCCC", width=1)
        _dashed_rect(draw, zx1,zy1,zx2,zy2, "#BBBBBB")
        draw.text((zx1+6,zy1+4), ZONE_LABEL.get(zone_key,zone_key.upper()),
                  fill="#888888", font=f_zone)

    # ── optional hard-mode left inventory panel ───────────────────────────────
    if left_w > 0:
        _hard_left_panel(draw, nodes, rng,
                         (18, dy1, 18 + left_w, dy2),
                         {"head": _font(8, bold=True), "body": _font(8)})

    # ── connectors ────────────────────────────────────────────────────────────
    nmap = {n["id"]:n for n in nodes}
    for e in edges:
        src=nmap.get(e["source"]); dst=nmap.get(e["target"])
        if src and dst:
            draw_connector(draw,(src["cx"],src["cy"]),(dst["cx"],dst["cy"]),
                           e, font=f_small, rng=rng, elbow_prob=elbow_p)

    # ── device icons + labels ─────────────────────────────────────────────────
    for n in nodes:
        cx,cy=n["cx"],n["cy"]
        fn=_DRAW.get(n["type"])
        if fn:
            s = iscale * rng.uniform(0.92, 1.08)
            bb = fn(draw, cx, cy, scale=s)
            n["bbox"] = [clamp(bb[0],0,IMG_W-1), clamp(bb[1],0,IMG_H-1),
                          clamp(bb[2],0,IMG_W-1), clamp(bb[3],0,IMG_H-1)]
        else:
            n["bbox"] = [cx-20,cy-20,cx+20,cy+20]

        # Label below icon
        lbl_y = n["bbox"][3]+4
        lw,lh = _tsz(draw, n["label"], f_label)
        draw.text((cx-lw//2, lbl_y), n["label"], fill="#1A1A1A", font=f_label)

    # ── legend ────────────────────────────────────────────────────────────────
    if has_legend:
        types_shown = sorted(set(n["type"] for n in nodes))
        nrows = len(types_shown)
        lx=dx1+8; ly=dy2-nrows*16-22
        draw.rectangle([lx-4,ly-18,lx+148,dy2-4],
                        fill="#FAFAFA", outline="#CCCCCC", width=1)
        draw.text((lx,ly-16), "Legend", fill="#333333", font=f_legb)
        for i,ntype in enumerate(types_shown):
            ry=ly+i*16
            draw.rectangle([lx,ry,lx+11,ry+10],
                            fill=DEV_COL[ntype][0], outline="#555555", width=1)
            draw.text((lx+15,ry), ntype.replace("_"," ").title(),
                      fill="#333333", font=f_leg)

    # ── optional callout annotation ───────────────────────────────────────────
    if rng.random() < callout_p and nodes:
        _callout(draw, nodes, rng, f_small, dx2)

    # ── optional watermark (hard mode) ────────────────────────────────────────
    if style.get("has_watermark"):
        wm_txt = "SYNTHETIC"
        wm_fnt = _font(52, bold=True)
        wm_w, wm_h = _tsz(draw, wm_txt, wm_fnt)
        draw.text(((IMG_W - wm_w) // 2, (IMG_H - wm_h) // 2),
                  wm_txt, fill="#F0F0F0", font=wm_fnt)

    # ── document noise (hard mode with --augment-document-noise) ─────────────
    if apply_ns:
        img = apply_document_noise(img, rng)

    img.save(str(img_path), "PNG")
    return nodes


def _callout(draw, nodes, rng, font, dx2):
    """Draw a small annotation callout on a random node."""
    n  = rng.choice(nodes)
    cx,cy = n["cx"],n["cy"]
    opts = [f"IP: 10.{rng.randint(0,255)}.{rng.randint(0,255)}.1",
            f"VLAN {rng.randint(10,400)}", f"Port {rng.randint(1,48)}/48",
            "Status: Active", f"BW: {rng.choice(['10G','1G','100M'])}"]
    txt = rng.choice(opts)
    ox=min(cx+65, dx2-85); oy=cy-45
    tw,th=_tsz(draw,txt,font); pad=4
    draw.rectangle([ox-pad,oy-pad,ox+tw+pad,oy+th+pad],
                    fill="#FFFDE7", outline="#F9A825", width=1)
    draw.text((ox,oy), txt, fill="#333333", font=font)
    draw.line([(ox,oy+th//2),(cx,cy)], fill="#F9A825", width=1)


def _hard_left_panel(draw, nodes, rng, bounds, fonts):
    """Draw a dense device-inventory side panel for hard-mode diagrams."""
    px1, py1, px2, py2 = bounds
    f_head = fonts.get("head") or _font(8, bold=True)
    f_body = fonts.get("body") or _font(8)
    _ABBR  = {"router": "RTR", "switch": "SW", "firewall": "FW",
              "server": "SRV", "database": "DB",
              "load_balancer": "LB", "cloud_or_wan": "WAN"}

    draw.rectangle([px1, py1, px2, py2], fill="#EEF2F7", outline="#BBBBBB", width=1)
    draw.text((px1 + 4, py1 + 4), "DEVICE INVENTORY", fill="#1A237E", font=f_head)
    draw.line([(px1 + 4, py1 + 22), (px2 - 4, py1 + 22)], fill="#AAAAAA", width=1)

    y, row_h = py1 + 28, 22
    for n in nodes:
        if y + row_h > py2 - 4: break
        abbr = _ABBR.get(n["type"], n["type"][:3].upper())
        ip   = f"10.{rng.randint(1,20)}.{rng.randint(0,255)}.{rng.randint(1,254)}"
        draw.text((px1 + 4,  y),      f"{abbr}  {n['id'][:14]}", fill="#212121", font=f_body)
        draw.text((px1 + 8,  y + 11), ip,                         fill="#666666", font=f_body)
        draw.line([(px1 + 4, y + row_h - 1), (px2 - 4, y + row_h - 1)],
                  fill="#DDDDDD", width=1)
        y += row_h


def apply_document_noise(img: "Image.Image", rng: random.Random) -> "Image.Image":
    """Apply mild scan/print noise to a PIL Image (hard-mode augmentation)."""
    import io

    # Slight Gaussian blur — simulates print or scan defocus
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.9)))

    # Brightness and contrast jitter
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.88, 1.08))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.90, 1.10))

    # Subtle horizontal scan-line texture (50 % chance)
    if rng.random() < 0.50:
        overlay = Image.new("RGB", img.size, (255, 255, 255))
        d = ImageDraw.Draw(overlay)
        step = rng.randint(6, 14)
        for y in range(0, img.size[1], step):
            d.line([(0, y), (img.size[0], y)], fill=(215, 215, 215), width=1)
        img = Image.blend(img, overlay, alpha=rng.uniform(0.03, 0.08))

    # JPEG compression artefact (40 % chance)
    if rng.random() < 0.40:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=rng.randint(68, 88))
        buf.seek(0)
        img = Image.open(buf).copy()

    return img


# ═════════════════════════════════════════════════════════════════════════════
# FILE WRITERS
# ═════════════════════════════════════════════════════════════════════════════

def save_yolo_labels(nodes, path):
    lines = []
    for n in nodes:
        bb = n.get("bbox")
        if bb is None: continue
        x1,y1,x2,y2 = bb
        if x2<=x1 or y2<=y1: continue
        cid = CLASS_ID.get(n["type"])
        if cid is None: continue
        cx=clamp((x1+x2)/2/IMG_W, 0,1)
        cy=clamp((y1+y2)/2/IMG_H, 0,1)
        w =clamp((x2-x1)/IMG_W,   0,1)
        h =clamp((y2-y1)/IMG_H,   0,1)
        if w<0.001 or h<0.001: continue
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    with open(path,"w") as f: f.write("\n".join(lines))


def save_graph_json(diagram_id, template, metadata, nodes, edges, path):
    doc = {
        "diagram_id": diagram_id,
        "template":   template,
        "metadata":   metadata,
        "nodes": [{"id":n["id"],"type":n["type"],
                    "bbox":[int(v) for v in (n["bbox"] or [0,0,0,0])],
                    "zone":n["zone"]} for n in nodes],
        "edges": [{"source":e["source"],"target":e["target"],
                    "label":e.get("label",""),"relationship":"connected_to"}
                   for e in edges],
    }
    with open(path,"w") as f: json.dump(doc, f, indent=2)


# ─── Alert / RCA ──────────────────────────────────────────────────────────────

_ALERT_PAT = {
    "database":      [("High DB latency","critical",0),
                      ("SQL connection timeout","major",2),
                      ("App API timeout","major",4),
                      ("LB backend failure","minor",6)],
    "firewall":      [("Packet drops elevated","critical",0),
                      ("Policy deny spike","major",2),
                      ("App unreachable","major",4),
                      ("VPN tunnel instability","minor",6)],
    "router":        [("BGP flap detected","critical",0),
                      ("WAN packet loss > 5%","major",2),
                      ("Site unreachable","critical",3),
                      ("Downstream service degradation","major",5)],
    "cloud_or_wan":  [("BGP session down","critical",0),
                      ("WAN packet loss","major",2),
                      ("Site unreachable","critical",4),
                      ("Downstream degradation","minor",6)],
    "switch":        [("Interface down","critical",0),
                      ("VLAN unreachable","major",2),
                      ("Multiple servers unreachable","major",3)],
    "server":        [("CPU utilization > 95%","critical",0),
                      ("Service unavailable","major",2),
                      ("API latency > 5s","major",3)],
    "load_balancer": [("Backend pool unhealthy","critical",0),
                      ("5xx spike detected","major",2),
                      ("Health check failed","major",3)],
}
_TYPE_W = {"database":3,"firewall":3,"router":2,
           "switch":2,"server":2,"load_balancer":2,"cloud_or_wan":1}


def generate_alert_scenario(diagram_id, nodes, edges, rng):
    if not nodes: return None
    pool = []
    for n in nodes: pool.extend([n]*_TYPE_W.get(n["type"],1))
    root = rng.choice(pool)
    pats = _ALERT_PAT.get(root["type"], _ALERT_PAT["server"])
    sel  = pats[:rng.randint(2, min(4,len(pats)))]

    adj = defaultdict(set)
    for e in edges:
        adj[e["source"]].add(e["target"])
        adj[e["target"]].add(e["source"])
    impacted = list(adj.get(root["id"],set()) - {root["id"]})
    others   = [n for n in nodes if n["id"] != root["id"]]

    alerts = []
    for (atype, sev, toff) in sel:
        if toff == 0:  nid = root["id"]
        elif impacted: nid = rng.choice(impacted)
        elif others:   nid = rng.choice(others)["id"]
        else:          nid = root["id"]
        alerts.append({"node":nid,"alert_type":atype,
                        "severity":sev,"time_offset_min":toff})

    return {"scenario_id":        f"{diagram_id}_incident_001",
            "root_cause":          root["id"],
            "root_cause_type":     root["type"],
            "alerts":              alerts,
            "expected_impacted_nodes": impacted[:5]}


def save_alert_json(scenario, path):
    with open(path,"w") as f: json.dump(scenario, f, indent=2)


# ═════════════════════════════════════════════════════════════════════════════
# CONTACT SHEET
# ═════════════════════════════════════════════════════════════════════════════

def create_contact_sheet(image_paths, out_path, n_cols=4, tw=306, th=192):
    n = len(image_paths)
    if n == 0: return
    n_rows  = math.ceil(n/n_cols)
    lbl_h   = 18
    sheet_w = n_cols*(tw+3)+3
    sheet_h = n_rows*(th+lbl_h+4)+4
    sheet   = Image.new("RGB",(sheet_w,sheet_h),"#1E1E1E")
    draw    = ImageDraw.Draw(sheet)
    fnt     = _font(9)

    for i,path in enumerate(image_paths):
        row=i//n_cols; col=i%n_cols
        x=col*(tw+3)+3; y=row*(th+lbl_h+4)+4
        try:
            thumb=Image.open(path).resize((tw,th),Image.LANCZOS)
            sheet.paste(thumb,(x,y))
        except Exception:
            draw.rectangle([x,y,x+tw,y+th],fill="#3C3C3C",outline="#555555")
        draw.rectangle([x,y+th,x+tw,y+th+lbl_h],fill="#111111")
        draw.text((x+3,y+th+2), Path(path).stem, fill="#AAAAAA", font=fnt)

    sheet.save(out_path)
    print(f"  Contact sheet saved: {out_path}")


def create_annotated_contact_sheet(image_paths, out_dir: Path, out_path,
                                   n_cols=4, tw=306, th=192):
    """Create a contact sheet with YOLO bounding boxes and class names overlaid."""
    _BOX_COLORS = {
        "router":        "#1565C0",
        "switch":        "#2E7D32",
        "firewall":      "#C62828",
        "server":        "#455A64",
        "database":      "#6A1B9A",
        "load_balancer": "#E65100",
        "cloud_or_wan":  "#0277BD",
    }
    n = len(image_paths)
    if n == 0: return
    n_rows  = math.ceil(n / n_cols)
    lbl_h   = 18
    sheet_w = n_cols * (tw + 3) + 3
    sheet_h = n_rows * (th + lbl_h + 4) + 4
    sheet   = Image.new("RGB", (sheet_w, sheet_h), "#1E1E1E")
    sdraw   = ImageDraw.Draw(sheet)
    fnt     = _font(9)
    fnt_box = _font(8)

    for i, img_path in enumerate(image_paths):
        row = i // n_cols; col = i % n_cols
        sx  = col * (tw + 3) + 3
        sy  = row * (th + lbl_h + 4) + 4

        p_img   = Path(img_path)
        lbl_path = out_dir / "labels" / p_img.parent.name / (p_img.stem + ".txt")

        try:
            img   = Image.open(img_path).copy()
            iw, ih = img.size
            idraw = ImageDraw.Draw(img)

            if lbl_path.exists():
                with open(lbl_path) as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) < 5: continue
                        cid  = int(parts[0])
                        if cid >= len(CLASSES): continue
                        cx_n = float(parts[1]); cy_n = float(parts[2])
                        w_n  = float(parts[3]); h_n  = float(parts[4])
                        bx1  = int((cx_n - w_n / 2) * iw)
                        by1  = int((cy_n - h_n / 2) * ih)
                        bx2  = int((cx_n + w_n / 2) * iw)
                        by2  = int((cy_n + h_n / 2) * ih)
                        cls_name = CLASSES[cid]
                        color    = _BOX_COLORS.get(cls_name, "#FFFFFF")
                        idraw.rectangle([bx1, by1, bx2, by2], outline=color, width=3)
                        txt_w, txt_h = _tsz(idraw, cls_name, fnt_box)
                        tag_y = max(0, by1 - txt_h - 2)
                        idraw.rectangle([bx1, tag_y, bx1 + txt_w + 4, tag_y + txt_h + 2],
                                        fill=color)
                        idraw.text((bx1 + 2, tag_y + 1), cls_name,
                                   fill="#FFFFFF", font=fnt_box)

            thumb = img.resize((tw, th), Image.LANCZOS)
            sheet.paste(thumb, (sx, sy))
        except Exception:
            sdraw.rectangle([sx, sy, sx + tw, sy + th],
                            fill="#3C3C3C", outline="#555555")

        sdraw.rectangle([sx, sy + th, sx + tw, sy + th + lbl_h], fill="#111111")
        sdraw.text((sx + 3, sy + th + 2), Path(img_path).stem, fill="#AAAAAA", font=fnt)

    sheet.save(out_path)
    print(f"  Annotated preview saved: {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# DATASET YAML + CLASSES.TXT
# ═════════════════════════════════════════════════════════════════════════════

def write_dataset_yaml(out_dir: Path, path_mode: str = "relative"):
    yp = out_dir / "dataset.yaml"
    cp = out_dir / "classes.txt"
    if path_mode == "absolute":
        path_str = out_dir.resolve().as_posix()
    else:
        p = out_dir.as_posix()
        path_str = p if p.startswith("./") or p.startswith("/") else "./" + p
    with open(yp,"w") as f:
        f.write(f"# InfraGraph AI - YOLO Dataset\n")
        f.write(f"path: {path_str}\n")
        f.write(f"train: images/train\nval:   images/val\ntest:  images/test\n\n")
        f.write(f"nc: {len(CLASSES)}\n\nnames:\n")
        for i,c in enumerate(CLASSES): f.write(f"  {i}: {c}\n")
    with open(cp,"w") as f: f.write("\n".join(CLASSES)+"\n")
    print(f"  dataset.yaml : {yp}")
    print(f"  classes.txt  : {cp}")


# ═════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="InfraGraph AI Dataset Generator")
    p.add_argument("--num",  type=int, default=20,
                   help="Number of diagrams to generate (default: 20)")
    p.add_argument("--out",  type=str, default="./infragraph_dataset",
                   help="Output directory (default: ./infragraph_dataset)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    p.add_argument("--yolo-path-mode", choices=["relative", "absolute"], default="relative",
                   help="Path mode for dataset.yaml (default: relative)")
    p.add_argument("--annotated-preview", action="store_true",
                   help="Generate previews/bbox_contact_sheet.png with YOLO bboxes")
    p.add_argument("--clean", action="store_true",
                   help="Delete existing dataset subfolders before generation")
    p.add_argument("--difficulty", choices=["easy", "medium", "hard", "mixed"],
                   default="mixed",
                   help="Curriculum difficulty (easy/medium/hard/mixed). "
                        "Mixed = 30%% easy / 50%% medium / 20%% hard. (default: mixed)")
    p.add_argument("--augment-document-noise", action="store_true",
                   help="Apply scan/print noise to hard-mode diagrams "
                        "(blur, brightness jitter, scan lines, JPEG artefact)")
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    args    = parse_args()
    rng     = random.Random(args.seed)
    out_dir = Path(args.out)
    total   = args.num

    print(f"\nInfraGraph AI - Synthetic Dataset Generator")
    print(f"  Output     : {out_dir.resolve()}")
    print(f"  Diagrams   : {total}")
    print(f"  Seed       : {args.seed}")
    print(f"  Difficulty : {args.difficulty}")
    if args.augment_document_noise:
        print(f"  Document noise augmentation: ON (hard diagrams only)")
    print()

    if args.clean:
        import shutil
        for sub in ("images", "labels", "graphs", "alerts", "previews"):
            target = out_dir / sub
            if target.exists():
                shutil.rmtree(target)
        print("  Cleaned existing dataset artifacts.\n")

    ensure_dirs(out_dir)
    write_dataset_yaml(out_dir, args.yolo_path_mode)
    print()

    class_cnt  = defaultdict(int)
    split_cnt  = defaultdict(int)
    diff_cnt   = defaultdict(int)
    all_imgs   = []

    for idx in range(total):
        diagram_id = f"diagram_{idx+1:04d}"
        split      = get_split(idx, total)
        split_cnt[split] += 1

        # ── pick difficulty for this diagram ──────────────────────────────────
        difficulty = _pick_difficulty(args.difficulty, rng)
        dp         = _DIFF_PROFILE[difficulty]
        diff_cnt[difficulty] += 1

        # Cycle through the difficulty's template pool, with ~50 % random override
        tmpl_pool = dp["templates"]
        template  = tmpl_pool[idx % len(tmpl_pool)]
        if rng.random() > 0.45:
            template = rng.choice(tmpl_pool)

        nodes, edges = generate_topology(template, rng, diff=dp)

        # ── build difficulty-driven style ─────────────────────────────────────
        has_meta  = rng.random() < dp["has_meta_p"]
        has_left  = rng.random() < dp["has_left_panel_p"]
        isc_lo, isc_hi = dp["icon_scale_range"]

        style = {
            "bg":                rng.choice(["#FFFFFF","#F8F9FA","#FAFAFA","#F5F5F5"]),
            "has_metadata":      has_meta,
            "has_legend":        rng.random() < dp["has_legend_p"],
            "has_footer":        rng.random() < dp["has_footer_p"],
            "icon_scale":        rng.uniform(isc_lo, isc_hi),
            "site_name":         rng.choice(_SITE_NAMES),
            "region":            rng.choice(_REGIONS),
            "site_type":         rng.choice(_SITE_TYPES),
            "as_number":         rng.randint(64512, 65534),
            "template_label":    template.replace("_"," ").title(),
            # difficulty params
            "difficulty":        difficulty,
            "connector_elbow_p": dp["connector_elbow_p"],
            "callout_p":         dp["callout_p"],
            "label_font_size":   dp["label_font_size"],
            "has_left_panel":    has_left,
            "left_panel_w":      165 if has_left else 0,
            "has_watermark":     rng.random() < dp.get("has_watermark_p", 0),
            "apply_noise":       args.augment_document_noise and difficulty == "hard",
        }
        left_panel_w = 165 if has_left else 0
        meta_w       = 195 if has_meta else 0
        diagram_area = (18 + left_panel_w, 78, IMG_W - meta_w - 12, IMG_H - 46)

        pos   = layout_topology(nodes, edges, template, rng, diagram_area,
                                jitter=dp["jitter"])
        img_p = out_dir/"images"/split/f"{diagram_id}.png"
        lbl_p = out_dir/"labels"/split/f"{diagram_id}.txt"
        grp_p = out_dir/"graphs"/split/f"{diagram_id}.json"
        alt_p = out_dir/"alerts"/split/f"{diagram_id}.json"

        nodes = render_diagram(nodes, edges, pos, style, img_p, diagram_id, rng)
        save_yolo_labels(nodes, lbl_p)

        meta = {"branch":style["site_name"],"region":style["region"],
                "site_type":style["site_type"],"version":"v0.1",
                "difficulty":difficulty}
        save_graph_json(diagram_id, template, meta, nodes, edges, grp_p)

        scenario = generate_alert_scenario(diagram_id, nodes, edges, rng)
        if scenario: save_alert_json(scenario, alt_p)

        all_imgs.append(str(img_p))
        for n in nodes: class_cnt[n["type"]] += 1

        if idx == 0 or (idx+1) % 10 == 0 or (idx+1) == total:
            print(f"  [{idx+1:4d}/{total}]  {diagram_id}.png  [{split}]"
                  f"  diff={difficulty:<6s}  tpl={template}  nodes={len(nodes)}")

    # ── Contact sheet ─────────────────────────────────────────────────────────
    if all_imgs:
        create_contact_sheet(all_imgs[:20], str(out_dir/"previews"/"contact_sheet.png"))

    # ── Annotated preview ─────────────────────────────────────────────────────
    if args.annotated_preview and all_imgs:
        create_annotated_contact_sheet(
            all_imgs[:20], out_dir,
            str(out_dir / "previews" / "bbox_contact_sheet.png"),
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  Dataset generation complete")
    print(sep)
    print(f"  Total images : {total}")
    print(f"  train / val / test : "
          f"{split_cnt['train']} / {split_cnt['val']} / {split_cnt['test']}")

    # ── Difficulty summary ────────────────────────────────────────────────────
    print(f"\n  Diagrams by difficulty ({args.difficulty} mode):")
    for d in ("easy", "medium", "hard"):
        cnt = diff_cnt.get(d, 0)
        pct = cnt / total * 100 if total else 0
        bar = "#" * max(1, round(cnt / total * 20)) if total else ""
        print(f"    {d:<8s}  {cnt:5d}  ({pct:4.1f}%)  {bar}")

    print(f"\n  Object instances per class:")
    total_objs = sum(class_cnt.values()) or 1
    mx = max(class_cnt.values()) if class_cnt else 1
    for cls in CLASSES:
        cnt = class_cnt.get(cls, 0)
        pct = cnt / total_objs * 100
        bar = "#" * max(1, round(cnt / mx * 28))
        print(f"    {cls:<20s}  {cnt:5d}  {pct:5.1f}%  {bar}")
    print()
    any_warn = False
    for cls in CLASSES:
        pct = class_cnt.get(cls, 0) / total_objs * 100
        if pct < 5.0:
            print(f"  WARNING: '{cls}' is {pct:.1f}% of total objects (<5% threshold)")
            any_warn = True
    if any_warn:
        print()
    print(f"  Output: {out_dir.resolve()}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
