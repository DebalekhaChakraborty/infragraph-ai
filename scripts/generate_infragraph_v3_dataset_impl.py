#!/usr/bin/env python3
"""
generate_infragraph_v3_dataset.py

V3 enterprise dataset: each scenario generates 3-5 related topology diagram
images. Those exact same diagrams drive RF-DETR detector training AND
enterprise graph stitching.

Pipeline story:
  Separate diagrams are onboarded -> each becomes a local graph ->
  local graphs are stitched into enterprise graph memory ->
  alerts are analyzed across the galaxy graph.

Usage:
    python scripts/generate_infragraph_v3_dataset.py \
        --num-scenarios 100 --out ./datasets/infragraph_v3 \
        --seed 2026 --clean
"""

import argparse
import datetime
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import networkx as nx
    _NX = True
except ImportError:
    _NX = False

# ============================================================
# LAYOUT CONSTANTS
# ============================================================
CANVAS_W      = 1600    # full image width
CANVAS_H      = 1000    # full image height
DPI           = 100
DEVICE_RADIUS = 28
BBOX_PAD      = 5       # tight bbox: wraps just the device shape
TITLE_H       = 72      # reserved top band (title + subtitle)
FOOT_H        = 38      # reserved bottom footer band
RIGHT_BAND_X  = 1388    # x where the right metadata panel begins
DRAW_X0       = 14      # left margin of the local drawing area
DRAW_Y0       = TITLE_H
DRAW_W        = RIGHT_BAND_X - DRAW_X0   # usable drawing width  ~1374
DRAW_H        = CANVAS_H - TITLE_H - FOOT_H  # usable drawing height  ~890

BG           = "#FFFFFF"
PAGE_BG      = "#F8F9FA"
HEADER_BLUE  = "#1A237E"
HEADER_SUB   = "#90CAF9"
CONNECTOR_COLOR = "#37474F"
CROSS_COLOR  = "#0277BD"
TEXT_COLOR   = "#212121"
LABEL_COLOR  = "#424242"
ZONE_LABEL_COLOR = "#607D8B"
GHOST_ALPHA  = 0.70

DEVICE_FILL = {
    "router":        "#1565C0",
    "switch":        "#2E7D32",
    "firewall":      "#C62828",
    "server":        "#455A64",
    "database":      "#6A1B9A",
    "load_balancer": "#E65100",
    "cloud_or_wan":  "#0277BD",
    "service":       "#00897B",
}
DEVICE_EDGE_C = {
    "router":        "#0D47A1",
    "switch":        "#1B5E20",
    "firewall":      "#B71C1C",
    "server":        "#263238",
    "database":      "#4A148C",
    "load_balancer": "#BF360C",
    "cloud_or_wan":  "#01579B",
    "service":       "#00695C",
}
DIAGRAM_TITLES = {
    "branch_topology":          "Branch Office Topology",
    "wan_topology":             "WAN / MPLS Topology",
    "datacenter_topology":      "Datacenter Topology",
    "app_db_topology":          "Application & Database Tier",
    "shared_services_topology": "Shared Services Topology",
}
DIAGRAM_TYPE_COLORS = {
    "branch_topology":          "#3ab89a",
    "wan_topology":             "#3a8fd4",
    "datacenter_topology":      "#e0424a",
    "app_db_topology":          "#6a55d4",
    "shared_services_topology": "#b040e0",
}

COCO_CATS = [
    {"id": 1, "name": "router",        "supercategory": "network_device"},
    {"id": 2, "name": "switch",        "supercategory": "network_device"},
    {"id": 3, "name": "firewall",      "supercategory": "network_device"},
    {"id": 4, "name": "server",        "supercategory": "network_device"},
    {"id": 5, "name": "database",      "supercategory": "network_device"},
    {"id": 6, "name": "load_balancer", "supercategory": "network_device"},
    {"id": 7, "name": "cloud_or_wan",  "supercategory": "network_device"},
    {"id": 8, "name": "service",       "supercategory": "network_device"},
]
COCO_CAT_ID = {c["name"]: c["id"] for c in COCO_CATS}
YOLO_NAMES  = [c["name"] for c in COCO_CATS]
YOLO_ID     = {name: i for i, name in enumerate(YOLO_NAMES)}

DIAGRAM_COMBOS = {
    3: [
        ["branch_topology", "wan_topology", "datacenter_topology"],
        ["branch_topology", "datacenter_topology", "app_db_topology"],
        ["wan_topology", "datacenter_topology", "app_db_topology"],
    ],
    4: [
        ["branch_topology", "wan_topology", "datacenter_topology", "app_db_topology"],
        ["branch_topology", "datacenter_topology", "app_db_topology", "shared_services_topology"],
        ["branch_topology", "wan_topology", "datacenter_topology", "shared_services_topology"],
    ],
    5: [
        ["branch_topology", "wan_topology", "datacenter_topology",
         "app_db_topology", "shared_services_topology"],
    ],
}
SIZE_PROBS = [0.25, 0.35, 0.40]
HERO_DIAGRAM_TYPES = [
    "branch_topology",
    "wan_topology",
    "datacenter_topology",
    "app_db_topology",
    "shared_services_topology",
]

ALERT_SEV_SCORE = {
    "critical": 1.0, "high": 0.8, "warning": 0.6,
    "medium": 0.5,   "low": 0.3,  "info": 0.1,
}

SHARED_ENTITY_DEFS = [
    {"canonical_id": "BR-RTR-01",     "primary": "branch_topology",         "secondary": ["wan_topology"]},
    {"canonical_id": "WAN-MPLS-CORE", "primary": "wan_topology",            "secondary": ["branch_topology"]},
    {"canonical_id": "DC-RTR-01",     "primary": "datacenter_topology",     "secondary": ["wan_topology"]},
    {"canonical_id": "DC-FW-01",      "primary": "datacenter_topology",     "secondary": ["wan_topology"]},
    {"canonical_id": "DC-CORE-SW-01", "primary": "datacenter_topology",     "secondary": ["app_db_topology"]},
    {"canonical_id": "DC-AGG-SW-01",  "primary": "datacenter_topology",     "secondary": ["app_db_topology"]},
    {"canonical_id": "BR-APP-01",     "primary": "branch_topology",         "secondary": ["shared_services_topology"]},
    {"canonical_id": "APP-01",        "primary": "app_db_topology",         "secondary": ["shared_services_topology"]},
    {"canonical_id": "IAM-01",        "primary": "shared_services_topology","secondary": ["app_db_topology"]},
    {"canonical_id": "DNS-01",        "primary": "shared_services_topology","secondary": ["app_db_topology"]},
]

STITCH_RULES = [
    ("branch_topology",     "BR-RTR-01",    "wan_topology",             "WAN-PE-01",    "connected_to",  "MPLS-L3VPN"),
    ("branch_topology",     "WAN-MPLS-CORE","wan_topology",             "WAN-MPLS-CORE","shared_entity", "same-node"),
    ("wan_topology",        "WAN-PE-02",    "datacenter_topology",      "DC-RTR-01",    "connected_to",  "MPLS-L3VPN"),
    ("datacenter_topology", "DC-CORE-SW-01","app_db_topology",          "DC-CORE-SW-01","shared_entity", "same-node"),
    ("datacenter_topology", "DC-CORE-SW-01","app_db_topology",          "APP-LB-01",    "routes_to",     "HTTPS"),
    ("datacenter_topology", "DC-AGG-SW-01", "app_db_topology",          "APP-LB-01",    "routes_to",     "10G"),
    ("branch_topology",     "BR-RTR-01",    "shared_services_topology", "DNS-01",       "depends_on",    "DNS"),
    ("app_db_topology",     "APP-LB-01",    "shared_services_topology", "IAM-01",       "depends_on",    "Auth"),
    ("app_db_topology",     "APP-01",       "shared_services_topology", "IAM-01",       "depends_on",    "LDAP/IAM"),
    ("app_db_topology",     "APP-01",       "shared_services_topology", "DNS-01",       "depends_on",    "DNS"),
    ("app_db_topology",     "DB-MASTER",    "shared_services_topology", "MON-01",       "depends_on",    "Monitor"),
]

ALERT_PATTERNS = {
    "wan_failure": {
        "required": {"wan_topology"},
        "root_cause_diagram": "wan_topology",
        "root_cause_candidates": ["WAN-MPLS-CORE", "WAN-PE-01"],
        "symptoms": {
            "wan_topology":    (["WAN-PE-01", "WAN-PE-02"], ["high", "high"]),
            "branch_topology": (["BR-RTR-01", "BR-FW-01"], ["high", "warning"]),
            "app_db_topology": (["APP-LB-01"], ["warning"]),
        },
    },
    "dc_fw_failure": {
        "required": {"datacenter_topology"},
        "root_cause_diagram": "datacenter_topology",
        "root_cause_candidates": ["DC-FW-01"],
        "symptoms": {
            "datacenter_topology": (["DC-CORE-SW-01", "DC-AGG-SW-01"], ["high", "warning"]),
            "app_db_topology":     (["APP-LB-01", "APP-01"], ["high", "medium"]),
        },
    },
    "shared_id_failure": {
        "required": {"shared_services_topology"},
        "root_cause_diagram": "shared_services_topology",
        "root_cause_candidates": ["IAM-01"],
        "symptoms": {
            "shared_services_topology": (["MON-01"], ["high"]),
            "branch_topology":          (["BR-WRK-01"], ["warning"]),
            "app_db_topology":          (["APP-01"], ["warning"]),
        },
    },
    "lb_failure": {
        "required": {"app_db_topology"},
        "root_cause_diagram": "app_db_topology",
        "root_cause_candidates": ["APP-LB-01"],
        "symptoms": {
            "app_db_topology": (["APP-01", "APP-02", "DB-MASTER"], ["high", "high", "warning"]),
        },
    },
    "db_failure": {
        "required": {"app_db_topology"},
        "root_cause_diagram": "app_db_topology",
        "root_cause_candidates": ["DB-MASTER"],
        "symptoms": {
            "app_db_topology": (["APP-01", "APP-02", "DB-REPLICA-01"], ["high", "high", "warning"]),
        },
    },
    "branch_rtr_failure": {
        "required": {"branch_topology"},
        "root_cause_diagram": "branch_topology",
        "root_cause_candidates": ["BR-RTR-01"],
        "symptoms": {
            "branch_topology": (["BR-FW-01", "BR-SW-01", "BR-WRK-01"], ["high", "warning", "medium"]),
            "wan_topology":    (["WAN-PE-01"], ["warning"]),
        },
    },
}


# ============================================================
# PIXEL COORDINATE HELPER
# coords are (0..1) relative to the local drawing area, not the full canvas
# ============================================================
def _px(nx_norm, ny_norm):
    cx = int(DRAW_X0 + nx_norm * DRAW_W)
    cy = int(DRAW_Y0 + ny_norm * DRAW_H)
    return cx, cy


# ============================================================
# IP GENERATION
# ============================================================
def _ip(zone_tag, idx, scenario_idx):
    s = scenario_idx % 256
    n = (idx % 253) + 1
    if zone_tag == "branch":     return f"10.{s}.1.{n}"
    if zone_tag == "wan":        return f"172.16.{s}.{n}"
    if zone_tag == "datacenter": return f"10.{(s+100)%256}.1.{n}"
    if zone_tag == "appdb":      return f"10.{(s+200)%256}.{(n//30)+1}.{n%30+1}"
    if zone_tag == "shared":     return f"10.255.0.{n}"
    return f"192.168.{s}.{n}"


# ============================================================
# DIAGRAM TEMPLATES
# Rules:
#   - primary_nodes: drawn in local PNG, included in annotation objects
#   - ghost_nodes:   cross-diagram shared entities; metadata only, NOT drawn in PNG
#   - edges with is_cross=True: in annotation JSON only, NOT drawn in PNG
#   - target_diagram key added to cross edges for metadata
# ============================================================

def _branch_template(rng, scenario_idx, diagram_types):
    n_wrk = rng.randint(3, 4)
    # Left-to-right flow: router -> firewall -> switch -> apps/workers
    primary = [
        {"id": "BR-RTR-01", "type": "router",   "zone": "Branch Edge", "pos": (0.13, 0.50),
         "ip_zone": "branch", "is_shared": True,  "canonical_id": "BR-RTR-01"},
        {"id": "BR-FW-01",  "type": "firewall", "zone": "Branch Edge", "pos": (0.30, 0.50),
         "ip_zone": "branch", "is_shared": False, "canonical_id": "BR-FW-01"},
        {"id": "BR-SW-01",  "type": "switch",   "zone": "Branch LAN",  "pos": (0.52, 0.50),
         "ip_zone": "branch", "is_shared": False, "canonical_id": "BR-SW-01"},
        {"id": "BR-APP-01", "type": "server",   "zone": "Branch Apps", "pos": (0.76, 0.28),
         "ip_zone": "branch", "is_shared": False, "canonical_id": "BR-APP-01"},
    ]
    wrk_ys = [0.56, 0.72, 0.42, 0.86][:n_wrk]
    for i, y in enumerate(wrk_ys, 1):
        primary.append({
            "id": f"BR-WRK-{i:02d}", "type": "server", "zone": "Branch LAN",
            "pos": (0.76 + (i % 2) * 0.14, y),
            "ip_zone": "branch", "is_shared": False, "canonical_id": f"BR-WRK-{i:02d}",
        })

    # Ghost nodes: cross-diagram shared entities (metadata only, not drawn in PNG)
    ghost = []
    if "wan_topology" in diagram_types:
        ghost.append({
            "id": "WAN-MPLS-CORE", "type": "cloud_or_wan", "zone": "WAN",
            "pos": (0.03, 0.50), "is_shared": True, "canonical_id": "WAN-MPLS-CORE",
            "_target_diagram": "wan_topology",
        })

    edges = [
        {"src": "WAN-MPLS-CORE", "dst": "BR-RTR-01", "rel": "wan_dependency",
         "label": "MPLS", "is_cross": True, "target_diagram": "wan_topology"},
        {"src": "BR-RTR-01", "dst": "BR-FW-01",  "rel": "routes_to",  "label": "VPN",   "is_cross": False},
        {"src": "BR-FW-01",  "dst": "BR-SW-01",  "rel": "secured_by", "label": "LAN",   "is_cross": False},
        {"src": "BR-SW-01",  "dst": "BR-APP-01", "rel": "serves",     "label": "HTTPS", "is_cross": False},
    ]
    for i in range(1, n_wrk + 1):
        edges.append({
            "src": "BR-SW-01", "dst": f"BR-WRK-{i:02d}",
            "rel": "connected_to", "label": "Access", "is_cross": False,
        })

    # Tighter zone bounds, no external WAN Handoff zone
    zones = [
        {"name": "Branch Edge", "color": "#FEF9E7", "bounds": (0.04, 0.26, 0.42, 0.74)},
        {"name": "Branch LAN",  "color": "#EAFAF1", "bounds": (0.44, 0.26, 0.96, 0.92)},
        {"name": "Branch Apps", "color": "#EBF5FB", "bounds": (0.66, 0.16, 0.96, 0.44)},
    ]
    return primary, ghost, edges, zones


def _wan_template(rng, scenario_idx, diagram_types):
    # Left-to-right: ISP -> PE routers -> MPLS core
    primary = [
        {"id": "WAN-ISP-01",    "type": "cloud_or_wan", "zone": "Internet/ISP",  "pos": (0.09, 0.30),
         "ip_zone": "wan", "is_shared": False, "canonical_id": "WAN-ISP-01"},
        {"id": "WAN-ISP-02",    "type": "cloud_or_wan", "zone": "Internet/ISP",  "pos": (0.09, 0.68),
         "ip_zone": "wan", "is_shared": False, "canonical_id": "WAN-ISP-02"},
        {"id": "WAN-PE-01",     "type": "router",       "zone": "Provider Edge", "pos": (0.30, 0.36),
         "ip_zone": "wan", "is_shared": False, "canonical_id": "WAN-PE-01"},
        {"id": "WAN-PE-02",     "type": "router",       "zone": "Provider Edge", "pos": (0.30, 0.62),
         "ip_zone": "wan", "is_shared": False, "canonical_id": "WAN-PE-02"},
        {"id": "WAN-MPLS-CORE", "type": "cloud_or_wan", "zone": "WAN Core",      "pos": (0.56, 0.50),
         "ip_zone": "wan", "is_shared": False, "canonical_id": "WAN-MPLS-CORE"},
    ]

    # Ghost nodes: cross-diagram refs, metadata only
    ghost = []
    if "branch_topology" in diagram_types:
        ghost.append({
            "id": "BR-RTR-01", "type": "router", "zone": "Branch",
            "pos": (0.80, 0.30), "is_shared": True, "canonical_id": "BR-RTR-01",
            "_target_diagram": "branch_topology",
        })
    if "datacenter_topology" in diagram_types:
        ghost.append({
            "id": "DC-RTR-01", "type": "router", "zone": "DC Edge",
            "pos": (0.80, 0.56), "is_shared": True, "canonical_id": "DC-RTR-01",
            "_target_diagram": "datacenter_topology",
        })
        ghost.append({
            "id": "DC-FW-01", "type": "firewall", "zone": "DC Edge",
            "pos": (0.92, 0.68), "is_shared": True, "canonical_id": "DC-FW-01",
            "_target_diagram": "datacenter_topology",
        })

    edges = [
        {"src": "WAN-ISP-01",    "dst": "WAN-PE-01",     "rel": "routes_to", "label": "BGP",  "is_cross": False},
        {"src": "WAN-ISP-02",    "dst": "WAN-PE-02",     "rel": "routes_to", "label": "BGP",  "is_cross": False},
        {"src": "WAN-PE-01",     "dst": "WAN-MPLS-CORE", "rel": "routes_to", "label": "MPLS", "is_cross": False},
        {"src": "WAN-PE-02",     "dst": "WAN-MPLS-CORE", "rel": "routes_to", "label": "MPLS", "is_cross": False},
    ]
    if "branch_topology" in diagram_types:
        edges.append({
            "src": "WAN-PE-01", "dst": "BR-RTR-01",
            "rel": "connected_to", "label": "L3VPN", "is_cross": True,
            "target_diagram": "branch_topology",
        })
    if "datacenter_topology" in diagram_types:
        edges.append({
            "src": "WAN-PE-02", "dst": "DC-RTR-01",
            "rel": "connected_to", "label": "L3VPN", "is_cross": True,
            "target_diagram": "datacenter_topology",
        })
        edges.append({
            "src": "DC-RTR-01", "dst": "DC-FW-01",
            "rel": "routes_to", "label": "DC Edge", "is_cross": True,
            "target_diagram": "datacenter_topology",
        })

    # No external ghost zones
    zones = [
        {"name": "Internet/ISP",  "color": "#EBF5FB", "bounds": (0.04, 0.18, 0.20, 0.82)},
        {"name": "Provider Edge", "color": "#FEF9E7", "bounds": (0.22, 0.22, 0.44, 0.78)},
        {"name": "WAN Core",      "color": "#EAF2F8", "bounds": (0.46, 0.28, 0.70, 0.72)},
    ]
    return primary, ghost, edges, zones


def _datacenter_template(rng, scenario_idx, diagram_types):
    n_srv = rng.randint(3, 5)
    primary = [
        {"id": "DC-RTR-01",     "type": "router",   "zone": "DC Edge",        "pos": (0.08, 0.50),
         "ip_zone": "datacenter", "is_shared": True,  "canonical_id": "DC-RTR-01"},
        {"id": "DC-FW-01",      "type": "firewall", "zone": "DC Edge",        "pos": (0.22, 0.36),
         "ip_zone": "datacenter", "is_shared": True,  "canonical_id": "DC-FW-01"},
        {"id": "DC-FW-02",      "type": "firewall", "zone": "DC Edge",        "pos": (0.22, 0.64),
         "ip_zone": "datacenter", "is_shared": False, "canonical_id": "DC-FW-02"},
        {"id": "DC-CORE-SW-01", "type": "switch",   "zone": "DC Core",        "pos": (0.40, 0.36),
         "ip_zone": "datacenter", "is_shared": False, "canonical_id": "DC-CORE-SW-01"},
        {"id": "DC-CORE-SW-02", "type": "switch",   "zone": "DC Core",        "pos": (0.40, 0.64),
         "ip_zone": "datacenter", "is_shared": False, "canonical_id": "DC-CORE-SW-02"},
        {"id": "DC-AGG-SW-01",  "type": "switch",   "zone": "DC Aggregation", "pos": (0.58, 0.50),
         "ip_zone": "datacenter", "is_shared": True,  "canonical_id": "DC-AGG-SW-01"},
    ]
    srv_ys = [0.20, 0.36, 0.52, 0.68, 0.82][:n_srv]
    for i, y in enumerate(srv_ys, 1):
        primary.append({
            "id": f"DC-SRV-{i:02d}", "type": "server", "zone": "DC Compute",
            "pos": (0.76 + (i % 2) * 0.12, y),
            "ip_zone": "datacenter", "is_shared": False, "canonical_id": f"DC-SRV-{i:02d}",
        })

    ghost = []  # datacenter has no ghost nodes needed in local PNG

    edges = [
        {"src": "DC-RTR-01",     "dst": "DC-FW-01",      "rel": "routes_to",    "label": "MPLS",   "is_cross": False},
        {"src": "DC-RTR-01",     "dst": "DC-FW-02",      "rel": "routes_to",    "label": "Backup", "is_cross": False},
        {"src": "DC-FW-01",      "dst": "DC-CORE-SW-01", "rel": "secured_by",   "label": "DMZ",    "is_cross": False},
        {"src": "DC-FW-02",      "dst": "DC-CORE-SW-02", "rel": "secured_by",   "label": "VPN",    "is_cross": False},
        {"src": "DC-CORE-SW-01", "dst": "DC-AGG-SW-01",  "rel": "connected_to", "label": "10G",    "is_cross": False},
        {"src": "DC-CORE-SW-02", "dst": "DC-AGG-SW-01",  "rel": "connected_to", "label": "10G",    "is_cross": False},
    ]
    for i in range(1, n_srv + 1):
        edges.append({
            "src": "DC-AGG-SW-01", "dst": f"DC-SRV-{i:02d}",
            "rel": "connected_to", "label": "VLAN", "is_cross": False,
        })

    zones = [
        {"name": "DC Edge",        "color": "#FDEDEC", "bounds": (0.03, 0.20, 0.32, 0.80)},
        {"name": "DC Core",        "color": "#FEF9E7", "bounds": (0.34, 0.20, 0.52, 0.80)},
        {"name": "DC Aggregation", "color": "#EAFAF1", "bounds": (0.54, 0.30, 0.66, 0.70)},
        {"name": "DC Compute",     "color": "#EBF5FB", "bounds": (0.68, 0.12, 0.96, 0.90)},
    ]
    return primary, ghost, edges, zones


def _app_db_template(rng, scenario_idx, diagram_types):
    n_app = rng.randint(2, 3)
    n_rep = rng.randint(1, 2)

    # Three clear tiers: LB ingress | app servers + service | DB tier
    primary = [
        {"id": "APP-LB-01",      "type": "load_balancer", "zone": "App Ingress", "pos": (0.15, 0.50),
         "ip_zone": "appdb", "is_shared": False, "canonical_id": "APP-LB-01"},
        {"id": "DB-MASTER",      "type": "database",      "zone": "DB Tier",     "pos": (0.80, 0.34),
         "ip_zone": "appdb", "is_shared": False, "canonical_id": "DB-MASTER"},
        {"id": "DB-CACHE-01",    "type": "server",        "zone": "DB Tier",     "pos": (0.80, 0.62),
         "ip_zone": "appdb", "is_shared": False, "canonical_id": "DB-CACHE-01"},
        {"id": "SERVICE-API-01", "type": "service",       "zone": "App Tier",    "pos": (0.54, 0.78),
         "ip_zone": "appdb", "is_shared": False, "canonical_id": "SERVICE-API-01"},
    ]
    app_ys = [0.26, 0.50, 0.66][:n_app]
    for i, y in enumerate(app_ys, 1):
        primary.insert(i, {
            "id": f"APP-{i:02d}", "type": "server", "zone": "App Tier",
            "pos": (0.54, y), "ip_zone": "appdb", "is_shared": False,
            "canonical_id": f"APP-{i:02d}",
        })
    for i in range(1, n_rep + 1):
        y = 0.20 + i * 0.46
        primary.append({
            "id": f"DB-REPLICA-{i:02d}", "type": "database", "zone": "DB Tier",
            "pos": (0.92, y), "ip_zone": "appdb", "is_shared": False,
            "canonical_id": f"DB-REPLICA-{i:02d}",
        })

    # Ghost nodes: cross-diagram shared entities (metadata only)
    ghost = []
    if "datacenter_topology" in diagram_types:
        ghost.append({
            "id": "DC-CORE-SW-01", "type": "switch", "zone": "Network",
            "pos": (0.04, 0.36), "is_shared": True, "canonical_id": "DC-CORE-SW-01",
            "_target_diagram": "datacenter_topology",
        })
        ghost.append({
            "id": "DC-AGG-SW-01", "type": "switch", "zone": "Network",
            "pos": (0.04, 0.62), "is_shared": True, "canonical_id": "DC-AGG-SW-01",
            "_target_diagram": "datacenter_topology",
        })
    if "shared_services_topology" in diagram_types:
        ghost.append({
            "id": "IAM-01", "type": "service", "zone": "Shared Services",
            "pos": (0.94, 0.28), "is_shared": True, "canonical_id": "IAM-01",
            "_target_diagram": "shared_services_topology",
        })
        ghost.append({
            "id": "DNS-01", "type": "service", "zone": "Shared Services",
            "pos": (0.94, 0.64), "is_shared": True, "canonical_id": "DNS-01",
            "_target_diagram": "shared_services_topology",
        })

    edges = []
    if "datacenter_topology" in diagram_types:
        edges.append({
            "src": "DC-CORE-SW-01", "dst": "APP-LB-01",
            "rel": "routes_to", "label": "HTTPS", "is_cross": True,
            "target_diagram": "datacenter_topology",
        })
        edges.append({
            "src": "DC-AGG-SW-01", "dst": "APP-LB-01",
            "rel": "routes_to", "label": "10G", "is_cross": True,
            "target_diagram": "datacenter_topology",
        })

    for i in range(1, n_app + 1):
        edges.append({"src": "APP-LB-01",        "dst": f"APP-{i:02d}",   "rel": "serves",     "label": "HTTPS", "is_cross": False})
        edges.append({"src": f"APP-{i:02d}",      "dst": "DB-MASTER",      "rel": "depends_on", "label": "JDBC",  "is_cross": False})
        edges.append({"src": f"APP-{i:02d}",      "dst": "DB-CACHE-01",    "rel": "depends_on", "label": "Cache", "is_cross": False})
        edges.append({"src": f"APP-{i:02d}",      "dst": "SERVICE-API-01", "rel": "depends_on", "label": "API",   "is_cross": False})

    edges.append({
        "src": "SERVICE-API-01", "dst": "IAM-01",
        "rel": "depends_on", "label": "LDAP/IAM", "is_cross": True,
        "target_diagram": "shared_services_topology",
    })
    edges.append({
        "src": "APP-01", "dst": "DNS-01",
        "rel": "depends_on", "label": "DNS", "is_cross": True,
        "target_diagram": "shared_services_topology",
    })
    for i in range(1, n_rep + 1):
        edges.append({
            "src": "DB-MASTER", "dst": f"DB-REPLICA-{i:02d}",
            "rel": "connected_to", "label": "DB Link", "is_cross": False,
        })

    # Clean three-tier zones; no DC network zone, no shared services zone
    zones = [
        {"name": "App Ingress", "color": "#FEF9E7", "bounds": (0.06, 0.28, 0.28, 0.72)},
        {"name": "App Tier",    "color": "#EAFAF1", "bounds": (0.34, 0.16, 0.66, 0.90)},
        {"name": "DB Tier",     "color": "#F5EEF8", "bounds": (0.70, 0.16, 0.96, 0.90)},
    ]
    return primary, ghost, edges, zones


def _shared_services_template(rng, scenario_idx, diagram_types):
    primary = [
        {"id": "AD-01",     "type": "service", "zone": "Identity",   "pos": (0.15, 0.42),
         "ip_zone": "shared", "is_shared": False, "canonical_id": "AD-01"},
        {"id": "IAM-01",    "type": "service", "zone": "Identity",   "pos": (0.34, 0.42),
         "ip_zone": "shared", "is_shared": False, "canonical_id": "IAM-01"},
        {"id": "DNS-01",    "type": "service", "zone": "DNS/NTP",    "pos": (0.56, 0.30),
         "ip_zone": "shared", "is_shared": False, "canonical_id": "DNS-01"},
        {"id": "NTP-01",    "type": "service", "zone": "DNS/NTP",    "pos": (0.56, 0.62),
         "ip_zone": "shared", "is_shared": False, "canonical_id": "NTP-01"},
        {"id": "MON-01",    "type": "service", "zone": "Monitoring", "pos": (0.78, 0.34),
         "ip_zone": "shared", "is_shared": False, "canonical_id": "MON-01"},
        {"id": "SYSLOG-01", "type": "server",  "zone": "Monitoring", "pos": (0.78, 0.64),
         "ip_zone": "shared", "is_shared": False, "canonical_id": "SYSLOG-01"},
    ]

    # Ghost nodes: external consumers (metadata only)
    ghost = []
    if "app_db_topology" in diagram_types:
        ghost.append({
            "id": "APP-01", "type": "server", "zone": "Consumers",
            "pos": (0.92, 0.30), "is_shared": True, "canonical_id": "APP-01",
            "_target_diagram": "app_db_topology",
        })
    if "branch_topology" in diagram_types:
        ghost.append({
            "id": "BR-APP-01", "type": "server", "zone": "Consumers",
            "pos": (0.92, 0.62), "is_shared": True, "canonical_id": "BR-APP-01",
            "_target_diagram": "branch_topology",
        })

    edges = [
        {"src": "AD-01",  "dst": "IAM-01",    "rel": "connected_to", "label": "LDAP",   "is_cross": False},
        {"src": "IAM-01", "dst": "MON-01",    "rel": "monitored_by", "label": "API",    "is_cross": False},
        {"src": "DNS-01", "dst": "NTP-01",    "rel": "connected_to", "label": "NTP",    "is_cross": False},
        {"src": "MON-01", "dst": "SYSLOG-01", "rel": "depends_on",   "label": "Syslog", "is_cross": False},
        {"src": "APP-01",    "dst": "IAM-01", "rel": "depends_on", "label": "OAuth",
         "is_cross": True, "target_diagram": "app_db_topology"},
        {"src": "BR-APP-01", "dst": "DNS-01", "rel": "depends_on", "label": "DNS",
         "is_cross": True, "target_diagram": "branch_topology"},
    ]

    # Three clean zones; no external consumers zone
    zones = [
        {"name": "Identity",   "color": "#F5EEF8", "bounds": (0.04, 0.22, 0.46, 0.62)},
        {"name": "DNS/NTP",    "color": "#EAFAF1", "bounds": (0.48, 0.18, 0.68, 0.78)},
        {"name": "Monitoring", "color": "#FEF9E7", "bounds": (0.70, 0.22, 0.92, 0.78)},
    ]
    return primary, ghost, edges, zones


TEMPLATE_FNS = {
    "branch_topology":          _branch_template,
    "wan_topology":             _wan_template,
    "datacenter_topology":      _datacenter_template,
    "app_db_topology":          _app_db_template,
    "shared_services_topology": _shared_services_template,
}


# ============================================================
# DRAWING FUNCTIONS
# ============================================================

def _draw_device(ax, device_type, cx, cy, is_shared=False):
    R   = DEVICE_RADIUS
    fc  = DEVICE_FILL.get(device_type, "#888888")
    ec  = DEVICE_EDGE_C.get(device_type, "#555555")
    lw  = 1.8
    ls  = "--" if is_shared else "-"
    alp = 0.88

    if device_type == "router":
        ax.add_patch(plt.Circle((cx, cy), R, fc=fc, ec=ec, lw=lw, ls=ls, alpha=alp, zorder=3))
        for dx2, dy2 in [(0, -(R-5)), (0, R-5), (-(R-5), 0), (R-5, 0)]:
            ax.plot([cx, cx+dx2], [cy, cy+dy2], color=ec, lw=1.4, alpha=alp, zorder=4)

    elif device_type == "switch":
        rect = mpatches.FancyBboxPatch((cx-R, cy-int(R*0.58)), R*2, int(R*1.16),
                                        boxstyle="round,pad=2", fc=fc, ec=ec,
                                        lw=lw, ls=ls, alpha=alp, zorder=3)
        ax.add_patch(rect)
        h16 = int(R*1.16)
        for k in range(1, 4):
            yl = cy - int(R*0.58) + k * h16 // 4
            ax.plot([cx-R+5, cx+R-5], [yl, yl], color=ec, lw=1.0, alpha=alp*0.65, zorder=4)

    elif device_type == "firewall":
        pts = [(cx, cy-R), (cx+R, cy-int(R*0.28)), (cx+int(R*0.68), cy+R),
               (cx-int(R*0.68), cy+R), (cx-R, cy-int(R*0.28))]
        ax.add_patch(plt.Polygon(pts, fc=fc, ec=ec, lw=lw, ls=ls, alpha=alp, zorder=3))
        o = int(R*0.35)
        ax.plot([cx-o, cx+o], [cy-o+4, cy+o-4], color="#ffffff", lw=1.5, alpha=alp*0.7, zorder=4)
        ax.plot([cx+o, cx-o], [cy-o+4, cy+o-4], color="#ffffff", lw=1.5, alpha=alp*0.7, zorder=4)

    elif device_type == "server":
        ax.add_patch(mpatches.FancyBboxPatch((cx-R, cy-int(R*0.72)), R*2, int(R*1.44),
                                              boxstyle="square,pad=0", fc=fc, ec=ec,
                                              lw=lw, ls=ls, alpha=alp, zorder=3))
        for k in range(3):
            yl = cy - int(R*0.72) + 7 + k*10
            ax.plot([cx-R+4, cx+R-4], [yl, yl], color=ec, lw=0.9, alpha=alp*0.55, zorder=4)
        ax.add_patch(plt.Circle((cx+R-7, cy-int(R*0.72)+5), 3,
                                fc="#00ff80", ec=None, alpha=alp, zorder=5))

    elif device_type == "database":
        ew = R - 4
        ax.add_patch(mpatches.FancyBboxPatch((cx-ew, cy-int(R*0.55)), ew*2, int(R*1.35),
                                              boxstyle="square,pad=0", fc=fc, ec=ec,
                                              lw=lw, ls=ls, alpha=alp, zorder=3))
        ax.add_patch(mpatches.Ellipse((cx, cy-int(R*0.55)), ew*2, int(R*0.45),
                                       fc=ec, ec=ec, lw=lw, ls=ls, alpha=alp, zorder=4))
        ax.add_patch(mpatches.Ellipse((cx, cy+int(R*0.80)), ew*2, int(R*0.45),
                                       fc=fc, ec=ec, lw=lw, ls=ls, alpha=alp*0.7, zorder=4))

    elif device_type == "load_balancer":
        pts = [(cx, cy-R), (cx+R, cy), (cx, cy+R), (cx-R, cy)]
        ax.add_patch(plt.Polygon(pts, fc=fc, ec=ec, lw=lw, ls=ls, alpha=alp, zorder=3))
        ax.annotate("", xy=(cx, cy+int(R*0.38)), xytext=(cx, cy-int(R*0.38)),
                    arrowprops=dict(arrowstyle="-|>", color="#ffffff", lw=1.4), zorder=5)

    elif device_type == "cloud_or_wan":
        for dx2, dy2, rf in [(-int(R*0.38), int(R*0.10), R*0.55),
                               (int(R*0.38),  int(R*0.10), R*0.55),
                               (0,           -int(R*0.15), R*0.60)]:
            ax.add_patch(plt.Circle((cx+dx2, cy+dy2), rf,
                                    fc=fc, ec=None, alpha=alp*0.75, zorder=3))
        ax.add_patch(plt.Circle((cx, cy), R*0.94, fc="none", ec=ec,
                                lw=lw, ls=ls, alpha=alp, zorder=4))

    elif device_type == "service":
        hex_pts = [(cx + R*math.cos(math.radians(60*i - 30)),
                    cy + R*math.sin(math.radians(60*i - 30))) for i in range(6)]
        ax.add_patch(plt.Polygon(hex_pts, fc=fc, ec=ec, lw=lw, ls=ls, alpha=alp, zorder=3))
        ax.add_patch(plt.Circle((cx, cy), int(R*0.30), fc="#ffffff", ec=None,
                                alpha=alp*0.65, zorder=4))

    else:
        ax.add_patch(plt.Circle((cx, cy), R, fc=fc, ec=ec, lw=lw, ls=ls, alpha=alp, zorder=3))


def _draw_connector(ax, x1, y1, x2, y2, label=""):
    color = CONNECTOR_COLOR
    dx = x2 - x1; dy = y2 - y1
    dist = math.sqrt(dx*dx + dy*dy)
    if dist < 2:
        return
    pad = DEVICE_RADIUS + 6
    sx = x1 + (dx/dist)*pad;  sy = y1 + (dy/dist)*pad
    ex = x2 - (dx/dist)*pad;  ey = y2 - (dy/dist)*pad
    ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                                linestyle="-", mutation_scale=11),
                zorder=2)
    if label:
        mx = (sx+ex)/2;  my = (sy+ey)/2
        # Offset label perpendicular to line
        pdx = -dy/dist*10;  pdy = dx/dist*10
        ax.text(mx+pdx, my+pdy, label, ha="center", va="center",
                fontsize=7, color=LABEL_COLOR, alpha=0.95, zorder=6,
                bbox=dict(fc="#FFFDE7", ec="#E0E0E0", pad=1.8, alpha=0.95))


def _draw_zone(ax, name, x0, y0, x1, y1, color):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0, y0), x1-x0, y1-y0,
        boxstyle="round,pad=3", fc=color, ec="#D0D7DE", lw=1.0, alpha=0.76, zorder=1))
    ax.text(x0+10, y0+16, name.upper(), fontsize=7.0, color=ZONE_LABEL_COLOR,
            fontweight="bold", alpha=0.95, zorder=5, fontfamily="monospace")


def _draw_title(ax, diagram_type, scenario_id):
    ax.add_patch(mpatches.Rectangle((0, 0), CANVAS_W, TITLE_H,
                                    fc=HEADER_BLUE, ec="none", zorder=6))
    ax.plot([0, CANVAS_W], [TITLE_H, TITLE_H], color="#0D47A1", lw=1.5, zorder=7)
    title = DIAGRAM_TITLES.get(diagram_type, diagram_type)
    ax.text(22, 24, f"Network Topology - {title}", ha="left", va="center",
            fontsize=16, color="#FFFFFF", fontweight="bold", zorder=8)
    ax.text(22, 54, f"Diagram Intelligence V3 | Scenario {scenario_id} | local topology view",
            ha="left", va="center", fontsize=8.5, color=HEADER_SUB, zorder=8)
    ax.text(CANVAS_W - 20, 54, diagram_type, ha="right", va="center",
            fontsize=8, color=HEADER_SUB, alpha=0.95, zorder=8)
    badge = DIAGRAM_TYPE_COLORS.get(diagram_type, "#888888")
    ax.add_patch(plt.Circle((CANVAS_W - 20, 24), 5, fc=badge, ec="#FFFFFF", lw=0.8, zorder=8))


def _draw_right_band(ax, scenario_id, diagram_type, primary_nodes, ghost_nodes, edges):
    """Metadata panel and legend in the reserved right band; never overlaps drawing area."""
    x0 = RIGHT_BAND_X + 6
    w  = CANVAS_W - x0 - 6

    # Vertical separator
    ax.plot([RIGHT_BAND_X, RIGHT_BAND_X],
            [DRAW_Y0 + 4, CANVAS_H - FOOT_H - 4],
            color="#D0D7DE", lw=1.0, zorder=3)

    # ---- Site information box ----
    y0 = DRAW_Y0 + 10
    local_e  = sum(1 for e in edges if not e.get("is_cross"))
    cross_e  = sum(1 for e in edges if e.get("is_cross"))
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0, y0), w, 162, boxstyle="round,pad=3",
        fc="#F8F9FA", ec="#E0E0E0", lw=1.0, alpha=0.96, zorder=4))
    ax.text(x0+8, y0+16, "SITE INFO", fontsize=8.0,
            color=HEADER_BLUE, fontweight="bold", zorder=5)
    short_sid = scenario_id.replace("enterprise_v3_", "v3-")
    rows = [
        ("Scenario", short_sid),
        ("Type",     diagram_type.replace("_topology", "").replace("_", " ")),
        ("Nodes",    str(len(primary_nodes))),
        ("Local E.", str(local_e)),
        ("Cross E.", str(cross_e)),
        ("Version",  "v3"),
    ]
    for i, (k, v) in enumerate(rows):
        y = y0 + 36 + i * 19
        ax.text(x0+8,  y, k, fontsize=6.8, color="#777777", zorder=5)
        ax.text(x0+62, y, v[:16], fontsize=6.8, color=TEXT_COLOR, zorder=5)

    # ---- Legend box ----
    y1 = y0 + 175
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0, y1), w, 148, boxstyle="round,pad=3",
        fc="#FFFFFF", ec="#E0E0E0", lw=1.0, alpha=0.96, zorder=4))
    ax.text(x0+8, y1+16, "LEGEND", fontsize=8.0, color=HEADER_BLUE,
            fontweight="bold", zorder=5)
    entries = ["router", "switch", "firewall", "server",
               "database", "load_balancer", "cloud_or_wan", "service"]
    for i, cls in enumerate(entries):
        y = y1 + 34 + i * 14
        ax.add_patch(plt.Circle((x0+14, y-2), 4,
                                fc=DEVICE_FILL[cls], ec=DEVICE_EDGE_C[cls],
                                lw=0.6, zorder=5))
        ax.text(x0+24, y, cls.replace("_", " "), fontsize=6.2, color=LABEL_COLOR,
                va="center", zorder=5)

    # ---- Cross-diagram reference card ----
    if ghost_nodes:
        y2 = y1 + 162
        n  = len(ghost_nodes)
        h  = 34 + n * 18
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y2), w, h, boxstyle="round,pad=3",
            fc="#EBF5FB", ec="#90CAF9", lw=0.9, alpha=0.92, zorder=4))
        ax.text(x0+8, y2+14, "EXT. REFS", fontsize=7.0, color="#0277BD",
                fontweight="bold", zorder=5)
        for i, g in enumerate(ghost_nodes):
            y = y2 + 28 + i * 18
            tgt = g.get("_target_diagram", g.get("zone", "")).replace("_topology", "")
            ax.text(x0+8, y, f"{g['id']}  [{tgt}]",
                    fontsize=5.8, color="#01579B", zorder=5, fontfamily="monospace")


def draw_diagram(scenario_id, diagram_type, primary_nodes, ghost_nodes, edges, zones, out_path):
    """Render local-topology diagram to PNG.

    Ghost nodes and cross-diagram edges are kept in JSON metadata but are NOT
    drawn in the local PNG.  Returns {node_id: (cx, cy)} pixel map (all nodes).
    """
    fig = plt.figure(figsize=(CANVAS_W/DPI, CANVAS_H/DPI), dpi=DPI, facecolor=BG)
    ax  = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(CANVAS_H, 0)
    ax.axis("off")

    # Canvas border
    ax.add_patch(mpatches.Rectangle((0, 0), CANVAS_W, CANVAS_H,
                                    fc=BG, ec="#9E9E9E", lw=1.0, zorder=0))
    # Drawing area background (left of right band, below title, above footer)
    ax.add_patch(mpatches.Rectangle(
        (DRAW_X0, DRAW_Y0 + 8),
        DRAW_W - 4,
        DRAW_H - 8,
        fc=PAGE_BG, ec="#ECEFF1", lw=0.8, zorder=0))

    _draw_title(ax, diagram_type, scenario_id)
    _draw_right_band(ax, scenario_id, diagram_type, primary_nodes, ghost_nodes, edges)

    # Compute pixel positions for ALL nodes (incl. ghosts — needed for connector endpoints)
    node_pixels = {}
    for n in primary_nodes + ghost_nodes:
        cx, cy = _px(*n["pos"])
        node_pixels[n["id"]] = (cx, cy)

    # Draw zones (within drawing area only)
    for z in zones:
        x0n, y0n, x1n, y1n = z["bounds"]
        px0 = int(DRAW_X0 + x0n * DRAW_W)
        py0 = int(DRAW_Y0 + y0n * DRAW_H)
        px1 = int(DRAW_X0 + x1n * DRAW_W)
        py1 = int(DRAW_Y0 + y1n * DRAW_H)
        _draw_zone(ax, z["name"], px0, py0, px1, py1, z["color"])

    # Draw LOCAL connectors only (skip cross-diagram edges)
    iface_names = ["Gi0/1", "eth0", "ge-0/0/1", "Gi0/2", "eth1"]
    for edge_idx, e in enumerate(edges):
        if e.get("is_cross"):
            continue   # cross-diagram edges belong in the enterprise graph view
        src, dst = e["src"], e["dst"]
        if src in node_pixels and dst in node_pixels:
            x1p, y1p = node_pixels[src]
            x2p, y2p = node_pixels[dst]
            _draw_connector(ax, x1p, y1p, x2p, y2p, label=e.get("label", ""))
            # Interface stub labels (small, monospace, near connector endpoints)
            iface_a = iface_names[edge_idx % len(iface_names)]
            iface_b = iface_names[(edge_idx + 2) % len(iface_names)]
            ax.text(x1p + 8, y1p + 12, iface_a, ha="left", va="center",
                    fontsize=4.8, color=ZONE_LABEL_COLOR, alpha=0.78,
                    fontfamily="monospace", zorder=5)
            ax.text(x2p - 8, y2p - 12, iface_b, ha="right", va="center",
                    fontsize=4.8, color=ZONE_LABEL_COLOR, alpha=0.78,
                    fontfamily="monospace", zorder=5)

    # Draw PRIMARY nodes only (ghost nodes are not rendered in local PNG)
    for n in primary_nodes:
        cx, cy = node_pixels[n["id"]]
        _draw_device(ax, n["type"], cx, cy, is_shared=n.get("is_shared", False))
        # Label below the device (single line — avoid overlapping connectors above)
        ax.text(cx, cy + DEVICE_RADIUS + 14, n["id"],
                ha="center", va="top",
                fontsize=7.5, color=TEXT_COLOR, fontweight="bold",
                zorder=6, fontfamily="monospace",
                bbox=dict(fc="#FFFFFF", ec="none", pad=0.7, alpha=0.82))
        # IP address smaller, below label
        ip = n.get("ip_address", "")
        if ip:
            ax.text(cx, cy + DEVICE_RADIUS + 28, ip,
                    ha="center", va="top",
                    fontsize=5.8, color=ZONE_LABEL_COLOR, alpha=0.90,
                    zorder=6, fontfamily="monospace",
                    bbox=dict(fc="#FFFFFF", ec="none", pad=0.4, alpha=0.65))
        # Small shared-entity marker
        if n.get("is_shared"):
            ax.text(cx + DEVICE_RADIUS + 4, cy - DEVICE_RADIUS,
                    "*", ha="left", va="top",
                    fontsize=9, color=CROSS_COLOR, fontweight="bold", zorder=6)

    # Footer
    footer_y = CANVAS_H - FOOT_H + 8
    ax.plot([0, CANVAS_W], [footer_y, footer_y], color="#E0E0E0", lw=1.0, zorder=6)
    ax.text(16, CANVAS_H - 14,
            f"InfraGraph AI | Diagram Intelligence V3 | {scenario_id} | {diagram_type}",
            fontsize=7, color="#9E9E9E", ha="left", va="center", zorder=7)

    fig.savefig(str(out_path), dpi=DPI, bbox_inches=None, facecolor=BG)
    plt.close(fig)
    return node_pixels


# ============================================================
# ANNOTATION GENERATION
# ============================================================

def make_annotation(scenario_id, diagram_id, diagram_type, image_path,
                    primary_nodes, ghost_nodes, edges, zones, node_pixels):
    """Generate V3 annotation JSON.

    Objects: primary nodes only (ghost nodes not visible in local PNG).
    Connectors: all edges, with is_cross_diagram flag; only local ones are drawn.
    """
    R = DEVICE_RADIUS + BBOX_PAD  # tight bbox around device shape

    objects = []
    for n in primary_nodes:
        cx, cy = node_pixels[n["id"]]
        x1 = max(0,        cx - R)
        y1 = max(0,        cy - R)
        x2 = min(CANVAS_W, cx + R)
        y2 = min(CANVAS_H, cy + R)
        objects.append({
            "object_id":        n["id"],
            "class_name":       n["type"],
            "bbox":             [x1, y1, x2, y2],
            "bbox_format":      "xyxy",
            "confidence":       1.0,
            "label_text":       n["id"],
            "ip_address":       n.get("ip_address", ""),
            "zone":             n["zone"],
            "is_shared_entity": n.get("is_shared", False),
            "canonical_id":     n.get("canonical_id", n["id"]),
        })
    # Ghost nodes are cross-diagram shared entities; not visible in local PNG,
    # so they are intentionally excluded from annotation objects.

    # Text blocks (primary nodes + zones only)
    text_blocks = []
    for n in primary_nodes:
        cx, cy = node_pixels[n["id"]]
        lbl = n["id"]
        ew  = max(20, len(lbl) * 5)
        text_blocks.append({
            "text":      lbl,
            "bbox":      [cx - ew, cy + DEVICE_RADIUS + 4, cx + ew, cy + DEVICE_RADIUS + 18],
            "text_type": "node_label",
        })
        ip = n.get("ip_address", "")
        if ip:
            iw = max(16, len(ip) * 4)
            text_blocks.append({
                "text":      ip,
                "bbox":      [cx - iw, cy + DEVICE_RADIUS + 18, cx + iw, cy + DEVICE_RADIUS + 30],
                "text_type": "ip_address",
            })
    for z in zones:
        x0n, y0n, _, _ = z["bounds"]
        zx = int(DRAW_X0 + x0n * DRAW_W) + 10
        zy = int(DRAW_Y0 + y0n * DRAW_H) + 8
        zw = max(42, len(z["name"]) * 6)
        text_blocks.append({
            "text":      z["name"],
            "bbox":      [zx, zy, min(CANVAS_W, zx + zw), min(CANVAS_H, zy + 16)],
            "text_type": "zone_label",
        })

    # Connectors — all edges kept in JSON, with is_cross_diagram flag
    all_pos = {n["id"]: node_pixels[n["id"]] for n in primary_nodes + ghost_nodes}
    connectors = []
    iface_names = ["Gi0/1", "eth0", "ge-0/0/1", "Gi0/2", "eth1"]
    for i, e in enumerate(edges):
        if e["src"] in all_pos and e["dst"] in all_pos:
            sx, sy = all_pos[e["src"]]
            dx2, dy2 = all_pos[e["dst"]]
            mx, my   = (sx + dx2) // 2, (sy + dy2) // 2
            is_cross  = e.get("is_cross", False)
            connectors.append({
                "connector_id":     f"edge_{i:03d}",
                "source":           e["src"],
                "target":           e["dst"],
                "relationship":     e["rel"],
                "label":            e.get("label", ""),
                "confidence":       1.0,
                "source_interface": iface_names[i % len(iface_names)],
                "target_interface": iface_names[(i + 2) % len(iface_names)],
                "points":           [[sx, sy], [mx, my], [dx2, dy2]],
                "bbox":             [min(sx,dx2)-4, min(sy,dy2)-4, max(sx,dx2)+4, max(sy,dy2)+4],
                "style":            "dashed" if is_cross else "solid",
                "is_cross_diagram": is_cross,
                "source_diagram":   diagram_type,
                "target_diagram":   e.get("target_diagram", diagram_type if not is_cross else ""),
            })
            if e.get("label"):
                ew = max(24, len(e["label"]) * 5)
                text_blocks.append({
                    "text":      e["label"],
                    "bbox":      [mx - ew, my - 10, mx + ew, my + 10],
                    "text_type": "connector_label",
                })

    return {
        "scenario_id":  scenario_id,
        "diagram_id":   diagram_id,
        "diagram_type": diagram_type,
        "image_path":   str(image_path),
        "width":        CANVAS_W,
        "height":       CANVAS_H,
        "objects":      objects,
        "text_blocks":  text_blocks,
        "connectors":   connectors,
    }


def make_yolo_label(annotation):
    lines = []
    for obj in annotation["objects"]:
        cid = YOLO_ID.get(obj["class_name"])
        if cid is None:
            continue
        x1, y1, x2, y2 = obj["bbox"]
        cx = ((x1 + x2) / 2) / CANVAS_W
        cy = ((y1 + y2) / 2) / CANVAS_H
        bw = (x2 - x1) / CANVAS_W
        bh = (y2 - y1) / CANVAS_H
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines)


# ============================================================
# LOCAL GRAPH
# ============================================================

def make_local_graph(scenario_id, diagram_id, diagram_type, primary_nodes, ghost_nodes, edges, node_pixels):
    diagram_nodes    = primary_nodes + ghost_nodes
    diagram_node_ids = {n["id"] for n in diagram_nodes}
    nodes_out = []
    for n in diagram_nodes:
        cx, cy = node_pixels[n["id"]]
        nodes_out.append({
            "id":               n["id"],
            "label":            n["id"],
            "type":             n["type"],
            "ip_address":       n.get("ip_address", ""),
            "zone":             n["zone"],
            "diagram_id":       diagram_id,
            "diagram_type":     diagram_type,
            "is_shared_entity": n.get("is_shared", False),
            "canonical_id":     n.get("canonical_id", n["id"]),
            "bbox":             [cx - DEVICE_RADIUS, cy - DEVICE_RADIUS,
                                 cx + DEVICE_RADIUS, cy + DEVICE_RADIUS],
        })
    edges_out = []
    for i, e in enumerate(edges):
        if e["src"] in diagram_node_ids and e["dst"] in diagram_node_ids:
            edges_out.append({
                "source":       e["src"],
                "target":       e["dst"],
                "relationship": e["rel"],
                "label":        e.get("label", ""),
                "edge_scope":   "local",
                "connector_id": f"edge_{i:03d}",
            })
    return {
        "scenario_id":  scenario_id,
        "diagram_id":   diagram_id,
        "diagram_type": diagram_type,
        "nodes":        nodes_out,
        "edges":        edges_out,
    }


# ============================================================
# STITCH MAP
# ============================================================

def make_stitch_map(scenario_id, diagram_types):
    present = set(diagram_types)
    shared_entities = []
    for sed in SHARED_ENTITY_DEFS:
        if sed["primary"] in present:
            secondary = [d for d in sed["secondary"] if d in present]
            if secondary:
                appears_in = [{"diagram_id": sed["primary"], "local_id": sed["canonical_id"]}]
                for d in secondary:
                    appears_in.append({"diagram_id": d, "local_id": sed["canonical_id"]})
                shared_entities.append({
                    "canonical_id": sed["canonical_id"],
                    "appears_in":   appears_in,
                })
    cross_edges = []
    for (sd, sn, td, tn, rel, lbl) in STITCH_RULES:
        if sd in present and td in present:
            cross_edges.append({
                "source":         sn, "source_diagram": sd,
                "target":         tn, "target_diagram":  td,
                "relationship":   rel, "label":           lbl,
            })
    return {
        "scenario_id":         scenario_id,
        "diagram_types":       diagram_types,
        "shared_entities":     shared_entities,
        "cross_diagram_edges": cross_edges,
    }


# ============================================================
# ENTERPRISE GRAPH
# ============================================================

def build_enterprise_graph(scenario_id, diagram_types, local_graphs, stitch_map):
    all_nodes       = {}
    diagram_clusters = {}
    for lg in local_graphs:
        did = lg["diagram_id"]
        diagram_clusters[did] = {
            "diagram_id":   did,
            "diagram_type": lg["diagram_type"],
            "node_ids":     [n.get("canonical_id", n["id"]) for n in lg["nodes"]],
        }
        for n in lg["nodes"]:
            cid = n.get("canonical_id", n["id"])
            if cid not in all_nodes:
                nn = dict(n)
                nn["id"] = cid
                all_nodes[cid] = nn
            else:
                all_nodes[cid]["is_shared_entity"] = True

    all_edges = []
    for lg in local_graphs:
        for e in lg["edges"]:
            all_edges.append(dict(
                e,
                edge_type="local",
                source_diagram=lg["diagram_id"],
                target_diagram=lg["diagram_id"],
            ))

    cross_edges = []
    for ce in stitch_map.get("cross_diagram_edges", []):
        entry = {
            "source":         ce["source"],
            "target":         ce["target"],
            "relationship":   ce["relationship"],
            "label":          ce.get("label", ""),
            "edge_scope":     "cross_diagram",
            "edge_type":      "cross_diagram",
            "source_diagram": ce["source_diagram"],
            "target_diagram": ce["target_diagram"],
        }
        all_edges.append(entry)
        cross_edges.append(entry)

    shared_ids = [n["id"] for n in all_nodes.values() if n.get("is_shared_entity")]
    return {
        "scenario_id":         scenario_id,
        "diagram_types":       diagram_types,
        "nodes":               list(all_nodes.values()),
        "edges":               all_edges,
        "cross_diagram_edges": cross_edges,
        "diagram_clusters":    list(diagram_clusters.values()),
        "shared_entities":     stitch_map.get("shared_entities", []),
        "stats": {
            "num_nodes":               len(all_nodes),
            "num_edges":               len(all_edges),
            "num_cross_diagram_edges": len(cross_edges),
            "num_shared_entities":     len(shared_ids),
        },
    }


# ============================================================
# ALERTS
# ============================================================

def make_alerts(scenario_id, diagram_types, enterprise_graph, rng):
    present      = set(diagram_types)
    all_node_ids = {n["id"] for n in enterprise_graph["nodes"]}

    valid = []
    for pname, pdef in ALERT_PATTERNS.items():
        if pdef["required"].issubset(present):
            cands = [c for c in pdef["root_cause_candidates"] if c in all_node_ids]
            if cands:
                valid.append((pname, pdef, cands))

    if not valid:
        nodes = enterprise_graph["nodes"]
        if not nodes:
            return None
        rc_node      = rng.choice(nodes)
        rc           = rc_node["id"]
        rc_diag      = rc_node.get("diagram_id", diagram_types[0])
        pattern_name = "generic_failure"
    else:
        pname, pdef, cands = rng.choice(valid)
        rc           = rng.choice(cands)
        rc_diag      = pdef["root_cause_diagram"]
        pattern_name = pname

    alert_types_rc  = ["packet_drop", "cpu_spike", "link_down", "interface_flap"]
    alert_types_sym = ["latency", "connection_timeout", "error_rate", "packet_loss"]

    alerts = [{
        "node":            rc,
        "diagram_id":      rc_diag,
        "alert_type":      rng.choice(alert_types_rc),
        "severity":        "critical",
        "time_offset_min": 0,
    }]

    impacted_nodes    = []
    impacted_diagrams = {rc_diag}
    impact_paths      = []
    t = 1

    pdef_syms = ALERT_PATTERNS.get(pattern_name, {}).get("symptoms", {})
    for diag_id, (sym_nodes, severities) in pdef_syms.items():
        if diag_id in present:
            for sn, sev in zip(sym_nodes, severities):
                if sn in all_node_ids and sn != rc:
                    alerts.append({
                        "node":            sn,
                        "diagram_id":      diag_id,
                        "alert_type":      rng.choice(alert_types_sym),
                        "severity":        sev,
                        "time_offset_min": t,
                    })
                    t += rng.randint(2, 8)
                    impacted_nodes.append(sn)
                    impacted_diagrams.add(diag_id)
                    impact_paths.append([rc, sn])

    sev_rank = {"critical": 4, "high": 3, "warning": 2, "medium": 1, "low": 0, "info": 0}
    sev_rev  = {4: "critical", 3: "high", 2: "warning", 1: "medium", 0: "low"}
    max_rank = max((sev_rank.get(a["severity"], 0) for a in alerts), default=1)
    return {
        "scenario_id":          scenario_id,
        "root_cause":           rc,
        "root_cause_diagram":   rc_diag,
        "root_cause_pattern":   pattern_name,
        "severity":             sev_rev.get(max_rank, "medium"),
        "alerts":               alerts,
        "impacted_nodes":       impacted_nodes,
        "impacted_diagrams":    list(impacted_diagrams),
        "impact_paths":         impact_paths,
    }


def make_hero_alerts(scenario_id, enterprise_graph):
    node_ids = {n["id"] for n in enterprise_graph.get("nodes", [])}
    preferred_path = [
        "DC-FW-01", "DC-CORE-SW-01", "DC-AGG-SW-01",
        "APP-LB-01", "APP-01", "SERVICE-API-01", "DB-MASTER", "IAM-01",
    ]
    path = [n for n in preferred_path if n in node_ids]
    root = path[0] if path else next(iter(node_ids), "DC-FW-01")
    alerts = []
    alert_plan = [
        (root,            "datacenter_topology",      "packet_drop",            "critical", 0),
        ("DC-CORE-SW-01", "datacenter_topology",      "link_errors",            "high",     3),
        ("APP-LB-01",     "app_db_topology",           "backend_pool_unhealthy", "high",     6),
        ("APP-01",        "app_db_topology",           "api_latency",            "warning",  9),
        ("DB-MASTER",     "app_db_topology",           "connection_timeout",     "warning", 12),
        ("IAM-01",        "shared_services_topology",  "auth_errors",            "warning", 15),
        ("BR-APP-01",     "branch_topology",           "user_timeout",           "warning", 18),
    ]
    for node, diagram_id, alert_type, severity, offset in alert_plan:
        if node in node_ids:
            alerts.append({
                "node":            node,
                "diagram_id":      diagram_id,
                "alert_type":      alert_type,
                "severity":        severity,
                "time_offset_min": offset,
            })
    impacted_nodes    = [a["node"] for a in alerts if a["node"] != root]
    impacted_diagrams = sorted({a["diagram_id"] for a in alerts})
    return {
        "scenario_id":          scenario_id,
        "root_cause":           root,
        "root_cause_diagram":   "datacenter_topology",
        "root_cause_pattern":   "hero_dc_fw_to_app_db_identity",
        "severity":             "critical",
        "alerts":               alerts,
        "impacted_nodes":       impacted_nodes,
        "impacted_diagrams":    impacted_diagrams,
        "impact_paths":         [path] if path else [],
    }


# ============================================================
# PREVIEW IMAGES
# ============================================================

def _load_png_as_array(path):
    try:
        import matplotlib.image as mpimg
        return mpimg.imread(str(path))
    except Exception:
        return None


def _graph_node_color(ntype):
    return DEVICE_FILL.get(ntype, "#78909C")


def _local_graph_positions(nodes, x0, y0, w, h):
    by_type = {}
    for n in nodes:
        by_type.setdefault(n.get("type", "server"), []).append(n)
    tiers = [
        ["cloud_or_wan", "router"],
        ["firewall", "load_balancer"],
        ["switch", "service"],
        ["server"],
        ["database"],
    ]
    pos  = {}
    used = set()
    for ti, types in enumerate(tiers):
        tier_nodes = []
        for t in types:
            tier_nodes.extend(by_type.get(t, []))
        if not tier_nodes:
            continue
        px = x0 + (ti + 1) * w / (len(tiers) + 1)
        for j, node in enumerate(tier_nodes):
            py = y0 + (j + 1) * h / (len(tier_nodes) + 1)
            pos[node["id"]] = (px, py)
            used.add(node["id"])
    leftovers = [n for n in nodes if n["id"] not in used]
    for j, node in enumerate(leftovers):
        pos[node["id"]] = (x0 + 0.15 * w, y0 + (j + 1) * h / (len(leftovers) + 1))
    return pos


def _draw_local_graph_card(ax, lg, x0, y0, w, h, color, title=None, badge=True, alerts_data=None):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0, y0), w, h, boxstyle="round,pad=0.035",
        fc="#FFFFFF", ec=color, lw=1.5, zorder=2))
    if title:
        ax.text(x0 + 0.12, y0 + h - 0.18, title, fontsize=8.5,
                color=TEXT_COLOR, fontweight="bold", ha="left", va="top", zorder=5)
    if badge:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0 + w - 1.55, y0 + h - 0.34), 1.35, 0.20,
            boxstyle="round,pad=0.03", fc="#2E7D32", ec="none", alpha=0.95, zorder=5))
        ax.text(x0 + w - 0.875, y0 + h - 0.24, "Local Graph Extracted",
                ha="center", va="center", fontsize=5.8, color="#FFFFFF", zorder=6)

    nodes    = lg.get("nodes", [])
    edges    = lg.get("edges", [])
    pos      = _local_graph_positions(nodes, x0 + 0.25, y0 + 0.18, w - 0.50, h - 0.55)
    rc       = (alerts_data or {}).get("root_cause")
    impacted = set((alerts_data or {}).get("impacted_nodes", []))
    alerting = {a.get("node") for a in (alerts_data or {}).get("alerts", [])}

    for e in edges:
        s, t = e.get("source"), e.get("target")
        if s in pos and t in pos:
            x1, y1 = pos[s]; x2, y2 = pos[t]
            ax.plot([x1, x2], [y1, y2], color="#607D8B", lw=1.0, alpha=0.75, zorder=3)

    for n in nodes:
        nid = n["id"]
        if nid not in pos:
            continue
        x, y = pos[nid]
        if nid == rc:
            fc, ec, size = "#E53935", "#B71C1C", 0.090
        elif nid in alerting:
            fc, ec, size = "#FB8C00", "#E65100", 0.078
        elif nid in impacted:
            fc, ec, size = "#FDD835", "#F9A825", 0.078
        else:
            fc, ec, size = _graph_node_color(n.get("type")), "#263238", 0.065
        ax.add_patch(plt.Circle((x, y), size, fc=fc, ec=ec, lw=0.7, zorder=5))
        if n.get("is_shared_entity"):
            ax.add_patch(plt.Circle((x, y), size + 0.035, fc="none", ec="#0277BD",
                                    lw=1.0, ls="--", zorder=5))
        ax.text(x, y - 0.12, nid, fontsize=4.7, ha="center", va="top",
                color=TEXT_COLOR, fontfamily="monospace", zorder=6)
    return pos


def make_preview_contact_sheet(scenario_id, diagram_types, diagram_dir, out_path):
    n    = len(diagram_types)
    cols = 2 if n <= 4 else 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*7.4, rows*5.0), facecolor="#F5F5F5")
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for idx, dtype in enumerate(diagram_types):
        r, c = divmod(idx, cols)
        ax   = axes[r][c] if rows > 1 else axes[0][c]
        arr  = _load_png_as_array(diagram_dir / f"{dtype}.png")
        if arr is not None:
            ax.imshow(arr, aspect="auto")
        else:
            ax.set_facecolor("#FFFFFF")
            ax.text(0.5, 0.5, dtype.replace("_", "\n"), ha="center", va="center",
                    transform=ax.transAxes, color=TEXT_COLOR, fontsize=10)
        badge_color = DIAGRAM_TYPE_COLORS.get(dtype, "#888888")
        ax.set_title(DIAGRAM_TITLES.get(dtype, dtype), fontsize=10, color=TEXT_COLOR,
                     pad=4, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_edgecolor(badge_color)
            spine.set_linewidth(2)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.015, 0.025), 0.24, 0.050, transform=ax.transAxes,
            boxstyle="round,pad=0.01", fc="#2E7D32", ec="none", alpha=0.95, zorder=5))
        ax.text(0.135, 0.050, "Local Graph Extracted", transform=ax.transAxes,
                ha="center", va="center", fontsize=5.8, color="white",
                fontweight="bold", zorder=6)
        lg_path = diagram_dir.parent / "local_graphs" / f"{dtype}.json"
        if lg_path.exists():
            try:
                with open(lg_path) as f:
                    lg = json.load(f)
                count_text = f"nodes={len(lg.get('nodes', []))}  edges={len(lg.get('edges', []))}"
                ax.text(0.99, 0.04, count_text, transform=ax.transAxes,
                        ha="right", va="center", fontsize=7, color=TEXT_COLOR,
                        bbox=dict(boxstyle="round,pad=0.2", fc="#FFFFFFDD", ec="#B0BEC5", lw=0.5),
                        zorder=6)
            except Exception:
                pass

    for idx in range(n, rows*cols):
        r, c = divmod(idx, cols)
        ax   = axes[r][c] if rows > 1 else axes[0][c]
        ax.set_visible(False)

    fig.suptitle(f"Diagram Intelligence V3 Contact Sheet - {scenario_id}", fontsize=14,
                 color=TEXT_COLOR, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(str(out_path), dpi=110, facecolor="#F5F5F5")
    plt.close(fig)


def make_preview_bbox_contact_sheet(scenario_id, diagram_types, diagram_dir, out_path):
    """Contact sheet with annotation bounding boxes overlaid on each diagram thumbnail."""
    n    = len(diagram_types)
    if n == 0:
        return
    ann_dir = diagram_dir.parent / "annotations"
    cols    = 2 if n <= 4 else 3
    rows    = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*7.4, rows*5.0), facecolor="#F5F5F5")
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for idx, dtype in enumerate(diagram_types):
        r, c = divmod(idx, cols)
        ax   = axes[r][c] if rows > 1 else axes[0][c]
        arr  = _load_png_as_array(diagram_dir / f"{dtype}.png")
        if arr is not None:
            ax.imshow(arr, aspect="auto")
        else:
            ax.set_facecolor("#FFFFFF")

        ann_path = ann_dir / f"{dtype}.json"
        if ann_path.exists():
            try:
                with open(ann_path) as f:
                    ann = json.load(f)
                orig_w = ann.get("width", CANVAS_W)
                orig_h = ann.get("height", CANVAS_H)
                h_px   = arr.shape[0] if arr is not None else orig_h
                w_px   = arr.shape[1] if arr is not None else orig_w
                sx, sy = w_px / orig_w, h_px / orig_h
                for obj in ann.get("objects", []):
                    x1, y1, x2, y2 = obj["bbox"]
                    cls   = obj.get("class_name", "")
                    color = DEVICE_FILL.get(cls, "#888888")
                    rect  = mpatches.Rectangle(
                        (x1*sx, y1*sy), (x2-x1)*sx, (y2-y1)*sy,
                        linewidth=1.5, edgecolor=color, facecolor="none", zorder=5)
                    ax.add_patch(rect)
                count = len(ann.get("objects", []))
                ax.text(0.99, 0.04, f"{count} objects",
                        transform=ax.transAxes, ha="right", va="center",
                        fontsize=7, color=TEXT_COLOR,
                        bbox=dict(boxstyle="round,pad=0.2", fc="#FFFFFFDD", ec="#B0BEC5", lw=0.5),
                        zorder=6)
            except Exception:
                pass

        badge_color = DIAGRAM_TYPE_COLORS.get(dtype, "#888888")
        ax.set_title(DIAGRAM_TITLES.get(dtype, dtype), fontsize=10, color=TEXT_COLOR,
                     pad=4, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_edgecolor(badge_color)
            spine.set_linewidth(2)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for idx in range(n, rows*cols):
        r, c = divmod(idx, cols)
        ax   = axes[r][c] if rows > 1 else axes[0][c]
        ax.set_visible(False)

    fig.suptitle(f"Annotation Bbox Sheet - {scenario_id}", fontsize=14,
                 color=TEXT_COLOR, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(str(out_path), dpi=110, facecolor="#F5F5F5")
    plt.close(fig)


def make_preview_stitching_story(scenario_id, diagram_types, diagram_dir, out_path):
    scenario_dir    = diagram_dir.parent
    local_dir       = scenario_dir / "local_graphs"
    enterprise_path = scenario_dir / "enterprise_graph.json"
    alerts_path     = scenario_dir / "alerts.json"
    enterprise_graph = json.load(open(enterprise_path)) if enterprise_path.exists() else {}
    alerts_data      = json.load(open(alerts_path))     if alerts_path.exists()     else {}

    fig = plt.figure(figsize=(22, 13), facecolor="#F5F5F5")
    ax  = fig.add_axes([0, 0, 1, 1], facecolor="#F5F5F5")
    ax.set_xlim(0, 22); ax.set_ylim(0, 13); ax.axis("off")

    ax.add_patch(mpatches.Rectangle((0, 12.25), 22, 0.75, fc=HEADER_BLUE, ec="none"))
    ax.text(11, 12.64, "Diagram Intelligence V3 - From Source Diagrams to Enterprise Graph Memory",
            ha="center", va="center", fontsize=19, color="#FFFFFF", fontweight="bold")
    ax.text(11, 12.20, scenario_id, ha="center", va="top", fontsize=10, color=HEADER_SUB)

    stage_x      = [0.55, 7.55, 14.35]
    stage_w      = [6.25, 5.65, 7.05]
    stage_titles = ["Stage 1: Source topology diagrams", "Stage 2: Local graph extraction", "Stage 3: Enterprise galaxy graph"]
    stage_colors = ["#1565C0", "#2E7D32", "#0277BD"]
    for x, w, title, clr in zip(stage_x, stage_w, stage_titles, stage_colors):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, 0.55), w, 11.35, boxstyle="round,pad=0.08",
            fc="#FFFFFF", ec=clr, lw=1.7, zorder=1))
        ax.text(x+0.18, 11.65, title, fontsize=12, color=clr,
                fontweight="bold", ha="left", va="center", zorder=4)

    for sx in [6.95, 13.55]:
        ax.annotate("", xy=(sx+0.42, 6.2), xytext=(sx, 6.2),
                    arrowprops=dict(arrowstyle="-|>", color="#37474F", lw=3.0, mutation_scale=22))

    n      = len(diagram_types)
    card_h = min(1.85, 9.7 / max(n, 1))
    start_y = 11.05 - card_h
    for idx, dtype in enumerate(diagram_types):
        y    = start_y - idx * (card_h + 0.18)
        clr  = DIAGRAM_TYPE_COLORS.get(dtype, "#607D8B")
        img_arr = _load_png_as_array(diagram_dir / f"{dtype}.png")
        ax.add_patch(mpatches.FancyBboxPatch((0.85, y), 5.65, card_h,
                                             boxstyle="round,pad=0.035",
                                             fc="#FFFFFF", ec=clr, lw=1.3, zorder=2))
        if img_arr is not None:
            ax.imshow(img_arr, extent=(1.00, 6.35, y+0.15, y+card_h-0.30),
                      aspect="auto", zorder=3)
        lg_path = local_dir / f"{dtype}.json"
        lg      = json.load(open(lg_path)) if lg_path.exists() else {"nodes": [], "edges": []}
        ax.text(1.02, y+card_h-0.12, DIAGRAM_TITLES.get(dtype, dtype),
                fontsize=8.4, color=TEXT_COLOR, fontweight="bold", ha="left", va="top", zorder=5,
                bbox=dict(fc="#FFFFFFDD", ec="none", pad=1.0))
        ax.text(6.32, y+0.07, f"nodes={len(lg.get('nodes', []))}  edges={len(lg.get('edges', []))}",
                fontsize=6.6, color=LABEL_COLOR, ha="right", va="bottom", zorder=5,
                bbox=dict(fc="#FFFFFFDD", ec="#CFD8DC", pad=1.0))
        _draw_local_graph_card(ax, lg, 7.88, y, 5.00, card_h, clr,
                               title=DIAGRAM_TITLES.get(dtype, dtype), alerts_data=alerts_data)

    _draw_enterprise_cluster_graph(ax, enterprise_graph, alerts_data, 14.75, 1.05, 6.45, 10.05)
    ax.text(17.98, 0.72, "Dashed cyan links stitch local graphs into enterprise memory",
            ha="center", va="center", fontsize=8.2, color=LABEL_COLOR)

    fig.savefig(str(out_path), dpi=100, facecolor="#F5F5F5")
    plt.close(fig)


def _draw_rca_summary_box(ax, alerts_data, x0, y0, w, h, title="Cross-Diagram RCA Context"):
    alerts_data = alerts_data or {}
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0, y0), w, h, boxstyle="round,pad=0.055",
        fc="#FFFFFF", ec="#B0BEC5", lw=1.0, zorder=20))
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0 + 0.12, y0 + h - 0.44), w - 0.24, 0.30,
        boxstyle="round,pad=0.03", fc="#E3F2FD", ec="none", zorder=21))
    ax.text(x0 + 0.22, y0 + h - 0.29, title, fontsize=8.4,
            color="#01579B", fontweight="bold", ha="left", va="center", zorder=22)

    rc        = alerts_data.get("root_cause", "unknown")
    rc_diag   = alerts_data.get("root_cause_diagram", "unknown")
    impacted  = alerts_data.get("impacted_diagrams", [])
    alert_count = len(alerts_data.get("alerts", []))
    path      = alerts_data.get("impact_paths", [[]])[0] if alerts_data.get("impact_paths") else []
    path_text = " -> ".join(path[:6])
    if len(path) > 6:
        path_text += " -> ..."

    if h < 2.5:
        impacted_text = ", ".join(d.replace("_topology", "") for d in impacted[:5]) or "none"
        rows = [
            f"Root Cause: {rc}  |  Diagram: {rc_diag.replace('_', ' ')}",
            f"Impacted: {impacted_text}  |  Alerts: {alert_count}",
            f"Key Path: {(path_text or 'none')[:92]}",
        ]
        yy = y0 + h - 0.76
        for row in rows:
            ax.text(x0 + 0.18, yy, row, fontsize=6.6, color=TEXT_COLOR,
                    ha="left", va="top", zorder=22)
            yy -= 0.36
    else:
        rows = [
            ("Root Cause",       rc),
            ("Root Diagram",     rc_diag.replace("_", " ")),
            ("Impacted Diagrams",", ".join(d.replace("_topology", "") for d in impacted[:4]) or "none"),
            ("Alert Count",      str(alert_count)),
            ("Key Impact Path",  path_text or "none"),
        ]
        yy = y0 + h - 0.74
        for label, value in rows:
            ax.text(x0 + 0.18, yy, label, fontsize=6.8, color=LABEL_COLOR,
                    fontweight="bold", ha="left", va="top", zorder=22)
            ax.text(x0 + 0.18, yy - 0.18, value, fontsize=6.3, color=TEXT_COLOR,
                    ha="left", va="top", zorder=22, wrap=True)
            yy -= 0.58 if label != "Key Impact Path" else 0.76


def make_preview_stitching_story_hero(scenario_id, diagram_types, diagram_dir, out_path):
    scenario_dir    = diagram_dir.parent
    local_dir       = scenario_dir / "local_graphs"
    enterprise_path = scenario_dir / "enterprise_graph.json"
    alerts_path     = scenario_dir / "alerts.json"
    enterprise_graph = json.load(open(enterprise_path)) if enterprise_path.exists() else {}
    alerts_data      = json.load(open(alerts_path))     if alerts_path.exists()     else {}

    fig = plt.figure(figsize=(24, 14), facecolor="#F5F7FA")
    ax  = fig.add_axes([0, 0, 1, 1], facecolor="#F5F7FA")
    ax.set_xlim(0, 24); ax.set_ylim(0, 14); ax.axis("off")

    ax.add_patch(mpatches.Rectangle((0, 13.05), 24, 0.95, fc=HEADER_BLUE, ec="none", zorder=1))
    ax.text(12, 13.58, "InfraGraph AI - Diagram Intelligence V3 to Enterprise Graph Memory",
            ha="center", va="center", fontsize=21, color="#FFFFFF", fontweight="bold", zorder=2)
    ax.text(12, 13.16,
            "Same source diagrams -> local graphs -> stitched galaxy graph -> cross-diagram RCA context",
            ha="center", va="center", fontsize=10.5, color=HEADER_SUB, zorder=2)

    stage_defs = [
        (0.45,  0.55, 6.95, 12.05, "#1565C0", "1  Source topology diagrams"),
        (7.95,  0.55, 5.95, 12.05, "#2E7D32", "2  Local graph extraction"),
        (14.45, 0.55, 9.10, 12.05, "#0277BD", "3  Enterprise galaxy + RCA"),
    ]
    for x, y, w, h, color, title in stage_defs:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08",
            fc="#FFFFFF", ec=color, lw=1.7, zorder=1))
        ax.text(x + 0.22, y + h - 0.34, title, fontsize=12.2, color=color,
                fontweight="bold", ha="left", va="center", zorder=4)

    for sx in [7.48, 13.98]:
        ax.annotate("", xy=(sx + 0.34, 6.70), xytext=(sx, 6.70),
                    arrowprops=dict(arrowstyle="-|>", color="#263238", lw=3.3,
                                    mutation_scale=24), zorder=12)

    n      = len(diagram_types)
    card_h = min(2.04, 10.65 / max(n, 1))
    start_y = 11.95 - card_h
    for idx, dtype in enumerate(diagram_types):
        y       = start_y - idx * (card_h + 0.18)
        clr     = DIAGRAM_TYPE_COLORS.get(dtype, "#607D8B")
        img_arr = _load_png_as_array(diagram_dir / f"{dtype}.png")
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.78, y), 6.20, card_h, boxstyle="round,pad=0.045",
            fc="#FFFFFF", ec=clr, lw=1.25, zorder=2))
        if img_arr is not None:
            ax.imshow(img_arr, extent=(0.96, 6.80, y + 0.16, y + card_h - 0.30),
                      aspect="auto", zorder=3)
        lg_path = local_dir / f"{dtype}.json"
        lg      = json.load(open(lg_path)) if lg_path.exists() else {"nodes": [], "edges": []}
        ax.text(1.00, y + card_h - 0.13, DIAGRAM_TITLES.get(dtype, dtype),
                fontsize=8.6, color=TEXT_COLOR, fontweight="bold", ha="left", va="top",
                bbox=dict(fc="#FFFFFFE6", ec="none", pad=1.2), zorder=6)
        ax.text(6.76, y + 0.08, f"{len(lg.get('nodes', []))} nodes  |  {len(lg.get('edges', []))} edges",
                fontsize=6.8, color=LABEL_COLOR, ha="right", va="bottom",
                bbox=dict(fc="#FFFFFFE6", ec="#CFD8DC", pad=1.0), zorder=6)
        _draw_local_graph_card(ax, lg, 8.28, y, 5.25, card_h, clr,
                               title=DIAGRAM_TITLES.get(dtype, dtype), alerts_data=alerts_data)

    _draw_enterprise_cluster_graph(ax, enterprise_graph, alerts_data, 14.82, 3.10, 8.22, 8.88)
    _draw_rca_summary_box(ax, alerts_data, 15.00, 0.82, 7.85, 1.95, title="RCA Summary")
    ax.text(19.0, 2.26,
            "Dashed cyan links are stitched cross-diagram dependencies from the same source diagrams.",
            fontsize=7.8, color=LABEL_COLOR, ha="center", va="top", zorder=22)

    fig.savefig(str(out_path), dpi=100, facecolor="#F5F7FA")
    plt.close(fig)


def _draw_enterprise_cluster_graph(ax, enterprise_graph, alerts_data, x0, y0, w, h):
    nodes      = enterprise_graph.get("nodes", [])
    edges      = enterprise_graph.get("edges", [])
    clusters   = enterprise_graph.get("diagram_clusters", [])
    node_by_id = {n["id"]: n for n in nodes}
    cluster_by_id = {c["diagram_id"]: c for c in clusters}
    order = [
        "branch_topology", "wan_topology", "datacenter_topology",
        "app_db_topology", "shared_services_topology",
    ]
    rel_panels = {
        "branch_topology":          (0.02, 0.60, 0.27, 0.32),
        "wan_topology":             (0.36, 0.62, 0.27, 0.30),
        "datacenter_topology":      (0.18, 0.28, 0.30, 0.28),
        "app_db_topology":          (0.58, 0.28, 0.34, 0.30),
        "shared_services_topology": (0.28, 0.02, 0.44, 0.22),
    }

    ax.add_patch(mpatches.FancyBboxPatch(
        (x0, y0), w, h, boxstyle="round,pad=0.06",
        fc="#F8FBFF", ec="#0277BD", lw=1.5, zorder=1))
    ax.text(x0 + 0.18, y0 + h - 0.22, "Unified Enterprise Graph Memory",
            fontsize=10.5, color="#01579B", fontweight="bold", ha="left", va="top", zorder=8)

    all_panel_pos = {}
    for did in order:
        if did not in cluster_by_id:
            continue
        rx, ry, rw, rh = rel_panels[did]
        px, py, pw, ph = x0 + rx*w, y0 + ry*h, rw*w, rh*h
        cluster  = cluster_by_id[did]
        node_ids = cluster.get("node_ids", [])
        lg_nodes = []
        for nid in node_ids:
            if nid in node_by_id:
                nn = dict(node_by_id[nid])
                nn["id"] = nid
                lg_nodes.append(nn)
        panel_edges = [
            e for e in edges
            if e.get("edge_scope") == "local"
            and e.get("source_diagram") == did
            and e.get("source") in node_ids
            and e.get("target") in node_ids
        ]
        lg    = {"nodes": lg_nodes, "edges": panel_edges}
        color = DIAGRAM_TYPE_COLORS.get(did, "#607D8B")
        pos   = _draw_local_graph_card(
            ax, lg, px, py, pw, ph, color,
            title=DIAGRAM_TITLES.get(did, did).replace(" Topology", ""),
            badge=False, alerts_data=alerts_data)
        for nid, xy in pos.items():
            all_panel_pos[(did, nid)] = xy

    cross_edges = [e for e in edges if e.get("edge_scope") == "cross_diagram"]
    for e in cross_edges:
        sd, td = e.get("source_diagram"), e.get("target_diagram")
        s,  t  = e.get("source"),         e.get("target")
        p1 = all_panel_pos.get((sd, s)) or next(
            (xy for (did, nid), xy in all_panel_pos.items() if nid == s), None)
        p2 = all_panel_pos.get((td, t)) or next(
            (xy for (did, nid), xy in all_panel_pos.items() if nid == t), None)
        if p1 and p2:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#00A8E8", lw=1.8,
                    ls=(0, (4, 3)), alpha=0.85, zorder=4)
            mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            lbl = e.get("label") or e.get("relationship", "")
            if lbl:
                ax.text(mx, my, lbl[:12], fontsize=5.8, color="#01579B",
                        ha="center", va="center",
                        bbox=dict(fc="#E1F5FE", ec="#B3E5FC", pad=0.8), zorder=9)

    rc_diag          = (alerts_data or {}).get("root_cause_diagram")
    shared_service_ids = {"AD-01", "DNS-01", "IAM-01", "NTP-01", "MON-01"}
    node_pos = {}
    for (_, nid), xy in all_panel_pos.items():
        node_pos.setdefault(nid, xy)
    for (did, nid), xy in all_panel_pos.items():
        if did == rc_diag:
            node_pos[nid] = xy
        elif did == "shared_services_topology" and nid in shared_service_ids:
            node_pos[nid] = xy
    key_path = (alerts_data or {}).get("impact_paths", [])
    if key_path:
        path = [nid for nid in key_path[0] if nid in node_pos]
        for a, b in zip(path, path[1:]):
            x1, y1 = node_pos[a]
            x2, y2 = node_pos[b]
            ax.plot([x1, x2], [y1, y2], color="#D32F2F", lw=3.2, alpha=0.88, zorder=8)
        if path:
            ax.text(node_pos[path[0]][0], node_pos[path[0]][1] + 0.16,
                    "ROOT", fontsize=5.8, color="#B71C1C", ha="center",
                    fontweight="bold", zorder=12)

    legend_x, legend_y = x0 + w - 1.72, y0 + 0.18
    ax.add_patch(mpatches.FancyBboxPatch((legend_x, legend_y), 1.48, 0.92,
                                         boxstyle="round,pad=0.035",
                                         fc="#FFFFFF", ec="#CFD8DC", lw=0.8, zorder=10))
    legend = [("Root", "#E53935"), ("Alert", "#FB8C00"), ("Impact", "#FDD835"), ("Shared", "#0277BD")]
    for i, (lbl, clr) in enumerate(legend):
        yy = legend_y + 0.72 - i * 0.19
        ax.add_patch(plt.Circle((legend_x + 0.16, yy), 0.045, fc=clr, ec="#263238", lw=0.4, zorder=11))
        ax.text(legend_x + 0.28, yy, lbl, fontsize=6.0, color=LABEL_COLOR, va="center", zorder=11)


def make_preview_enterprise_graph(scenario_id, enterprise_graph, alerts_data, out_path):
    fig = plt.figure(figsize=(16, 10), facecolor="#F5F5F5")
    ax  = fig.add_axes([0, 0, 1, 1], facecolor="#F5F5F5")
    ax.set_xlim(0, 16); ax.set_ylim(0, 10); ax.axis("off")
    ax.add_patch(mpatches.Rectangle((0, 9.25), 16, 0.75, fc=HEADER_BLUE, ec="none"))
    ax.text(8, 9.62, f"Enterprise Galaxy Graph - {scenario_id}",
            ha="center", va="center", fontsize=16, color="#FFFFFF", fontweight="bold")
    _draw_enterprise_cluster_graph(ax, enterprise_graph, alerts_data, 0.55, 0.55, 14.9, 8.35)
    fig.savefig(str(out_path), dpi=110, facecolor="#F5F5F5")
    plt.close(fig)


def make_preview_rca_overlay(scenario_id, enterprise_graph, alerts_data, out_path):
    nodes = enterprise_graph.get("nodes", [])
    if not nodes:
        return
    fig = plt.figure(figsize=(16, 10), facecolor="#F5F5F5")
    ax  = fig.add_axes([0, 0, 1, 1], facecolor="#F5F5F5")
    ax.set_xlim(0, 16); ax.set_ylim(0, 10); ax.axis("off")
    ax.add_patch(mpatches.Rectangle((0, 9.25), 16, 0.75, fc=HEADER_BLUE, ec="none"))
    ax.text(8, 9.62, f"Cross-Diagram RCA Overlay - {scenario_id}",
            ha="center", va="center", fontsize=16, color="#FFFFFF", fontweight="bold")
    _draw_enterprise_cluster_graph(ax, enterprise_graph, alerts_data, 0.45, 0.55, 11.25, 8.35)
    _draw_rca_summary_box(ax, alerts_data, 12.05, 5.40, 3.45, 3.45, title="Root Cause")
    ax.add_patch(mpatches.FancyBboxPatch(
        (12.05, 0.75), 3.45, 4.25, boxstyle="round,pad=0.055",
        fc="#FFFFFF", ec="#B0BEC5", lw=1.0, zorder=20))
    ax.text(12.25, 4.70, "Alert Timeline", fontsize=8.5, color="#01579B",
            fontweight="bold", ha="left", va="top", zorder=21)
    y = 4.28
    for alert in (alerts_data or {}).get("alerts", [])[:7]:
        color = {"critical": "#E53935", "high": "#FB8C00", "warning": "#FDD835"}.get(
            alert.get("severity"), "#90A4AE")
        ax.add_patch(plt.Circle((12.28, y + 0.02), 0.055, fc=color, ec="#263238", lw=0.4, zorder=22))
        ax.text(12.42, y + 0.06, f"t+{alert.get('time_offset_min', 0)}m {alert.get('node', '?')}",
                fontsize=6.4, color=TEXT_COLOR, ha="left", va="center", zorder=22)
        ax.text(12.42, y - 0.12, alert.get("alert_type", "alert"),
                fontsize=5.8, color=LABEL_COLOR, ha="left", va="center", zorder=22)
        y -= 0.46
    fig.savefig(str(out_path), dpi=110, facecolor="#F5F5F5")
    plt.close(fig)


# ============================================================
# SCENARIO GENERATION
# ============================================================

def generate_scenario(scenario_idx, split, out_root, master_rng):
    scenario_id = f"enterprise_v3_{scenario_idx:04d}"
    rng         = random.Random(master_rng.randint(0, 2**31))

    # First scenario is the deterministic reference scenario with the complete
    # five-diagram enterprise chain; remaining scenarios vary size and combo.
    if scenario_idx == 0:
        diagram_types = list(HERO_DIAGRAM_TYPES)
    else:
        size  = rng.choices([3, 4, 5], weights=SIZE_PROBS)[0]
        combos = DIAGRAM_COMBOS[size]
        diagram_types = list(rng.choice(combos))

    scenario_dir = out_root / "scenarios" / split / scenario_id
    diag_dir     = scenario_dir / "diagrams"
    ann_dir      = scenario_dir / "annotations"
    yolo_dir     = scenario_dir / "labels_yolo"
    lg_dir       = scenario_dir / "local_graphs"
    for d in [diag_dir, ann_dir, yolo_dir, lg_dir]:
        d.mkdir(parents=True, exist_ok=True)

    local_graphs     = []
    all_annotations  = {}

    for dtype in diagram_types:
        primary, ghost, edges, zones = TEMPLATE_FNS[dtype](rng, scenario_idx, diagram_types)
        for i, n in enumerate(primary):
            n["ip_address"] = _ip(n["ip_zone"], i, scenario_idx)

        img_path    = diag_dir / f"{dtype}.png"
        node_pixels = draw_diagram(scenario_id, dtype, primary, ghost, edges, zones, img_path)

        ann      = make_annotation(scenario_id, dtype, dtype, img_path,
                                   primary, ghost, edges, zones, node_pixels)
        ann_path = ann_dir / f"{dtype}.json"
        with open(ann_path, "w") as f:
            json.dump(ann, f, indent=2)
        all_annotations[dtype] = ann

        yolo_txt  = make_yolo_label(ann)
        yolo_path = yolo_dir / f"{dtype}.txt"
        with open(yolo_path, "w") as f:
            f.write(yolo_txt)

        lg      = make_local_graph(scenario_id, dtype, dtype, primary, ghost, edges, node_pixels)
        lg_path = lg_dir / f"{dtype}.json"
        with open(lg_path, "w") as f:
            json.dump(lg, f, indent=2)
        local_graphs.append(lg)

    stitch_map = make_stitch_map(scenario_id, diagram_types)
    with open(scenario_dir / "stitch_map.json", "w") as f:
        json.dump(stitch_map, f, indent=2)

    enterprise_graph = build_enterprise_graph(scenario_id, diagram_types, local_graphs, stitch_map)
    with open(scenario_dir / "enterprise_graph.json", "w") as f:
        json.dump(enterprise_graph, f, indent=2)

    alerts_data = (
        make_hero_alerts(scenario_id, enterprise_graph)
        if scenario_idx == 0
        else make_alerts(scenario_id, diagram_types, enterprise_graph, rng)
    )
    with open(scenario_dir / "alerts.json", "w") as f:
        json.dump(alerts_data or {}, f, indent=2)

    stats    = enterprise_graph.get("stats", {})
    metadata = {
        "scenario_id":             scenario_id,
        "diagram_types":           diagram_types,
        "num_diagrams":            len(diagram_types),
        "num_nodes":               stats.get("num_nodes", 0),
        "num_edges":               stats.get("num_edges", 0),
        "num_cross_diagram_edges": stats.get("num_cross_diagram_edges", 0),
        "num_shared_entities":     stats.get("num_shared_entities", 0),
        "root_cause":              (alerts_data or {}).get("root_cause", ""),
        "root_cause_diagram":      (alerts_data or {}).get("root_cause_diagram", ""),
        "root_cause_pattern":      (alerts_data or {}).get("root_cause_pattern", ""),
        "severity":                (alerts_data or {}).get("severity", ""),
        "num_impacted_nodes":      len((alerts_data or {}).get("impacted_nodes", [])),
        "num_impacted_diagrams":   len((alerts_data or {}).get("impacted_diagrams", [])),
        "split":                   split,
    }
    with open(scenario_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    try:
        make_preview_contact_sheet(scenario_id, diagram_types, diag_dir,
                                   scenario_dir / "preview_contact_sheet.png")
        make_preview_bbox_contact_sheet(scenario_id, diagram_types, diag_dir,
                                        scenario_dir / "preview_bbox_contact_sheet.png")
        make_preview_stitching_story(scenario_id, diagram_types, diag_dir,
                                     scenario_dir / "preview_stitching_story.png")
        make_preview_stitching_story_hero(scenario_id, diagram_types, diag_dir,
                                          scenario_dir / "preview_stitching_story_hero.png")
        make_preview_enterprise_graph(scenario_id, enterprise_graph, alerts_data,
                                      scenario_dir / "preview_enterprise_graph.png")
        make_preview_rca_overlay(scenario_id, enterprise_graph, alerts_data,
                                 scenario_dir / "preview_rca_overlay.png")
    except Exception as e:
        print(f"[warn] preview failed for {scenario_id}: {e}")

    return metadata


# ============================================================
# YOLO DATASET ASSEMBLY
# ============================================================

def build_yolo_structure(out_root, all_metadata):
    yolo_root = out_root / "yolo"
    for split in ["train", "val", "test"]:
        (yolo_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for meta in all_metadata:
        split = meta["split"]
        sid   = meta["scenario_id"]
        for dtype in meta["diagram_types"]:
            src_img  = out_root / "scenarios" / split / sid / "diagrams"     / f"{dtype}.png"
            src_lbl  = out_root / "scenarios" / split / sid / "labels_yolo"  / f"{dtype}.txt"
            stem     = f"{sid}__{dtype}"
            dst_img  = yolo_root / "images" / split / f"{stem}.png"
            dst_lbl  = yolo_root / "labels" / split / f"{stem}.txt"
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)

    yaml_content = (
        f"path: {out_root / 'yolo'}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n\n"
        f"nc: {len(YOLO_NAMES)}\n"
        f"names: {YOLO_NAMES}\n"
    )
    with open(yolo_root / "dataset.yaml", "w") as f:
        f.write(yaml_content)


def make_global_preview_sheet(out_root, all_metadata, preview_name, out_name, max_items=12):
    candidates = []
    for meta in all_metadata:
        preview_path = out_root / "scenarios" / meta["split"] / meta["scenario_id"] / preview_name
        if preview_path.exists():
            candidates.append((meta, preview_path))
        if len(candidates) >= max_items:
            break
    if not candidates:
        return

    cols = min(3, len(candidates))
    rows = math.ceil(len(candidates) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 3.4), facecolor="#080c10")
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for idx, (meta, preview_path) in enumerate(candidates):
        r, c = divmod(idx, cols)
        ax   = axes[r][c] if rows > 1 else axes[0][c]
        arr  = _load_png_as_array(preview_path)
        if arr is not None:
            ax.imshow(arr, aspect="auto")
        ax.set_title(f"{meta['scenario_id']} | {meta['num_diagrams']} diagrams",
                     fontsize=8, color=TEXT_COLOR, pad=4)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_edgecolor("#3a4a5a")

    for idx in range(len(candidates), rows * cols):
        r, c = divmod(idx, cols)
        ax   = axes[r][c] if rows > 1 else axes[0][c]
        ax.set_visible(False)

    fig.suptitle(out_name.replace("_", " ").replace(".png", "").title(),
                 fontsize=12, color=TEXT_COLOR, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = out_root / "previews" / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=80, facecolor="#080c10")
    plt.close(fig)


# ============================================================
# DATASET SUMMARY
# ============================================================

def write_dataset_summary(out_root, all_metadata, args):
    splits          = {"train": 0, "val": 0, "test": 0}
    pattern_dist    = {}
    dtype_dist      = {}
    class_dist      = {}
    root_cause_type_dist    = {}
    impacted_diagram_dist   = {}
    total_images            = 0
    total_diagram_nodes     = 0
    total_diagram_edges     = 0
    total_enterprise_nodes  = 0
    total_cross_edges       = 0
    total_text_blocks       = 0
    total_connectors        = 0
    full_5_diagram_scenario_count = 0
    hero_scenario_id   = "enterprise_v3_0000"
    hero_scenario_path = ""
    hero_visual_path   = ""

    for m in all_metadata:
        splits[m["split"]] = splits.get(m["split"], 0) + 1
        if m.get("num_diagrams", 0) == 5:
            full_5_diagram_scenario_count += 1
        p = m.get("root_cause_pattern", "unknown")
        pattern_dist[p] = pattern_dist.get(p, 0) + 1
        for dt in m.get("diagram_types", []):
            dtype_dist[dt] = dtype_dist.get(dt, 0) + 1
        total_images           += m.get("num_diagrams", 0)
        total_enterprise_nodes += m.get("num_nodes", 0)
        total_cross_edges      += m.get("num_cross_diagram_edges", 0)

        scenario_dir = out_root / "scenarios" / m["split"] / m["scenario_id"]
        if m["scenario_id"] == hero_scenario_id:
            hero_scenario_path  = str(scenario_dir)
            candidate_hero      = scenario_dir / "preview_stitching_story_hero.png"
            hero_visual_path    = str(candidate_hero) if candidate_hero.exists() else ""

        alerts_path = scenario_dir / "alerts.json"
        if alerts_path.exists():
            with open(alerts_path) as f:
                alerts = json.load(f)
            root_cause = alerts.get("root_cause", "")
            for diag in alerts.get("impacted_diagrams", []):
                impacted_diagram_dist[diag] = impacted_diagram_dist.get(diag, 0) + 1
        else:
            alerts     = {}
            root_cause = ""

        enterprise_path = scenario_dir / "enterprise_graph.json"
        node_types      = {}
        if enterprise_path.exists():
            with open(enterprise_path) as f:
                eg = json.load(f)
            for node in eg.get("nodes", []):
                node_types[node.get("id", "")] = node.get("type", "unknown")
        if root_cause:
            rc_type = node_types.get(root_cause, "unknown")
            root_cause_type_dist[rc_type] = root_cause_type_dist.get(rc_type, 0) + 1

        for ann_path in sorted((scenario_dir / "annotations").glob("*.json")):
            with open(ann_path) as f:
                ann = json.load(f)
            objects    = ann.get("objects",    [])
            connectors = ann.get("connectors", [])
            total_diagram_nodes  += len(objects)
            total_diagram_edges  += len(connectors)
            total_text_blocks    += len(ann.get("text_blocks", []))
            total_connectors     += len(connectors)
            for obj in objects:
                cls = obj.get("class_name", "unknown")
                class_dist[cls] = class_dist.get(cls, 0) + 1

    summary = {
        "version":            "v3",
        "dataset_root":       str(out_root),
        "generated_at":       datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "canvas_width":       CANVAS_W,
        "canvas_height":      CANVAS_H,
        "seed":               args.seed,
        "total_scenarios":    len(all_metadata),
        "num_scenarios":      len(all_metadata),
        "splits":             splits,
        "total_images":       total_images,
        "total_nodes":        total_enterprise_nodes,
        "diagram_types":      list(TEMPLATE_FNS.keys()),
        "device_classes":     YOLO_NAMES,
        "coco_categories":    COCO_CATS,
        "class_distribution": class_dist,
        "diagram_type_distribution": dtype_dist,
        "average_nodes_per_diagram": round(total_diagram_nodes / max(total_images, 1), 2),
        "average_edges_per_diagram": round(total_diagram_edges / max(total_images, 1), 2),
        "average_nodes_per_enterprise_graph": round(total_enterprise_nodes / max(len(all_metadata), 1), 2),
        "average_cross_diagram_edges": round(total_cross_edges / max(len(all_metadata), 1), 2),
        "ocr_text_block_count":   total_text_blocks,
        "connector_count":         total_connectors,
        "root_cause_type_distribution":  root_cause_type_dist,
        "impacted_diagram_distribution": impacted_diagram_dist,
        "alert_pattern_distribution":    pattern_dist,
        "full_5_diagram_scenario_count": full_5_diagram_scenario_count,
        "hero_scenario_id":    hero_scenario_id,
        "hero_scenario_path":  hero_scenario_path,
        "hero_visual_path":    hero_visual_path,
        "app_db_topology_count": dtype_dist.get("app_db_topology", 0),
    }
    with open(out_root / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    p = argparse.ArgumentParser(description="Generate V3 enterprise diagram dataset")
    p.add_argument("--num-scenarios", type=int, default=100)
    p.add_argument("--out",           type=str, default="./datasets/infragraph_v3")
    p.add_argument("--seed",          type=int, default=2026)
    p.add_argument("--clean",         action="store_true",
                   help="Delete output directory before generating")
    args = p.parse_args()

    out_root = Path(args.out).resolve()
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
        print(f"Cleaned: {out_root}")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "previews").mkdir(exist_ok=True)

    master_rng = random.Random(args.seed)
    n          = args.num_scenarios
    n_train    = int(n * 0.80)
    n_val      = int(n * 0.10)
    n_test     = n - n_train - n_val

    splits_list = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test

    print(f"Generating {n} scenarios -> {out_root}")
    print(f"  Canvas: {CANVAS_W}x{CANVAS_H}  Drawing area: {DRAW_W}x{DRAW_H}")
    print(f"  Train={n_train}  Val={n_val}  Test={n_test}")

    all_metadata = []
    for idx, split in enumerate(splits_list):
        meta = generate_scenario(idx, split, out_root, master_rng)
        all_metadata.append(meta)
        if (idx + 1) % 10 == 0 or (idx + 1) == n:
            print(f"  [{idx+1}/{n}] {meta['scenario_id']}  split={split}  "
                  f"types={len(meta['diagram_types'])}  "
                  f"pattern={meta.get('root_cause_pattern', '?')}")

    print("Building YOLO structure...")
    build_yolo_structure(out_root, all_metadata)

    print("Building global preview sheets...")
    make_global_preview_sheet(out_root, all_metadata, "preview_contact_sheet.png",     "contact_sheet.png")
    make_global_preview_sheet(out_root, all_metadata, "preview_bbox_contact_sheet.png", "bbox_contact_sheet.png")
    make_global_preview_sheet(out_root, all_metadata, "preview_stitching_story.png",   "stitching_story_contact_sheet.png")
    make_global_preview_sheet(out_root, all_metadata, "preview_enterprise_graph.png",  "enterprise_contact_sheet.png")

    summary = write_dataset_summary(out_root, all_metadata, args)
    print(f"Done. total_images={summary['total_images']}  "
          f"total_nodes={summary['total_nodes']}")
    print(f"Dataset: {out_root}")


if __name__ == "__main__":
    main()
