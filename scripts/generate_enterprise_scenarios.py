#!/usr/bin/env python3
"""
generate_enterprise_scenarios.py — Enterprise multi-diagram graph scenario generator.

Each scenario = one enterprise = 3–5 local infrastructure diagrams stitched
into one unified enterprise graph.  Designed for cross-diagram GNN RCA training.

Usage:
    python scripts/generate_enterprise_scenarios.py \
        --num 120 --out ./datasets/infragraph_v1/enterprise_graph \
        --seed 2026 --clean
"""
from __future__ import annotations

import argparse, json, math, random, shutil, sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# optional visuals; JSON generation continues if these packages are unavailable
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NODE_TYPE_COLOR = {
    "router": "#60a5fa", "switch": "#818cf8", "firewall": "#ef4444",
    "server": "#22d3ee", "database": "#10b981", "load_balancer": "#f59e0b",
    "cloud_or_wan": "#94a3b8", "service": "#a78bfa",
}
ALERT_COLORS = {
    "root": "#ef4444", "alerting": "#f97316", "impacted": "#fbbf24", "normal": "#475569",
}
ALERT_TYPES  = ["packet_drop", "latency", "cpu_spike", "link_degradation", "connection_timeout"]
SEVERITIES   = ["critical", "high", "medium"]

# Cluster layout angles on enterprise graph circle (radians)
CLUSTER_ANGLE = {
    "branch_topology":          math.pi * 0.5,
    "wan_topology":             math.pi * 1.1,
    "datacenter_topology":      math.pi * 1.9,
    "app_db_topology":          math.pi * 1.5,
    "shared_services_topology": math.pi * 0.9,
}

# Root-cause patterns: (name, required_diagram_types, root_node_id, default_severity)
ROOT_CAUSE_PATTERNS = [
    ("wan_mpls_failure",     ["wan_topology"],              "WAN-MPLS-CORE",   "critical"),
    ("dc_fw_failure",        ["datacenter_topology"],        "DC-FW-01",        "critical"),
    ("db_master_failure",    ["app_db_topology"],            "DB-MASTER",       "high"),
    ("lb_failure",           ["app_db_topology"],            "APP-LB-01",       "critical"),
    ("identity_svc_failure", ["shared_services_topology"],   "SVC-IDENTITY-01", "high"),
    ("wan_pe_failure",       ["wan_topology"],              "WAN-PE-01",       "high"),
    ("branch_rtr_failure",   ["branch_topology"],            "BR-RTR-01",       "medium"),
    ("dc_core_sw_failure",   ["datacenter_topology"],        "DC-CORE-SW-01",   "high"),
]

# Curated diagram combinations per scenario size
DIAGRAM_CONFIGS: dict[int, list[list[str]]] = {
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
SIZE_WEIGHTS = {3: 0.35, 4: 0.45, 5: 0.20}


# ---------------------------------------------------------------------------
# Local graph generators
# ---------------------------------------------------------------------------

def _branch_graph(rng: random.Random) -> tuple[list[dict], list[dict], list[str]]:
    """Branch office topology: router → firewall → switch → workers."""
    n_wkr = rng.randint(2, 4)
    nodes: list[dict] = [
        {"id": "BR-RTR-01",  "type": "router",   "zone": "branch-edge"},
        {"id": "BR-FW-01",   "type": "firewall",  "zone": "branch-edge"},
        {"id": "BR-SW-01",   "type": "switch",    "zone": "branch-lan"},
    ]
    for i in range(1, n_wkr + 1):
        nodes.append({"id": f"BR-WRK-{i:02d}", "type": "server", "zone": "branch-lan"})

    edges: list[dict] = [
        {"source": "BR-RTR-01", "target": "BR-FW-01",  "relation": "routes_to"},
        {"source": "BR-FW-01",  "target": "BR-SW-01",  "relation": "secured_by"},
    ]
    for i in range(1, n_wkr + 1):
        edges.append({"source": "BR-SW-01", "target": f"BR-WRK-{i:02d}", "relation": "connected_to"})

    bridge_ids = ["BR-RTR-01"]
    return nodes, edges, bridge_ids


def _wan_graph(rng: random.Random) -> tuple[list[dict], list[dict], list[str]]:
    """WAN/MPLS topology: ISP → MPLS core → PE routers."""
    n_pe = rng.randint(2, 4)
    nodes: list[dict] = [
        {"id": "WAN-MPLS-CORE", "type": "cloud_or_wan", "zone": "wan-core"},
        {"id": "WAN-ISP",       "type": "cloud_or_wan", "zone": "wan-edge"},
    ]
    pe_ids = []
    for i in range(1, n_pe + 1):
        pe_id = f"WAN-PE-{i:02d}"
        nodes.append({"id": pe_id, "type": "router", "zone": "wan-edge"})
        pe_ids.append(pe_id)

    edges: list[dict] = [
        {"source": "WAN-ISP", "target": "WAN-MPLS-CORE", "relation": "routes_to"},
    ]
    for pe_id in pe_ids:
        edges.append({"source": "WAN-MPLS-CORE", "target": pe_id, "relation": "routes_to"})

    bridge_ids = pe_ids
    return nodes, edges, bridge_ids


def _datacenter_graph(rng: random.Random) -> tuple[list[dict], list[dict], list[str]]:
    """Datacenter topology: firewalls → core switches → agg switch → servers."""
    n_srv = rng.randint(3, 6)
    nodes: list[dict] = [
        {"id": "DC-FW-01",       "type": "firewall", "zone": "dc-edge"},
        {"id": "DC-FW-02",       "type": "firewall", "zone": "dc-edge"},
        {"id": "DC-CORE-SW-01",  "type": "switch",   "zone": "dc-core"},
        {"id": "DC-CORE-SW-02",  "type": "switch",   "zone": "dc-core"},
        {"id": "DC-AGG-SW-01",   "type": "switch",   "zone": "dc-agg"},
    ]
    for i in range(1, n_srv + 1):
        nodes.append({"id": f"DC-SRV-{i:02d}", "type": "server", "zone": "dc-compute"})

    edges: list[dict] = [
        {"source": "DC-FW-01",      "target": "DC-CORE-SW-01", "relation": "secured_by"},
        {"source": "DC-FW-02",      "target": "DC-CORE-SW-02", "relation": "secured_by"},
        {"source": "DC-CORE-SW-01", "target": "DC-AGG-SW-01",  "relation": "connected_to"},
        {"source": "DC-CORE-SW-02", "target": "DC-AGG-SW-01",  "relation": "connected_to"},
    ]
    for i in range(1, n_srv + 1):
        edges.append({"source": "DC-AGG-SW-01", "target": f"DC-SRV-{i:02d}", "relation": "connected_to"})

    bridge_ids = ["DC-FW-01", "DC-FW-02", "DC-CORE-SW-01"]
    return nodes, edges, bridge_ids


def _app_db_graph(rng: random.Random) -> tuple[list[dict], list[dict], list[str]]:
    """Application/database tier: LB → app servers → DB master/replicas/cache."""
    n_app = rng.randint(2, 4)
    n_rep = rng.randint(1, 2)
    nodes: list[dict] = [
        {"id": "APP-LB-01",    "type": "load_balancer", "zone": "app-ingress"},
        {"id": "DB-MASTER",    "type": "database",      "zone": "db-tier"},
        {"id": "DB-CACHE-01",  "type": "server",        "zone": "db-tier"},
    ]
    for i in range(1, n_app + 1):
        nodes.append({"id": f"APP-{i:02d}", "type": "server", "zone": "app-tier"})
    for i in range(1, n_rep + 1):
        nodes.append({"id": f"DB-REPLICA-{i:02d}", "type": "database", "zone": "db-tier"})

    edges: list[dict] = []
    for i in range(1, n_app + 1):
        edges.append({"source": "APP-LB-01",   "target": f"APP-{i:02d}",   "relation": "serves"})
        edges.append({"source": f"APP-{i:02d}", "target": "DB-MASTER",     "relation": "depends_on"})
        edges.append({"source": f"APP-{i:02d}", "target": "DB-CACHE-01",   "relation": "depends_on"})
    for i in range(1, n_rep + 1):
        edges.append({"source": "DB-MASTER", "target": f"DB-REPLICA-{i:02d}", "relation": "connected_to"})

    bridge_ids = ["APP-LB-01"]
    return nodes, edges, bridge_ids


def _shared_services_graph(_rng: random.Random) -> tuple[list[dict], list[dict], list[str]]:
    """Shared services: identity, DNS, NTP, monitoring."""
    nodes: list[dict] = [
        {"id": "SVC-IDENTITY-01", "type": "service", "zone": "shared-svc"},
        {"id": "SVC-DNS-01",      "type": "server",  "zone": "shared-svc"},
        {"id": "SVC-NTP-01",      "type": "server",  "zone": "shared-svc"},
        {"id": "SVC-MONITOR-01",  "type": "service", "zone": "shared-svc"},
    ]
    edges: list[dict] = [
        {"source": "SVC-IDENTITY-01",  "target": "SVC-DNS-01",      "relation": "depends_on"},
        {"source": "SVC-MONITOR-01",   "target": "SVC-IDENTITY-01", "relation": "depends_on"},
        {"source": "SVC-MONITOR-01",   "target": "SVC-NTP-01",      "relation": "depends_on"},
    ]
    bridge_ids = ["SVC-IDENTITY-01", "SVC-DNS-01"]
    return nodes, edges, bridge_ids


GRAPH_FACTORIES = {
    "branch_topology":          _branch_graph,
    "wan_topology":             _wan_graph,
    "datacenter_topology":      _datacenter_graph,
    "app_db_topology":          _app_db_graph,
    "shared_services_topology": _shared_services_graph,
}


# ---------------------------------------------------------------------------
# Local graph JSON builder
# ---------------------------------------------------------------------------

def _local_graph_json(diagram_id: str, nodes: list[dict], edges: list[dict]) -> dict:
    """Return JSON schema dict for a single local diagram graph."""
    annotated_nodes = []
    for n in nodes:
        node = dict(n)
        node["diagram_id"] = diagram_id
        node["is_shared_entity"] = False
        annotated_nodes.append(node)
    return {
        "diagram_id":   diagram_id,
        "diagram_type": diagram_id,
        "nodes":        annotated_nodes,
        "edges":        edges,
    }


# ---------------------------------------------------------------------------
# Stitch map builder
# ---------------------------------------------------------------------------

def _build_stitch_map(
    local_graphs: dict[str, dict],
    diagram_types: list[str],
) -> dict:
    """Build cross-diagram stitching map with shared entities and cross-diagram edges."""
    has = lambda dt: dt in diagram_types  # noqa: E731

    shared_entities: list[dict] = []
    cross_diagram_edges: list[dict] = []

    # branch ↔ wan
    if has("branch_topology") and has("wan_topology"):
        cross_diagram_edges.append({
            "source":        "BR-RTR-01",
            "target":        "WAN-PE-01",
            "relation":      "wan_dependency",
            "source_diagram": "branch_topology",
            "target_diagram": "wan_topology",
        })

    # wan ↔ datacenter
    if has("wan_topology") and has("datacenter_topology"):
        cross_diagram_edges.append({
            "source":        "WAN-MPLS-CORE",
            "target":        "DC-FW-01",
            "relation":      "routes_to",
            "source_diagram": "wan_topology",
            "target_diagram": "datacenter_topology",
        })
        shared_entities.append({
            "canonical_id": "DC-FW-01",
            "appears_in": [
                {"diagram_id": "wan_topology",        "local_alias": "WAN-DC-FW-01"},
                {"diagram_id": "datacenter_topology", "local_alias": "DC-FW-01"},
            ],
        })

    # branch ↔ datacenter (no wan)
    if has("branch_topology") and has("datacenter_topology") and not has("wan_topology"):
        cross_diagram_edges.append({
            "source":        "BR-RTR-01",
            "target":        "DC-FW-01",
            "relation":      "wan_dependency",
            "source_diagram": "branch_topology",
            "target_diagram": "datacenter_topology",
        })

    # datacenter ↔ app_db
    if has("datacenter_topology") and has("app_db_topology"):
        cross_diagram_edges.append({
            "source":        "DC-CORE-SW-01",
            "target":        "APP-LB-01",
            "relation":      "connected_to",
            "source_diagram": "datacenter_topology",
            "target_diagram": "app_db_topology",
        })

    # wan ↔ app_db (no datacenter)
    if has("wan_topology") and has("app_db_topology") and not has("datacenter_topology"):
        cross_diagram_edges.append({
            "source":        "WAN-MPLS-CORE",
            "target":        "APP-LB-01",
            "relation":      "routes_to",
            "source_diagram": "wan_topology",
            "target_diagram": "app_db_topology",
        })

    # shared_services ↔ app_db
    if has("shared_services_topology") and has("app_db_topology"):
        # check if APP-02 exists in app_db local graph
        app_db_nodes = {n["id"] for n in local_graphs.get("app_db_topology", {}).get("nodes", [])}
        cross_diagram_edges.append({
            "source":        "SVC-IDENTITY-01",
            "target":        "APP-01",
            "relation":      "serves",
            "source_diagram": "shared_services_topology",
            "target_diagram": "app_db_topology",
        })
        if "APP-02" in app_db_nodes:
            cross_diagram_edges.append({
                "source":        "SVC-IDENTITY-01",
                "target":        "APP-02",
                "relation":      "serves",
                "source_diagram": "shared_services_topology",
                "target_diagram": "app_db_topology",
            })
        shared_entities.append({
            "canonical_id": "SVC-DNS-01",
            "appears_in": [
                {"diagram_id": "shared_services_topology", "local_alias": "SVC-DNS-01"},
                {"diagram_id": "app_db_topology",          "local_alias": "APP-DNS-REF"},
            ],
        })

    # shared_services ↔ branch
    if has("shared_services_topology") and has("branch_topology"):
        cross_diagram_edges.append({
            "source":        "SVC-DNS-01",
            "target":        "BR-SW-01",
            "relation":      "serves",
            "source_diagram": "shared_services_topology",
            "target_diagram": "branch_topology",
        })

    return {
        "shared_entities":    shared_entities,
        "cross_diagram_edges": cross_diagram_edges,
    }


# ---------------------------------------------------------------------------
# Enterprise graph builder
# ---------------------------------------------------------------------------

def _build_enterprise_graph(
    scenario_id: str,
    local_graphs: dict[str, dict],
    stitch_map: dict,
    diagram_types: list[str],
) -> dict:
    """Merge all local graphs into one unified enterprise graph."""
    seen: set[str] = set()
    all_nodes: list[dict] = []
    all_edges: list[dict] = []
    diagram_clusters: list[dict] = []

    # Collect canonical shared entity IDs
    shared_canonical: set[str] = {
        se["canonical_id"] for se in stitch_map.get("shared_entities", [])
    }

    # Collect bridge targets from cross_diagram_edges
    bridge_targets: set[str] = set()
    for cde in stitch_map.get("cross_diagram_edges", []):
        bridge_targets.add(cde["source"])
        bridge_targets.add(cde["target"])

    # Merge local graphs
    for diagram_id in diagram_types:
        lg = local_graphs.get(diagram_id)
        if lg is None:
            continue
        cluster_node_ids: list[str] = []
        for n in lg["nodes"]:
            nid = n["id"]
            if nid in seen:
                continue
            seen.add(nid)
            node = dict(n)
            node["diagram_id"]             = diagram_id
            node["diagram_type"]           = diagram_id
            node["is_shared_entity"]       = nid in shared_canonical
            node["is_cross_diagram_bridge"] = nid in bridge_targets
            all_nodes.append(node)
            cluster_node_ids.append(nid)

        for e in lg["edges"]:
            edge = dict(e)
            edge["edge_scope"] = "local"
            edge["diagram_id"] = diagram_id
            all_edges.append(edge)

        diagram_clusters.append({
            "diagram_id": diagram_id,
            "node_ids":   cluster_node_ids,
        })

    # Add cross_diagram_edges (only if both endpoints exist)
    cross_edges_added: list[dict] = []
    for cde in stitch_map.get("cross_diagram_edges", []):
        if cde["source"] in seen and cde["target"] in seen:
            edge = dict(cde)
            edge["edge_scope"] = "cross_diagram"
            all_edges.append(edge)
            cross_edges_added.append(cde)

    return {
        "scenario_id":        scenario_id,
        "nodes":              all_nodes,
        "edges":              all_edges,
        "diagram_clusters":   diagram_clusters,
        "cross_diagram_edges": cross_edges_added,
        "shared_entities":    stitch_map.get("shared_entities", []),
    }


# ---------------------------------------------------------------------------
# BFS impact propagation
# ---------------------------------------------------------------------------

def _bfs_impact(
    root_id: str,
    edges: list[dict],
) -> tuple[list[str], list[list[str]]]:
    """BFS from root_id following directed edges. Returns (impacted_nodes, impact_paths)."""
    MAX_DEPTH = 7
    adj: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        adj[e["source"]].append(e["target"])

    visited: set[str]       = set()
    impacted_nodes: list[str] = []
    impact_paths: list[list[str]] = []

    # BFS queue: (current_node, path_so_far)
    queue: list[tuple[str, list[str]]] = [(root_id, [root_id])]
    visited.add(root_id)

    while queue:
        current, path = queue.pop(0)
        for neighbour in adj[current]:
            if neighbour in visited:
                continue
            if len(path) >= MAX_DEPTH:
                continue
            visited.add(neighbour)
            new_path = path + [neighbour]
            impacted_nodes.append(neighbour)
            if len(new_path) >= 2:
                impact_paths.append(new_path)
            queue.append((neighbour, new_path))

    return impacted_nodes, impact_paths


# ---------------------------------------------------------------------------
# Alert generator
# ---------------------------------------------------------------------------

def _generate_alerts(
    scenario_id: str,
    enterprise_graph: dict,
    diagram_types: list[str],
    rng: random.Random,
) -> dict:
    """Generate a root-cause alert scenario for the enterprise graph."""
    node_ids = {n["id"] for n in enterprise_graph["nodes"]}

    # Filter patterns to those satisfying diagram-type and node existence requirements
    eligible = [
        p for p in ROOT_CAUSE_PATTERNS
        if all(dt in diagram_types for dt in p[1]) and p[2] in node_ids
    ]
    if not eligible:
        eligible = ROOT_CAUSE_PATTERNS[:1]

    pattern = rng.choice(eligible)
    pattern_name, _req_types, root_cause, default_severity = pattern

    # Possibly override severity
    severity = rng.choice(SEVERITIES) if rng.random() < 0.25 else default_severity

    root_node = next((n for n in enterprise_graph["nodes"] if n["id"] == root_cause), {})
    root_cause_diagram = root_node.get("diagram_id", diagram_types[0] if diagram_types else "unknown")

    # BFS impact propagation
    impacted_nodes, impact_paths = _bfs_impact(root_cause, enterprise_graph["edges"])

    # Flat alerts list — spec schema
    alert_type = "packet_drop" if ("FW" in root_cause or "RTR" in root_cause) else rng.choice(ALERT_TYPES)
    alerts_list: list[dict] = [{
        "node":           root_cause,
        "diagram_id":     root_cause_diagram,
        "alert_type":     alert_type,
        "severity":       "critical",
        "time_offset_min": 0,
    }]
    n_secondary = rng.randint(2, 4)
    if impacted_nodes:
        sample_nodes = rng.sample(impacted_nodes, min(n_secondary, len(impacted_nodes)))
        for offset, nid in enumerate(sample_nodes, start=1):
            node_info = next((n for n in enterprise_graph["nodes"] if n["id"] == nid), {})
            alerts_list.append({
                "node":           nid,
                "diagram_id":     node_info.get("diagram_id", "unknown"),
                "alert_type":     rng.choice(ALERT_TYPES),
                "severity":       rng.choice(["medium", "high"]),
                "time_offset_min": offset * rng.randint(1, 5),
            })

    # Impacted diagrams
    impacted_diagrams = sorted({
        n.get("diagram_id", "")
        for nid in impacted_nodes
        for n in enterprise_graph["nodes"]
        if n["id"] == nid and n.get("diagram_id")
    })

    return {
        "scenario_id":         scenario_id,
        "root_cause":          root_cause,
        "root_cause_diagram":  root_cause_diagram,
        "root_cause_pattern":  pattern_name,   # internal field kept for dataset stats
        "severity":            severity,
        "alerts":              alerts_list,
        "impacted_nodes":      impacted_nodes,
        "impacted_diagrams":   impacted_diagrams,
        "impact_paths":        impact_paths,
    }


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _node_color(
    node_id: str,
    ntype: str,
    root: str,
    alerting_set: set[str],
    impacted_set: set[str],
) -> str:
    """Return display colour for a node based on alert status."""
    if node_id == root:
        return ALERT_COLORS["root"]
    if node_id in alerting_set:
        return ALERT_COLORS["alerting"]
    if node_id in impacted_set:
        return ALERT_COLORS["impacted"]
    return NODE_TYPE_COLOR.get(ntype, ALERT_COLORS["normal"])


def draw_local_diagram(
    local_graph: dict,
    out_path: Path,
    alerts: dict | None = None,
) -> None:
    """Draw a single local infrastructure diagram as a dark-theme PNG."""
    if not (HAS_MPL and HAS_NX):
        return

    G = nx.DiGraph()
    for n in local_graph["nodes"]:
        G.add_node(n["id"], ntype=n.get("type", "server"))
    for e in local_graph["edges"]:
        G.add_edge(e["source"], e["target"], relation=e.get("relation", ""))

    root_id      = alerts.get("root_cause", "") if alerts else ""
    alerting_set: set[str] = {a["node"] for a in (alerts or {}).get("alerts", [])}
    impacted_set: set[str] = set((alerts or {}).get("impacted_nodes", []))

    pos = nx.spring_layout(G, seed=42, k=2.5)
    node_colors = [
        _node_color(n, G.nodes[n].get("ntype", "server"), root_id, alerting_set, impacted_set)
        for n in G.nodes()
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0b0f1c")
    ax.set_facecolor("#0b0f1c")

    nx.draw_networkx(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=900,
        font_size=6.5,
        font_color="#e2e8f0",
        edge_color="#334155",
        arrows=True,
        connectionstyle="arc3,rad=0.1",
        arrowsize=14,
        width=1.2,
    )

    diagram_type = local_graph.get("diagram_type", local_graph.get("diagram_id", ""))
    pretty = diagram_type.replace("_", " ").title()
    ax.set_title(pretty, color="#94a3b8", fontsize=10, pad=8)
    ax.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_enterprise_graph(
    enterprise_graph: dict,
    alerts: dict,
    out_path: Path,
) -> None:
    """Draw the unified enterprise graph with cluster halos and cross-diagram edges."""
    if not (HAS_MPL and HAS_NX):
        return

    G = nx.DiGraph()
    for n in enterprise_graph["nodes"]:
        G.add_node(n["id"], ntype=n.get("type", "server"), diagram_id=n.get("diagram_id", ""))
    for e in enterprise_graph["edges"]:
        G.add_edge(e["source"], e["target"],
                   edge_scope=e.get("edge_scope", "local"),
                   relation=e.get("relation", ""))

    # Build cluster → node list map
    cluster_map: dict[str, list[str]] = {}
    for cl in enterprise_graph.get("diagram_clusters", []):
        cluster_map[cl["diagram_id"]] = cl["node_ids"]

    # Position nodes by cluster using CLUSTER_ANGLE
    rng_pos = random.Random(42)
    pos: dict[str, tuple[float, float]] = {}
    CLUSTER_R = 4.5
    SUB_R = 1.8
    for diagram_id, node_ids in cluster_map.items():
        angle = CLUSTER_ANGLE.get(diagram_id, 0.0)
        cx = CLUSTER_R * math.cos(angle)
        cy = CLUSTER_R * math.sin(angle)
        n_nodes = len(node_ids)
        for i, nid in enumerate(node_ids):
            sub_angle = (2 * math.pi * i / max(n_nodes, 1))
            jitter_x = rng_pos.uniform(-0.25, 0.25)
            jitter_y = rng_pos.uniform(-0.25, 0.25)
            px = cx + SUB_R * math.cos(sub_angle) + jitter_x
            py = cy + SUB_R * math.sin(sub_angle) + jitter_y
            pos[nid] = (px, py)

    root_id      = alerts.get("root_cause", "")
    alerting_set: set[str] = {a["node"] for a in alerts.get("alerts", [])}
    impacted_set: set[str] = set(alerts.get("impacted_nodes", []))

    node_colors = [
        _node_color(n, G.nodes[n].get("ntype", "server"), root_id, alerting_set, impacted_set)
        for n in G.nodes()
    ]
    node_sizes = []
    for n in G.nodes():
        if n == root_id:
            node_sizes.append(2200)
        elif n in alerting_set:
            node_sizes.append(1600)
        elif n in impacted_set:
            node_sizes.append(1100)
        else:
            node_sizes.append(800)

    fig, ax = plt.subplots(figsize=(14, 11))
    fig.patch.set_facecolor("#070b16")
    ax.set_facecolor("#070b16")

    # Draw cluster halos
    for diagram_id, node_ids in cluster_map.items():
        if not node_ids:
            continue
        angle = CLUSTER_ANGLE.get(diagram_id, 0.0)
        cx = CLUSTER_R * math.cos(angle)
        cy = CLUSTER_R * math.sin(angle)
        halo = mpatches.Circle(
            (cx, cy), radius=2.5,
            color="#1a2744", alpha=0.45, zorder=0,
        )
        ax.add_patch(halo)
        label_y = cy + 2.7
        ax.text(cx, label_y, diagram_id.replace("_", "\n"),
                ha="center", va="bottom", fontsize=6, color="#64748b", zorder=1)

    # Separate local vs cross-diagram edges
    local_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_scope") != "cross_diagram"]
    cross_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_scope") == "cross_diagram"]

    # Draw local edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=local_edges,
        edge_color="#1e3a5f",
        width=1.0,
        arrows=True,
        arrowsize=10,
        connectionstyle="arc3,rad=0.08",
    )
    # Draw cross-diagram edges
    if cross_edges:
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=cross_edges,
            edge_color="#60a5fa",
            width=2.2,
            arrows=True,
            arrowsize=14,
            style="dashed",
            alpha=0.85,
            connectionstyle="arc3,rad=0.15",
        )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.92,
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=5.5,
        font_color="#e2e8f0",
    )

    # Legend
    legend_items = [
        mpatches.Patch(color=ALERT_COLORS["root"],      label="Root Cause"),
        mpatches.Patch(color=ALERT_COLORS["alerting"],  label="Alerting"),
        mpatches.Patch(color=ALERT_COLORS["impacted"],  label="Impacted"),
        mpatches.Patch(color=ALERT_COLORS["normal"],    label="Normal"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower right",
        facecolor="#0f172a",
        edgecolor="#334155",
        labelcolor="#94a3b8",
        fontsize=8,
    )

    n_nodes = len(enterprise_graph["nodes"])
    n_edges = len(enterprise_graph["edges"])
    ax.set_title(
        f"Enterprise Graph  |  {enterprise_graph['scenario_id']}  |  "
        f"{n_nodes} nodes  {n_edges} edges",
        color="#94a3b8", fontsize=10, pad=10,
    )
    ax.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_contact_sheet(
    image_paths: list[Path],
    out_path: Path,
    title: str = "",
) -> None:
    """Tile multiple PNG previews into a single contact sheet."""
    if not HAS_PIL:
        return
    if not image_paths:
        return

    THUMB_W, THUMB_H = 560, 420
    HEADER_H = 50 if title else 0
    BG_COLOR = (7, 11, 22)

    images = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            img = img.resize((THUMB_W, THUMB_H), Image.LANCZOS)
            images.append(img)
        except Exception:
            pass

    if not images:
        return

    n = len(images)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    sheet_w = cols * THUMB_W
    sheet_h = rows * THUMB_H + HEADER_H

    sheet = Image.new("RGB", (sheet_w, sheet_h), BG_COLOR)
    draw = ImageDraw.Draw(sheet)

    if title:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.text((10, 14), title, fill=(148, 163, 184), font=font)

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * THUMB_W
        y = row * THUMB_H + HEADER_H
        sheet.paste(img, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


# ---------------------------------------------------------------------------
# Scenario write function
# ---------------------------------------------------------------------------

def _write_scenario(
    scenario: dict,
    split_dir: Path,
    scenario_id: str,
    diagram_types: list[str],
    split: str,
) -> Path:
    """Write all files for one enterprise scenario to disk."""
    sc_dir = split_dir / scenario_id
    diagrams_dir   = sc_dir / "diagrams"
    local_graphs_dir = sc_dir / "local_graphs"
    diagrams_dir.mkdir(parents=True, exist_ok=True)
    local_graphs_dir.mkdir(parents=True, exist_ok=True)

    # Write local graphs
    for dt in diagram_types:
        lg = scenario["local_graphs"].get(dt)
        if lg is not None:
            (local_graphs_dir / f"{dt}.json").write_text(
                json.dumps(lg, indent=2), encoding="utf-8"
            )

    # Write stitch map
    (sc_dir / "stitch_map.json").write_text(
        json.dumps(scenario["stitch_map"], indent=2), encoding="utf-8"
    )

    # Write enterprise graph
    (sc_dir / "enterprise_graph.json").write_text(
        json.dumps(scenario["enterprise_graph"], indent=2), encoding="utf-8"
    )

    # Write alerts
    (sc_dir / "alerts.json").write_text(
        json.dumps(scenario["alerts"], indent=2), encoding="utf-8"
    )

    # Write metadata
    meta = dict(scenario["metadata"])
    meta["split"] = split
    (sc_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    # Draw local diagrams
    local_diagram_paths: list[Path] = []
    for dt in diagram_types:
        lg = scenario["local_graphs"].get(dt)
        if lg is not None:
            out_png = diagrams_dir / f"{dt}.png"
            draw_local_diagram(lg, out_png, alerts=scenario["alerts"])
            if out_png.exists():
                local_diagram_paths.append(out_png)

    # Draw enterprise graph preview
    enterprise_png = sc_dir / "preview_enterprise_graph.png"
    draw_enterprise_graph(scenario["enterprise_graph"], scenario["alerts"], enterprise_png)

    # Draw contact sheet of local diagrams
    contact_sheet_png = sc_dir / "preview_contact_sheet.png"
    draw_contact_sheet(
        local_diagram_paths,
        contact_sheet_png,
        title=f"{scenario_id} — local diagrams",
    )

    return sc_dir


# ---------------------------------------------------------------------------
# Scenario generator
# ---------------------------------------------------------------------------

def generate_scenario(
    scenario_id: str,
    diagram_types: list[str],
    rng: random.Random,
) -> dict:
    """Generate one enterprise scenario: local graphs + stitch + enterprise + alerts."""
    # Build local graphs
    raw_local: dict[str, tuple[list[dict], list[dict], list[str]]] = {}
    for dt in diagram_types:
        factory = GRAPH_FACTORIES[dt]
        nodes, edges, bridge_ids = factory(rng)
        raw_local[dt] = (nodes, edges, bridge_ids)

    # Build local graph JSON objects
    local_graphs: dict[str, dict] = {}
    for dt in diagram_types:
        nodes, edges, _bridge = raw_local[dt]
        local_graphs[dt] = _local_graph_json(dt, nodes, edges)

    # Build stitch map
    stitch_map = _build_stitch_map(local_graphs, diagram_types)

    # Build enterprise graph
    enterprise_graph = _build_enterprise_graph(
        scenario_id, local_graphs, stitch_map, diagram_types
    )

    # Generate alerts
    alerts = _generate_alerts(scenario_id, enterprise_graph, diagram_types, rng)

    # Compute metadata metrics
    n_nodes = len(enterprise_graph["nodes"])
    n_edges = len(enterprise_graph["edges"])
    n_cross  = len(enterprise_graph["cross_diagram_edges"])
    n_shared = len(enterprise_graph["shared_entities"])

    metadata = {
        "scenario_id":         scenario_id,
        "diagram_types":       diagram_types,
        "num_diagrams":        len(diagram_types),
        "num_nodes":           n_nodes,
        "num_edges":           n_edges,
        "num_cross_diagram_edges": n_cross,
        "num_shared_entities":     n_shared,
        "root_cause":              alerts["root_cause"],
        "root_cause_diagram":      alerts["root_cause_diagram"],
        "root_cause_pattern":      alerts["root_cause_pattern"],
        "severity":                alerts["severity"],
        "num_impacted_nodes":      len(alerts["impacted_nodes"]),
        "num_impacted_diagrams":   len(alerts["impacted_diagrams"]),
    }

    return {
        "local_graphs":     local_graphs,
        "stitch_map":       stitch_map,
        "enterprise_graph": enterprise_graph,
        "alerts":           alerts,
        "metadata":         metadata,
    }


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def _assign_splits(total: int) -> list[str]:
    """Assign 80/10/10 train/val/test splits ensuring at least 1 of each."""
    n_val  = max(1, round(total * 0.10))
    n_test = max(1, round(total * 0.10))
    n_train = max(1, total - n_val - n_test)
    return ["train"] * n_train + ["val"] * n_val + ["test"] * n_test


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------

def generate_dataset(
    out_dir: Path,
    num: int,
    seed: int,
    clean: bool,
    min_diagrams: int,
    max_diagrams: int,
    preview_count: int,
) -> dict:
    """Generate the full enterprise graph dataset."""
    out_dir = Path(out_dir)

    if clean and out_dir.exists():
        shutil.rmtree(out_dir)

    # Create directory structure
    for split in ("train", "val", "test"):
        (out_dir / "scenarios" / split).mkdir(parents=True, exist_ok=True)
    (out_dir / "previews").mkdir(parents=True, exist_ok=True)

    main_rng = random.Random(seed)

    # Assign splits and shuffle
    splits = _assign_splits(num)
    main_rng.shuffle(splits)

    # Counters for summary
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    total_diagrams = 0
    total_nodes    = 0
    total_cross    = 0
    root_cause_dist: dict[str, int] = defaultdict(int)
    impacted_diagram_dist: dict[int, int] = defaultdict(int)
    scenarios_meta: list[dict] = []
    preview_paths: list[Path] = []

    for idx in range(num):
        split = splits[idx]
        scenario_id = f"enterprise_{idx:04d}"
        split_dir   = out_dir / "scenarios" / split

        # Choose diagram count using SIZE_WEIGHTS
        n_diagrams = main_rng.choices(
            list(SIZE_WEIGHTS.keys()),
            weights=list(SIZE_WEIGHTS.values()),
            k=1,
        )[0]
        n_diagrams = max(min_diagrams, min(max_diagrams, n_diagrams))

        # Pick diagram type combination
        config_options = DIAGRAM_CONFIGS.get(n_diagrams, DIAGRAM_CONFIGS[3])
        diagram_types  = main_rng.choice(config_options)

        # Per-scenario deterministic rng
        sc_rng = random.Random(seed + idx * 1000)

        # Generate scenario
        scenario = generate_scenario(scenario_id, diagram_types, sc_rng)

        # Write to disk
        sc_dir = _write_scenario(scenario, split_dir, scenario_id, diagram_types, split)

        # Collect preview
        enterprise_png = sc_dir / "preview_enterprise_graph.png"
        if enterprise_png.exists() and len(preview_paths) < preview_count:
            preview_paths.append(enterprise_png)

        # Accumulate stats
        meta = scenario["metadata"]
        split_counts[split] += 1
        total_diagrams += meta["num_diagrams"]
        total_nodes    += meta["num_nodes"]
        total_cross    += meta["num_cross_diagram_edges"]
        root_cause_dist[meta["root_cause_diagram"]] += 1
        n_impacted_diags = len(scenario["alerts"].get("impacted_diagrams", []))
        impacted_diagram_dist[n_impacted_diags] += 1
        scenarios_meta.append({
            "scenario_id": scenario_id,
            "split":       split,
            **{k: v for k, v in meta.items() if k != "scenario_id"},
        })

        # Progress
        if (idx + 1) % 10 == 0 or (idx + 1) == num:
            print(f"  [{idx+1:4d}/{num}] {scenario_id}  split={split}  "
                  f"diagrams={meta['num_diagrams']}  nodes={meta['num_nodes']}  "
                  f"cross_edges={meta['num_cross_diagram_edges']}")

    # Global contact sheet
    contact_sheet_path = out_dir / "previews" / "enterprise_contact_sheet.png"
    draw_contact_sheet(
        preview_paths,
        contact_sheet_path,
        title="Enterprise Graph Dataset — Preview",
    )

    summary = {
        "dataset":                       "infragraph_v1_enterprise_graph",
        "total_scenarios":               num,
        "splits":                        split_counts,
        "avg_diagrams_per_scenario":     round(total_diagrams / max(num, 1), 2),
        "avg_nodes_per_enterprise":      round(total_nodes / max(num, 1), 2),
        "avg_cross_diagram_edges":       round(total_cross / max(num, 1), 2),
        "root_cause_diagram_distribution": dict(root_cause_dist),
        "impacted_diagram_count_distribution": {
            str(k): v for k, v in impacted_diagram_dist.items()
        },
        "scenarios": scenarios_meta,
    }

    (out_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    return summary


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enterprise multi-diagram graph scenario generator for cross-diagram GNN RCA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num",           type=int, default=120,
                        help="Number of enterprise scenarios to generate")
    parser.add_argument("--out",           type=str, default="./datasets/infragraph_v1/enterprise_graph",
                        help="Output directory")
    parser.add_argument("--seed",          type=int, default=2026,
                        help="Global random seed")
    parser.add_argument("--clean",         action="store_true",
                        help="Wipe output directory before generating")
    parser.add_argument("--min-diagrams",  type=int, default=3,
                        help="Minimum diagrams per scenario")
    parser.add_argument("--max-diagrams",  type=int, default=5,
                        help="Maximum diagrams per scenario")
    parser.add_argument("--preview-count", type=int, default=12,
                        help="Number of enterprise previews in the contact sheet")
    args = parser.parse_args()

    print("=" * 70)
    print("  InfraGraph AI — Enterprise Scenario Generator")
    print("=" * 70)
    print(f"  num={args.num}  seed={args.seed}  out={args.out}")
    print(f"  diagrams/scenario={args.min_diagrams}–{args.max_diagrams}  "
          f"preview_count={args.preview_count}")
    print()

    if not HAS_MPL:
        print("  WARNING: matplotlib not available — PNG visualisations will be skipped.")
    if not HAS_NX:
        print("  WARNING: networkx not available — graph layout will be skipped.")
    if not HAS_PIL:
        print("  WARNING: Pillow not available — contact sheets will be skipped.")
    print()

    out_dir = Path(args.out)
    summary = generate_dataset(
        out_dir        = out_dir,
        num            = args.num,
        seed           = args.seed,
        clean          = args.clean,
        min_diagrams   = args.min_diagrams,
        max_diagrams   = args.max_diagrams,
        preview_count  = args.preview_count,
    )

    print()
    print("=" * 70)
    print("  DONE")
    print("=" * 70)
    print(f"  Dataset path : {out_dir.resolve()}")
    splits = summary["splits"]
    print(f"  train={splits['train']}  val={splits['val']}  test={splits['test']}")

    # First train scenario details
    train_scenarios = [s for s in summary["scenarios"] if s["split"] == "train"]
    if train_scenarios:
        first = train_scenarios[0]
        sc_path = out_dir / "scenarios" / "train" / first["scenario_id"]
        print(f"  Sample scenario : {sc_path}")
        print(f"  Enterprise graph: {first['num_nodes']} nodes, {first['num_edges']} edges")
        print(f"  Root cause      : {first['root_cause_pattern']} @ {first['root_cause']}")
        print(f"  Cross-diagram edges: {first['num_cross_diagram_edges']}")

    # Preview paths
    contact_sheet = out_dir / "previews" / "enterprise_contact_sheet.png"
    if contact_sheet.exists():
        print(f"  Contact sheet   : {contact_sheet}")
    else:
        print("  Contact sheet   : not generated (Pillow missing or no previews)")

    first_enterprise_png = None
    if train_scenarios:
        p = out_dir / "scenarios" / "train" / train_scenarios[0]["scenario_id"] / "preview_enterprise_graph.png"
        if p.exists():
            first_enterprise_png = p
    if first_enterprise_png:
        print(f"  Sample preview  : {first_enterprise_png}")
    else:
        print("  Sample preview  : not generated (matplotlib/networkx missing)")

    print(f"  Avg diagrams/scenario : {summary['avg_diagrams_per_scenario']}")
    print(f"  Avg nodes/enterprise  : {summary['avg_nodes_per_enterprise']}")
    print(f"  Avg cross-diagram edges: {summary['avg_cross_diagram_edges']}")
    print()


if __name__ == "__main__":
    main()
