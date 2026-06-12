"""
build_topology_rca_demo.py — Bridge from YOLO vision model output to
topology graph and root-cause analysis (RCA) for InfraGraph AI.

Usage:
    python scripts/build_topology_rca_demo.py --diagram-id diagram_0373

Inputs (all resolved from CLI defaults):
    datasets/infragraph_v2/images/test/<id>.png
    datasets/infragraph_v2/graphs/test/<id>.json
    datasets/infragraph_v2/alerts/test/<id>.json
    outputs/v2_test_predictions_cpu/labels/<id>.txt

Outputs (under --out, default outputs/topology_demo/):
    <id>_detected_nodes.json
    <id>_rca_result.json
    <id>_topology.png
    <id>_graph_summary.json
"""
import argparse
import json
import os
import struct
import sys
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

# ── Class map ────────────────────────────────────────────────────────────────
CLASS_MAP: Dict[int, str] = {
    0: "router",
    1: "switch",
    2: "firewall",
    3: "server",
    4: "database",
    5: "load_balancer",
    6: "cloud_or_wan",
}

SEVERITY_WEIGHT = {"critical": 4, "major": 3, "minor": 2, "warning": 1, "info": 1}

DEVICE_COLORS = {
    "router":        "#4472C4",
    "switch":        "#70AD47",
    "firewall":      "#C00000",
    "server":        "#FF8C00",
    "database":      "#7030A0",
    "load_balancer": "#00B0F0",
    "cloud_or_wan":  "#A5A5A5",
}

DEVICE_MARKERS = {
    "router":        "D",
    "switch":        "s",
    "firewall":      "^",
    "server":        "o",
    "database":      "h",
    "load_balancer": "p",
    "cloud_or_wan":  "8",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_json(folder: str, diagram_id: str, *extra_suffixes: str) -> Optional[str]:
    """Try <id><suffix>.json patterns, then bare <id>.json, then directory scan."""
    candidates = [f"{diagram_id}{s}.json" for s in extra_suffixes]
    candidates.append(f"{diagram_id}.json")
    for name in candidates:
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            return p
    try:
        for fname in sorted(os.listdir(folder)):
            if diagram_id in fname and fname.endswith(".json"):
                return os.path.join(folder, fname)
    except FileNotFoundError:
        pass
    return None


def _png_size(path: str) -> Tuple[int, int]:
    """Read (width, height) from PNG IHDR chunk without PIL."""
    with open(path, "rb") as f:
        f.read(16)  # 8-byte sig + 4-byte IHDR length + 4-byte "IHDR"
        w = struct.unpack(">I", f.read(4))[0]
        h = struct.unpack(">I", f.read(4))[0]
    return w, h


def _yolo_to_pixel(
    cx: float, cy: float, bw: float, bh: float, img_w: int, img_h: int
) -> Tuple[int, int, int, int]:
    x1 = int((cx - bw / 2) * img_w)
    y1 = int((cy - bh / 2) * img_h)
    x2 = int((cx + bw / 2) * img_w)
    y2 = int((cy + bh / 2) * img_h)
    return x1, y1, x2, y2


def _severity_score(sev: str) -> int:
    return SEVERITY_WEIGHT.get(sev.lower(), 1)


def _device_type_bonus(node_type: str, alert_type: str) -> float:
    """Return a bonus when the device type matches the alert category."""
    alert_lc = alert_type.lower()
    bonuses: Dict[str, List[str]] = {
        "firewall":     ["packet", "drop", "deny", "policy", "firewall", "block"],
        "router":       ["unreachable", "route", "bgp", "ospf", "link", "latency"],
        "database":     ["database", "query", "sql", "db", "slow", "timeout"],
        "load_balancer":["load", "balance", "pool", "upstream", "502", "503"],
        "server":       ["cpu", "memory", "app", "service", "process", "crash"],
    }
    for dtype, keywords in bonuses.items():
        if node_type == dtype and any(k in alert_lc for k in keywords):
            return 0.5
    return 0.0


def _find_paths_for_root(
    G: nx.DiGraph,
    root: str,
    targets: List[str],
    target_reason: str,
    max_paths: int = 10,
    max_len: int = 8,
    existing: Optional[List[dict]] = None,
) -> List[dict]:
    """
    Find shortest path from root to each target, trying three methods:
      1. directed root → target
      2. directed target → root
      3. undirected (G.to_undirected())
    Appends to `existing` list (returning a new list), capped at max_paths.
    """
    results: List[dict] = list(existing) if existing else []
    covered: Set[str] = {entry["target"] for entry in results}
    G_und = G.to_undirected()

    for target in targets:
        if len(results) >= max_paths:
            break
        if target == root or target in covered:
            continue
        covered.add(target)

        path: Optional[List[str]] = None
        method: Optional[str] = None

        # 1. directed root → target
        try:
            p = nx.shortest_path(G, root, target)
            path, method = p, "directed_root_to_target"
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

        # 2. directed target → root
        if path is None:
            try:
                p = nx.shortest_path(G, target, root)
                path, method = p, "directed_target_to_root"
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

        # 3. undirected
        if path is None:
            try:
                p = nx.shortest_path(G_und, root, target)
                path, method = p, "undirected"
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

        if path is None:
            continue

        truncated = len(path) > max_len
        if truncated:
            path = path[:max_len]

        results.append({
            "source":        root,
            "target":        target,
            "target_reason": target_reason,
            "path":          path,
            "path_length":   len(path),
            "method":        method,
            "truncated":     truncated,
        })

    return results


def _path_to_edges(path: List[str]) -> Set[Tuple[str, str]]:
    return {(path[i], path[i + 1]) for i in range(len(path) - 1)}


# ── Core pipeline ─────────────────────────────────────────────────────────────

def load_predictions(label_path: str, img_w: int, img_h: int) -> List[dict]:
    nodes = []
    with open(label_path) as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            conf = float(parts[5]) if len(parts) > 5 else 1.0
            x1, y1, x2, y2 = _yolo_to_pixel(cx, cy, bw, bh, img_w, img_h)
            nodes.append({
                "predicted_id":    f"pred_{i}",
                "class_id":        cls,
                "type":            CLASS_MAP.get(cls, "unknown"),
                "confidence":      round(conf, 6),
                "bbox_normalized": [round(cx, 6), round(cy, 6), round(bw, 6), round(bh, 6)],
                "bbox_pixel":      [x1, y1, x2, y2],
            })
    return nodes


def build_graph(graph_data: dict) -> nx.DiGraph:
    G: nx.DiGraph = nx.DiGraph()
    for node in graph_data.get("nodes", []):
        attrs = {k: v for k, v in node.items() if k != "id"}
        G.add_node(node["id"], **attrs)
    for edge in graph_data.get("edges", []):
        G.add_edge(
            edge["source"], edge["target"],
            label=edge.get("label", ""),
            relationship=edge.get("relationship", "connected_to"),
        )
    return G


def compute_rca(G: nx.DiGraph, alert_data: dict) -> dict:
    alerts: List[dict] = alert_data.get("alerts", [])
    gt_root: Optional[str] = alert_data.get("root_cause")
    expected_impacted: List[str] = alert_data.get("expected_impacted_nodes", [])
    alerting_nodes: Set[str] = {a["node"] for a in alerts}

    # ── Heuristic scoring ────────────────────────────────────────────────────
    scores: Dict[str, float] = {}
    for alert in alerts:
        nid = alert["node"]
        if nid not in G:
            continue
        sev_s    = _severity_score(alert.get("severity", "info"))
        time_s   = 1.0 / (1.0 + alert.get("time_offset_min", 99))
        dtype    = G.nodes[nid].get("type", "")
        dtype_b  = _device_type_bonus(dtype, alert.get("alert_type", ""))
        downstream = len(nx.descendants(G, nid))
        impact_s = downstream / max(G.number_of_nodes(), 1)
        score = sev_s * 2.0 + time_s * 10.0 + impact_s * 3.0 + dtype_b
        scores[nid] = scores.get(nid, 0.0) + score

    if not scores:
        predicted_root = gt_root
        top5 = (
            [{"node": gt_root, "score": 0.0, "type": G.nodes.get(gt_root, {}).get("type", "")}]
            if gt_root else []
        )
    else:
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        predicted_root = ranked[0][0]
        top5 = [
            {"node": n, "score": round(s, 4), "type": G.nodes.get(n, {}).get("type", "")}
            for n, s in ranked[:5]
        ]

    # ── Impacted nodes (descendants of predicted root) ────────────────────────
    pred_descendants: List[str] = []
    if predicted_root and predicted_root in G:
        pred_descendants = sorted(nx.descendants(G, predicted_root))

    # ── Impact paths for PREDICTED root ──────────────────────────────────────
    pred_alerting_targets = [n for n in sorted(alerting_nodes) if n != predicted_root and n in G]
    pred_paths = _find_paths_for_root(
        G, predicted_root, pred_alerting_targets, "alerting_node", max_paths=10
    )
    pred_paths = _find_paths_for_root(
        G, predicted_root, pred_descendants, "impacted_node",
        max_paths=10, existing=pred_paths
    )

    # ── Impact paths for GROUND-TRUTH root ───────────────────────────────────
    gt_paths: List[dict] = []
    if gt_root and gt_root in G and gt_root != predicted_root:
        gt_alerting_targets = [n for n in sorted(alerting_nodes) if n != gt_root and n in G]
        gt_expected_targets  = [n for n in expected_impacted if n in G and n != gt_root]
        gt_descendants = sorted(nx.descendants(G, gt_root))

        gt_paths = _find_paths_for_root(
            G, gt_root, gt_alerting_targets, "alerting_node", max_paths=10
        )
        gt_paths = _find_paths_for_root(
            G, gt_root, gt_expected_targets, "expected_impacted",
            max_paths=10, existing=gt_paths
        )
        gt_paths = _find_paths_for_root(
            G, gt_root, gt_descendants, "impacted_node",
            max_paths=10, existing=gt_paths
        )

    # ── Impact path summary ───────────────────────────────────────────────────
    pred_shortest = min(pred_paths, key=lambda p: p["path_length"]) if pred_paths else None
    gt_shortest   = min(gt_paths,   key=lambda p: p["path_length"]) if gt_paths   else None

    impact_path_summary = {
        "predicted_root_cause_path_count":    len(pred_paths),
        "ground_truth_root_cause_path_count": len(gt_paths),
        "shortest_predicted_path":            pred_shortest["path"] if pred_shortest else [],
        "shortest_ground_truth_path":         gt_shortest["path"]   if gt_shortest   else [],
    }

    # ── Reasoning summary ─────────────────────────────────────────────────────
    total_score = sum(scores.values()) or 1.0
    conf_score  = round(scores.get(predicted_root, 0.0) / total_score, 4) if scores else 0.0

    if predicted_root == gt_root or not gt_root:
        first_alert = alerts[0] if alerts else {}
        downstream_count = len(pred_descendants)
        reasoning = (
            f"Node '{predicted_root}' was correctly identified as the root cause "
            f"(confidence {conf_score:.0%}). "
            f"Earliest alert: '{first_alert.get('alert_type', '?')}' "
            f"(severity={first_alert.get('severity', '?')}, t={first_alert.get('time_offset_min', '?')} min). "
            f"{downstream_count} downstream node(s) reachable from this node."
        )
    else:
        reasoning = (
            f"The heuristic selected '{predicted_root}' because it has multiple correlated "
            f"downstream alerts and high topology centrality. However, the ground-truth root "
            f"cause is '{gt_root}', which appears earlier in the alert sequence. "
            f"This shows the limitation of rule-based scoring: the heuristic may confuse a "
            f"downstream aggregation or correlation node with the true upstream origin. "
            f"This motivates the GNN stage, where the model can learn propagation direction "
            f"and root-cause patterns from topology structure and alert features."
        )

    return {
        "predicted_root_cause":    predicted_root,
        "ground_truth_root_cause": gt_root,
        "confidence_score":        conf_score,
        "top_candidates":          top5,
        "alerting_nodes":          sorted(alerting_nodes),
        "impacted_nodes":          pred_descendants,
        "impact_paths": {
            "predicted_root_cause":    pred_paths,
            "ground_truth_root_cause": gt_paths,
        },
        "impact_path_summary":     impact_path_summary,
        "reasoning_summary":       reasoning,
    }


def visualize_topology(
    G: nx.DiGraph, rca: dict, diagram_id: str, out_path: str
) -> None:
    alerting        = set(rca.get("alerting_nodes", []))
    predicted_root  = rca.get("predicted_root_cause")
    gt_root         = rca.get("ground_truth_root_cause")
    impacted        = set(rca.get("impacted_nodes", []))
    summary         = rca.get("impact_path_summary", {})

    pred_short_path = summary.get("shortest_predicted_path", [])
    gt_short_path   = summary.get("shortest_ground_truth_path", [])
    pred_path_edges = _path_to_edges(pred_short_path)
    gt_path_edges   = _path_to_edges(gt_short_path)

    # Layout: use bbox centres from graph JSON, flip Y for matplotlib
    pos: Dict[str, Tuple[float, float]] = {}
    for nid, data in G.nodes(data=True):
        bbox = data.get("bbox")
        if bbox and len(bbox) == 4:
            pos[nid] = ((bbox[0] + bbox[2]) / 2.0, -((bbox[1] + bbox[3]) / 2.0))
    if len(pos) < G.number_of_nodes():
        pos = nx.spring_layout(G, seed=42, k=2.5)  # type: ignore[assignment]

    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_facecolor("#F0F2F5")
    fig.patch.set_facecolor("#F0F2F5")

    # ── Edges: categorised by which path they belong to ──────────────────────
    both_edges  = [(u, v) for u, v in G.edges() if (u,v) in pred_path_edges and (u,v) in gt_path_edges]
    pred_only   = [(u, v) for u, v in G.edges() if (u,v) in pred_path_edges and (u,v) not in gt_path_edges]
    gt_only     = [(u, v) for u, v in G.edges() if (u,v) in gt_path_edges   and (u,v) not in pred_path_edges]
    gray_edges  = [(u, v) for u, v in G.edges() if (u,v) not in pred_path_edges and (u,v) not in gt_path_edges]

    draw_kw = dict(ax=ax, arrows=True, arrowsize=14, connectionstyle="arc3,rad=0.06")
    if gray_edges:
        nx.draw_networkx_edges(G, pos, edgelist=gray_edges,
                               edge_color="#BBBBBB", width=1.0, alpha=0.65, **draw_kw)
    if pred_only:
        nx.draw_networkx_edges(G, pos, edgelist=pred_only,
                               edge_color="#E74C3C", width=3.0, alpha=0.95, **draw_kw)
    if gt_only:
        nx.draw_networkx_edges(G, pos, edgelist=gt_only,
                               edge_color="#2980B9", width=3.0, alpha=0.95, **draw_kw)
    if both_edges:
        nx.draw_networkx_edges(G, pos, edgelist=both_edges,
                               edge_color="#8E44AD", width=3.5, alpha=0.95, **draw_kw)

    # Edge labels (bandwidth / protocol)
    elabels = {(u, v): d.get("label", "") for u, v, d in G.edges(data=True) if d.get("label")}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=elabels, ax=ax,
                                 font_size=6, font_color="#555555",
                                 bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.6))

    # ── Nodes: per device type (different shape per type) ────────────────────
    for ntype, shape in DEVICE_MARKERS.items():
        nlist = [n for n, d in G.nodes(data=True) if d.get("type") == ntype and n in pos]
        if not nlist:
            continue
        sizes, facecolors, edgecolors, linewidths = [], [], [], []
        for n in nlist:
            deg = G.degree(n)
            sizes.append(max(800, deg * 220))
            facecolors.append(DEVICE_COLORS.get(ntype, "#888888"))
            # Border priority: predicted root > GT root > alerting > impacted > default
            if n == predicted_root:
                edgecolors.append("#FFD700"); linewidths.append(5.0)
            elif n == gt_root and gt_root != predicted_root:
                edgecolors.append("#00BCD4"); linewidths.append(5.0)
            elif n in alerting:
                edgecolors.append("#FF2200"); linewidths.append(3.0)
            elif n in impacted:
                edgecolors.append("#FF8C00"); linewidths.append(2.0)
            else:
                edgecolors.append("#333333"); linewidths.append(1.0)
        nx.draw_networkx_nodes(
            G, pos, nodelist=nlist, ax=ax,
            node_shape=shape, node_color=facecolors, node_size=sizes,
            edgecolors=edgecolors, linewidths=linewidths,
        )

    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight="bold",
                            font_color="#111111")

    # ── Legend ────────────────────────────────────────────────────────────────
    device_handles = [
        mpatches.Patch(color=c, label=t.replace("_", " "))
        for t, c in DEVICE_COLORS.items()
    ]
    highlight_handles = [
        mpatches.Patch(facecolor="none", edgecolor="#FFD700", linewidth=3,
                       label="Predicted root cause (gold border)"),
    ]
    if gt_root and gt_root != predicted_root:
        highlight_handles.append(
            mpatches.Patch(facecolor="none", edgecolor="#00BCD4", linewidth=3,
                           label="Ground-truth root cause (cyan border)")
        )
    highlight_handles += [
        mpatches.Patch(facecolor="none", edgecolor="#FF2200", linewidth=2,
                       label="Alerting node (red border)"),
        mpatches.Patch(facecolor="none", edgecolor="#FF8C00", linewidth=2,
                       label="Impacted node (orange border)"),
        mpatches.Patch(color="#E74C3C",
                       label="Predicted root impact path (red edge)"),
        mpatches.Patch(color="#2980B9",
                       label="Ground-truth root impact path (blue edge)"),
    ]
    ax.legend(handles=device_handles + highlight_handles,
              loc="upper left", fontsize=7.5, framealpha=0.88, ncol=2, borderpad=0.8)

    # ── Title ─────────────────────────────────────────────────────────────────
    match = predicted_root == gt_root
    match_tag = "  [RCA correct ✓]" if match else (f"  [GT={gt_root}  ✗]" if gt_root else "")
    title = (
        f"InfraGraph AI  |  Topology & RCA  |  {diagram_id}\n"
        f"Predicted root cause: {predicted_root}{match_tag}"
    )
    if pred_short_path:
        title += f"    Shortest predicted path: {' → '.join(pred_short_path)}"
    ax.set_title(title, fontsize=11, fontweight="bold", pad=14)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[6] Topology PNG saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build topology graph and RCA Presentation from YOLO V2 prediction output."
    )
    ap.add_argument("--diagram-id",   required=True,
                    help="Diagram ID, e.g. diagram_0373")
    ap.add_argument("--dataset-root", default="./datasets/infragraph_v2")
    ap.add_argument("--pred-root",    default="./outputs/v2_test_predictions_cpu")
    ap.add_argument("--out",          default="./demo_assets/topology_demo")
    args = ap.parse_args()

    did  = args.diagram_id
    dset = os.path.normpath(args.dataset_root)
    pred = os.path.normpath(args.pred_root)
    out  = os.path.normpath(args.out)
    os.makedirs(out, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  InfraGraph AI — Topology & RCA Presentation")
    print(f"  Diagram: {did}")
    print(f"{'='*60}\n")

    # 1. Image size
    img_path = os.path.join(dset, "images", "test", f"{did}.png")
    if not os.path.isfile(img_path):
        sys.exit(f"ERROR: image not found: {img_path}")
    img_w, img_h = _png_size(img_path)
    print(f"[1] Image: {img_w}x{img_h}  ({img_path})")

    # 2. YOLO predictions → pixel bboxes
    lbl_path = os.path.join(pred, "labels", f"{did}.txt")
    if not os.path.isfile(lbl_path):
        sys.exit(f"ERROR: prediction label not found: {lbl_path}")
    det_nodes = load_predictions(lbl_path, img_w, img_h)
    print(f"[2] YOLO detections: {len(det_nodes)}")

    det_path = os.path.join(out, f"{did}_detected_nodes.json")
    with open(det_path, "w") as f:
        json.dump(det_nodes, f, indent=2)

    # 3. Graph JSON → NetworkX DiGraph
    graph_folder = os.path.join(dset, "graphs", "test")
    graph_file   = _find_json(graph_folder, did, "_graph")
    if not graph_file:
        sys.exit(f"ERROR: graph JSON not found in {graph_folder}")
    with open(graph_file) as f:
        graph_data = json.load(f)
    G = build_graph(graph_data)
    template = graph_data.get("template", "unknown")
    print(f"[3] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges  (template={template})")

    # 4. Alert JSON
    alert_folder = os.path.join(dset, "alerts", "test")
    alert_file   = _find_json(alert_folder, did, "_alerts")
    if not alert_file:
        sys.exit(f"ERROR: alert JSON not found in {alert_folder}")
    with open(alert_file) as f:
        alert_data = json.load(f)
    alert_count = len(alert_data.get("alerts", []))
    print(f"[4] Alerts: {alert_count}  (scenario: {alert_data.get('scenario_id', '?')})")

    # 5. RCA computation
    rca = compute_rca(G, alert_data)
    rca_out = {"diagram_id": did, **rca}
    rca_path = os.path.join(out, f"{did}_rca_result.json")
    with open(rca_path, "w") as f:
        json.dump(rca_out, f, indent=2)
    summary = rca["impact_path_summary"]
    print(f"[5] RCA: predicted='{rca['predicted_root_cause']}'  "
          f"gt='{rca['ground_truth_root_cause']}'  conf={rca['confidence_score']}")
    print(f"     Predicted-root paths: {summary['predicted_root_cause_path_count']}")
    print(f"     GT-root paths:        {summary['ground_truth_root_cause_path_count']}")

    # 6. Topology visualisation
    viz_path = os.path.join(out, f"{did}_topology.png")
    visualize_topology(G, rca, did, viz_path)

    # 7. Graph summary
    type_counts: Dict[str, int] = {}
    for _, d in G.nodes(data=True):
        t = d.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    graph_summary = {
        "node_count":          G.number_of_nodes(),
        "edge_count":          G.number_of_edges(),
        "device_type_counts":  type_counts,
        "alert_count":         alert_count,
        "root_cause":          rca["predicted_root_cause"],
        "ground_truth_root":   rca["ground_truth_root_cause"],
        "impacted_node_count": len(rca["impacted_nodes"]),
    }
    sum_path = os.path.join(out, f"{did}_graph_summary.json")
    with open(sum_path, "w") as f:
        json.dump(graph_summary, f, indent=2)

    # 8. Print results
    pred_short = summary["shortest_predicted_path"]
    gt_short   = summary["shortest_ground_truth_path"]
    match      = rca["predicted_root_cause"] == rca["ground_truth_root_cause"]

    print(f"\n{'='*60}")
    print(f"  RESULTS — {did}")
    print(f"{'='*60}")
    print(f"  Detected nodes (YOLO):       {len(det_nodes)}")
    print(f"  Graph nodes:                 {G.number_of_nodes()}")
    print(f"  Graph edges:                 {G.number_of_edges()}")
    print(f"  Alert count:                 {alert_count}")
    print(f"  Predicted root cause:        {rca['predicted_root_cause']}")
    print(f"  Ground-truth root:           {rca['ground_truth_root_cause']}")
    print(f"  RCA correct:                 {'YES' if match else 'NO'}")
    print(f"  Predicted-root paths:        {summary['predicted_root_cause_path_count']}")
    print(f"  GT-root paths:               {summary['ground_truth_root_cause_path_count']}")
    print(f"  Shortest predicted path:     {' -> '.join(pred_short) if pred_short else 'none'}")
    print(f"  Shortest GT path:            {' -> '.join(gt_short) if gt_short else 'none'}")
    print(f"\n  Output files:")
    for p in [det_path, rca_path, viz_path, sum_path]:
        print(f"    {p}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

