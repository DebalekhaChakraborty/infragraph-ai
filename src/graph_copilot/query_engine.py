"""Deterministic graph query engine for Graph Copilot.

Supports exact investigative queries against the normalized graph context
produced by graph_context.load_global_graph_context().
"""
from __future__ import annotations

import re
from typing import Any


# ── Node lookups ──────────────────────────────────────────────────────────────

def find_node_by_id(ctx: dict, node_id: str) -> "dict | None":
    return ctx["nodes_by_id"].get(node_id)


def find_nodes_by_ip(ctx: dict, ip: str) -> list[dict]:
    return ctx["nodes_by_ip"].get(ip, [])


def get_node_info(ctx: dict, node_id: str) -> dict:
    n = find_node_by_id(ctx, node_id)
    if not n:
        return {"found": False, "node_id": node_id}
    return {
        "found":      True,
        "node_id":    node_id,
        "ip_address": n.get("ip_address", ""),
        "interfaces": n.get("interfaces") or n.get("ports") or [],
        "zone":       n.get("zone", ""),
        "type":       n.get("type", ""),
        "diagram_id": n.get("diagram_id", ""),
        "is_shared":  bool(n.get("is_shared_entity")),
    }


# ── Neighbor / edge queries ───────────────────────────────────────────────────

def get_neighbors(ctx: dict, node_id: str) -> dict:
    outbound = [
        {"target": e.get("target"), "relation": e.get("relationship") or e.get("label")}
        for e in ctx["edges_by_source"].get(node_id, [])
    ]
    inbound = [
        {"source": e.get("source"), "relation": e.get("relationship") or e.get("label")}
        for e in ctx["edges_by_target"].get(node_id, [])
    ]
    cross_out = [
        {
            "target":         e.get("target") or e.get("target_node"),
            "target_diagram": e.get("target_diagram", ""),
            "relation":       e.get("label") or "cross_link",
        }
        for e in ctx["cross_by_source"].get(node_id, [])
    ]
    cross_in = [
        {
            "source":         e.get("source") or e.get("source_node"),
            "source_diagram": e.get("source_diagram", ""),
            "relation":       e.get("label") or "cross_link",
        }
        for e in ctx["cross_by_target"].get(node_id, [])
    ]
    return {
        "node_id":        node_id,
        "outbound":       outbound,
        "inbound":        inbound,
        "cross_outbound": cross_out,
        "cross_inbound":  cross_in,
        "total_neighbors": len(outbound) + len(inbound) + len(cross_out) + len(cross_in),
    }


def get_edges_between(ctx: dict, node_a: str, node_b: str) -> list[dict]:
    result = []
    for e in ctx["edges_by_source"].get(node_a, []):
        if e.get("target") == node_b:
            result.append(e)
    for e in ctx["edges_by_source"].get(node_b, []):
        if e.get("target") == node_a:
            result.append(e)
    return result


def get_diagrams_for_node(ctx: dict, node_id: str) -> list[str]:
    n = find_node_by_id(ctx, node_id)
    if not n:
        return []
    diag = n.get("diagram_id")
    return [diag] if diag else []


def get_all_cross_diagram_links(ctx: dict) -> list[dict]:
    return ctx.get("cross_diagram_edges") or []


# ── RCA / incident queries ────────────────────────────────────────────────────

def get_root_cause(ctx: dict) -> dict:
    rca = ctx.get("gnn_rca") or {}
    return {
        "root_cause":          rca.get("root_cause"),
        "root_cause_diagram":  rca.get("root_cause_diagram"),
        "mode":                rca.get("mode"),
        "impacted_diagrams":   rca.get("impacted_diagrams") or [],
        "impacted_nodes":      rca.get("impacted_nodes") or [],
        "alert_nodes":         rca.get("alert_nodes") or [],
        "alert_count":         rca.get("alert_count", 0),
        "impact_path":         rca.get("impact_path") or [],
        "top_candidates":      rca.get("top_candidates") or [],
    }


def get_impacted_diagrams(ctx: dict) -> list[str]:
    return list((ctx.get("gnn_rca") or {}).get("impacted_diagrams") or [])


def get_impact_path(ctx: dict) -> list[str]:
    return list(ctx.get("impact_paths") or [])


# ── Graph traversal ───────────────────────────────────────────────────────────

def get_blast_radius(ctx: dict, node_id: str, max_nodes: int = 50) -> list[str]:
    """BFS downstream — nodes reachable from node_id following outbound edges."""
    try:
        import networkx as nx  # type: ignore
        G = _build_nx(ctx)
        if node_id not in G:
            return []
        return list(nx.descendants(G, node_id))[:max_nodes]
    except ImportError:
        visited: set[str] = set()
        queue = [node_id]
        while queue:
            cur = queue.pop(0)
            if cur in visited:
                continue
            visited.add(cur)
            for e in ctx["edges_by_source"].get(cur, []):
                tgt = e.get("target")
                if tgt and tgt not in visited:
                    queue.append(tgt)
        visited.discard(node_id)
        return list(visited)[:max_nodes]


def get_upstream_path(ctx: dict, node_id: str, max_hops: int = 8) -> list[str]:
    """Shortest upstream path to node_id from a graph root."""
    try:
        import networkx as nx  # type: ignore
        G = _build_nx(ctx)
        if node_id not in G:
            return [node_id]
        roots = [n for n in G.nodes if G.in_degree(n) == 0]
        best: list[str] = []
        for r in roots:
            try:
                p = nx.shortest_path(G, r, node_id)
                if len(p) <= max_hops + 1 and (not best or len(p) < len(best)):
                    best = p
            except Exception:
                pass
        return best or [node_id]
    except ImportError:
        visited: set[str] = set()
        path: list[str]   = []
        queue = [node_id]
        while queue and len(path) < max_hops:
            cur = queue.pop(0)
            if cur in visited:
                continue
            visited.add(cur)
            path.append(cur)
            for e in ctx["edges_by_target"].get(cur, []):
                src = e.get("source")
                if src and src not in visited:
                    queue.append(src)
        return list(reversed(path))


def _build_nx(ctx: dict):
    import networkx as nx  # type: ignore
    G = nx.DiGraph()
    for nid in ctx["nodes_by_id"]:
        G.add_node(nid)
    for nid, edges in ctx["edges_by_source"].items():
        for e in edges:
            tgt = e.get("target")
            if tgt:
                G.add_edge(nid, tgt)
    return G


# ── Natural-language dispatch ─────────────────────────────────────────────────

_IP_RE    = re.compile(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b')
_NODE_RE  = re.compile(r'[`"\']([A-Z][A-Z0-9\-_]{1,})[`"\']')


def run_query(ctx: dict, question: str) -> "dict[str, Any] | None":
    """Run a deterministic graph query from a natural-language question.

    Returns a structured result dict or None when no query pattern matched.
    """
    q = question.lower()

    # IP lookup
    ip_m = _IP_RE.search(question)
    if ip_m:
        ip    = ip_m.group(1)
        nodes = find_nodes_by_ip(ctx, ip)
        return {"type": "nodes_by_ip", "ip": ip, "nodes": nodes}

    node_m = _NODE_RE.search(question)
    nid    = node_m.group(1) if node_m else None

    # Blast radius / downstream
    if any(kw in q for kw in ("blast radius", "blast_radius", "downstream", "if.*fail", "impact if")):
        if nid:
            return {"type": "blast_radius", "node_id": nid,
                    "affected": get_blast_radius(ctx, nid)}

    # Upstream / dependency path
    if any(kw in q for kw in ("upstream", "dependency", "depends on", "path from", "path to")):
        if nid:
            return {"type": "upstream_path", "node_id": nid,
                    "path": get_upstream_path(ctx, nid)}

    # Neighbors / connected to
    if any(kw in q for kw in ("connected to", "neighbors of", "what is connected", "interfaces")):
        if nid:
            return {"type": "neighbors", **get_neighbors(ctx, nid)}

    # Edges between two nodes
    node_ms = _NODE_RE.findall(question)
    if len(node_ms) >= 2 and any(kw in q for kw in ("between", "edge", "link")):
        edges = get_edges_between(ctx, node_ms[0], node_ms[1])
        return {"type": "edges_between", "node_a": node_ms[0], "node_b": node_ms[1],
                "edges": edges}

    # Which diagrams does a node appear in
    if nid and any(kw in q for kw in ("which diagram", "diagram.*node", "node.*diagram")):
        return {"type": "node_diagrams", "node_id": nid,
                "diagrams": get_diagrams_for_node(ctx, nid)}

    # Impacted diagrams
    if "impacted" in q and "diagram" in q:
        return {"type": "impacted_diagrams",
                "impacted_diagrams": get_impacted_diagrams(ctx)}

    # Root cause
    if "root cause" in q or "root_cause" in q:
        return {"type": "root_cause", **get_root_cause(ctx)}

    # Alert / impact propagation path
    if any(kw in q for kw in ("propagation path", "alert path", "impact path")):
        return {"type": "propagation_path", "path": get_impact_path(ctx)}

    # Cross-diagram links
    if any(kw in q for kw in ("cross-diagram", "cross diagram", "cross_diagram")):
        links = get_all_cross_diagram_links(ctx)
        return {"type": "cross_diagram_links", "count": len(links), "links": links[:20]}

    # Generic node info (IP, zone, type)
    if nid and any(kw in q for kw in ("node", "ip", "zone", "type", "info", "what is")):
        return {"type": "node_info", **get_node_info(ctx, nid)}

    return None


# ── Result formatting ─────────────────────────────────────────────────────────

def format_query_result(result: "dict | None") -> str:
    if not result:
        return ""
    rtype = result.get("type", "")

    if rtype == "nodes_by_ip":
        nodes = result.get("nodes", [])
        if not nodes:
            return f"No node found with IP `{result['ip']}`."
        lines = [f"Nodes with IP `{result['ip']}`:\n"]
        for n in nodes:
            lines.append(
                f"- **{n.get('id')}** — type: {n.get('type','?')}, "
                f"diagram: `{n.get('diagram_id','?')}`, zone: {n.get('zone','?')}"
            )
        return "\n".join(lines)

    if rtype == "blast_radius":
        affected = result.get("affected", [])
        if not affected:
            return f"No downstream nodes found for `{result['node_id']}`."
        tail = f"\n\n... and {len(affected)-30} more" if len(affected) > 30 else ""
        return (
            f"Blast radius from **`{result['node_id']}`** — "
            f"{len(affected)} downstream nodes affected:\n\n"
            + "\n".join(f"- `{n}`" for n in affected[:30]) + tail
        )

    if rtype == "upstream_path":
        path = result.get("path", [])
        if not path:
            return f"No upstream path found for `{result['node_id']}`."
        return (
            f"Upstream dependency path to **`{result['node_id']}`**:\n\n"
            + " → ".join(f"`{n}`" for n in path)
        )

    if rtype == "neighbors":
        lines = [f"Connections of **`{result['node_id']}`** ({result['total_neighbors']} total):\n"]
        for e in result.get("outbound", [])[:10]:
            lines.append(f"- → `{e['target']}` ({e['relation']})")
        for e in result.get("inbound", [])[:10]:
            lines.append(f"- ← `{e['source']}` ({e['relation']})")
        for e in result.get("cross_outbound", [])[:5]:
            lines.append(
                f"- ⇒ `{e.get('target')}` [{e.get('target_diagram','')}] "
                f"(cross-diagram: {e['relation']})"
            )
        return "\n".join(lines)

    if rtype == "edges_between":
        edges = result.get("edges", [])
        if not edges:
            return f"No direct edges found between `{result['node_a']}` and `{result['node_b']}`."
        lines = [f"Edges between **`{result['node_a']}`** and **`{result['node_b']}`**:\n"]
        for e in edges:
            lines.append(f"- {e.get('source')} → {e.get('target')} ({e.get('relationship') or e.get('label','link')})")
        return "\n".join(lines)

    if rtype == "node_diagrams":
        diags = result.get("diagrams", [])
        return (
            f"Node **`{result['node_id']}`** appears in: "
            + (", ".join(f"`{d}`" for d in diags) or "no diagrams found")
        )

    if rtype == "impacted_diagrams":
        diags = result.get("impacted_diagrams", [])
        return (
            f"Impacted diagrams ({len(diags)}): "
            + (", ".join(f"`{d}`" for d in diags) or "none")
        )

    if rtype == "root_cause":
        rca = result
        lines = [
            f"**Root Cause: `{rca.get('root_cause') or '—'}`** "
            f"in `{rca.get('root_cause_diagram','?')}`\n",
            f"- RCA mode: {rca.get('mode','unknown')}",
            f"- Impacted diagrams: {', '.join(rca.get('impacted_diagrams',[]) or ['none'])}",
            f"- Alert nodes: {', '.join(rca.get('alert_nodes',[]) or ['none'])}",
            f"- Alert count: {rca.get('alert_count',0)}",
            f"- Impact path: {' → '.join(rca.get('impact_path',[]) or ['N/A'])}",
        ]
        for c in (rca.get("top_candidates") or [])[:5]:
            lines.append(f"- GNN candidate: `{c.get('node_id','?')}` score={c.get('score','?')}")
        return "\n".join(lines)

    if rtype == "propagation_path":
        path = result.get("path", [])
        return (
            "Impact propagation path:\n\n"
            + (" → ".join(f"`{n}`" for n in path) or "Path not available.")
        )

    if rtype == "cross_diagram_links":
        links = result.get("links", [])
        lines = [f"Cross-diagram links ({result['count']} total):\n"]
        for e in links:
            src  = e.get("source") or e.get("source_node", "?")
            tgt  = e.get("target") or e.get("target_node", "?")
            sd   = e.get("source_diagram", "")
            td   = e.get("target_diagram", "")
            rel  = e.get("label") or e.get("relationship", "link")
            lines.append(f"- `{sd}:{src}` → `{td}:{tgt}` ({rel})")
        return "\n".join(lines)

    if rtype == "node_info":
        if not result.get("found"):
            return f"Node `{result.get('node_id')}` not found in graph memory."
        return (
            f"**Node `{result['node_id']}`**\n"
            f"- Type: {result.get('type','?')}\n"
            f"- IP address: `{result.get('ip_address') or '—'}`\n"
            f"- Zone: {result.get('zone') or '—'}\n"
            f"- Diagram: `{result.get('diagram_id') or '—'}`\n"
            f"- Shared entity: {'yes' if result.get('is_shared') else 'no'}\n"
        )

    return ""
