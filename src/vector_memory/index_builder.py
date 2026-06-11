"""Build semantic vector documents from graph-memory evidence."""
from __future__ import annotations

import json
from hashlib import sha1
from typing import Any


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _base_metadata(
    source_type: str,
    *,
    scenario_id: str = "",
    diagram_id: str = "",
    node_id: str = "",
    edge_id: str = "",
    incident_id: str = "",
    rca_source: str = "",
    scope: str = "",
    path: str = "",
) -> dict[str, str]:
    return {
        "source_type": source_type,
        "scenario_id": scenario_id,
        "diagram_id": diagram_id,
        "node_id": node_id,
        "edge_id": edge_id,
        "incident_id": incident_id,
        "rca_source": rca_source,
        "scope": scope,
        "path": path,
    }


def _doc_id(metadata: dict[str, str], text: str) -> str:
    raw = "|".join(str(metadata.get(k, "")) for k in sorted(metadata)) + "|" + text
    return sha1(raw.encode("utf-8")).hexdigest()


def _make_doc(text: str, metadata: dict[str, str]) -> dict[str, Any]:
    return {"id": _doc_id(metadata, text), "text": text, "metadata": metadata}


def _node_id(node: dict) -> str:
    return str(node.get("id") or node.get("node_id") or node.get("object_id") or "")


def _edge_text(edge: dict) -> tuple[str, str, str]:
    source = str(edge.get("source", ""))
    target = str(edge.get("target", ""))
    relation = str(edge.get("relationship") or edge.get("relation") or edge.get("label") or "connected_to")
    return source, target, relation


def build_vector_docs_from_graph_memory(
    graph_memory_packet: dict,
    local_graph: dict | None = None,
    enterprise_graph: dict | None = None,
    local_incident: dict | None = None,
    enterprise_incident: dict | None = None,
    local_rca_result: dict | None = None,
    enterprise_rca_result: dict | None = None,
    ai_resolution_plan: dict | None = None,
) -> list[dict]:
    packet = _as_dict(graph_memory_packet)
    local_graph = _as_dict(local_graph)
    enterprise_graph = _as_dict(enterprise_graph)
    local_incident = _as_dict(local_incident)
    enterprise_incident = _as_dict(enterprise_incident)
    local_rca_result = _as_dict(local_rca_result)
    enterprise_rca_result = _as_dict(enterprise_rca_result)
    ai_resolution_plan = _as_dict(ai_resolution_plan)

    scenario_id = str(packet.get("scenario_id") or enterprise_graph.get("scenario_id") or "")
    diagram_id = str(packet.get("diagram_id") or local_graph.get("diagram_id") or "")
    run_id = str(packet.get("run_id") or packet.get("diagram_id") or scenario_id or "infragraph")
    docs: list[dict] = []

    local_nodes = _as_list(local_graph.get("nodes")) or _as_list(packet.get("nodes"))
    local_edges = _as_list(local_graph.get("edges")) or _as_list(packet.get("edges"))
    edge_targets: dict[str, list[str]] = {}
    for edge in local_edges:
        if not isinstance(edge, dict):
            continue
        source, target, relation = _edge_text(edge)
        if source and target:
            edge_targets.setdefault(source, []).append(f"{target} ({relation})")

    for node in local_nodes:
        if not isinstance(node, dict):
            continue
        nid = _node_id(node)
        ntype = str(node.get("type") or node.get("class_name") or "device")
        zone = str(node.get("zone") or "")
        ip = str(node.get("ip_address") or "")
        connected = ", ".join(edge_targets.get(nid, [])[:8])
        text = f"Device {nid} is a {ntype} in {diagram_id}."
        if zone:
            text += f" It is in zone {zone}."
        if ip:
            text += f" Its IP address is {ip}."
        if connected:
            text += f" It is connected to {connected}."
        docs.append(_make_doc(
            text,
            _base_metadata("device_memory", scenario_id=scenario_id, diagram_id=diagram_id, node_id=nid, scope="local", path=run_id),
        ))

    for idx, edge in enumerate(local_edges):
        if not isinstance(edge, dict):
            continue
        source, target, relation = _edge_text(edge)
        label = str(edge.get("label") or relation)
        text = f"Connector {source} to {target} in {diagram_id} represents {relation}."
        if label and label != relation:
            text += f" Connector label is {label}."
        docs.append(_make_doc(
            text,
            _base_metadata("connector_memory", scenario_id=scenario_id, diagram_id=diagram_id, edge_id=f"{source}->{target}:{idx}", scope="local", path=run_id),
        ))

    for row in _as_list(packet.get("nodes")):
        if not isinstance(row, dict):
            continue
        nid = _node_id(row)
        ip = str(row.get("ip_address") or "")
        if ip:
            docs.append(_make_doc(
                f"Interface and IP evidence: node {nid} has IP address {ip} in {diagram_id}.",
                _base_metadata("interface_ip_memory", scenario_id=scenario_id, diagram_id=diagram_id, node_id=nid, scope="local", path=run_id),
            ))

    for idx, block in enumerate(_as_list(packet.get("text_blocks"))):
        if not isinstance(block, dict):
            continue
        text_value = str(block.get("text") or "")
        if text_value:
            docs.append(_make_doc(
                f"OCR text evidence in {diagram_id}: {text_value}",
                _base_metadata("ocr_text_evidence", scenario_id=scenario_id, diagram_id=diagram_id, edge_id=f"text_{idx}", scope="local", path=run_id),
            ))

    def add_incident_docs(incident: dict, scope: str) -> None:
        incident_id = str(incident.get("incident_id") or incident.get("scenario_id") or f"{run_id}_{scope}")
        for idx, alert in enumerate(_as_list(incident.get("alerts") or incident.get("alert_timeline"))):
            if not isinstance(alert, dict):
                continue
            node = str(alert.get("node") or alert.get("node_id") or "")
            atype = str(alert.get("alert_type") or alert.get("title") or "alert")
            severity = str(alert.get("severity") or "")
            text = f"{scope.title()} incident alert {atype} on node {node}."
            if severity:
                text += f" Severity is {severity}."
            docs.append(_make_doc(
                text,
                _base_metadata(f"{scope}_incident_alert", scenario_id=scenario_id, diagram_id=diagram_id, node_id=node, incident_id=incident_id, scope=scope, path=run_id),
            ))

    add_incident_docs(local_incident, "local")
    add_incident_docs(enterprise_incident, "enterprise")

    def add_rca_doc(rca: dict, scope: str) -> None:
        if not rca:
            return
        root = str(rca.get("root_cause") or rca.get("predicted_root_cause") or "")
        mode = str(rca.get("mode") or rca.get("rca_source") or rca.get("model_type") or "")
        path = rca.get("impact_path") or rca.get("traversal_steps") or []
        text = f"{scope.title()} RCA selected {root} as root cause."
        if mode:
            text += f" RCA source is {mode}."
        if path:
            text += f" Impact path: {' -> '.join(map(str, path))}."
        docs.append(_make_doc(
            text,
            _base_metadata("rca_result", scenario_id=scenario_id, diagram_id=diagram_id, node_id=root, rca_source=mode, scope=scope, path=run_id),
        ))
        for rank, candidate in enumerate(_as_list(rca.get("top_candidates")), 1):
            if not isinstance(candidate, dict):
                continue
            node = str(candidate.get("node_id") or candidate.get("id") or "")
            score = candidate.get("score", "")
            ctype = str(candidate.get("type") or "")
            docs.append(_make_doc(
                f"GNN candidate rank {rank}: node {node} type {ctype} score {score}.",
                _base_metadata("gnn_candidate_ranking", scenario_id=scenario_id, diagram_id=diagram_id, node_id=node, rca_source=mode, scope=scope, path=run_id),
            ))

    add_rca_doc(local_rca_result, "local")
    add_rca_doc(enterprise_rca_result, "enterprise")

    if ai_resolution_plan:
        response = _as_dict(ai_resolution_plan.get("response"))
        plan_text = json.dumps(response or ai_resolution_plan, sort_keys=True)[:4000]
        docs.append(_make_doc(
            f"AI resolution plan for {diagram_id or scenario_id}: {plan_text}",
            _base_metadata("ai_resolution_plan", scenario_id=scenario_id, diagram_id=diagram_id, scope=str(ai_resolution_plan.get("scope") or ""), path=run_id),
        ))

    if enterprise_graph:
        nodes = _as_list(enterprise_graph.get("nodes"))
        edges = _as_list(enterprise_graph.get("edges"))
        clusters = _as_list(enterprise_graph.get("diagram_clusters"))
        docs.append(_make_doc(
            f"Scenario summary {scenario_id}: enterprise graph has {len(nodes)} nodes, {len(edges)} edges, and {len(clusters)} diagram clusters.",
            _base_metadata("scenario_summary", scenario_id=scenario_id, diagram_id=diagram_id, scope="enterprise", path=run_id),
        ))

    return docs
