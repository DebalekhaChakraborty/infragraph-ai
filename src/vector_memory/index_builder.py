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


def _assign_evidence_ids(docs: list[dict]) -> list[dict]:
    assigned: list[dict] = []
    for idx, doc in enumerate(docs, 1):
        metadata = dict(doc.get("metadata") or {})
        evidence_id = metadata.get("evidence_id") or f"E{idx:03d}"
        metadata["evidence_id"] = str(evidence_id)
        text = str(doc.get("text") or "")
        if not text.startswith(f"{evidence_id}:"):
            text = f"{evidence_id}: {text}"
        assigned.append({
            "id": _doc_id(metadata, text),
            "text": text,
            "metadata": metadata,
        })
    return assigned


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
        candidates = _as_list(rca.get("top_candidates")) or _as_list(rca.get("ranking"))
        for rank, candidate in enumerate(candidates, 1):
            if not isinstance(candidate, dict):
                continue
            node = str(candidate.get("node_id") or candidate.get("node") or candidate.get("id") or "")
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
        section_map = {
            "executive_summary": "AI resolution executive summary",
            "risk_level": "AI resolution risk level",
            "automation_eligibility": "AI resolution automation eligibility",
            "blast_radius": "AI resolution blast radius",
            "pre_checks": "AI resolution pre-checks",
            "validation_steps": "AI resolution validation steps",
            "remediation_steps": "AI resolution remediation steps",
            "post_checks": "AI resolution post-checks",
            "do_not_execute_if": "AI resolution do-not-execute conditions",
            "rollback_or_safety_notes": "AI resolution rollback and safety notes",
            "escalation_recommendation": "AI resolution escalation recommendation",
            "itsm_ticket_summary": "AI resolution ITSM summary",
            "audit_summary": "AI resolution audit summary",
            "confidence_notes": "AI resolution confidence notes",
        }
        for key, label in section_map.items():
            value = response.get(key)
            if not value:
                continue
            if isinstance(value, list):
                value_text = " ".join(str(v) for v in value)
            elif isinstance(value, dict):
                value_text = json.dumps(value, sort_keys=True)
            else:
                value_text = str(value)
            docs.append(_make_doc(
                f"{label}: {value_text}",
                _base_metadata(
                    f"ai_resolution_{key}",
                    scenario_id=scenario_id,
                    diagram_id=diagram_id,
                    node_id=str(response.get("probable_root_cause") or ""),
                    rca_source=str(ai_resolution_plan.get("source") or ""),
                    scope=str(response.get("scope") or ai_resolution_plan.get("scope") or ""),
                    path=run_id,
                ),
            ))

    if enterprise_graph:
        nodes = _as_list(enterprise_graph.get("nodes"))
        edges = _as_list(enterprise_graph.get("edges"))
        clusters = _as_list(enterprise_graph.get("diagram_clusters"))
        docs.append(_make_doc(
            f"Scenario summary {scenario_id}: enterprise graph has {len(nodes)} nodes, {len(edges)} edges, and {len(clusters)} diagram clusters.",
            _base_metadata("scenario_summary", scenario_id=scenario_id, diagram_id=diagram_id, scope="enterprise", path=run_id),
        ))

    return _assign_evidence_ids(docs)


def build_vector_docs_from_enterprise_graph(
    enterprise_graph: dict,
    scenario_id: str = "",
    alert_timeline: "list[dict] | None" = None,
    propagation_steps: "list[dict] | None" = None,
    enterprise_rca: "dict | None" = None,
) -> list[dict]:
    """Build fine-grained vector docs from an enterprise graph — one doc per
    node, edge, cross-diagram edge, diagram cluster, IP/interface, alert event,
    propagation step, GNN candidate, and impact-path edge.

    This produces the dense index needed for the Graph Copilot to answer
    exact investigative questions about any node/IP/edge in the graph.
    """
    docs: list[dict] = []
    eg    = _as_dict(enterprise_graph)
    rca   = _as_dict(enterprise_rca)
    scen  = scenario_id or str(eg.get("scenario_id") or "")
    run_id = scen or "enterprise"

    # ── Build node → diagram map ─────────────────────────────────────────────
    node_diag: dict[str, str] = {}
    _clusters_raw = eg.get("diagram_clusters", {})
    if isinstance(_clusters_raw, dict):
        for did, cl in _clusters_raw.items():
            nids = cl if isinstance(cl, list) else (cl.get("node_ids", []) if isinstance(cl, dict) else [])
            for nid in nids:
                node_diag[nid] = did
    elif isinstance(_clusters_raw, list):
        for cl in _clusters_raw:
            if isinstance(cl, dict):
                for nid in cl.get("node_ids", []):
                    node_diag[nid] = cl.get("diagram_id", "")

    # ── One doc per enterprise node ──────────────────────────────────────────
    for n in _as_list(eg.get("nodes")):
        if not isinstance(n, dict):
            continue
        nid    = str(n.get("id") or n.get("node_id") or "")
        ntype  = str(n.get("type") or "device")
        ip     = str(n.get("ip_address") or "")
        zone   = str(n.get("zone") or "")
        diag   = node_diag.get(nid) or str(n.get("diagram_id") or "")
        shared = bool(n.get("is_shared_entity"))
        text   = f"Enterprise node {nid} is a {ntype} in diagram {diag}."
        if zone:
            text += f" Zone: {zone}."
        if ip:
            text += f" IP address: {ip}."
        if shared:
            text += " It is a shared entity bridging multiple diagrams."
        docs.append(_make_doc(text, {
            **_base_metadata("enterprise_node", scenario_id=scen, diagram_id=diag,
                             node_id=nid, scope="enterprise", path=run_id),
            "ip_address": ip,
        }))

    # ── One doc per intra-diagram edge ───────────────────────────────────────
    for idx, e in enumerate(_as_list(eg.get("edges"))):
        if not isinstance(e, dict):
            continue
        src    = str(e.get("source") or "")
        tgt    = str(e.get("target") or "")
        rel    = str(e.get("relationship") or e.get("label") or "connected_to")
        diag   = node_diag.get(src) or str(e.get("diagram_id") or "")
        scope  = str(e.get("edge_scope") or "intra_diagram")
        text   = f"Enterprise edge: {src} {rel} {tgt} within diagram {diag} (scope: {scope})."
        docs.append(_make_doc(text, {
            **_base_metadata("enterprise_edge", scenario_id=scen, diagram_id=diag,
                             edge_id=f"{src}->{tgt}:{idx}", scope="enterprise", path=run_id),
            "source_node":  src,
            "target_node":  tgt,
            "relationship": rel,
            "edge_scope":   scope,
        }))

    # ── One doc per cross-diagram edge ───────────────────────────────────────
    for idx, e in enumerate(_as_list(eg.get("cross_diagram_edges"))):
        if not isinstance(e, dict):
            continue
        src    = str(e.get("source") or e.get("source_node") or "")
        tgt    = str(e.get("target") or e.get("target_node") or "")
        sd     = str(e.get("source_diagram") or node_diag.get(src) or "")
        td     = str(e.get("target_diagram") or node_diag.get(tgt) or "")
        rel    = str(e.get("label") or e.get("relationship") or "cross_link")
        text   = (
            f"Cross-diagram edge: {sd}:{src} → {td}:{tgt} ({rel}). "
            f"Bridges {sd} and {td}."
        )
        docs.append(_make_doc(text, {
            **_base_metadata("cross_diagram_edge", scenario_id=scen, diagram_id=sd,
                             edge_id=f"cross:{src}->{tgt}:{idx}", scope="cross_diagram",
                             path=run_id),
            "source_node":    src,
            "target_node":    tgt,
            "source_diagram": sd,
            "target_diagram": td,
            "relationship":   rel,
            "edge_scope":     "cross_diagram",
        }))

    # ── One doc per diagram cluster ──────────────────────────────────────────
    if isinstance(_clusters_raw, dict):
        for did, cl in _clusters_raw.items():
            nids = cl if isinstance(cl, list) else (cl.get("node_ids", []) if isinstance(cl, dict) else [])
            text = (
                f"Diagram cluster {did} contains {len(nids)} nodes in scenario {scen}: "
                + ", ".join(str(n) for n in nids[:20])
                + (f" ... and {len(nids)-20} more" if len(nids) > 20 else "") + "."
            )
            docs.append(_make_doc(text, _base_metadata(
                "diagram_cluster", scenario_id=scen, diagram_id=did, scope="enterprise", path=run_id,
            )))

    # ── One doc per IP / interface ───────────────────────────────────────────
    for n in _as_list(eg.get("nodes")):
        if not isinstance(n, dict):
            continue
        nid = str(n.get("id") or "")
        ip  = str(n.get("ip_address") or "")
        if ip and nid:
            diag = node_diag.get(nid) or str(n.get("diagram_id") or "")
            docs.append(_make_doc(
                f"IP/interface: node {nid} in diagram {diag} has IP address {ip}.",
                {
                    **_base_metadata("enterprise_ip", scenario_id=scen, diagram_id=diag,
                                     node_id=nid, scope="enterprise", path=run_id),
                    "ip_address": ip,
                },
            ))

    # ── One doc per alert timeline event ─────────────────────────────────────
    for idx, ev in enumerate(_as_list(alert_timeline)):
        if not isinstance(ev, dict):
            continue
        node   = str(ev.get("node") or ev.get("node_id") or "")
        atype  = str(ev.get("alert_type") or ev.get("title") or "alert")
        sev    = str(ev.get("severity") or "")
        diag   = str(ev.get("diagram_id") or node_diag.get(node) or "")
        ts     = str(ev.get("timestamp") or ev.get("time") or "")
        text   = f"Alert timeline event {idx+1}: {atype} on {node} in {diag}."
        if sev:
            text += f" Severity: {sev}."
        if ts:
            text += f" Timestamp: {ts}."
        docs.append(_make_doc(text, _base_metadata(
            "alert_timeline_event", scenario_id=scen, diagram_id=diag,
            node_id=node, incident_id=scen, scope="enterprise", path=run_id,
        )))

    # ── One doc per propagation step ─────────────────────────────────────────
    for idx, step in enumerate(_as_list(propagation_steps)):
        if not isinstance(step, dict):
            continue
        node   = str(step.get("node") or "")
        ts     = str(step.get("timestamp") or "")
        diag   = node_diag.get(node) or ""
        text   = f"Propagation step {idx+1}: reached node {node}."
        if ts:
            text += f" Timestamp: {ts}."
        if diag:
            text += f" Diagram: {diag}."
        docs.append(_make_doc(text, _base_metadata(
            "propagation_step", scenario_id=scen, diagram_id=diag,
            node_id=node, scope="enterprise", path=run_id,
        )))

    # ── One doc per GNN RCA candidate ────────────────────────────────────────
    rca_mode = str(rca.get("mode") or rca.get("rca_source") or "")
    for rank, cand in enumerate(_as_list(rca.get("top_candidates")), 1):
        if not isinstance(cand, dict):
            continue
        nid   = str(cand.get("node_id") or cand.get("node") or cand.get("id") or "")
        score = cand.get("score", "")
        ctype = str(cand.get("type") or "")
        diag  = node_diag.get(nid) or ""
        text  = f"GNN RCA candidate rank {rank}: node {nid}"
        if ctype:
            text += f" (type {ctype})"
        if score:
            text += f" score {score}"
        if diag:
            text += f" in diagram {diag}"
        text += "."
        docs.append(_make_doc(text, _base_metadata(
            "gnn_candidate", scenario_id=scen, diagram_id=diag,
            node_id=nid, rca_source=rca_mode, scope="enterprise", path=run_id,
        )))

    # ── One doc per impact-path edge ─────────────────────────────────────────
    impact_path = _as_list(rca.get("impact_path"))
    for i, (a, b) in enumerate(zip(impact_path, impact_path[1:])):
        a, b = str(a), str(b)
        diag = node_diag.get(a) or node_diag.get(b) or ""
        docs.append(_make_doc(
            f"Impact path edge {i+1}: {a} → {b} (RCA propagation in scenario {scen}).",
            _base_metadata("impact_path_edge", scenario_id=scen, diagram_id=diag,
                           edge_id=f"path:{a}->{b}", rca_source=rca_mode,
                           scope="enterprise", path=run_id),
        ))

    return _assign_evidence_ids(docs)
