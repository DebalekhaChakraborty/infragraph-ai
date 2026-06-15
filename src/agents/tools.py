"""
Thin wrapper functions around existing InfraGraph pipeline artifacts.
Each function wraps one existing capability and returns a plain dict.
No LLM is used for root cause reasoning — LLM is only used for remediation generation.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

# ── Lazy imports from sibling src/ packages ───────────────────────────────────
# These are available when REPO_ROOT/src is in sys.path (set by streamlit_app.py).

try:
    from incident_simulation.local_incidents import build_local_incident as _build_local_incident
    _INCIDENT_LOCAL_OK = True
except ImportError:
    _INCIDENT_LOCAL_OK = False
    _build_local_incident = None  # type: ignore

try:
    from incident_simulation.enterprise_incidents import (
        build_cross_diagram_hero_incident as _build_hero_incident,
    )
    _INCIDENT_ENT_OK = True
except ImportError:
    _INCIDENT_ENT_OK = False
    _build_hero_incident = None  # type: ignore

try:
    from ai_remediation import (
        generate_resolution_plan as _generate_resolution_plan,
        make_remediation_input   as _make_remediation_input,
        get_qwen_runtime_config  as _get_qwen_runtime_config,
    )
    _REM_OK = True
except ImportError:
    _generate_resolution_plan = None  # type: ignore
    _make_remediation_input   = None  # type: ignore
    _get_qwen_runtime_config  = None  # type: ignore
    _REM_OK = False

try:
    from runbook_retrieval import (  # type: ignore
        retrieve_candidate_runbooks as _retrieve_runbooks,
        rerank_runbooks             as _rerank_runbooks,
        apply_runbook_policy        as _apply_runbook_policy,
    )
    _RUNBOOK_OK = True
except ImportError:
    _retrieve_runbooks    = None  # type: ignore
    _rerank_runbooks      = None  # type: ignore
    _apply_runbook_policy = None  # type: ignore
    _RUNBOOK_OK = False


# ── Internal helpers ──────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_scenario_dir(repo_root: Path, scenario_id: str) -> Path | None:
    base = repo_root / "datasets" / "infragraph_v3" / "scenarios"
    for split in ("train", "val", "test"):
        p = base / split / scenario_id
        if (p / "enterprise_graph.json").exists():
            return p
    return None


def _default_scenario_id(repo_root: Path) -> str | None:
    base = repo_root / "datasets" / "infragraph_v3" / "scenarios" / "train"
    if base.exists():
        candidates = sorted(p.name for p in base.iterdir() if p.is_dir())
        if candidates:
            return candidates[0]
    return None


# ── Public tool functions ─────────────────────────────────────────────────────

def load_selected_context(
    repo_root: Path,
    selected_diagram_id: str | None = None,
    scenario_id: str | None = None,
) -> dict:
    """
    Load scenario / diagram context for the agent run.
    Returns: diagram_id, scenario_id, local_graph, enterprise_graph,
             alerts_data, graph_memory_packet, topology_source.
    """
    result: dict = {
        "diagram_id":         selected_diagram_id or "",
        "scenario_id":        scenario_id or "",
        "local_graph":        {},
        "enterprise_graph":   {},
        "alerts_data":        {},
        "graph_memory_packet": {},
        "topology_source":    "not_loaded",
    }

    # 1. Global graph memory built by previous runs
    global_graph = repo_root / "runtime_state" / "global_graph_memory" / "infragraph_global_graph.json"
    if global_graph.exists():
        try:
            result["enterprise_graph"] = json.loads(global_graph.read_text(encoding="utf-8"))
            result["topology_source"] = "global_graph_memory"
        except Exception:
            pass

    # 2. Dataset scenario (overrides global if found)
    if not scenario_id:
        scenario_id = _default_scenario_id(repo_root)
    if scenario_id:
        result["scenario_id"] = scenario_id
        sdir = _find_scenario_dir(repo_root, scenario_id)
        if sdir:
            for fname, key in (
                ("enterprise_graph.json", "enterprise_graph"),
                ("alerts.json",           "alerts_data"),
            ):
                p = sdir / fname
                if p.exists():
                    try:
                        result[key] = json.loads(p.read_text(encoding="utf-8"))
                    except Exception:
                        pass
            if result["enterprise_graph"]:
                result["topology_source"] = f"scenario:{scenario_id}"

            # Pick a local_graph from local_graphs/ subfolder
            lg_dir = sdir / "local_graphs"
            if lg_dir.exists():
                for dtype in (
                    "branch_topology", "datacenter_topology",
                    "app_db_topology", "shared_services_topology", "wan_topology",
                ):
                    lg_p = lg_dir / f"{dtype}.json"
                    if lg_p.exists():
                        try:
                            result["local_graph"] = json.loads(lg_p.read_text(encoding="utf-8"))
                            if not selected_diagram_id:
                                result["diagram_id"] = f"{scenario_id}__{dtype}"
                        except Exception:
                            pass
                        break

            # Scenario metadata may carry a canonical diagram_id
            meta_p = sdir / "metadata.json"
            if meta_p.exists() and not selected_diagram_id:
                try:
                    meta = json.loads(meta_p.read_text(encoding="utf-8"))
                    result["diagram_id"] = meta.get("scenario_id", scenario_id)
                except Exception:
                    pass

    # 3. Live ingestion packet for selected diagram
    if selected_diagram_id:
        li_p = (
            repo_root / "runtime_state" / "live_ingestion"
            / selected_diagram_id / "graph_memory_packet.json"
        )
        if li_p.exists():
            try:
                result["graph_memory_packet"] = json.loads(li_p.read_text(encoding="utf-8"))
            except Exception:
                pass

    return result


def simulate_alert_intake(
    local_graph: dict,
    diagram_id: str,
    diagram_type: str | None = None,
) -> dict:
    """
    Simulate topology-aware alert stream from a local diagram graph.
    Uses build_local_incident when available; deterministic fallback otherwise.
    """
    if _INCIDENT_LOCAL_OK and _build_local_incident and local_graph:
        try:
            inc = _build_local_incident(local_graph, diagram_id, diagram_type)
            return {
                "incident":      inc,
                "alert_timeline": inc.get("alert_timeline", []),
                "alert_source":  "simulated_topology_aware_alerts",
                "ok":            True,
            }
        except Exception:
            pass

    nodes = local_graph.get("nodes", [])
    first = nodes[0].get("id", "UNKNOWN") if nodes else "UNKNOWN"
    return {
        "incident": {
            "root_cause":         first,
            "alert_timeline":     [],
            "impact_path":        [first],
            "candidate_ranking":  [{"node_id": first, "score": 0.5}],
            "reasoning_steps":    ["Fallback: incident simulation not available"],
            "recommended_actions": [],
            "rca_source":         "local_graph_fallback",
        },
        "alert_timeline": [],
        "alert_source":   "local_graph_fallback",
        "ok":             False,
    }


def simulate_enterprise_alert_intake(
    enterprise_graph: dict,
    selected_diagram_id: str,
    scenario_id: str,
    alerts_data: dict | None = None,
    gnn_result:  dict | None = None,
) -> dict:
    """
    Build a cross-diagram enterprise incident using existing hero simulation.
    """
    if _INCIDENT_ENT_OK and _build_hero_incident and enterprise_graph:
        try:
            inc = _build_hero_incident(
                enterprise_graph,
                alerts_data or {},
                selected_diagram_id,
                gnn_result,
            )
            return {
                "incident":    inc,
                "alert_source": "simulated_cross_diagram_alerts",
                "ok":          True,
            }
        except Exception:
            pass

    return {
        "incident": {
            "incident_id":      f"ENT-{scenario_id or 'DEMO'}",
            "scenario_id":      scenario_id or "",
            "alert_timeline":   [],
            "root_cause":       "",
            "impacted_diagrams": [],
            "rca_source":       "enterprise_graph_fallback",
        },
        "alert_source": "enterprise_graph_fallback",
        "ok":           False,
    }


def _parse_gnn_result(data: dict, path: str) -> dict:
    """Normalise a raw GNN RCA JSON into a consistent agent result dict."""
    # Accept both 'predicted_root_cause' (inference output) and 'root_cause' (older schema)
    root_cause = (
        data.get("predicted_root_cause")
        or data.get("root_cause")
        or ""
    )
    top_candidates = data.get("top_candidates") or data.get("ranking") or []

    # Derive confidence: explicit field first, else top candidate score
    confidence = data.get("confidence")
    if confidence is None and top_candidates:
        confidence = top_candidates[0].get("score", 0.0)
    confidence = float(confidence or 0.0)

    impacted_diagrams = data.get("impacted_diagrams") or []
    # Also collect unique diagram_ids from top candidates when impacted_diagrams is sparse
    if len(impacted_diagrams) < 2 and top_candidates:
        from_candidates = list(dict.fromkeys(
            c["diagram_id"] for c in top_candidates[:10]
            if c.get("diagram_id")
        ))
        impacted_diagrams = impacted_diagrams or from_candidates

    return {
        "gnn_result":         data,
        "root_cause":         root_cause,
        "root_cause_diagram": data.get("root_cause_diagram", ""),
        "confidence":         confidence,
        "top_candidates":     top_candidates,
        "impacted_diagrams":  impacted_diagrams,
        "rca_source":         data.get("rca_source") or "Enterprise GNN RCA",
        "model_notes":        data.get("model_notes", {}),
        "ok":                 True,
        "path":               path,
    }


def load_or_run_enterprise_gnn_rca(repo_root: Path, scenario_id: str) -> dict:
    """
    Load pre-computed GNN RCA result JSON for scenario.
    Searches all known output locations and normalises both old and new schemas.
    Falls back with a warning listing all checked paths — never crashes the orchestrator.
    """
    sid = scenario_id or ""

    # Priority 1: V2 result (Temporal Relation-Aware GNN)
    if sid:
        _v2_path = (
            repo_root / "outputs" / "enterprise_gnn_rca_v2"
            / f"{sid}_enterprise_gnn_v2_rca_result.json"
        )
        if _v2_path.exists():
            try:
                _v2_data = json.loads(_v2_path.read_text(encoding="utf-8"))
                if isinstance(_v2_data, dict) and (
                    _v2_data.get("predicted_root_cause") or _v2_data.get("root_cause")
                ):
                    return _parse_gnn_result(_v2_data, str(_v2_path))
            except Exception:
                pass  # corrupt file — fall through to V1

    # Priority 2–5: V1 result locations (unchanged)
    # All directories to search, in priority order
    search_dirs = [
        repo_root / "outputs"  / "enterprise_gnn_rca",
        repo_root / "assets"   / "preloaded" / "enterprise_gnn_rca",
        repo_root / "demo_assets" / "enterprise_gnn_rca",
        repo_root / "runtime_state" / "enterprise_gnn_rca",
    ]
    # Filename variants to try per directory
    name_variants = (
        [
            f"{sid}_enterprise_gnn_rca_result.json",
            f"{sid}.json",
        ]
        if sid else []
    ) + ["enterprise_gnn_rca_result.json"]

    checked: list[str] = []
    for d in search_dirs:
        for name in name_variants:
            path = d / name
            checked.append(str(path))
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(data, dict):
                        continue
                    # Must have at least a root-cause field to be a valid result
                    if not (data.get("predicted_root_cause") or data.get("root_cause")):
                        continue
                    return _parse_gnn_result(data, str(path))
                except Exception:
                    continue

    model_exists = (
        repo_root / "model_artifacts" / "enterprise_gnn_rca" / "enterprise_gnn_rca.pt"
    ).exists()
    checked_str = "; ".join(checked[:6])
    return {
        "gnn_result":         None,
        "root_cause":         "",
        "root_cause_diagram": "",
        "confidence":         0.0,
        "top_candidates":     [],
        "impacted_diagrams":  [],
        "rca_source":         "scenario_grounded_fallback",
        "ok":                 False,
        "model_exists":       model_exists,
        "checked_paths":      checked,
        "warning": (
            f"No GNN RCA result found for scenario '{sid}'. "
            + ("Model exists — run Enterprise GNN RCA tab first." if model_exists
               else "Model not found in model_artifacts/.")
            + f" Checked: {checked_str}"
        ),
    }


def validate_rca_evidence(
    enterprise_graph: dict,
    incident: dict,
    gnn_result: dict | None,
) -> dict:
    """
    Deterministic evidence validation — no LLM involved.
    Returns a list of human-readable evidence bullets.
    """
    nodes       = enterprise_graph.get("nodes", [])
    edges       = enterprise_graph.get("edges", [])
    cross_edges = enterprise_graph.get("cross_diagram_edges", [])
    timeline    = incident.get("alert_timeline", [])
    imp_diag    = (
        (gnn_result or {}).get("impacted_diagrams")
        or incident.get("impacted_diagrams", [])
    )
    impact_path = incident.get("impact_path", incident.get("propagation_steps", []))
    root_cause  = (gnn_result or {}).get("root_cause") or incident.get("root_cause", "unknown")

    evidence = [
        f"{len(timeline)} alert event(s) in simulated timeline",
        f"{len(nodes)} nodes, {len(edges)} intra-diagram edges in enterprise graph",
    ]
    if cross_edges:
        evidence.append(f"{len(cross_edges)} cross-diagram edge(s) detected")
    if imp_diag:
        evidence.append(
            f"{len(imp_diag)} diagram(s) impacted: "
            + ", ".join(str(d) for d in imp_diag[:5])
        )
    if impact_path:
        evidence.append(
            "Propagation path: " + " → ".join(str(n) for n in impact_path[:6])
        )
    if gnn_result:
        candidates = gnn_result.get("ranking", gnn_result.get("top_candidates", []))
        if candidates:
            evidence.append(
                "Top GNN candidates: "
                + ", ".join(
                    f"{c.get('node_id', c.get('diagram_id', '?'))} "
                    f"(score {c.get('score', c.get('gnn_score', '?'))})"
                    for c in candidates[:3]
                )
            )
        for reason in (gnn_result.get("correlation_reasons") or [])[:3]:
            evidence.append(str(reason))
    else:
        for c in (incident.get("candidate_ranking") or [])[:3]:
            evidence.append(
                f"Candidate: {c.get('node_id', '?')} (score {c.get('score', '?')})"
            )

    return {
        "evidence_summary":  evidence,
        "root_cause":        root_cause,
        "impacted_diagrams": list(imp_diag),
        "alert_count":       len(timeline),
        "ok":                True,
    }


def build_remediation_context(
    run_id: str,
    selected_diagram_id: str,
    scenario_id: str,
    enterprise_graph: dict,
    ent_incident: dict,
    impacted_diagrams: list,
    root_cause: str,
    root_cause_diagram: str,
    rca_source: str,
    gnn_result: dict | None,
) -> dict:
    """
    Assemble a make_remediation_input() context from agent state.
    Returns empty dict if ai_remediation is unavailable.
    """
    if not _REM_OK or _make_remediation_input is None:
        return {
            "root_cause":   root_cause,
            "rca_source":   rca_source,
            "scenario_id":  scenario_id,
        }

    nodes = enterprise_graph.get("nodes", [])
    edges = enterprise_graph.get("edges", [])
    cross = enterprise_graph.get("cross_diagram_edges", [])
    clusters = enterprise_graph.get("diagram_clusters", {})
    if isinstance(clusters, dict):
        n_domains = len(clusters)
    elif isinstance(clusters, list):
        n_domains = len(clusters)
    else:
        n_domains = 0
    # Derive from unique node diagram_ids if clusters missing
    if n_domains == 0 and nodes:
        n_domains = len(set(n.get("diagram_id", "") for n in nodes if n.get("diagram_id")))

    # Resolve root-cause node type — prefer the actual root-cause node first
    _root_node_type = ""
    _root_lower = root_cause.lower()
    for _n in nodes:
        _nid = str(_n.get("id") or _n.get("node_id") or _n.get("canonical_id") or "")
        if _nid == root_cause:
            _root_node_type = str(_n.get("type") or _n.get("node_type") or "")
            break
    if not _root_node_type:
        if any(k in _root_lower for k in ("fw", "firewall")):
            _root_node_type = "firewall"
        elif any(k in _root_lower for k in ("rtr", "router", "wan")):
            _root_node_type = "router"
        elif any(k in _root_lower for k in ("lb", "load_balancer", "load balancer")):
            _root_node_type = "load_balancer"
        elif any(k in _root_lower for k in ("db", "database")):
            _root_node_type = "database"
        elif "dns" in _root_lower:
            _root_node_type = "dns"
        elif any(k in _root_lower for k in ("srv", "server", "app")):
            _root_node_type = "server"
        else:
            # Final fallback: dominant type across all graph nodes
            _node_type_counts: dict[str, int] = {}
            for _n in nodes:
                _nt = _n.get("type", _n.get("node_type", ""))
                if _nt:
                    _node_type_counts[_nt] = _node_type_counts.get(_nt, 0) + 1
            _root_node_type = (
                max(_node_type_counts, key=_node_type_counts.get)  # type: ignore[arg-type]
                if _node_type_counts else ""
            )

    # Retrieve and rank runbook chain
    runbook_chain: list[dict] = []
    if _RUNBOOK_OK and _retrieve_runbooks and _rerank_runbooks and _apply_runbook_policy:
        try:
            _conf_for_policy = float((gnn_result or {}).get("confidence") or 0.5)
            _candidates = _retrieve_runbooks(
                root_cause=root_cause,
                root_cause_diagram=root_cause_diagram,
                node_type=_root_node_type,
                alert_timeline=ent_incident.get("alert_timeline", []),
                impacted_diagrams=list(impacted_diagrams),
                evidence_summary=[],
            )
            _rca_ctx = {
                "rca_source":        rca_source,
                "confidence":        _conf_for_policy,
                "impacted_diagrams": list(impacted_diagrams),
                "node_type":         _root_node_type,
            }
            _ranked = _rerank_runbooks(_candidates, _rca_ctx)
            runbook_chain = _apply_runbook_policy(
                _ranked,
                calibrated_confidence=_conf_for_policy,
            )
        except Exception:
            runbook_chain = []

    ctx = _make_remediation_input(
        incident_id=ent_incident.get("incident_id", f"AGT-{run_id[:6]}"),
        scope="enterprise",
        selected_diagram_id=selected_diagram_id,
        scenario_id=scenario_id,
        alert_timeline=ent_incident.get("alert_timeline", []),
        graph_memory_summary=(
            f"{len(nodes)} nodes, {len(edges)} intra-diagram edges, "
            f"{len(cross)} cross-diagram edges across {n_domains} topology domain(s)."
        ),
        root_cause=root_cause,
        root_cause_diagram=root_cause_diagram,
        impacted_nodes=[],
        impacted_diagrams=list(impacted_diagrams),
        impact_path=ent_incident.get("impact_path", ent_incident.get("propagation_steps", [])),
        candidate_ranking=ent_incident.get("candidate_ranking", []),
        gnn_result_available=bool(gnn_result),
        rca_source=rca_source,
        device_context=[
            {
                "node_id":     n.get("id", ""),
                "device_type": n.get("type", ""),
                "diagram_id":  n.get("diagram_id", ""),
            }
            for n in nodes[:15]
        ],
        connector_context=[
            {
                "source":     e.get("source", ""),
                "target":     e.get("target", ""),
                "type":       e.get("label", ""),
                "diagram_id": e.get("diagram_id", ""),
            }
            for e in edges[:10]
        ],
        cluster_id=(gnn_result or {}).get("cluster_id", ""),
        cluster_score=(gnn_result or {}).get("cluster_score"),
        correlation_reasons=(gnn_result or {}).get("correlation_reasons", []),
        causal_evidence=(gnn_result or {}).get("causal_evidence", []),
        runbook_chain=runbook_chain or None,
    )
    ctx["root_node_type"] = _root_node_type
    return ctx


def generate_ai_remediation(
    context: dict,
    prefer_qwen: bool = True,
    base_url: str | None = None,
    model:    str | None = None,
    timeout:  int | None = None,
) -> dict:
    """
    Generate remediation using Qwen/vLLM or template fallback.
    Source field: qwen_vllm | template | template_fallback | unavailable.
    """
    if not context:
        return {"source": "skipped", "ok": False, "error": "No context", "response": {}}

    if _REM_OK and _get_qwen_runtime_config and not (base_url and model):
        try:
            cfg = _get_qwen_runtime_config()
            base_url = base_url or cfg.get("base_url")
            model    = model    or cfg.get("model")
            timeout  = timeout  or cfg.get("timeout")
        except Exception:
            pass

    if _REM_OK and _generate_resolution_plan:
        try:
            return _generate_resolution_plan(
                context,
                scope="enterprise",
                prefer_qwen=prefer_qwen,
                base_url=base_url,
                model=model,
                timeout=timeout,
            )
        except Exception as e:
            return {"source": "template_fallback", "ok": False, "error": str(e), "response": {}}

    return {"source": "unavailable", "ok": False, "error": "ai_remediation not available", "response": {}}


def draft_itsm_ticket(
    root_cause: str,
    rca_result: dict,
    remediation_result: dict,
    incident: dict,
    approval_status: str = "pending",
) -> dict:
    """
    Generate a demo ITSM ticket draft.
    Does NOT call any external ITSM / ServiceNow API.
    """
    impacted   = rca_result.get("impacted_diagrams", incident.get("impacted_diagrams", []))
    n_impacted = len(impacted)
    confidence = float(rca_result.get("confidence", 0.0))

    if n_impacted >= 3 or confidence >= 0.8:
        priority         = "P1 — Critical"
        assignment_group = "Network Operations — Tier 3"
    elif n_impacted >= 2 or confidence >= 0.6:
        priority         = "P2 — High"
        assignment_group = "Network Operations — Tier 2"
    else:
        priority         = "P3 — Medium"
        assignment_group = "Network Operations — Tier 1"

    h         = hashlib.md5(f"{root_cause}{rca_result.get('rca_source', '')}".encode()).hexdigest()[:6].upper()
    ticket_id = f"DEMO-INC-{h}"
    rc_label  = root_cause or "unknown device"
    short_desc = f"[InfraGraph AI] Cross-diagram fault — root cause: {rc_label}"[:120]
    rca_src   = rca_result.get("rca_source", "graph-based RCA")
    rem_src   = remediation_result.get("source", "template")

    description = (
        f"InfraGraph AI detected a cross-diagram infrastructure incident affecting "
        f"{n_impacted} topology domain(s). Root cause: '{root_cause}' via {rca_src} "
        f"(confidence: {confidence:.0%}). Remediation: {rem_src}.\n\n"
        f"Impacted: {', '.join(str(d) for d in impacted[:10]) or 'unknown'}.\n\n"
        f"DEMO ONLY — no external ITSM system was contacted. "
        f"Human approval required before execution."
    )

    ev_items = rca_result.get("evidence_summary", [])
    evidence_summary = "\n".join(f"• {e}" for e in ev_items) if ev_items else "See agent trace."

    rem_response = remediation_result.get("response", {})
    if isinstance(rem_response, dict):
        steps = rem_response.get("resolution_steps", rem_response.get("steps", []))
        if steps:
            rem_summary = "\n".join(
                f"{i+1}. {s.get('action', s) if isinstance(s, dict) else s}"
                for i, s in enumerate(steps[:5])
            )
        else:
            rem_summary = str(rem_response.get("summary", ""))[:500]
    else:
        rem_summary = str(rem_response)[:500]

    return {
        "ticket_id":           ticket_id,
        "short_description":   short_desc,
        "description":         description,
        "priority":            priority,
        "category":            "Infrastructure / Network",
        "assignment_group":    assignment_group,
        "affected_ci":         root_cause or incident.get("root_cause", "Unknown"),
        "impacted_diagrams":   list(impacted),
        "evidence_summary":    evidence_summary,
        "remediation_summary": rem_summary,
        "approval_status":     approval_status,
    }


def persist_agent_run(repo_root: Path, agent_run: dict) -> str:
    """
    Save agent_run.json and ticket_draft.json to runtime_state/agent_runs/<run_id>/.
    Returns the directory path as a string.
    """
    run_id  = agent_run.get("run_id", "unknown")
    run_dir = repo_root / "runtime_state" / "agent_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "agent_run.json").write_text(
        json.dumps(agent_run, indent=2, default=str), encoding="utf-8"
    )
    ticket = agent_run.get("ticket_draft", {})
    if ticket:
        (run_dir / "ticket_draft.json").write_text(
            json.dumps(ticket, indent=2, default=str), encoding="utf-8"
        )
    return str(run_dir)
