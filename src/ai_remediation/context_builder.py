"""
context_builder.py — Build remediation input from a clean RCA JSON output.

Reads:
  assets/preloaded/enterprise_gnn_rca/<scenario_id>.json  (RCA result)
  scenario_library/enterprise_gnn_rca/<case_id>/events.json (observable events)

Optionally retrieves SOP/KB evidence from the ChromaDB vector index.

Never reads labels.json.
Never adds ground_truth or evaluation fields to the returned context.

Public API
----------
load_json(path: Path) -> dict
build_enterprise_remediation_context(...) -> dict
"""
from __future__ import annotations

import json
from pathlib import Path

from .response_schema import make_remediation_input


def load_json(path: Path) -> dict:
    """Load and parse a JSON file.  Raises FileNotFoundError or json.JSONDecodeError on failure."""
    return json.loads(path.read_text(encoding="utf-8"))


def build_enterprise_remediation_context(
    *,
    repo_root: Path,
    scenario_id: str,
    rca_path: Path | None = None,
    scenario_library_root: Path | None = None,
    # KB retrieval options
    kb_index_dir: Path | None = None,
    retrieve_kb: bool = True,
    kb_top_k: int = 5,
    strict_kb: bool = False,
) -> dict:
    """
    Build a remediation input context dict from a clean Enterprise GNN RCA output.

    Parameters
    ----------
    repo_root              : repository root (used to resolve relative paths)
    scenario_id            : e.g. "enterprise_v3_0000"
    rca_path               : override path to RCA JSON
                             (default: assets/preloaded/enterprise_gnn_rca/<scenario_id>.json)
    scenario_library_root  : override path to scenario_library root
                             (default: repo_root/scenario_library)
    kb_index_dir           : override path to ChromaDB KB index
                             (default: repo_root/runtime_state/kb_index)
    retrieve_kb            : if True, query the KB index and inject evidence
    kb_top_k               : max KB evidence chunks to retrieve
    strict_kb              : if True and no KB evidence found, raise RuntimeError

    Returns
    -------
    dict from make_remediation_input() — safe to pass to generate_resolution_plan()

    Integrity
    ---------
    This function never reads labels.json and never adds ground_truth or
    evaluation fields to the returned context.
    """
    rca_path = rca_path or (
        repo_root / "assets" / "preloaded" / "enterprise_gnn_rca" / f"{scenario_id}.json"
    )
    lib_root = scenario_library_root or (repo_root / "scenario_library")

    rca = load_json(rca_path)

    case_id = rca.get("case_id", f"ent_{scenario_id}")

    # Load observable events — never labels
    events_path = lib_root / "enterprise_gnn_rca" / case_id / "events.json"
    alert_timeline: list[dict] = []
    if events_path.exists():
        events_doc  = load_json(events_path)
        alert_timeline = events_doc.get("events", [])

    # Extract fields from RCA output
    root_cause       = rca.get("predicted_root_cause", "")
    root_cause_diagram = rca.get("root_cause_diagram", "")
    impacted_diagrams  = rca.get("impacted_diagrams", [])
    candidate_ranking  = rca.get("top_candidates", [])
    rca_source         = rca.get("rca_source", "")
    cluster_id         = rca.get("cluster_id", "")
    cluster_score      = rca.get("cluster_score")
    correlation_reasons = rca.get("correlation_reasons", [])
    causal_evidence    = rca.get("causal_evidence", [])

    # Impacted nodes: unique supporting_nodes from causal_evidence first,
    # then fall back to unique node_ids from top_candidates
    impacted_nodes: list[str] = []
    for item in causal_evidence:
        for n in item.get("supporting_nodes", []):
            if n and n not in impacted_nodes:
                impacted_nodes.append(n)
    if not impacted_nodes:
        for c in candidate_ranking:
            n = c.get("node_id", "")
            if n and n not in impacted_nodes:
                impacted_nodes.append(n)

    # Impact path: from first causal_evidence supporting_nodes list
    # or fall back to top candidate node IDs
    impact_path: list[str] = []
    if causal_evidence:
        impact_path = causal_evidence[0].get("supporting_nodes", [])
    if not impact_path:
        impact_path = [c.get("node_id", "") for c in candidate_ranking if c.get("node_id")]

    # First observed node from alert timeline
    first_observed_node = alert_timeline[0].get("node", "") if alert_timeline else ""

    # ── KB / SOP retrieval ────────────────────────────────────────────────────
    kb_evidence: list[dict] = []
    if retrieve_kb:
        _idx_dir = kb_index_dir or (repo_root / "runtime_state" / "kb_index")
        kb_evidence = _retrieve_kb_evidence_safe(
            context_preview={
                "root_cause":          root_cause,
                "root_cause_diagram":  root_cause_diagram,
                "impacted_diagrams":   impacted_diagrams,
                "alert_timeline":      alert_timeline,
                "candidate_ranking":   candidate_ranking,
                "correlation_reasons": correlation_reasons,
                "causal_evidence":     causal_evidence,
            },
            index_dir=_idx_dir,
            top_k=kb_top_k,
        )
        if strict_kb and not kb_evidence:
            raise RuntimeError(
                f"strict_kb=True but no KB evidence retrieved for scenario {scenario_id}. "
                "Run: python scripts/build_kb_index.py --reset"
            )

    # Build graph memory summary (includes KB chunk count if retrieved)
    cluster_score_str = f"{cluster_score:.4f}" if cluster_score is not None else "—"
    kb_note = f"; kb_chunks_retrieved={len(kb_evidence)}" if kb_evidence else ""
    graph_memory_summary = (
        f"Enterprise RCA scenario {scenario_id}; "
        f"root={root_cause}; "
        f"diagram={root_cause_diagram}; "
        f"cluster={cluster_id}; "
        f"cluster_score={cluster_score_str}; "
        f"impacted_diagrams={impacted_diagrams}{kb_note}."
    )

    context = make_remediation_input(
        incident_id=f"INC-{scenario_id}",
        scope="enterprise",
        selected_diagram_id=root_cause_diagram,
        diagram_type=root_cause_diagram,
        scenario_id=scenario_id,
        alert_timeline=alert_timeline,
        graph_memory_summary=graph_memory_summary,
        root_cause=root_cause,
        root_cause_diagram=root_cause_diagram,
        first_observed_node=first_observed_node,
        impacted_nodes=impacted_nodes,
        impacted_diagrams=impacted_diagrams,
        impact_path=impact_path,
        candidate_ranking=candidate_ranking,
        gnn_result_available=(rca_source == "Enterprise GNN RCA"),
        rca_source=rca_source,
        cluster_id=cluster_id,
        cluster_score=cluster_score,
        correlation_reasons=correlation_reasons,
        causal_evidence=causal_evidence,
    )

    # Inject KB evidence into both tracking fields
    if kb_evidence:
        existing = list(context.get("retrieved_graph_memory_evidence", []) or [])
        context["retrieved_graph_memory_evidence"] = existing + kb_evidence
        context["retrieved_kb_evidence"] = kb_evidence

    return context


def _retrieve_kb_evidence_safe(
    *,
    context_preview: dict,
    index_dir: Path,
    top_k: int,
) -> list[dict]:
    """Attempt KB retrieval; return [] on any import or retrieval error."""
    try:
        from kb_retrieval.retriever import retrieve_kb_evidence
        from kb_retrieval.schema import DEFAULT_COLLECTION
        return retrieve_kb_evidence(
            context=context_preview,
            index_dir=index_dir,
            collection_name=DEFAULT_COLLECTION,
            top_k=top_k,
        )
    except Exception:
        return []
