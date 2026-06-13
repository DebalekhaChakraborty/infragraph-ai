"""
inspect_gnn_rca_proof.py — Inspect the provenance of an Enterprise GNN RCA result.

Usage
-----
    python scripts/inspect_gnn_rca_proof.py --scenario enterprise_v3_0072

Reads:
  assets/preloaded/enterprise_gnn_rca/<scenario>.json         (main RCA artifact)
  assets/preloaded/enterprise_gnn_rca/<scenario>_enterprise_gnn_rca_result.json  (model output)
  assets/preloaded/enterprise_gnn_rca/enterprise_gnn_metrics.json
  assets/preloaded/remediation/<scenario>.json                (remediation output, if exists)

Never reads:
  labels.json, ground_truth files, evaluation outputs, or any annotation file.

Output schema
-------------
  scenario_id            : str
  rca_source             : str
  inference_mode         : "precomputed_gnn_inference_artifact" | "no_trained_result"
  predicted_root_cause   : str
  root_cause_diagram     : str
  confidence             : float
  model_backend          : str
  model_architecture     : str
  model_path             : str
  metrics_path           : str
  train_top1 / val_top1 / test_top1 : float
  top_candidates         : list[{rank, node, diagram, type, score}]
  evidence_ids           : list[str]  (CE-* from causal_evidence)
  propagation_path       : list[str]  (supporting_nodes from propagation_hypothesis stage)
  impacted_diagrams      : list[str]
  runbook_availability   : "kb_index_required" | list[str]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Repository root ──────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent

# ── Forbidden fields (never read or display) ─────────────────────────────────
_FORBIDDEN_KEYS = frozenset({
    "ground_truth_root_cause",
    "ground_truth_node",
    "expected_root_cause",
    "is_correct",
    "correct_top1",
    "correct_top_k",
    "reciprocal_rank",
    "evaluation",
    "label",
    "labels",
})


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  [WARN] Could not read {path}: {exc}", file=sys.stderr)
        return {}


def _strip_forbidden(obj: dict) -> dict:
    """Return obj with all forbidden keys removed (top-level only)."""
    return {k: v for k, v in obj.items() if k not in _FORBIDDEN_KEYS}


def _check_runbook_availability(scenario_id: str, remediation: dict) -> list[str] | str:
    """Return KB IDs from remediation if present, else indicate KB index is needed."""
    evidence_ids: list[str] = remediation.get("evidence_ids_used", [])
    kb_ids = [e for e in evidence_ids if e.startswith("KB-") or e.startswith("RB-")]
    if kb_ids:
        return kb_ids
    kb_index_dir = REPO_ROOT / "runtime_state" / "kb_index"
    if not kb_index_dir.exists():
        return "kb_index_required -- run: python scripts/build_kb_index.py --reset"
    return "kb_available -- re-generate remediation with --prefer-qwen or --template-only --strict-kb"


def _propagation_path_from_evidence(causal_evidence: list[dict]) -> list[str]:
    """Extract propagation path from propagation_hypothesis stage if present."""
    for item in causal_evidence:
        if item.get("stage") == "propagation_hypothesis":
            return item.get("supporting_nodes", [])
    if causal_evidence:
        return causal_evidence[0].get("supporting_nodes", [])
    return []


def inspect(scenario_id: str, verbose: bool = False) -> dict:
    gnn_dir    = REPO_ROOT / "assets" / "preloaded" / "enterprise_gnn_rca"
    rem_dir    = REPO_ROOT / "assets" / "preloaded" / "remediation"

    main_rca_path    = gnn_dir / f"{scenario_id}.json"
    result_rca_path  = gnn_dir / f"{scenario_id}_enterprise_gnn_rca_result.json"
    metrics_path     = gnn_dir / "enterprise_gnn_metrics.json"
    model_path       = gnn_dir / "enterprise_gnn_model.pt"
    remediation_path = rem_dir / f"{scenario_id}.json"

    main_rca    = _strip_forbidden(_load_json(main_rca_path))
    result_rca  = _strip_forbidden(_load_json(result_rca_path))
    metrics     = _load_json(metrics_path)
    remediation = _load_json(remediation_path)
    rem_content = remediation.get("remediation") or remediation

    if not main_rca and not result_rca:
        print(f"[ERROR] No GNN RCA artifact found for scenario '{scenario_id}'", file=sys.stderr)
        print(f"  Checked: {main_rca_path}", file=sys.stderr)
        print(f"  Checked: {result_rca_path}", file=sys.stderr)
        sys.exit(1)

    # Inference mode
    inference_source = result_rca.get("inference_source", "")
    if result_rca and inference_source in ("trained_enterprise_gnn", "precomputed"):
        inference_mode = "precomputed_gnn_inference_artifact"
    elif result_rca and result_rca.get("predicted_root_cause"):
        inference_mode = "precomputed_gnn_inference_artifact"
    else:
        inference_mode = "no_trained_result"

    # Model info
    backend      = metrics.get("backend")      or result_rca.get("backend")      or "torch"
    architecture = metrics.get("architecture") or result_rca.get("model_type")   or "—"
    model_type   = metrics.get("model_type")   or result_rca.get("model_type")   or "Enterprise GCN RCA"

    # Root cause — from main_rca (causal evidence), fall back to result_rca
    predicted_rc   = main_rca.get("predicted_root_cause") or result_rca.get("predicted_root_cause") or "—"
    rc_diagram     = main_rca.get("root_cause_diagram")   or result_rca.get("root_cause_diagram")   or "—"
    confidence     = main_rca.get("confidence", result_rca.get("confidence", None))
    rca_source     = main_rca.get("rca_source") or ("Enterprise GNN RCA" if result_rca else "—")
    impacted_diags = main_rca.get("impacted_diagrams") or result_rca.get("impacted_diagrams", [])

    # Causal evidence IDs (CE-*)
    causal_evidence = main_rca.get("causal_evidence", [])
    evidence_ids    = [e.get("evidence_id", "") for e in causal_evidence if e.get("evidence_id")]

    # Evidence IDs from remediation (CE-*, KB-*)
    rem_evidence_ids = rem_content.get("evidence_ids_used", [])

    # Propagation path
    propagation_path = _propagation_path_from_evidence(causal_evidence)

    # Top candidates — prefer main_rca (softmax normalised), fall back to result_rca (raw scores)
    top_candidates = []
    for c in (main_rca.get("top_candidates") or result_rca.get("top_candidates") or []):
        top_candidates.append({
            "rank":    c.get("rank", "—"),
            "node":    c.get("node_id") or c.get("node", "—"),
            "diagram": c.get("diagram_id") or c.get("diagram_type", "—"),
            "type":    c.get("node_type") or c.get("type", "—"),
            "score":   round(float(c.get("score", 0)), 6),
        })

    # Runbook availability
    runbook_avail = _check_runbook_availability(scenario_id, rem_content)

    proof = {
        "scenario_id":          scenario_id,
        "rca_source":           rca_source,
        "inference_mode":       inference_mode,
        "predicted_root_cause": predicted_rc,
        "root_cause_diagram":   rc_diagram,
        "confidence":           confidence,
        "model_backend":        backend,
        "model_type":           model_type,
        "model_architecture":   architecture,
        "model_path":           str(model_path),
        "model_available":      model_path.exists(),
        "metrics_path":         str(metrics_path),
        "metrics_available":    metrics_path.exists(),
        "train_top1":           metrics.get("train_metrics", {}).get("top1"),
        "val_top1":             metrics.get("val_metrics",  {}).get("top1"),
        "test_top1":            metrics.get("test_metrics", {}).get("top1"),
        "val_mrr":              metrics.get("val_metrics",  {}).get("mrr"),
        "epochs_trained":       metrics.get("epochs_trained"),
        "top_candidates":       top_candidates,
        "evidence_ids_gnn":     evidence_ids,
        "evidence_ids_remediation": rem_evidence_ids,
        "propagation_path":     propagation_path,
        "impacted_diagrams":    impacted_diags,
        "runbook_availability": runbook_avail,
        "remediation_available": remediation_path.exists(),
        "remediation_source":   remediation.get("remediation_source") or remediation.get("source") or "—",
    }

    if verbose:
        proof["causal_evidence_stages"] = [
            {"id": e.get("evidence_id"), "stage": e.get("stage"), "confidence": e.get("confidence")}
            for e in causal_evidence
        ]
        proof["correlation_reasons"] = main_rca.get("correlation_reasons", [])
        proof["cluster_id"]          = main_rca.get("cluster_id", "—")
        proof["cluster_score"]       = main_rca.get("cluster_score")

    return proof


def _fmt_proof(proof: dict) -> str:
    sep = "=" * 64
    lines = [
        sep,
        f"  GNN RCA Proof: {proof['scenario_id']}",
        sep,
        f"  rca_source           : {proof['rca_source']}",
        f"  inference_mode       : {proof['inference_mode']}",
        f"  predicted_root_cause : {proof['predicted_root_cause']}",
        f"  root_cause_diagram   : {proof['root_cause_diagram']}",
        f"  confidence           : {proof['confidence']}",
        "",
        f"  model_type           : {proof['model_type']}",
        f"  model_architecture   : {proof['model_architecture']}",
        f"  model_backend        : {proof['model_backend']}",
        f"  epochs_trained       : {proof['epochs_trained']}",
        f"  train_top1           : {proof['train_top1']}",
        f"  val_top1             : {proof['val_top1']}",
        f"  test_top1            : {proof['test_top1']}",
        f"  val_mrr              : {proof['val_mrr']}",
        f"  model_path           : {proof['model_path']}  ({'OK' if proof['model_available'] else 'MISSING'})",
        f"  metrics_path         : {proof['metrics_path']}  ({'OK' if proof['metrics_available'] else 'MISSING'})",
        "",
        "  top_candidates:",
    ]
    for c in proof.get("top_candidates", []):
        lines.append(f"    [{c['rank']}] {c['node']} ({c['type']}) in {c['diagram']}  score={c['score']}")
    _prop_path = " -> ".join(proof['propagation_path']) if proof['propagation_path'] else "—"
    lines += [
        "",
        f"  evidence_ids (GNN)         : {', '.join(proof['evidence_ids_gnn']) or 'none'}",
        f"  evidence_ids (remediation) : {', '.join(str(e) for e in proof['evidence_ids_remediation']) or 'none'}",
        f"  propagation_path           : {_prop_path}",
        f"  impacted_diagrams          : {', '.join(proof['impacted_diagrams']) or '—'}",
        "",
        f"  remediation_available : {proof['remediation_available']}",
        f"  remediation_source    : {proof['remediation_source']}",
        f"  runbook_availability  : {proof['runbook_availability']}",
    ]
    if "causal_evidence_stages" in proof:
        lines += ["", "  causal_evidence_stages:"]
        for s in proof["causal_evidence_stages"]:
            lines.append(f"    {s['id']}  stage={s['stage']}  confidence={s['confidence']}")
        if proof.get("correlation_reasons"):
            lines += ["", "  correlation_reasons:"]
            for r in proof["correlation_reasons"]:
                lines.append(f"    - {r}")
        lines += [
            f"  cluster_id    : {proof.get('cluster_id')}",
            f"  cluster_score : {proof.get('cluster_score')}",
        ]
    lines.append(sep)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the provenance of an Enterprise GNN RCA result.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scenario", "-s",
        required=True,
        help="Scenario ID, e.g. enterprise_v3_0072",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output raw JSON instead of formatted text",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Include causal evidence stages and correlation reasons",
    )
    args = parser.parse_args()

    proof = inspect(args.scenario, verbose=args.verbose)
    if args.json:
        print(json.dumps(proof, indent=2, default=str))
    else:
        print(_fmt_proof(proof))


if __name__ == "__main__":
    main()
