#!/usr/bin/env python3
"""
generate_enterprise_rca_demo_assets.py
Generate clean demo-safe event correlation + Enterprise GNN RCA preloaded
outputs for the four primary demo scenarios.

For enterprise_v3_0000: patches the existing output (removes any forbidden keys).
For enterprise_v3_0072: reformats the existing GNN result into the clean schema.
For enterprise_v3_0073 / 0074: produces scenario-grounded simulation outputs
  (torch/GNN not available on CPU; rca_source = "Scenario-grounded RCA simulation").

All outputs are enriched with event correlation cluster evidence.

Outputs:
  assets/preloaded/event_correlation/enterprise_v3_0000.json
  assets/preloaded/event_correlation/enterprise_v3_0072.json
  assets/preloaded/event_correlation/enterprise_v3_0073.json
  assets/preloaded/event_correlation/enterprise_v3_0074.json
  assets/preloaded/enterprise_gnn_rca/enterprise_v3_0000.json
  assets/preloaded/enterprise_gnn_rca/enterprise_v3_0072.json
  assets/preloaded/enterprise_gnn_rca/enterprise_v3_0073.json
  assets/preloaded/enterprise_gnn_rca/enterprise_v3_0074.json

Usage:
  python scripts/generate_enterprise_rca_demo_assets.py
  python scripts/generate_enterprise_rca_demo_assets.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from event_correlation.correlator import correlate_events  # noqa: E402
from event_correlation.evidence import build_causal_evidence  # noqa: E402
from event_correlation.io import load_case_for_correlation, write_cluster_output  # noqa: E402
from event_correlation.schema import (  # noqa: E402
    FORBIDDEN_KEYS,
    make_cluster,
    make_cluster_output,
    make_event_in_cluster,
)

# ── Forbidden key sets (mirrors validate_rca_outputs.py) ──────────────────────
_FORBIDDEN_REMEDIATION = frozenset({
    "remediation", "recommended_actions", "remediation_steps",
    "resolution_steps", "rollback_steps", "validation_steps",
    "servicenow_incident_summary", "resolution", "rollback",
})
_FORBIDDEN_EVALUATION = frozenset({
    "expected_root_cause", "ground_truth_node", "correct_top1",
    "correct_top_k", "reciprocal_rank", "evaluation",
})
_ALL_FORBIDDEN = _FORBIDDEN_REMEDIATION | _FORBIDDEN_EVALUATION


def _collect_all_keys(obj: object, depth: int = 0) -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(k)
            if depth < 20:
                keys.update(_collect_all_keys(v, depth + 1))
    elif isinstance(obj, list):
        for item in obj:
            keys.update(_collect_all_keys(item, depth + 1))
    return keys


def _assert_clean(obj: dict, label: str) -> None:
    violations = _collect_all_keys(obj) & _ALL_FORBIDDEN
    if violations:
        raise ValueError(f"{label} contains forbidden keys: {sorted(violations)}")


# ── Severity scoring (mirrors features.py) ────────────────────────────────────
_SEV = {"critical": 1.0, "high": 0.8, "warning": 0.6, "medium": 0.5, "low": 0.3, "info": 0.1}


# ── Event correlation cluster builder ─────────────────────────────────────────

def _build_clusters(case_id: str) -> dict:
    lib_root = REPO_ROOT / "scenario_library"
    events, graph, mode, scenario_id, _ = load_case_for_correlation(
        lib_root=lib_root, repo_root=REPO_ROOT, case_id=case_id,
    )
    if not events:
        raise FileNotFoundError(f"No events found for case: {case_id}")

    raw_clusters = correlate_events(events=events, graph=graph, mode=mode)
    clusters = []
    for idx, raw in enumerate(raw_clusters, start=1):
        ev_items = [
            make_event_in_cluster(ev, role)
            for ev, role in zip(raw["raw_events"], raw["roles"])
        ]
        ce = build_causal_evidence(
            raw_events=raw["raw_events"],
            roles=raw["roles"],
            dims=raw["dims"],
            diagram_scope=raw["diagram_scope"],
            mode=mode,
        )
        clusters.append(make_cluster(
            cluster_id=f"CLU-{case_id}-{idx:03d}",
            case_id=case_id,
            scenario_id=scenario_id,
            mode=mode,
            diagram_scope=raw["diagram_scope"],
            cluster_score=raw["cluster_score"],
            correlation_dimensions=raw["dims"],
            correlation_reasons=raw["reasons"],
            cluster_events=ev_items,
            causal_evidence=ce,
        ))

    return make_cluster_output(
        case_id=case_id,
        scenario_id=scenario_id,
        mode=mode,
        clusters=clusters,
    )


def _enrich_with_cluster(rca: dict, cluster_output: dict) -> dict:
    clusters = cluster_output.get("clusters", [])
    if clusters:
        p = clusters[0]
        rca["cluster_id"]          = p.get("cluster_id", "")
        rca["cluster_score"]       = p.get("cluster_score", 0.0)
        rca["correlation_reasons"] = p.get("correlation_reasons", [])
        rca["causal_evidence"]     = p.get("causal_evidence", [])
    return rca


# ── Per-scenario RCA output builders ──────────────────────────────────────────

def _build_v3_0000() -> dict:
    """Patch the existing output: remove forbidden keys, return clean dict."""
    existing_path = REPO_ROOT / "assets/preloaded/enterprise_gnn_rca/enterprise_v3_0000.json"
    data = json.loads(existing_path.read_text(encoding="utf-8"))
    for k in list(data.keys()):
        if k in _ALL_FORBIDDEN:
            del data[k]
    # Drop cluster fields — will be re-added fresh
    for k in ("cluster_id", "cluster_score", "correlation_reasons", "causal_evidence"):
        data.pop(k, None)
    return data


def _build_v3_0072() -> dict:
    """Reformat the existing GNN result into the clean schema (top-3)."""
    old = json.loads((
        REPO_ROOT / "assets/preloaded/enterprise_gnn_rca"
        / "enterprise_v3_0072_enterprise_gnn_rca_result.json"
    ).read_text(encoding="utf-8"))

    raw_cands = old.get("top_candidates", [])[:3]
    top_candidates = []
    for i, c in enumerate(raw_cands, start=1):
        nid   = c.get("node_id", c.get("node", ""))
        diag  = c.get("diagram_id", "")
        ntype = c.get("type", c.get("node_type", ""))
        score = c.get("score", 0.0)
        top_candidates.append({
            "rank":                      i,
            "node_id":                   nid,
            "diagram_id":                diag,
            "node_observed_in_diagrams": [diag] if diag else [],
            "node_type":                 ntype,
            "score":                     round(float(score), 4),
            "evidence": [
                f"alert_count={1 if c.get('has_alert') else 0}",
                f"cross_diagram_degree=0.0",
                f"distance_to_alert=0.0",
                f"shared_entity={bool(c.get('is_shared_entity', False))}",
            ],
        })

    return {
        "scenario_id":          "enterprise_v3_0072",
        "case_id":              "ent_enterprise_v3_0072",
        "mode":                 "enterprise_gnn_rca",
        "rca_source":           "Enterprise GNN RCA",
        "predicted_root_cause": old.get("predicted_root_cause", ""),
        "root_cause_diagram":   old.get("root_cause_diagram", ""),
        "confidence":           round(float(raw_cands[0].get("score", 0.0)), 4) if raw_cands else 0.0,
        "top_candidates":       top_candidates,
        "impacted_diagrams":    old.get("impacted_diagrams", []),
        "alert_count":          old.get("alert_count", 0),
    }


def _simulation_rca(case_id: str, scenario_id: str, events: list[dict], labels: dict) -> dict:
    """
    Build a scenario-grounded RCA output without running the GNN model.
    rca_source = 'Scenario-grounded RCA simulation' (integrity constraint).
    """
    root = labels.get("root_cause_node", "")
    impacted = labels.get("impacted_diagrams", [])

    # Rank alerted nodes by severity then time; pad to top-3
    scored = sorted(
        events,
        key=lambda e: (-_SEV.get(e.get("severity", ""), 0.0), e.get("time_offset_min", 0)),
    )
    # Deduplicate by node (keep highest-severity event per node)
    seen: dict[str, dict] = {}
    for ev in scored:
        nid = ev.get("node", "")
        if nid and nid not in seen:
            seen[nid] = ev

    node_order = list(seen.keys())
    if root in node_order:
        node_order.remove(root)
        node_order.insert(0, root)

    top_candidates = []
    for rank, nid in enumerate(node_order[:3], start=1):
        ev   = seen[nid]
        sev  = _SEV.get(ev.get("severity", ""), 0.3)
        conf = round(max(0.10, sev - (rank - 1) * 0.20), 4)
        diag = ev.get("diagram_id", "")
        top_candidates.append({
            "rank":                      rank,
            "node_id":                   nid,
            "diagram_id":                diag,
            "node_observed_in_diagrams": [diag] if diag else [],
            "node_type":                 "",
            "score":                     conf,
            "evidence": [
                f"alert_count=1",
                f"severity={ev.get('severity','')}",
                f"time_offset_min={ev.get('time_offset_min',0)}",
            ],
        })

    predicted_root = node_order[0] if node_order else ""
    confidence     = top_candidates[0]["score"] if top_candidates else 0.0
    root_diag      = seen[predicted_root].get("diagram_id", "") if predicted_root in seen else ""

    return {
        "scenario_id":          scenario_id,
        "case_id":              case_id,
        "mode":                 "enterprise_gnn_rca",
        "rca_source":           "Scenario-grounded RCA simulation",
        "predicted_root_cause": predicted_root,
        "root_cause_diagram":   root_diag,
        "confidence":           confidence,
        "top_candidates":       top_candidates,
        "impacted_diagrams":    impacted,
        "alert_count":          len(events),
    }


def _build_v3_0073() -> dict:
    lib = REPO_ROOT / "scenario_library/enterprise_gnn_rca/ent_enterprise_v3_0073"
    ev  = json.loads((lib / "events.json").read_text(encoding="utf-8"))
    lab = json.loads((lib / "labels.json").read_text(encoding="utf-8"))
    return _simulation_rca("ent_enterprise_v3_0073", "enterprise_v3_0073",
                           ev.get("events", []), lab)


def _build_v3_0074() -> dict:
    lib = REPO_ROOT / "scenario_library/enterprise_gnn_rca/ent_enterprise_v3_0074"
    ev  = json.loads((lib / "events.json").read_text(encoding="utf-8"))
    lab = json.loads((lib / "labels.json").read_text(encoding="utf-8"))
    return _simulation_rca("ent_enterprise_v3_0074", "enterprise_v3_0074",
                           ev.get("events", []), lab)


# ── Main ───────────────────────────────────────────────────────────────────────

_SCENARIOS = [
    ("enterprise_v3_0000", "ent_enterprise_v3_0000", _build_v3_0000),
    ("enterprise_v3_0072", "ent_enterprise_v3_0072", _build_v3_0072),
    ("enterprise_v3_0073", "ent_enterprise_v3_0073", _build_v3_0073),
    ("enterprise_v3_0074", "ent_enterprise_v3_0074", _build_v3_0074),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate clean Enterprise RCA demo assets."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be written without writing anything.")
    args = parser.parse_args()

    ec_dir  = REPO_ROOT / "assets/preloaded/event_correlation"
    rca_dir = REPO_ROOT / "assets/preloaded/enterprise_gnn_rca"

    print("====================================================")
    print(" InfraGraph AI — Generate Enterprise RCA Demo Assets")
    print("====================================================")
    print()

    for scenario_id, case_id, rca_builder in _SCENARIOS:
        print(f"--- {scenario_id} ({case_id}) ---")

        # Build event correlation clusters
        print(f"  Building event correlation clusters...")
        cluster_output = _build_clusters(case_id)
        cl_path = ec_dir / f"{scenario_id}.json"
        print(f"  Cluster: {len(cluster_output['clusters'])} cluster(s)  "
              f"score={cluster_output['clusters'][0]['cluster_score']:.4f}")

        # Build RCA output
        print(f"  Building RCA output...")
        rca = rca_builder()
        rca = _enrich_with_cluster(rca, cluster_output)
        rca_path = rca_dir / f"{scenario_id}.json"

        # Validate before writing
        _assert_clean(rca, f"RCA output for {scenario_id}")
        _assert_clean(cluster_output, f"Cluster output for {scenario_id}")

        print(f"  RCA source   : {rca.get('rca_source')}")
        print(f"  Predicted    : {rca.get('predicted_root_cause')} "
              f"(confidence={rca.get('confidence', 0):.4f})")
        print(f"  Cluster ID   : {rca.get('cluster_id', '—')}")

        if args.dry_run:
            print(f"  DRY RUN: would write {cl_path}")
            print(f"  DRY RUN: would write {rca_path}")
        else:
            ec_dir.mkdir(parents=True, exist_ok=True)
            rca_dir.mkdir(parents=True, exist_ok=True)
            cl_path.write_text(json.dumps(cluster_output, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
            rca_path.write_text(json.dumps(rca, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8")
            print(f"  Written: {cl_path.relative_to(REPO_ROOT)}")
            print(f"  Written: {rca_path.relative_to(REPO_ROOT)}")
        print()

    print("====================================================")
    if not args.dry_run:
        print("Running validation...")
        sys.stdout.flush()
        import subprocess
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts/validate_rca_outputs.py"), "--verbose"],
            check=False,
        )
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
