#!/usr/bin/env python3
"""
build_event_correlation_clusters.py — Build event correlation clusters for one case.

Reads events and graph data from the scenario library, correlates alert events
into coherent clusters with causal evidence, and writes the output to:

  assets/preloaded/event_correlation/<stem>.json        (default)
  reports/event_correlation/manual_eval/<stem>.json     (with --with-eval)

Where <stem> is the scenario_id for enterprise cases or the case_id otherwise.

No root-cause labels, remediation steps, or evaluation fields appear in output.

Usage:
  python scripts/build_event_correlation_clusters.py \\
      --case-id ent_enterprise_v3_0000

  python scripts/build_event_correlation_clusters.py \\
      --case-id topo_enterprise_v3_0000_datacenter_topology

  python scripts/build_event_correlation_clusters.py \\
      --case-id ent_enterprise_v3_0000 --with-eval
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


def _assert_cluster_clean(output: dict) -> None:
    """Verify no forbidden keys appear anywhere in the cluster output."""
    def _collect_keys(obj: object, depth: int = 0) -> set[str]:
        found: set[str] = set()
        if isinstance(obj, dict):
            for k, v in obj.items():
                found.add(k)
                if depth < 20:
                    found.update(_collect_keys(v, depth + 1))
        elif isinstance(obj, list):
            for item in obj:
                found.update(_collect_keys(item, depth + 1))
        return found

    violations = _collect_keys(output) & FORBIDDEN_KEYS
    if violations:
        raise ValueError(f"Cluster output contains forbidden keys: {sorted(violations)}")


def _build_clusters(
    events: list[dict],
    graph: dict | None,
    mode: str,
    case_id: str,
    scenario_id: str,
) -> list[dict]:
    """Run correlation → evidence → schema assembly for one case."""
    raw_clusters = correlate_events(events=events, graph=graph, mode=mode)

    clusters: list[dict] = []
    for idx, raw in enumerate(raw_clusters, start=1):
        raw_events  = raw["raw_events"]
        roles       = raw["roles"]
        dims        = raw["dims"]
        reasons     = raw["reasons"]
        diag_scope  = raw["diagram_scope"]

        causal_ev = build_causal_evidence(
            raw_events=raw_events,
            roles=roles,
            dims=dims,
            diagram_scope=diag_scope,
            mode=mode,
        )

        cluster_event_dicts = [
            make_event_in_cluster(ev, role)
            for ev, role in zip(raw_events, roles)
        ]

        cluster = make_cluster(
            cluster_id=f"CLU-{case_id}-{idx:03d}",
            case_id=case_id,
            scenario_id=scenario_id,
            mode=mode,
            diagram_scope=diag_scope,
            cluster_score=raw["cluster_score"],
            correlation_dimensions=dims,
            correlation_reasons=reasons,
            cluster_events=cluster_event_dicts,
            causal_evidence=causal_ev,
        )
        clusters.append(cluster)

    return clusters


def _print_cluster_summary(clusters: list[dict]) -> None:
    for cl in clusters:
        tw  = cl.get("time_window", {})
        print(
            f"  {cl['cluster_id']}: {cl.get('event_count', 0)} event(s)  "
            f"score={cl.get('cluster_score', 0):.4f}  "
            f"t={tw.get('start_offset_min', 0)}..{tw.get('end_offset_min', 0)} min  "
            f"diagrams={cl.get('diagram_scope', [])}"
        )
        dims = cl.get("correlation_dimensions", {})
        print(
            f"    dims: temporal={dims.get('temporal', 0):.2f}  "
            f"topology={dims.get('topology', 0):.2f}  "
            f"seq={dims.get('alert_type_seq', 0):.2f}  "
            f"peer={dims.get('source_peer', 0):.2f}  "
            f"cross={dims.get('cross_diagram', 0):.2f}"
        )
        for ev in cl.get("events", []):
            print(
                f"    [{ev.get('event_id','?')}]  "
                f"{ev.get('node','?')} ({ev.get('alert_type','?')})  "
                f"t={ev.get('time_offset_min','?')}  "
                f"role={ev.get('correlation_role','?')}"
            )
        print(f"    causal_evidence items: {len(cl.get('causal_evidence', []))}")
        for ce in cl.get("causal_evidence", []):
            print(
                f"      [{ce.get('evidence_id','?')}] {ce.get('stage','?')} "
                f"(conf={ce.get('confidence', 0):.2f}): "
                f"{ce.get('summary','')[:90]}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build event correlation clusters for one scenario case.",
    )
    parser.add_argument(
        "--case-id", required=True,
        help="Case ID (e.g. ent_enterprise_v3_0000 or topo_enterprise_v3_0000_datacenter_topology)",
    )
    parser.add_argument(
        "--mode", default=None,
        choices=["topology_rca", "enterprise_gnn_rca"],
        help="Processing mode.  Inferred from case-id prefix if omitted.",
    )
    parser.add_argument("--scenario-library", default="scenario_library")
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory override.  "
             "Default: assets/preloaded/event_correlation "
             "(or reports/event_correlation/manual_eval with --with-eval).",
    )
    parser.add_argument(
        "--with-eval", action="store_true",
        help="Write output to reports/event_correlation/manual_eval instead of "
             "assets/preloaded/event_correlation.",
    )
    args = parser.parse_args()

    repo_root = REPO_ROOT
    lib_root  = (repo_root / args.scenario_library).resolve()

    # Resolve output directory
    if args.out_dir:
        out_dir = (repo_root / args.out_dir).resolve()
    elif args.with_eval:
        out_dir = repo_root / "reports" / "event_correlation" / "manual_eval"
        print("[INFO] --with-eval set; writing to reports/event_correlation/manual_eval")
    else:
        out_dir = repo_root / "assets" / "preloaded" / "event_correlation"

    # Load case
    events, graph, mode, scenario_id, diagram_id = load_case_for_correlation(
        lib_root=lib_root,
        repo_root=repo_root,
        case_id=args.case_id,
        mode=args.mode,
    )

    if not events:
        print(f"[ERROR] No events found for case: {args.case_id!r}")
        print(f"        Looked in: {lib_root}")
        sys.exit(1)

    print(f"Case ID          : {args.case_id}")
    print(f"Mode             : {mode}")
    print(f"Scenario         : {scenario_id or '—'}")
    print(f"Events loaded    : {len(events)}")
    print(f"Graph nodes      : {len(graph.get('nodes', [])) if graph else 0}")

    # Build clusters
    clusters = _build_clusters(
        events=events,
        graph=graph,
        mode=mode,
        case_id=args.case_id,
        scenario_id=scenario_id,
    )

    print(f"Clusters formed  : {len(clusters)}")
    _print_cluster_summary(clusters)

    # Assemble top-level output
    output = make_cluster_output(
        case_id=args.case_id,
        scenario_id=scenario_id,
        mode=mode,
        clusters=clusters,
    )

    # Integrity guard
    _assert_cluster_clean(output)

    # Write
    out_path = write_cluster_output(
        output=output,
        case_id=args.case_id,
        scenario_id=scenario_id,
        mode=mode,
        out_dir=out_dir,
    )

    print(f"\nCluster output   : {out_path}")


if __name__ == "__main__":
    main()
