#!/usr/bin/env python3
"""
generate_enterprise_rca_demo_assets.py
Generate clean Enterprise GNN RCA + event correlation preloaded assets.

Default behavior:
  For each scenario, calls:
    1. scripts/build_event_correlation_clusters.py --case-id <case_id>
    2. scripts/predict_enterprise_gnn_rca.py --scenario-id <scenario_id>
         --cluster-file assets/preloaded/event_correlation/<scenario_id>.json

  All four default scenarios must produce rca_source="Enterprise GNN RCA".

Requires:
  model_artifacts/enterprise_gnn_rca/enterprise_gnn_rca.pt
  model_artifacts/enterprise_gnn_rca/enterprise_gnn_config.json

Fails immediately if these model artifacts are missing.

Usage:
  python scripts/generate_enterprise_rca_demo_assets.py
  python scripts/generate_enterprise_rca_demo_assets.py --scenarios enterprise_v3_0000 enterprise_v3_0072
  python scripts/generate_enterprise_rca_demo_assets.py --dry-run
  python scripts/generate_enterprise_rca_demo_assets.py --allow-simulation-fallback
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_MODEL_PT  = "model_artifacts/enterprise_gnn_rca/enterprise_gnn_rca.pt"
_MODEL_CFG = "model_artifacts/enterprise_gnn_rca/enterprise_gnn_config.json"

_DEFAULT_SCENARIOS = [
    "enterprise_v3_0000",
    "enterprise_v3_0072",
    "enterprise_v3_0073",
    "enterprise_v3_0074",
]

_SEV = {"critical": 1.0, "high": 0.8, "warning": 0.6, "medium": 0.5, "low": 0.3, "info": 0.1}


# ── Model artifact check ──────────────────────────────────────────────────────

def _check_model_artifacts() -> None:
    missing = []
    for rel in (_MODEL_PT, _MODEL_CFG):
        if not (REPO_ROOT / rel).exists():
            missing.append(rel)
    if missing:
        print("[ERROR] Enterprise GNN model artifact not found. "
              "Train or restore the model before generating demo assets.")
        for rel in missing:
            print(f"        Expected: {REPO_ROOT / rel}")
        print(
            "        Run: python scripts/train_enterprise_gnn_rca.py "
            "--epochs 80 --lr 0.001 --hidden-dim 64 --dropout 0.2 --top-k 3 --seed 42"
        )
        sys.exit(1)


# ── Simulation fallback (--allow-simulation-fallback only) ────────────────────

def _run_simulation_fallback(
    scenario_id: str,
    case_id: str,
    cluster_file: str,
) -> None:
    """
    Build a scenario-grounded simulation output from events.json only.
    rca_source = "Scenario-grounded RCA simulation".
    Never reads labels.json.
    """
    print(f"  [SIM] Building scenario-grounded simulation for {scenario_id} ...")

    lib = REPO_ROOT / "scenario_library/enterprise_gnn_rca" / case_id
    ev_path = lib / "events.json"
    if not ev_path.exists():
        print(f"  [SIM][ERROR] events.json not found: {ev_path}")
        sys.exit(1)

    events = json.loads(ev_path.read_text(encoding="utf-8")).get("events", [])

    # Rank nodes by (severity_score desc, time_offset_min asc) — no labels used
    scored = sorted(
        events,
        key=lambda e: (-_SEV.get(str(e.get("severity", "")).lower(), 0.3),
                       e.get("time_offset_min", 0)),
    )
    seen: dict[str, dict] = {}
    for ev in scored:
        nid = ev.get("node", "")
        if nid and nid not in seen:
            seen[nid] = ev

    node_order = list(seen.keys())

    top_candidates = []
    for rank, nid in enumerate(node_order[:3], start=1):
        ev   = seen[nid]
        sev  = _SEV.get(str(ev.get("severity", "")).lower(), 0.3)
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
                "alert_count=1",
                f"severity={ev.get('severity', '')}",
                f"time_offset_min={ev.get('time_offset_min', 0)}",
            ],
        })

    predicted_root = node_order[0] if node_order else ""
    confidence     = top_candidates[0]["score"] if top_candidates else 0.0
    root_diag      = seen[predicted_root].get("diagram_id", "") if predicted_root in seen else ""
    impacted_diags = sorted({ev.get("diagram_id", "") for ev in events if ev.get("diagram_id")})

    rca = {
        "scenario_id":          scenario_id,
        "case_id":              case_id,
        "mode":                 "enterprise_gnn_rca",
        "rca_source":           "Scenario-grounded RCA simulation",
        "predicted_root_cause": predicted_root,
        "root_cause_diagram":   root_diag,
        "confidence":           confidence,
        "top_candidates":       top_candidates,
        "impacted_diagrams":    impacted_diags,
        "alert_count":          len(events),
    }

    # Enrich with cluster if available
    cf_path = REPO_ROOT / cluster_file
    if cf_path.exists():
        try:
            cluster_data = json.loads(cf_path.read_text(encoding="utf-8"))
            clusters = cluster_data.get("clusters", [])
            if clusters:
                p = clusters[0]
                rca["cluster_id"]          = p.get("cluster_id", "")
                rca["cluster_score"]       = p.get("cluster_score", 0.0)
                rca["correlation_reasons"] = p.get("correlation_reasons", [])
                rca["causal_evidence"]     = p.get("causal_evidence", [])
        except Exception as exc:
            print(f"  [SIM][WARN] Could not load cluster file: {exc}")

    out_dir = REPO_ROOT / "assets/preloaded/enterprise_gnn_rca"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scenario_id}.json"
    out_path.write_text(json.dumps(rca, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  [SIM] Written: {out_path.relative_to(REPO_ROOT)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate clean Enterprise GNN RCA + event correlation demo assets."
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=_DEFAULT_SCENARIOS,
        metavar="SCENARIO_ID",
        help="Scenario IDs to process (default: all four).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands that would be run without executing them.",
    )
    parser.add_argument(
        "--allow-simulation-fallback", action="store_true",
        help=(
            "Allow scenario-grounded simulation when GNN prediction fails. "
            "Output is clearly labelled 'Scenario-grounded RCA simulation'. "
            "NOT the default — GNN is always required unless this flag is set."
        ),
    )
    args = parser.parse_args()

    # Always check model artifacts (dry-run shows what would be needed)
    if not args.dry_run:
        _check_model_artifacts()
    else:
        missing = [rel for rel in (_MODEL_PT, _MODEL_CFG) if not (REPO_ROOT / rel).exists()]
        if missing:
            print("[DRY-RUN WARN] Model artifacts missing — script would fail in live mode:")
            for rel in missing:
                print(f"               {REPO_ROOT / rel}")
        else:
            print("[DRY-RUN INFO] Model artifacts present.")

    print("====================================================")
    print(" InfraGraph AI — Generate Enterprise RCA Demo Assets")
    print("====================================================")
    print(f"Scenarios       : {', '.join(args.scenarios)}")
    print(f"Dry run         : {args.dry_run}")
    print(f"Sim fallback    : {args.allow_simulation_fallback}")
    print()

    any_sim = False

    for scenario_id in args.scenarios:
        case_id      = f"ent_{scenario_id}"
        cluster_file = f"assets/preloaded/event_correlation/{scenario_id}.json"

        print(f"--- {scenario_id} ---")

        # Step 1: event correlation clusters
        cmd_cluster = [
            sys.executable,
            str(REPO_ROOT / "scripts/build_event_correlation_clusters.py"),
            "--case-id", case_id,
        ]
        print(f"  [1/2] {' '.join(cmd_cluster[1:])}")

        if not args.dry_run:
            subprocess.run(cmd_cluster, check=True, cwd=str(REPO_ROOT))

        # Step 2: GNN prediction with cluster enrichment
        cmd_predict = [
            sys.executable,
            str(REPO_ROOT / "scripts/predict_enterprise_gnn_rca.py"),
            "--scenario-id", scenario_id,
            "--cluster-file", cluster_file,
        ]
        print(f"  [2/2] {' '.join(cmd_predict[1:])}")

        if not args.dry_run:
            try:
                subprocess.run(cmd_predict, check=True, cwd=str(REPO_ROOT))
            except subprocess.CalledProcessError as exc:
                if args.allow_simulation_fallback:
                    print(f"  [WARN] GNN prediction failed for {scenario_id}: {exc}")
                    print("  [WARN] --allow-simulation-fallback active; "
                          "output will be labelled 'Scenario-grounded RCA simulation'.")
                    _run_simulation_fallback(scenario_id, case_id, cluster_file)
                    any_sim = True
                else:
                    print(f"[ERROR] GNN prediction failed for {scenario_id}.")
                    print(
                        "        Pass --allow-simulation-fallback to allow simulation outputs, "
                        "or restore model artifacts and retry."
                    )
                    sys.exit(1)
        print()

    if any_sim:
        print("[WARN] Some outputs used 'Scenario-grounded RCA simulation' (not real GNN).")
        print()

    print("====================================================")

    if args.dry_run:
        print("DRY RUN complete — no files written, no commands executed.")
        return

    print("Running validation...")
    sys.stdout.flush()
    result = subprocess.run(
        [sys.executable,
         str(REPO_ROOT / "scripts/validate_rca_outputs.py"),
         "--verbose"],
        check=False, cwd=str(REPO_ROOT),
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
