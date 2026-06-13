#!/usr/bin/env python3
"""
generate_remediation_demo_assets.py
Generate remediation outputs from clean Enterprise GNN RCA outputs.

Reads  : assets/preloaded/enterprise_gnn_rca/<scenario_id>.json
Writes : assets/preloaded/remediation/<scenario_id>.json

RCA files are never modified.
Remediation outputs live separately in assets/preloaded/remediation/.

Usage:
  python scripts/generate_remediation_demo_assets.py
  python scripts/generate_remediation_demo_assets.py --scenarios enterprise_v3_0000 enterprise_v3_0072
  python scripts/generate_remediation_demo_assets.py --prefer-qwen
  python scripts/generate_remediation_demo_assets.py --template-only
  python scripts/generate_remediation_demo_assets.py --out-dir assets/preloaded/remediation
  python scripts/generate_remediation_demo_assets.py --strict-qwen
  python scripts/generate_remediation_demo_assets.py --include-raw
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ai_remediation.context_builder import build_enterprise_remediation_context  # noqa: E402
from ai_remediation.qwen_client import (  # noqa: E402
    check_vllm_available,
    generate_remediation_with_qwen,
    generate_resolution_plan,
    get_qwen_runtime_config,
)
from ai_remediation.template_mode import generate_template_remediation  # noqa: E402

_DEFAULT_SCENARIOS = [
    "enterprise_v3_0000",
    "enterprise_v3_0072",
    "enterprise_v3_0073",
    "enterprise_v3_0074",
]


def _build_envelope(
    scenario_id: str,
    context: dict,
    plan_result: dict,
    *,
    include_raw: bool = False,
) -> dict:
    """Wrap the plan result in the standard output envelope."""
    envelope: dict = {
        "scenario_id":        scenario_id,
        "case_id":            f"ent_{scenario_id}",
        "incident_id":        context.get("incident_id", f"INC-{scenario_id}"),
        "scope":              "enterprise",
        "rca_source":         context.get("rca_source", ""),
        "cluster_id":         context.get("cluster_id", ""),
        "cluster_score":      context.get("cluster_score"),
        "remediation_source": plan_result.get("source", "template"),
        "model":              plan_result.get("model", "—"),
        "ok":                 bool(plan_result.get("ok", False)),
        "error":              plan_result.get("error", ""),
        "input_context_summary": {
            "root_cause":              context.get("root_cause", ""),
            "root_cause_diagram":      context.get("root_cause_diagram", ""),
            "candidate_count":         len(context.get("candidate_ranking", [])),
            "causal_evidence_count":   len(context.get("causal_evidence", [])),
            "correlation_reason_count": len(context.get("correlation_reasons", [])),
        },
        "remediation": plan_result.get("response", {}),
    }
    # Preserve Qwen error if template fallback was used
    if plan_result.get("source") == "template" and plan_result.get("error"):
        envelope["qwen_error"] = plan_result["error"]
    if include_raw:
        envelope["raw_model_output"] = plan_result.get("raw", "")
    return envelope


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate remediation outputs from clean RCA outputs."
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=_DEFAULT_SCENARIOS, metavar="SCENARIO_ID",
        help="Scenario IDs to process (default: all four).",
    )
    parser.add_argument(
        "--prefer-qwen", action="store_true", default=False,
        help="Prefer Qwen/vLLM (falls back to template if unavailable). "
             "This is the default non-template mode.",
    )
    parser.add_argument(
        "--template-only", action="store_true",
        help="Use deterministic template mode only — do not call vLLM.",
    )
    parser.add_argument(
        "--strict-qwen", action="store_true",
        help="Fail if vLLM/Qwen is unavailable. Do not fall back to template.",
    )
    parser.add_argument(
        "--out-dir", default="assets/preloaded/remediation",
        help="Output directory (default: assets/preloaded/remediation).",
    )
    parser.add_argument(
        "--include-raw", action="store_true",
        help="Include raw_model_output field in the envelope.",
    )
    args = parser.parse_args()

    if args.strict_qwen and args.template_only:
        parser.error("--strict-qwen and --template-only are mutually exclusive.")

    prefer_qwen = args.prefer_qwen or (not args.template_only)

    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    qwen_config = get_qwen_runtime_config()

    print("====================================================")
    print(" InfraGraph AI — Generate Remediation Demo Assets")
    print("====================================================")
    print(f"Scenarios    : {', '.join(args.scenarios)}")
    print(f"Mode         : {'template-only' if args.template_only else ('strict-qwen' if args.strict_qwen else 'prefer-qwen (fallback=template)')}")
    print(f"Output dir   : {out_dir.relative_to(REPO_ROOT)}")
    print()

    failures: list[str] = []

    for scenario_id in args.scenarios:
        rca_path = REPO_ROOT / "assets/preloaded/enterprise_gnn_rca" / f"{scenario_id}.json"
        if not rca_path.exists():
            print(f"[ERROR] RCA output not found: {rca_path.relative_to(REPO_ROOT)}")
            failures.append(scenario_id)
            continue

        print(f"--- {scenario_id} ---")

        try:
            context = build_enterprise_remediation_context(
                repo_root=REPO_ROOT,
                scenario_id=scenario_id,
                rca_path=rca_path,
            )
        except Exception as exc:
            print(f"  [ERROR] Failed to build remediation context: {exc}")
            failures.append(scenario_id)
            continue

        print(f"  RCA source      : {context.get('rca_source', '—')}")
        print(f"  Root cause      : {context.get('root_cause', '—')} "
              f"({context.get('root_cause_diagram', '—')})")
        print(f"  Cluster         : {context.get('cluster_id', '—')} "
              f"(score={context.get('cluster_score')})")
        print(f"  Alert events    : {len(context.get('alert_timeline', []))}")
        print(f"  Causal evidence : {len(context.get('causal_evidence', []))} item(s)")

        try:
            if args.template_only:
                template_out = generate_template_remediation(context)
                plan_result = {
                    "source":   "template",
                    "model":    "—",
                    "ok":       bool(template_out),
                    "response": template_out,
                    "error":    "",
                    "raw":      "",
                }
            elif args.strict_qwen:
                plan_result = generate_remediation_with_qwen(context)
                if not plan_result.get("ok"):
                    print(f"  [ERROR] Qwen failed: {plan_result.get('error', '')}")
                    print("          --strict-qwen is set; no template fallback.")
                    failures.append(scenario_id)
                    continue
            else:
                plan_result = generate_resolution_plan(
                    context,
                    scope="enterprise",
                    prefer_qwen=prefer_qwen,
                    base_url=qwen_config["base_url"],
                    model=qwen_config["model"],
                    timeout=qwen_config["timeout"],
                )
        except Exception as exc:
            print(f"  [ERROR] Remediation generation failed: {exc}")
            failures.append(scenario_id)
            continue

        print(f"  Remediation src : {plan_result.get('source', '—')}")
        print(f"  OK              : {plan_result.get('ok', False)}")
        if plan_result.get("error"):
            print(f"  Error           : {plan_result['error']}")

        envelope = _build_envelope(
            scenario_id, context, plan_result, include_raw=args.include_raw
        )

        out_path = out_dir / f"{scenario_id}.json"
        out_path.write_text(
            json.dumps(envelope, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"  Written         : {out_path.relative_to(REPO_ROOT)}")
        print()

    print("====================================================")
    if failures:
        print(f"[FAIL] {len(failures)} scenario(s) failed: {', '.join(failures)}")
        sys.exit(1)
    else:
        print(f"[OK] {len(args.scenarios)} remediation output(s) written to "
              f"{out_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
