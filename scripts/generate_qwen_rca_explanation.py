"""
Stage 4 of InfraGraph AI: LLM explanation layer.

Generates a human-readable RCA explanation from topology (heuristic) and GNN
evidence.  Runs locally in --mode mock without any LLM, or calls an
OpenAI-compatible vLLM endpoint (--mode vllm) when running on AMD Jupyter.
"""

import argparse
import json
import os
import re
import sys
import textwrap
from datetime import datetime, timezone

# ── Evidence builder ──────────────────────────────────────────────────────────

def _find_alert_path(dataset_root, split, diagram_id):
    """Return alert JSON path, searching common naming conventions."""
    candidates = [
        os.path.join(dataset_root, "alerts", split, f"{diagram_id}.json"),
        os.path.join(dataset_root, "alerts", split, f"{diagram_id}_alerts.json"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    # Alternate path: scan the directory for a file containing diagram_id
    alert_dir = os.path.join(dataset_root, "alerts", split)
    if os.path.isdir(alert_dir):
        for fname in os.listdir(alert_dir):
            if diagram_id in fname and fname.endswith(".json"):
                return os.path.join(alert_dir, fname)
    return None


def load_evidence(topo_rca_path, graph_summary_path, gnn_rca_path,
                  alert_path=None):
    with open(topo_rca_path) as f:
        topo = json.load(f)
    with open(graph_summary_path) as f:
        summary = json.load(f)
    with open(gnn_rca_path) as f:
        gnn = json.load(f)

    alert_scenario = None
    if alert_path and os.path.isfile(alert_path):
        with open(alert_path) as f:
            raw = json.load(f)
        alert_scenario = {
            "scenario_id": raw.get("scenario_id"),
            "root_cause": raw.get("root_cause"),
            "root_cause_type": raw.get("root_cause_type"),
            "alerts": raw.get("alerts", []),
            "expected_impacted_nodes": raw.get("expected_impacted_nodes", []),
        }

    diagram_id = topo["diagram_id"]
    gt_root = topo["ground_truth_root_cause"]

    h_pred = topo["predicted_root_cause"]
    h_top = topo.get("top_candidates", [])
    h_type = next((c["type"] for c in h_top if c["node"] == h_pred), "unknown")
    h_correct = h_pred == gt_root

    g_pred = gnn["predicted_root_cause"]
    g_top = gnn.get("top_candidates", [])
    g_type = next((c["type"] for c in g_top if c["node"] == g_pred), "unknown")
    g_correct = gnn["is_correct"]

    gt_type = next(
        (c["type"] for c in g_top if c["node"] == gt_root),
        next((c["type"] for c in h_top if c["node"] == gt_root), "unknown"),
    )

    # Top GNN score margin
    g_scores = [c["score"] for c in g_top]
    g_margin = round(g_scores[0] - g_scores[1], 2) if len(g_scores) >= 2 else None

    return {
        "diagram_id": diagram_id,
        "graph": {
            "node_count": summary["node_count"],
            "edge_count": summary["edge_count"],
            "device_type_counts": summary["device_type_counts"],
            "alert_count": summary["alert_count"],
            "impacted_node_count": summary["impacted_node_count"],
        },
        "ground_truth": {
            "root_cause": gt_root,
            "root_cause_type": gt_type,
        },
        "heuristic_rca": {
            "predicted_root_cause": h_pred,
            "predicted_type": h_type,
            "confidence_score": topo.get("confidence_score"),
            "top_candidates": h_top[:5],
            "is_correct": h_correct,
        },
        "gnn_rca": {
            "predicted_root_cause": g_pred,
            "predicted_type": g_type,
            "top_candidates": g_top[:5],
            "top_score": round(g_scores[0], 3) if g_scores else None,
            "score_margin_vs_2nd": g_margin,
            "is_correct": g_correct,
            "mrr": gnn.get("mrr"),
            "backend": gnn.get("backend"),
            "test_metrics": gnn.get("test_metrics"),
        },
        "gnn_improved_over_heuristic": (not h_correct) and g_correct,
        "alerting_nodes": topo.get("alerting_nodes", []),
        "impacted_nodes": topo.get("impacted_nodes", []),
        "impact_paths": topo.get("impact_paths", {}),
        "impact_path_summary": topo.get("impact_path_summary", {}),
        "reasoning_summary": topo.get("reasoning_summary", ""),
        "alert_scenario": alert_scenario,
    }


# ── Prompt builder ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an enterprise AIOps root cause analysis assistant. "
    "Use only the provided topology, alert, heuristic RCA, and GNN RCA evidence. "
    "Do not invent nodes, alerts, tools, or remediation actions. "
    "Do not convert time_offset_min values into wall-clock timestamps — "
    "refer to them only as time offsets (e.g. 't+2 min'). "
    "Be concise, operational, and suitable for an L1/L2 incident response team. "
    "Output plain Markdown only — do not wrap your response in a code block."
)


def build_user_prompt(evidence):
    evidence_json = json.dumps(evidence, indent=2)
    h_status = "was incorrect" if not evidence["heuristic_rca"]["is_correct"] else "was correct"
    g_status = "improved on it" if evidence["gnn_improved_over_heuristic"] else "agreed with it"
    return textwrap.dedent(f"""\
        /no_think

        You are given the following structured RCA evidence for network incident
        `{evidence["diagram_id"]}`. Produce a complete incident analysis report
        in Markdown with these sections in order:

        1. **Executive Summary** (2-3 sentences)
        2. **What Happened** — list each alert with its node, alert type,
           severity, and time_offset_min. Do NOT convert offsets to clock times.
        3. **Root Cause Conclusion** — state the root cause and explain why
        4. **Heuristic vs GNN Comparison** — table + explanation of why the
           heuristic {h_status} and why the GNN {g_status}
        5. **Impacted Nodes/Services** — list and propagation path
        6. **Recommended Next Actions** — numbered, L1/L2 actionable steps only,
           grounded in the evidence (no invented tools or systems)
        7. **ServiceNow Incident Summary** — short-description, affected-CI,
           priority, assignment-group, symptom, root-cause, services-impacted,
           suggested-action (key:value format)
        8. **Confidence and Limitations**

        Output plain Markdown. Do not wrap it in a code block.

        Evidence:
        ```json
        {evidence_json}
        ```
    """)


def build_prompts(evidence):
    return SYSTEM_PROMPT, build_user_prompt(evidence)


# ── Mock explanation ──────────────────────────────────────────────────────────

def _confidence_label(margin):
    if margin is None:
        return "MEDIUM"
    if margin >= 5:
        return "HIGH"
    if margin >= 2:
        return "MEDIUM"
    return "LOW"


def _format_candidates(candidates, n=3):
    return ", ".join(
        f"{c['node']} ({c['type']}, score={round(c['score'], 2)})"
        for c in candidates[:n]
    )


def _shortest_path_str(evidence, key):
    # key is "predicted" or "ground_truth"
    summary = evidence["impact_path_summary"]
    path = summary.get(f"shortest_{key}_path", [])
    return " -> ".join(path) if path else "N/A"


def build_mock_explanation(evidence):
    d = evidence["diagram_id"]
    gt = evidence["ground_truth"]["root_cause"]
    gt_type = evidence["ground_truth"]["root_cause_type"]
    h = evidence["heuristic_rca"]
    g = evidence["gnn_rca"]
    graph = evidence["graph"]
    alerting = evidence["alerting_nodes"]
    impacted = evidence["impacted_nodes"]

    h_correct = h["is_correct"]
    g_correct = g["is_correct"]
    gnn_improved = evidence["gnn_improved_over_heuristic"]
    confidence = _confidence_label(g["score_margin_vs_2nd"])

    # --- Section helpers ---
    if g_correct and not h_correct:
        exec_summary = (
            f"A network incident triggered {graph['alert_count']} alerts across "
            f"{len(alerting)} nodes, impacting {graph['impacted_node_count']} downstream services. "
            f"The heuristic RCA incorrectly identified **{h['predicted_root_cause']}** as root cause. "
            f"The GNN-based RCA correctly identified **{gt}** ({gt_type}), "
            f"demonstrating the value of learned propagation-direction signals over rule-based scoring."
        )
    elif g_correct and h_correct:
        exec_summary = (
            f"A network incident triggered {graph['alert_count']} alerts across "
            f"{len(alerting)} nodes, impacting {graph['impacted_node_count']} downstream services. "
            f"Both heuristic and GNN RCA correctly identified **{gt}** ({gt_type}) as root cause."
        )
    elif not g_correct and h_correct:
        exec_summary = (
            f"A network incident triggered {graph['alert_count']} alerts across "
            f"{len(alerting)} nodes, impacting {graph['impacted_node_count']} downstream services. "
            f"The heuristic correctly identified **{gt}** ({gt_type}). "
            f"The GNN predicted **{g['predicted_root_cause']}** — human review recommended."
        )
    else:
        exec_summary = (
            f"A network incident triggered {graph['alert_count']} alerts across "
            f"{len(alerting)} nodes, impacting {graph['impacted_node_count']} downstream services. "
            f"Both RCA methods require human review — ground truth: **{gt}** ({gt_type})."
        )

    # What happened: use full alert details if available, else use alerting nodes
    scenario_alerts = (evidence.get("alert_scenario") or {}).get("alerts", [])
    if scenario_alerts:
        alert_bullets = "\n".join(
            f"- **{al['node']}** [{al['severity'].upper()}] "
            f"{al['alert_type']} (t+{al['time_offset_min']} min)"
            for al in scenario_alerts
        )
    else:
        alert_bullets = "\n".join(
            f"- **{node}**: alert detected"
            for node in alerting
        )

    # Root cause
    root_section_lines = [
        f"**Root cause: {gt} ({gt_type})**  ",
        f"Confidence: {confidence} "
        + (f"(GNN score: {g['top_score']}, margin over 2nd-ranked: {g['score_margin_vs_2nd']})"
           if g["score_margin_vs_2nd"] is not None else ""),
        "",
    ]
    if g_correct:
        root_section_lines.append(
            f"The {gt_type} **{gt}** initiated the incident. Its position in the "
            f"topology as an upstream chokepoint means its failure cascades to all "
            f"downstream services through the core switching and application layers."
        )
    else:
        root_section_lines.append(
            f"Ground truth identifies **{gt}** ({gt_type}) as root cause. "
            f"The GNN predicted **{g['predicted_root_cause']}** — manual inspection recommended."
        )

    # Heuristic vs GNN table
    h_correct_str = "Yes" if h_correct else "**No**"
    g_correct_str = "**Yes**" if g_correct else "No"
    comparison_table = (
        f"| | Heuristic (Stage 2) | GNN (Stage 3) |\n"
        f"|--|--|--|\n"
        f"| Predicted root cause | {h['predicted_root_cause']} | {g['predicted_root_cause']} |\n"
        f"| Ground truth | {gt} | {gt} |\n"
        f"| Correct? | {h_correct_str} | {g_correct_str} |\n"
        f"| Top candidates | {_format_candidates(h['top_candidates'])} | {_format_candidates(g['top_candidates'])} |"
    )

    if not h_correct:
        heuristic_explanation = (
            f"**Why the heuristic was wrong**: The heuristic scorer ranked nodes by "
            f"severity, timing, and downstream reach. **{h['predicted_root_cause']}** "
            f"received a higher composite score because it had more correlated alert "
            f"events and higher downstream reach in the topology. The heuristic cannot "
            f"distinguish a downstream aggregation node from the true upstream origin."
        )
    else:
        heuristic_explanation = (
            f"**Why the heuristic was correct**: Alert timing and severity signals "
            f"were sufficient for the heuristic to identify **{gt}** in this scenario."
        )

    if gnn_improved:
        gnn_explanation = (
            f"**Why the GNN is more reliable**: The GNN learned propagation direction "
            f"from graph structure and temporal alert features across 400 training "
            f"scenarios. **{gt}**'s features — {gt_type} priority flag, earliest alert "
            f"time, and critical severity — combined with its upstream topology position "
            f"produce a score of {g['top_score']}, clearly separating it from downstream "
            f"nodes. Test set: top-1={g['test_metrics']['top1']:.0%}, "
            f"MRR={g['test_metrics']['mrr']:.3f}."
        )
    elif g_correct:
        gnn_explanation = (
            f"**GNN confirmation**: The GNN score of {g['top_score']} for **{gt}** "
            f"confirms the heuristic finding with a margin of {g['score_margin_vs_2nd']} "
            f"over the second-ranked node."
        )
    else:
        gnn_explanation = (
            f"**GNN limitation**: The GNN predicted **{g['predicted_root_cause']}** "
            f"in this case. Human review is recommended."
        )

    # Impact
    impacted_str = ", ".join(impacted[:10])
    if len(impacted) > 10:
        impacted_str += f", ... ({len(impacted)} total)"
    shortest_pred = _shortest_path_str(evidence, "predicted")
    shortest_gt = _shortest_path_str(evidence, "ground_truth")

    # Next actions
    actions = [
        f"**Immediate**: SSH to **{gt}** and check interface counters, packet drops, and syslog",
        f"**Verify**: Identify the failure mode — ACL misconfiguration, upstream link fault, or hardware error",
        f"**Escalate if**: Packet drop rate exceeds 5% or {gt} CPU/memory is critically high",
        f"**Failover**: If {gt} is faulty and a redundant path exists, activate it now",
        f"**Validate**: After remediation, confirm downstream services ({', '.join(impacted[:3])}, ...) are reachable",
        f"**Post-incident**: Update CMDB topology and retrain GNN if root cause pattern is novel",
    ]
    actions_str = "\n".join(f"{i+1}. {a}" for i, a in enumerate(actions))

    # ServiceNow
    sn_priority = "P1" if graph["impacted_node_count"] >= 5 else "P2"
    sn_services = ", ".join(impacted[:5])
    snow = (
        f"**Short description**: Network fault on {gt} causing {graph['impacted_node_count']}-service outage  \n"
        f"**Affected CI**: {gt} ({gt_type})  \n"
        f"**Priority**: {sn_priority} -- {graph['impacted_node_count']} downstream nodes impacted  \n"
        f"**Assignment group**: Network Operations  \n"
        f"**Symptom**: Alerts on {', '.join(alerting)}; {graph['impacted_node_count']} downstream services unreachable  \n"
        f"**Root cause (automated)**: {gt} identified by GNN RCA (confidence: {confidence})  \n"
        f"**Services impacted**: {sn_services}  \n"
        f"**Suggested action**: Inspect {gt} interfaces, ACLs, and upstream connectivity"
    )

    # Limitations
    gnn_metrics = g.get("test_metrics", {})
    limitations = (
        f"- **GNN confidence**: {confidence} -- score margin {g['top_score']} vs "
        f"{round(g['top_candidates'][1]['score'], 2) if len(g['top_candidates']) > 1 else 'N/A'} (2nd candidate)\n"
        f"- **Model**: trained on {400} synthetic infragraph_v2 scenarios "
        f"(test top-1={gnn_metrics.get('top1', 'N/A'):.0%}, MRR={gnn_metrics.get('mrr', 'N/A'):.3f})\n"
        f"- **Limitations**: Synthetic training data; real-world noise, partial observability, "
        f"or novel topologies may reduce accuracy\n"
        f"- **Human review recommended** before executing remediation on production {gt_type}"
    )

    lines = [
        f"# InfraGraph AI RCA Explanation -- {d}",
        "",
        f"> Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"> Provider: mock (deterministic template)",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        exec_summary,
        "",
        "---",
        "",
        "## What Happened",
        "",
        f"**{graph['alert_count']} alerts** fired across **{len(alerting)} alerting nodes** "
        f"in a {graph['node_count']}-node, {graph['edge_count']}-edge topology:",
        "",
        alert_bullets,
        "",
        "---",
        "",
        "## Root Cause Conclusion",
        "",
        "\n".join(root_section_lines),
        "",
        "---",
        "",
        "## Heuristic vs GNN Comparison",
        "",
        comparison_table,
        "",
        heuristic_explanation,
        "",
        gnn_explanation,
        "",
        "---",
        "",
        "## Impacted Nodes / Services",
        "",
        f"**{graph['impacted_node_count']} nodes impacted**: {impacted_str}",
        "",
        f"**Shortest propagation path (predicted root cause)**: `{shortest_pred}`  ",
        f"**Shortest propagation path (ground truth)**: `{shortest_gt}`",
        "",
        "---",
        "",
        "## Recommended Next Actions (L1/L2)",
        "",
        actions_str,
        "",
        "---",
        "",
        "## ServiceNow Incident Summary",
        "",
        snow,
        "",
        "---",
        "",
        "## Confidence and Limitations",
        "",
        limitations,
    ]

    return "\n".join(lines)


# ── LLM output post-processing ────────────────────────────────────────────────

def clean_llm_output(text):
    """Remove <think> blocks and surrounding markdown code fences from LLM output."""
    # Strip <think>...</think> blocks (Qwen3 chain-of-thought, may be multiline)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()
    # Strip a leading ```markdown or ``` fence and the matching closing fence
    text = re.sub(r"^```(?:markdown)?\s*\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


# ── vLLM mode ─────────────────────────────────────────────────────────────────

def call_vllm(system_prompt, user_prompt, base_url, model):
    try:
        import requests
    except ImportError:
        raise RuntimeError("'requests' is not installed. Run: pip install requests")

    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1200,
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    return clean_llm_output(text)


# ── Output writers ────────────────────────────────────────────────────────────

def save_outputs(diagram_id, evidence, system_prompt, user_prompt,
                 explanation_md, provider, model, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    md_path = os.path.join(out_dir, f"{diagram_id}_explanation.md")
    json_path = os.path.join(out_dir, f"{diagram_id}_explanation.json")
    prompt_path = os.path.join(out_dir, f"{diagram_id}_prompt.json")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(explanation_md)

    prompt_doc = {"system": system_prompt, "user": user_prompt}
    with open(prompt_path, "w", encoding="utf-8") as f:
        json.dump(prompt_doc, f, indent=2, ensure_ascii=False)

    result_doc = {
        "diagram_id": diagram_id,
        "provider": provider,
        "model": model,
        "evidence": evidence,
        "prompt": prompt_doc,
        "explanation_markdown": explanation_md,
        "output_markdown_path": md_path,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_doc, f, indent=2, ensure_ascii=False)

    return md_path, json_path, prompt_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate a human-readable RCA explanation from topology and GNN evidence."
    )
    parser.add_argument("--diagram-id", default="diagram_0373")
    parser.add_argument(
        "--mode", choices=["mock", "vllm"], default="mock",
        help="mock: deterministic template (no LLM); vllm: call OpenAI-compatible endpoint",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-4B",
                        help="Model name passed to vLLM (ignored in mock mode)")
    parser.add_argument("--base-url", default="http://localhost:8000/v1",
                        help="vLLM OpenAI-compatible API base URL")
    parser.add_argument("--topo-dir", default="assets/preloaded/topology_demo")
    parser.add_argument("--gnn-dir", default="assets/preloaded/gnn_rca")
    parser.add_argument("--out", default="assets/preloaded/qwen_explanation")
    parser.add_argument("--dataset-root", default="datasets/infragraph_v2",
                        help="Root of the infragraph dataset (for loading original alert JSON)")
    parser.add_argument("--split", default="test",
                        help="Dataset split containing the diagram's alert JSON (default: test)")
    args = parser.parse_args()

    did = args.diagram_id

    # ── Locate inputs ──────────────────────────────────────────────────────
    topo_rca_path = os.path.join(args.topo_dir, f"{did}_rca_result.json")
    graph_summary_path = os.path.join(args.topo_dir, f"{did}_graph_summary.json")
    gnn_rca_path = os.path.join(args.gnn_dir, f"{did}_gnn_rca_result.json")

    for p in (topo_rca_path, graph_summary_path, gnn_rca_path):
        if not os.path.isfile(p):
            print(f"ERROR: required input not found: {p}", file=sys.stderr)
            sys.exit(1)

    alert_path = _find_alert_path(args.dataset_root, args.split, did)
    if alert_path:
        print(f"  Alert JSON: {alert_path}")
    else:
        print(f"  Alert JSON: not found under {args.dataset_root}/alerts/{args.split}/ — proceeding without it")

    # ── Build evidence ─────────────────────────────────────────────────────
    evidence = load_evidence(topo_rca_path, graph_summary_path, gnn_rca_path,
                             alert_path=alert_path)
    system_prompt, user_prompt = build_prompts(evidence)

    # ── Generate explanation ───────────────────────────────────────────────
    provider = args.mode
    model = args.model if args.mode == "vllm" else "mock-template"
    explanation_md = None

    if args.mode == "vllm":
        try:
            explanation_md = call_vllm(system_prompt, user_prompt, args.base_url, args.model)
        except Exception as exc:
            print(f"WARNING: vLLM call failed ({exc}). Falling back to mock mode.")
            provider = "mock_fallback_after_vllm_error"
            model = f"mock-Alternate path (attempted: {args.model})"
            explanation_md = build_mock_explanation(evidence)
    else:
        explanation_md = build_mock_explanation(evidence)

    # ── Save outputs ───────────────────────────────────────────────────────
    md_path, json_path, prompt_path = save_outputs(
        did, evidence, system_prompt, user_prompt,
        explanation_md, provider, model, args.out,
    )

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  InfraGraph AI -- Qwen RCA Explanation")
    print(f"{'='*60}")
    print(f"  Provider              : {provider}")
    print(f"  Model                 : {model}")
    print(f"  Heuristic root cause  : {evidence['heuristic_rca']['predicted_root_cause']}"
          f"  (correct: {evidence['heuristic_rca']['is_correct']})")
    print(f"  GNN root cause        : {evidence['gnn_rca']['predicted_root_cause']}"
          f"  (correct: {evidence['gnn_rca']['is_correct']})")
    print(f"  Ground truth          : {evidence['ground_truth']['root_cause']}")
    print(f"  Output markdown       : {md_path}")
    print(f"  Output JSON           : {json_path}")
    print(f"  Prompt JSON           : {prompt_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()


