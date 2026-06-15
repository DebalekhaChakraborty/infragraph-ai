"""collect_final_submission_metrics.py

Collect Slide-4 style evidence for the InfraGraph AI hackathon submission.

Gathers:
  A. GNN V2 training evidence (from training_report.json)
  B. GNN V2 inference latency (3 timed runs)
  C. Optional GNN training benchmark (--run-training-benchmark flag)
  D. Qwen/vLLM latency (live endpoint or committed GRPO evidence)
  E. AMD GPU telemetry (amd-smi / rocm-smi)
  F. RF-DETR evaluation evidence (from eval report if present)
  G. Slide-4 summary markdown table

Usage:
    python scripts/collect_final_submission_metrics.py [--run-training-benchmark]

Outputs to: docs/evidence/final_submission_metrics/
  - final_submission_metrics.json
  - final_submission_metrics.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
GNN_V2_TRAINING_REPORT = (
    REPO_ROOT / "model_artifacts" / "enterprise_gnn_rca_v2" / "training_report.json"
)
RFDETR_EVAL_REPORT = (
    REPO_ROOT / "reports" / "rfdetr_v3_eval" / "rfdetr_v3_eval_report.json"
)
QWEN_EVIDENCE_FILE = (
    REPO_ROOT
    / "training"
    / "verl_grpo"
    / "runs"
    / "qwen3_4b_grpo_lora_amd"
    / "completion_evidence.md"
)
OUT_DIR = REPO_ROOT / "docs" / "evidence" / "final_submission_metrics"
INFERENCE_SCRIPT = REPO_ROOT / "scripts" / "run_enterprise_gnn_v2_inference.py"
TRAINING_SCRIPT = REPO_ROOT / "scripts" / "train_enterprise_gnn_v2_rca.py"


# ---------------------------------------------------------------------------
# Section A: GNN V2 training evidence
# ---------------------------------------------------------------------------

def collect_gnn_v2_training_evidence() -> dict:
    print("  [A] Collecting GNN V2 training evidence ...")
    if not GNN_V2_TRAINING_REPORT.exists():
        return {"status": "training_report_unavailable", "path": str(GNN_V2_TRAINING_REPORT)}

    try:
        data = json.loads(GNN_V2_TRAINING_REPORT.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "training_report_read_error", "error": str(exc)}

    result: dict = {
        "status": "ok",
        "source_file": str(GNN_V2_TRAINING_REPORT),
    }

    for key in (
        "model_type",
        "gnn_architecture",
        "uses_edge_type",
        "uses_temporal_features",
        "num_graphs",
        "train_count",
        "val_count",
        "test_count",
        "epochs",
        "best_epoch",
        "best_val_mrr",
    ):
        if key in data:
            result[key] = data[key]

    # Extract test metrics
    test_metrics: dict = {}
    for possible_key in ("test_metrics", "test_results", "eval_results"):
        if possible_key in data and isinstance(data[possible_key], dict):
            raw = data[possible_key]
            for m_key in ("top1", "top3", "mrr", "top_1", "top_3"):
                if m_key in raw:
                    test_metrics[m_key.replace("_", "")] = raw[m_key]
            break

    # Also try flat keys
    for flat_key in ("test_top1", "test_top3", "test_mrr"):
        if flat_key in data:
            short = flat_key.replace("test_", "")
            test_metrics[short] = data[flat_key]

    result["test_metrics"] = test_metrics
    return result


# ---------------------------------------------------------------------------
# Section B: GNN V2 inference latency
# ---------------------------------------------------------------------------

def collect_gnn_v2_inference_latency(num_runs: int = 3) -> dict:
    print(f"  [B] Collecting GNN V2 inference latency ({num_runs} runs) ...")
    cmd = [
        sys.executable,
        str(INFERENCE_SCRIPT),
        "--scenario-id",
        "enterprise_v3_0079",
        "--split",
        "test",
    ]
    cmd_str = " ".join(cmd)

    if not INFERENCE_SCRIPT.exists():
        return {
            "status": "inference_error",
            "error": f"Inference script not found: {INFERENCE_SCRIPT}",
            "command": cmd_str,
        }

    latencies_ms: list[float] = []
    errors: list[str] = []

    for run_idx in range(num_runs):
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if proc.returncode != 0:
                stderr_snippet = (proc.stderr or "")[:400]
                errors.append(
                    f"Run {run_idx + 1}: exit code {proc.returncode}: {stderr_snippet}"
                )
            else:
                latencies_ms.append(elapsed_ms)
                print(f"      Run {run_idx + 1}: {elapsed_ms:.1f} ms")
        except subprocess.TimeoutExpired:
            errors.append(f"Run {run_idx + 1}: timed out")
        except Exception as exc:
            errors.append(f"Run {run_idx + 1}: {exc}")

    if not latencies_ms:
        return {
            "status": "inference_error",
            "runs_attempted": num_runs,
            "errors": errors,
            "command": cmd_str,
        }

    return {
        "status": "ok",
        "runs_attempted": num_runs,
        "runs_succeeded": len(latencies_ms),
        "min_ms": min(latencies_ms),
        "avg_ms": sum(latencies_ms) / len(latencies_ms),
        "max_ms": max(latencies_ms),
        "all_latencies_ms": latencies_ms,
        "errors": errors,
        "command": cmd_str,
    }


# ---------------------------------------------------------------------------
# Section C: Optional GNN training benchmark
# ---------------------------------------------------------------------------

def collect_gnn_training_benchmark(run_it: bool) -> dict:
    if not run_it:
        return {"status": "not_run", "note": "Pass --run-training-benchmark to enable."}

    print("  [C] Running GNN V2 training benchmark (this may take several minutes) ...")

    if not TRAINING_SCRIPT.exists():
        return {
            "status": "inference_error",
            "error": f"Training script not found: {TRAINING_SCRIPT}",
        }

    cmd = [
        sys.executable,
        str(TRAINING_SCRIPT),
        "--epochs", "80",
        "--eval-every", "5",
        "--hidden-dim", "64",
        "--num-layers", "2",
    ]
    cmd_str = " ".join(cmd)

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        elapsed_s = time.perf_counter() - t0

        if proc.returncode != 0:
            stderr_snippet = (proc.stderr or "")[:600]
            return {
                "status": "training_error",
                "exit_code": proc.returncode,
                "stderr_snippet": stderr_snippet,
                "training_time_seconds": elapsed_s,
                "command": cmd_str,
            }

        return {
            "status": "ok",
            "training_time_seconds": elapsed_s,
            "command": cmd_str,
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "training_error",
            "error": "Training timed out after 3600 s",
            "command": cmd_str,
        }
    except Exception as exc:
        return {
            "status": "training_error",
            "error": str(exc),
            "command": cmd_str,
        }


# ---------------------------------------------------------------------------
# Section D: Qwen/vLLM latency
# ---------------------------------------------------------------------------

_SAMPLE_PROMPT = (
    "You are an IT operations AI. The GNN root cause analysis identified DC-FW-01 "
    "(firewall, datacenter_topology) as the root cause of enterprise incident cluster "
    "enterprise_v3_0079. Impacted nodes include APP-01, APP-02 (app_db_topology). "
    "Alert timeline: DC-FW-01 CRITICAL packet-loss at T+0s, APP-01 WARNING latency "
    "at T+15s, APP-02 WARNING latency at T+20s. "
    "Runbook: FW-SOP-001 firewall restart procedure. "
    "Draft a concise remediation plan with validation steps and rollback notes."
)


def _count_tokens(text: str) -> tuple[int, str]:
    """Try transformers tokenizer, fall back to char/4 approximation."""
    try:
        from transformers import AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-4B", trust_remote_code=True
        )
        ids = tokenizer.encode(text)
        return len(ids), "transformers_AutoTokenizer"
    except Exception:
        count = math.ceil(len(text) / 4)
        return count, "approximate_char_div_4"


def collect_qwen_latency() -> dict:
    print("  [D] Collecting Qwen/vLLM latency ...")

    base_url = os.environ.get("INFRAGRAPH_QWEN_BASE_URL", "").strip()
    model = os.environ.get("INFRAGRAPH_QWEN_MODEL", "Qwen/Qwen3-4B").strip()

    if base_url:
        endpoint = base_url.rstrip("/") + "/v1/chat/completions"
        payload = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": _SAMPLE_PROMPT}],
                "max_tokens": 512,
                "temperature": 0.1,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
                latency_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:
            return {
                "status": "live_latency_error",
                "endpoint": endpoint,
                "model": model,
                "error": str(exc),
                "live_latency": "unavailable",
            }

        try:
            resp_data = json.loads(raw)
        except Exception as exc:
            return {
                "status": "live_latency_error",
                "endpoint": endpoint,
                "model": model,
                "error": f"JSON parse error: {exc}",
                "live_latency": "unavailable",
            }

        usage = resp_data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", None)
        completion_tokens = usage.get("completion_tokens", None)
        total_tokens = usage.get("total_tokens", None)

        if total_tokens is None:
            completion_text = ""
            choices = resp_data.get("choices", [])
            if choices:
                completion_text = (
                    choices[0].get("message", {}).get("content", "")
                )
            prompt_token_count, method = _count_tokens(_SAMPLE_PROMPT)
            completion_token_count, _ = _count_tokens(completion_text)
            total_tokens = prompt_token_count + completion_token_count
            token_count_method = method
            prompt_tokens = prompt_token_count
            completion_tokens = completion_token_count
        else:
            token_count_method = "api_reported"

        print(
            f"      Qwen live latency: {latency_ms:.0f} ms, "
            f"tokens: {total_tokens} ({token_count_method})"
        )

        return {
            "status": "ok",
            "live_latency": "available",
            "endpoint": endpoint,
            "model": model,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "token_count_method": token_count_method,
        }

    else:
        # No live endpoint — use committed GRPO evidence
        committed_evidence = ""
        if QWEN_EVIDENCE_FILE.exists():
            committed_evidence = QWEN_EVIDENCE_FILE.read_text(encoding="utf-8")

        return {
            "status": "ok",
            "live_latency": "unavailable",
            "note": (
                "INFRAGRAPH_QWEN_BASE_URL not set. "
                "Committed GRPO training evidence included below."
            ),
            "committed_grpo_evidence_path": str(QWEN_EVIDENCE_FILE),
            "committed_grpo_evidence": committed_evidence,
            "model": "Qwen/Qwen3-4B",
            "alignment": "LoRA rank 16 + GRPO/vERL (32/32 steps)",
        }


# ---------------------------------------------------------------------------
# Section E: AMD GPU telemetry
# ---------------------------------------------------------------------------

def collect_amd_gpu_telemetry() -> dict:
    print("  [E] Collecting AMD GPU telemetry ...")

    for cmd_name in ("amd-smi", "rocm-smi"):
        try:
            proc = subprocess.run(
                [cmd_name],
                capture_output=True,
                text=True,
                timeout=15,
            )
            output = proc.stdout or proc.stderr or ""
            output_snippet = output[:800]
            return {
                "status": "ok",
                "available": True,
                "command_used": cmd_name,
                "output_snippet": output_snippet,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "available": False,
                "command_used": cmd_name,
                "note": "AMD telemetry command timed out.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            return {
                "status": "error",
                "available": False,
                "command_used": cmd_name,
                "error": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    return {
        "status": "unavailable",
        "available": False,
        "command_used": None,
        "note": (
            "AMD telemetry command unavailable in this environment; "
            "see committed AMD evidence files."
        ),
        "committed_evidence_paths": [
            "docs/evidence/amd_qwen3_grpo_run/training_summary.md",
            "docs/evidence/amd_mi300x_enterprise_gnn_v2_run/training_summary.md",
            "training/verl_grpo/runs/qwen3_4b_grpo_lora_amd/completion_evidence.md",
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Section F: RF-DETR evidence
# ---------------------------------------------------------------------------

def collect_rfdetr_evidence() -> dict:
    print("  [F] Collecting RF-DETR evidence ...")

    if RFDETR_EVAL_REPORT.exists():
        try:
            data = json.loads(RFDETR_EVAL_REPORT.read_text(encoding="utf-8"))
            return {
                "status": "eval_report_present",
                "source_file": str(RFDETR_EVAL_REPORT),
                "metrics": data.get("metrics", {}),
                "summary_note": data.get("summary_note", ""),
                "split": data.get("split", ""),
                "num_images_processed": data.get("num_images_processed"),
                "per_class_metrics": data.get("per_class_metrics", {}),
            }
        except Exception as exc:
            return {
                "status": "eval_report_read_error",
                "source_file": str(RFDETR_EVAL_REPORT),
                "error": str(exc),
            }

    return {
        "status": "eval_report_absent",
        "note": (
            "RF-DETR checkpoints and inference script are present; "
            "detector accuracy is not claimed without eval report. "
            "Run: python scripts/evaluate_rfdetr_v3_detector.py"
        ),
        "checkpoint_path": "model_artifacts/rfdetr_v3/checkpoint_best_total.pth",
        "inference_script": "scripts/run_rfdetr_inference.py",
        "eval_script": "scripts/evaluate_rfdetr_v3_detector.py",
    }


# ---------------------------------------------------------------------------
# Section G: Slide-4 summary table
# ---------------------------------------------------------------------------

def _safe_fmt_float(v: object, fmt: str = ".1f") -> str:
    if v is None:
        return "N/A"
    try:
        fv = float(v)  # type: ignore[arg-type]
        if math.isnan(fv):
            return "N/A"
        return format(fv, fmt)
    except (TypeError, ValueError):
        return str(v)


def build_slide4_table(
    gnn_training: dict,
    gnn_latency: dict,
    qwen: dict,
    amd_gpu: dict,
    rfdetr: dict,
) -> str:
    # GNN training values
    num_graphs = gnn_training.get("num_graphs", "N/A")
    train_count = gnn_training.get("train_count", "N/A")
    val_count = gnn_training.get("val_count", "N/A")
    test_count = gnn_training.get("test_count", "N/A")
    best_epoch = gnn_training.get("best_epoch", "N/A")
    test_metrics = gnn_training.get("test_metrics", {})
    test_top1 = test_metrics.get("top1", test_metrics.get("top_1", "N/A"))

    # GNN inference latency
    if gnn_latency.get("status") == "ok":
        avg_ms = _safe_fmt_float(gnn_latency.get("avg_ms"), ".1f")
        min_ms = _safe_fmt_float(gnn_latency.get("min_ms"), ".1f")
        max_ms = _safe_fmt_float(gnn_latency.get("max_ms"), ".1f")
        latency_str = f"{avg_ms} ms avg ({min_ms}–{max_ms} ms range)"
    else:
        latency_str = "not measured"

    # Qwen token/latency
    if qwen.get("live_latency") == "available":
        total_tokens = qwen.get("total_tokens")
        token_method = qwen.get("token_count_method", "")
        latency_ms_val = qwen.get("latency_ms")
        qwen_tokens_str = (
            f"{total_tokens} ({token_method})" if total_tokens is not None else "N/A"
        )
        qwen_latency_str = (
            f"{_safe_fmt_float(latency_ms_val, '.0f')} ms"
            if latency_ms_val is not None
            else "N/A"
        )
    else:
        qwen_tokens_str = "not measured live"
        qwen_latency_str = "unavailable"

    # AMD GPU
    if amd_gpu.get("available"):
        amd_evidence_str = f"Live telemetry captured via `{amd_gpu.get('command_used')}`"
    else:
        amd_evidence_str = "MI300X / ROCm — GPU 100% utilization, VRAM ~42%, Power ~278W (training evidence)"

    # RF-DETR detector
    if rfdetr.get("status") == "eval_report_present":
        metrics = rfdetr.get("metrics", {})
        p = _safe_fmt_float(metrics.get("precision"), ".4f")
        r = _safe_fmt_float(metrics.get("recall"), ".4f")
        f1 = _safe_fmt_float(metrics.get("f1"), ".4f")
        detector_str = f"precision={p}, recall={r}, F1={f1} (prototype benchmark)"
    else:
        detector_str = "not claimed — eval report absent"

    rows = [
        ("Diagram model", "RF-DETR-supported detector + verified fallback + vision connector extraction"),
        ("RCA model", "EnterpriseRcaTemporalRelGNN / Temporal Relation-Aware GraphSAGE"),
        ("RCA dataset", f"{num_graphs} generated enterprise RCA scenarios"),
        ("Split", f"{train_count} train / {val_count} val / {test_count} test"),
        ("GNN feature dim", "54"),
        ("GNN V2 best epoch", str(best_epoch)),
        ("GNN V2 test top-1", f"{test_top1} (synthetic/generated enterprise benchmark)"),
        ("GNN V2 inference latency", latency_str),
        ("Qwen model", "Qwen/Qwen3-4B"),
        ("Alignment", "LoRA rank 16 + GRPO/vERL (32/32 steps)"),
        ("Qwen tokens", qwen_tokens_str),
        ("Qwen latency", qwen_latency_str),
        ("AMD GPU evidence", amd_evidence_str),
        ("Detector metrics", detector_str),
    ]

    header = "| Category | Evidence |\n|----------|----------|\n"
    body = "".join(f"| {cat} | {val} |\n" for cat, val in rows)
    return header + body


# ---------------------------------------------------------------------------
# Markdown report writer
# ---------------------------------------------------------------------------

def write_markdown_report(
    out_dir: Path,
    all_sections: dict,
    slide4_table: str,
) -> None:
    gnn = all_sections.get("gnn_v2_training", {})
    latency = all_sections.get("gnn_v2_inference_latency", {})
    training_bench = all_sections.get("gnn_training_benchmark", {})
    qwen = all_sections.get("qwen_latency", {})
    amd = all_sections.get("amd_gpu_telemetry", {})
    rfdetr = all_sections.get("rfdetr_evidence", {})

    lines = [
        "# InfraGraph AI — Final Submission Metrics\n",
        f"Generated: {all_sections.get('generated_at', 'N/A')}\n",
        "\n## Slide-4 Summary Table\n",
        slide4_table,
        "\n---\n",
        "\n## A. GNN V2 Training Evidence\n",
        f"- Source: `{gnn.get('source_file', GNN_V2_TRAINING_REPORT)}`\n",
        f"- Status: `{gnn.get('status', 'N/A')}`\n",
        f"- Model type: {gnn.get('model_type', 'N/A')}\n",
        f"- Architecture: {gnn.get('gnn_architecture', 'N/A')}\n",
        f"- Graphs: {gnn.get('num_graphs', 'N/A')} "
        f"(train={gnn.get('train_count', 'N/A')} / "
        f"val={gnn.get('val_count', 'N/A')} / "
        f"test={gnn.get('test_count', 'N/A')})\n",
        f"- Epochs: {gnn.get('epochs', 'N/A')}, best epoch: {gnn.get('best_epoch', 'N/A')}\n",
        f"- Best val MRR: {gnn.get('best_val_mrr', 'N/A')}\n",
        f"- Test metrics: {json.dumps(gnn.get('test_metrics', {}))}\n",
        f"- uses_edge_type: {gnn.get('uses_edge_type', 'N/A')}\n",
        f"- uses_temporal_features: {gnn.get('uses_temporal_features', 'N/A')}\n",
        "\n## B. GNN V2 Inference Latency\n",
    ]

    if latency.get("status") == "ok":
        lines += [
            f"- Status: `ok`\n",
            f"- Runs: {latency.get('runs_succeeded')}/{latency.get('runs_attempted')}\n",
            f"- Min: {_safe_fmt_float(latency.get('min_ms'), '.1f')} ms\n",
            f"- Avg: {_safe_fmt_float(latency.get('avg_ms'), '.1f')} ms\n",
            f"- Max: {_safe_fmt_float(latency.get('max_ms'), '.1f')} ms\n",
            f"- Command: `{latency.get('command', 'N/A')}`\n",
        ]
    else:
        lines += [
            f"- Status: `{latency.get('status', 'N/A')}`\n",
            f"- Errors: {latency.get('errors', [])}\n",
        ]

    lines += [
        "\n## C. GNN Training Benchmark\n",
        f"- Status: `{training_bench.get('status', 'N/A')}`\n",
    ]
    if training_bench.get("status") == "ok":
        lines.append(
            f"- Training time: {_safe_fmt_float(training_bench.get('training_time_seconds'), '.1f')} s\n"
        )
    elif training_bench.get("status") == "not_run":
        lines.append(f"- Note: {training_bench.get('note', '')}\n")

    lines += [
        "\n## D. Qwen/vLLM Latency\n",
        f"- Live latency: {qwen.get('live_latency', 'N/A')}\n",
        f"- Model: {qwen.get('model', 'N/A')}\n",
    ]
    if qwen.get("live_latency") == "available":
        lines += [
            f"- Endpoint: {qwen.get('endpoint', 'N/A')}\n",
            f"- Latency: {_safe_fmt_float(qwen.get('latency_ms'), '.0f')} ms\n",
            f"- Tokens: {qwen.get('total_tokens', 'N/A')} ({qwen.get('token_count_method', 'N/A')})\n",
        ]
    else:
        lines += [
            f"- Note: {qwen.get('note', '')}\n",
            f"- Committed evidence: `{qwen.get('committed_grpo_evidence_path', 'N/A')}`\n",
        ]

    lines += [
        "\n## E. AMD GPU Telemetry\n",
        f"- Available: {amd.get('available', False)}\n",
        f"- Command used: {amd.get('command_used', 'N/A')}\n",
        f"- Timestamp: {amd.get('timestamp', 'N/A')}\n",
    ]
    if amd.get("output_snippet"):
        lines += [
            "- Output snippet:\n",
            "```\n",
            amd.get("output_snippet", "") + "\n",
            "```\n",
        ]
    else:
        lines.append(f"- Note: {amd.get('note', '')}\n")

    lines += [
        "\n## F. RF-DETR Evidence\n",
        f"- Status: `{rfdetr.get('status', 'N/A')}`\n",
    ]
    if rfdetr.get("status") == "eval_report_present":
        m = rfdetr.get("metrics", {})
        lines += [
            f"- Source: `{rfdetr.get('source_file', 'N/A')}`\n",
            f"- Split: {rfdetr.get('split', 'N/A')}\n",
            f"- Precision: {_safe_fmt_float(m.get('precision'), '.4f')}\n",
            f"- Recall: {_safe_fmt_float(m.get('recall'), '.4f')}\n",
            f"- F1: {_safe_fmt_float(m.get('f1'), '.4f')}\n",
            f"- mAP@0.5: {_safe_fmt_float(m.get('mean_ap_at_50'), '.4f')}\n",
        ]
    else:
        lines.append(f"- Note: {rfdetr.get('note', '')}\n")

    md_path = out_dir / "final_submission_metrics.md"
    md_path.write_text("".join(lines), encoding="utf-8")
    print(f"  Markdown report written to {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect Slide-4 style evidence for InfraGraph AI hackathon submission."
    )
    parser.add_argument(
        "--run-training-benchmark",
        action="store_true",
        help="Run the GNN V2 training benchmark and record wall-clock time.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()

    print("Collecting final submission metrics ...")

    gnn_training = collect_gnn_v2_training_evidence()
    gnn_latency = collect_gnn_v2_inference_latency(num_runs=3)
    training_bench = collect_gnn_training_benchmark(run_it=args.run_training_benchmark)
    qwen = collect_qwen_latency()
    amd_gpu = collect_amd_gpu_telemetry()
    rfdetr = collect_rfdetr_evidence()

    all_sections = {
        "generated_at": generated_at,
        "gnn_v2_training": gnn_training,
        "gnn_v2_inference_latency": gnn_latency,
        "gnn_training_benchmark": training_bench,
        "qwen_latency": qwen,
        "amd_gpu_telemetry": amd_gpu,
        "rfdetr_evidence": rfdetr,
    }

    slide4_table = build_slide4_table(
        gnn_training=gnn_training,
        gnn_latency=gnn_latency,
        qwen=qwen,
        amd_gpu=amd_gpu,
        rfdetr=rfdetr,
    )
    all_sections["slide4_markdown_table"] = slide4_table

    json_path = OUT_DIR / "final_submission_metrics.json"
    json_path.write_text(json.dumps(all_sections, indent=2), encoding="utf-8")
    print(f"  JSON report written to {json_path}")

    write_markdown_report(OUT_DIR, all_sections, slide4_table)

    print("\nDone. Outputs:")
    print(f"  {json_path}")
    print(f"  {OUT_DIR / 'final_submission_metrics.md'}")


if __name__ == "__main__":
    main()
