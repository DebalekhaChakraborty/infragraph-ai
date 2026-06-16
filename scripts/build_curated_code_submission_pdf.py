from __future__ import annotations

import fnmatch
import html
import subprocess
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(".").resolve()
OUT_DIR = REPO_ROOT / "dist"
OUT_BASE = OUT_DIR / "InfraGraphAI_Codebase_Submission"

GITHUB_REPO_URL = "https://github.com/DebalekhaChakraborty/infragraph-ai"

CURATED_INCLUDE_PREFIXES = {
    # Main documentation
    "README.md",
    "docs/submission/deck_claims.md",

    # Requirements and runtime setup
    "requirements/",

    # Application / cockpit
    "app/README.md",
    "app/rfdetr_subprocess_bridge.py",
    "app/streamlit_app.py",

    # Core source code
    "src/agents/",
    "src/ai_remediation/",
    "src/event_correlation/",
    "src/governance/",
    "src/graph_copilot/",
    "src/incident_simulation/",
    "src/kb_retrieval/",
    "src/rca/",
    "src/rca_ml/",
    "src/runbook_retrieval/",
    "src/topology/",
    "src/vector_memory/",
    "src/vision/",
    "src/enterprise_gnn_rca.py",
    "src/live_detector.py",
    "src/live_rfdetr_detector.py",
    "src/runtime_ingestion.py",
    "src/paths.py",

    # Dataset/model generation logic - code only, not generated dataset
    "data_generator/generate_infragraph_dataset.py",

    # Training, inference, evaluation, and evidence scripts
    "scripts/collect_final_submission_metrics.py",
    "scripts/evaluate_rfdetr_v3_detector.py",
    "scripts/generate_rfdetr_runtime_evidence.py",
    "scripts/run_enterprise_gnn_v2_inference.py",
    "scripts/train_enterprise_gnn_v2_rca.py",

    # AMD / ROCm / Qwen execution path, excluding S3 upload/offload utilities
    "scripts/amd_rocm/bootstrap_grpo_env.sh",
    "scripts/amd_rocm/bootstrap_rca_gnn_env.sh",
    "scripts/amd_rocm/generate_qwen_sop_remediation_after_reset.sh",
    "scripts/amd_rocm/patch_verl_runtime_for_rocm.sh",
    "scripts/amd_rocm/run_qwen3_grpo_success_path.sh",
    "scripts/amd_rocm/start_qwen_sop_lora_vllm.sh",
    "scripts/amd_rocm/start_streamlit_with_external_rfdetr.sh",

    # Runbook/SOP knowledge base used for RAG grounding
    "assets/kb/runbooks/",
    "assets/kb/sops/",
    "assets/kb/known_resolutions/",

    # Evidence documents and reports
    "docs/evidence/final_submission_metrics/",
    "docs/evidence/amd_mi300x_enterprise_gnn_v2_run/",
    "docs/evidence/amd_qwen3_grpo_run/README.md",
    "docs/evidence/amd_qwen3_grpo_run/training_summary.md",
    "docs/evidence/amd_qwen3_grpo_run/completion_evidence.md",
    "docs/evidence/amd_qwen3_grpo_run/live_lora_vllm_verification.md",
    "docs/evidence/amd_qwen3_grpo_run/lora_train_meta.json",
    "docs/evidence/amd_qwen3_grpo_run/fsdp_config.json",
    "docs/evidence/amd_qwen3_grpo_run/torch_runtime.txt",
    "docs/evidence/amd_qwen3_grpo_run/python_version.txt",

    # Technical documentation
    "docs/ai_training_and_remediation_story.md",
    "docs/diagram_intelligence_v3_dataset.md",
    "docs/diagram_onboarding.md",
    "docs/enterprise_gnn_rca.md",
    "docs/enterprise_gnn_v2_temporal_relation_aware.md",
    "docs/enterprise_graph_dataset.md",
    "docs/event_correlation_and_causal_evidence.md",
    "docs/live_rfdetr_external_runtime.md",
    "docs/qwen_sop_lora_training.md",
    "docs/remediation_pipeline.md",
    "docs/rfdetr_v3_detector.md",
    "docs/run_qwen_vllm_amd.md",
    "docs/sop_kb_rag_remediation.md",
    "docs/topology_rca_pipeline.md",

    # Evaluation reports
    "reports/rfdetr_v3_eval/rfdetr_v3_eval_report.md",
    "reports/rfdetr_v3_eval/rfdetr_v3_eval_report.json",
    "reports/rfdetr_runtime_evidence/rfdetr_runtime_evidence.md",
    "reports/rfdetr_runtime_evidence/rfdetr_runtime_evidence.json",
    "reports/kb_index/build_summary.json",
    "reports/enterprise_gnn_rca/evaluation.json",
    "reports/enterprise_gnn_rca_v2/evaluation.json",
    "reports/topology_rca/eval_metrics.json",
    "reports/v3_annotation_qa/annotation_quality_report.json",

    # Model metadata / reports only, not weights
    "model_artifacts/enterprise_gnn_rca/enterprise_gnn_config.json",
    "model_artifacts/enterprise_gnn_rca/feature_columns.json",
    "model_artifacts/enterprise_gnn_rca_v2/training_report.json",
    "model_artifacts/enterprise_gnn_rca_v2/enterprise_gnn_v2_config.json",
    "model_artifacts/enterprise_gnn_rca_v2/feature_columns.json",
    "model_artifacts/qwen3_grpo_lora_adapter/README.md",
    "model_artifacts/qwen3_grpo_lora_adapter/adapter_config.json",

    # Representative final output only
    "outputs/enterprise_gnn_rca_v2/enterprise_v3_0079_enterprise_gnn_v2_rca_result.json",

    # GRPO training/eval proof
    "training/verl_grpo/reward_eval_report.json",
    "training/verl_grpo/runs/qwen3_4b_grpo_lora_amd/completion_evidence.md",
}

ALWAYS_EXCLUDE_PREFIXES = {
    ".git/",
    ".venv/",
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    "dist/",
    "datasets/",
    "runs/",
    "wandb/",
    "node_modules/",
    "lib/",
    "assets/gallery/",
    "assets/onboarding/",
    "assets/preloaded/",
    "runtime_state/",
    "scenario_library/",
    "data/qwen_sop_grounded/previews/",
    "data/qwen_sop_grounded_expanded/previews/",
    "reports/rfdetr_runtime_evidence/annotated/",
    "reports/rfdetr_runtime_evidence/predictions/",
    "docs/evidence/s3_offload/",
}

ALWAYS_EXCLUDE_FILE_PATTERNS = {
    "*.pt",
    "*.pth",
    "*.safetensors",
    "*.onnx",
    "*.pkl",
    "*.joblib",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.webp",
    "*.svg",
    "*.mp4",
    "*.mov",
    "*.avi",
    "*.zip",
    "*.tar",
    "*.gz",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "package-lock.json",
}

TEXT_EXTENSIONS = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml",
    ".ini", ".cfg", ".sh", ".ps1", ".bat", ".html", ".css",
    ".js", ".ts", ".tsx", ".jsx", ".sql", ".csv"
}


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def is_included(rel_path: str) -> bool:
    return any(rel_path == p or rel_path.startswith(p) for p in CURATED_INCLUDE_PREFIXES)


def is_excluded(rel_path: str) -> tuple[bool, str]:
    for p in ALWAYS_EXCLUDE_PREFIXES:
        if rel_path.startswith(p):
            return True, f"Excluded prefix: {p}"

    name = Path(rel_path).name
    for pattern in ALWAYS_EXCLUDE_FILE_PATTERNS:
        if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel_path, pattern):
            return True, f"Excluded file pattern: {pattern}"

    return False, ""


def is_text_file(path: Path) -> bool:
    if path.name in {".gitignore", ".dockerignore"}:
        return True
    return path.suffix.lower() in TEXT_EXTENSIONS


def read_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            pass
    return "[Could not decode this file as text]"


def language_for(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".py": "python",
        ".md": "markdown",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".sh": "bash",
        ".ps1": "powershell",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".jsx": "jsx",
        ".html": "html",
        ".css": "css",
        ".sql": "sql",
        ".toml": "toml",
        ".ini": "ini",
        ".cfg": "ini",
        ".csv": "csv",
        ".txt": "text",
    }.get(ext, "text")


def collect_files() -> tuple[list[Path], list[tuple[str, str]], list[tuple[str, str]]]:
    included: list[Path] = []
    excluded_manifest: list[tuple[str, str]] = []
    skipped_manifest: list[tuple[str, str]] = []

    for path in sorted(REPO_ROOT.rglob("*")):
        if not path.is_file():
            continue

        r = rel(path)

        excluded, reason = is_excluded(r)
        if excluded:
            excluded_manifest.append((r, reason))
            continue

        if is_included(r):
            if is_text_file(path):
                included.append(path)
            else:
                excluded_manifest.append((r, "Included prefix but non-text/binary file"))
            continue

        skipped_manifest.append((r, "Not part of curated submission appendix"))

    return included, excluded_manifest, skipped_manifest


def sort_included_files(files: list[Path]) -> list[Path]:
    def priority(path: Path) -> tuple[int, str]:
        r = rel(path)
        if r == "README.md":
            return (0, r)
        if r.startswith("app/"):
            return (1, r)
        if r.startswith("requirements/"):
            return (2, r)
        if r.startswith("src/"):
            return (3, r)
        if r.startswith("scripts/"):
            return (4, r)
        if r.startswith("assets/kb/"):
            return (5, r)
        if r.startswith("docs/"):
            return (6, r)
        if r.startswith("reports/"):
            return (7, r)
        if r.startswith("model_artifacts/"):
            return (8, r)
        if r.startswith("outputs/"):
            return (9, r)
        if r.startswith("training/"):
            return (10, r)
        return (99, r)

    return sorted(files, key=priority)


def make_file_index(files: list[Path]) -> str:
    lines = []
    last_parent = None
    for f in files:
        r = rel(f)
        parent = str(Path(r).parent)
        if parent != last_parent:
            lines.append(f"\n### {parent if parent != '.' else 'root'}")
            last_parent = parent
        lines.append(f"- `{r}`")
    return "\n".join(lines)


def build_markdown() -> Path:
    included, excluded, skipped = collect_files()

    included = sort_included_files(included)
    excluded = sorted(excluded, key=lambda x: x[0])
    skipped = sorted(skipped, key=lambda x: x[0])

    md_path = OUT_BASE.with_suffix(".md")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# InfraGraph AI — Curated Codebase Submission Appendix\n")
    lines.append(f"Generated: {datetime.utcnow().isoformat()}Z\n")
    lines.append(f"Repository: {GITHUB_REPO_URL}\n")

    # ── 1. Submission Scope ────────────────────────────────────────────────────
    lines.append("\n## 1. Submission Scope\n")
    lines.append(
        "This PDF is a curated source-code and evidence appendix generated from the InfraGraph AI repository. "
        "It includes active application code, training scripts, inference scripts, evaluation scripts, GNN/RCA logic, "
        "RF-DETR runtime/evaluation logic, Qwen GRPO/vERL execution path, orchestrator code, governance/remediation modules, "
        "requirements, runbooks/SOPs, and final evidence reports. "
        "It excludes raw datasets, binary model weights, tokenizer binaries, generated runtime state, repeated scenario samples, "
        "generated images, cache/build folders, vendored JS libraries, S3/offload utility artifacts, and optional experimental "
        "renderers to keep the submission reviewable. "
        f"The full repository can be reviewed at: {GITHUB_REPO_URL}\n"
    )

    # ── 2. What Is Excluded From This PDF ─────────────────────────────────────
    lines.append("\n## 2. What Is Excluded From This PDF\n")
    lines.append(
        "To keep the portal submission reviewable and within upload constraints, this PDF excludes raw datasets, binary model weights, "
        "tokenizer binaries, generated runtime state, repeated scenario samples, generated images, cache/build folders, vendored JS libraries, "
        "and S3/offload utility artifacts. Where these files are committed or referenced, they can be reviewed in the full GitHub repository:\n"
    )
    lines.append(f"- Full repository: `{GITHUB_REPO_URL}`\n")

    lines.append("\n### Excluded categories\n")
    excluded_categories = [
        "Raw/generated datasets and scenario libraries",
        "Runtime state and repeated incident execution outputs",
        "Model binaries and weights: .pt, .pth, .safetensors, ONNX, pickle/joblib",
        "Tokenizer binaries: tokenizer.json, vocab.json, merges.txt",
        "Generated images, annotated diagrams, videos, and slide/PDF artifacts",
        "Large repeated JSON previews and onboarding samples",
        "Vendor libraries, node modules, cache folders, and build outputs",
        "S3 upload/offload helper scripts and offload manifests",
        "Optional experimental renderers (e.g. 3D WebGL visualisers)",
    ]
    for item in excluded_categories:
        lines.append(f"- {item}")

    # ── 3. Included File Index ─────────────────────────────────────────────────
    lines.append("\n## 3. Included File Index\n")
    lines.append(f"Included files: **{len(included)}**\n")
    lines.append(make_file_index(included))

    # ── 4. Source Code, Training/Evaluation Scripts, and Evidence Appendix ─────
    lines.append("\n## 4. Source Code, Training/Evaluation Scripts, and Evidence Appendix\n")
    for path in included:
        r = rel(path)
        lang = language_for(path)
        text = read_text(path).replace("```", "'''")

        lines.append("\n---\n")
        lines.append(f"### `{r}`\n")
        lines.append(f"```{lang}")
        lines.append(text)
        lines.append("```")

    # ── 5. Complete Excluded / Skipped Manifest ────────────────────────────────
    lines.append("\n---\n")
    lines.append("\n## 5. Complete Excluded / Skipped Manifest\n")
    lines.append(
        "This final section lists every repository path that was not printed in the PDF. "
        "These files are excluded from this curated appendix because they are binary files, generated artifacts, "
        "runtime state, repeated data samples, image/model artifacts, vendor libraries, optional experimental renderers, "
        "or outside the selected submission-review scope. Where available, these files can still be reviewed in the GitHub repository:\n"
    )
    lines.append(f"- Full repository: `{GITHUB_REPO_URL}`\n")

    lines.append("\n### 5.1 Explicitly excluded files and folders\n")
    lines.append(f"Total explicitly excluded paths: **{len(excluded)}**\n")
    lines.append("| File | Reason |")
    lines.append("|------|--------|")
    for r, reason in excluded:
        lines.append(f"| `{r}` | {reason} |")

    lines.append("\n### 5.2 Skipped files outside curated PDF scope\n")
    lines.append(f"Total skipped paths outside curated scope: **{len(skipped)}**\n")
    lines.append("| File | Reason |")
    lines.append("|------|--------|")
    for r, reason in skipped:
        lines.append(f"| `{r}` | {reason} |")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def markdown_to_html(md_path: Path) -> Path:
    html_path = OUT_BASE.with_suffix(".html")
    text = md_path.read_text(encoding="utf-8")

    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<style>",
        "body{font-family:Arial, sans-serif; margin:28px; color:#222;}",
        "h1{font-size:28px; color:#111;}",
        "h2{border-top:2px solid #ddd; padding-top:14px; margin-top:28px; font-size:20px;}",
        "h3{font-size:13px; color:#0645ad; margin-top:18px;}",
        "p,li{font-size:10.5px; line-height:1.35;}",
        "pre{background:#f8f8f8; border:1px solid #ddd; padding:8px; font-size:7.5px; white-space:pre-wrap; word-break:break-word;}",
        "code{font-family:Consolas, monospace;}",
        "table{border-collapse:collapse; width:100%; font-size:7.5px;}",
        "td,th{border:1px solid #ccc; padding:3px; vertical-align:top;}",
        "@page{size:A4; margin:12mm;}",
        "</style></head><body>",
    ]

    in_code = False
    code_buf = []
    para_buf = []

    def flush_para():
        if para_buf:
            html_parts.append("<p>" + html.escape(" ".join(para_buf)) + "</p>")
            para_buf.clear()

    for line in text.splitlines():
        if line.startswith("```"):
            flush_para()
            if not in_code:
                in_code = True
                code_buf = []
            else:
                in_code = False
                html_parts.append("<pre><code>")
                html_parts.append(html.escape("\n".join(code_buf)))
                html_parts.append("</code></pre>")
            continue

        if in_code:
            code_buf.append(line)
            continue

        if line.startswith("# "):
            flush_para()
            html_parts.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            flush_para()
            html_parts.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            flush_para()
            html_parts.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("- "):
            flush_para()
            html_parts.append(f"<li>{html.escape(line[2:])}</li>")
        elif line.startswith("|"):
            flush_para()
            html_parts.append(f"<p><code>{html.escape(line)}</code></p>")
        elif line.strip() == "":
            flush_para()
        else:
            para_buf.append(line.strip())

    flush_para()
    html_parts.append("</body></html>")
    html_path.write_text("\n".join(html_parts), encoding="utf-8")
    return html_path


def try_pdf(html_path: Path) -> Path | None:
    pdf_path = OUT_BASE.with_suffix(".pdf")
    commands = [
        ["wkhtmltopdf", str(html_path), str(pdf_path)],
        ["chromium", "--headless", "--disable-gpu", f"--print-to-pdf={pdf_path}", str(html_path)],
        ["chromium-browser", "--headless", "--disable-gpu", f"--print-to-pdf={pdf_path}", str(html_path)],
        ["google-chrome", "--headless", "--disable-gpu", f"--print-to-pdf={pdf_path}", str(html_path)],
    ]

    for cmd in commands:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            if proc.returncode == 0 and pdf_path.exists() and pdf_path.stat().st_size > 0:
                return pdf_path
        except Exception:
            pass
    return None


def main() -> None:
    md_path = build_markdown()
    html_path = markdown_to_html(md_path)
    pdf_path = try_pdf(html_path)

    print(f"Created markdown: {md_path}")
    print(f"Created HTML:     {html_path}")
    if pdf_path:
        print(f"Created PDF:      {pdf_path}")
    else:
        print("PDF auto-conversion tool not found.")
        print("Open the HTML in a browser and use Print -> Save as PDF.")
        print(f"HTML path: {html_path}")


if __name__ == "__main__":
    main()
