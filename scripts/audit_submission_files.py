from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(".").resolve()

CORE_INCLUDE_PREFIXES = [
    "README.md",
    "requirements/",
    "app/",
    "src/",
    "scripts/collect_final_submission_metrics.py",
    "scripts/evaluate_rfdetr_v3_detector.py",
    "scripts/generate_rfdetr_runtime_evidence.py",
    "scripts/run_enterprise_gnn_v2_inference.py",
    "scripts/train_enterprise_gnn_v2_rca.py",
    "scripts/amd_rocm/",
    "docs/evidence/final_submission_metrics/",
    "docs/evidence/amd_qwen3_grpo_run/",
    "docs/evidence/amd_mi300x_enterprise_gnn_v2_run/",
    "reports/rfdetr_v3_eval/",
    "reports/rfdetr_runtime_evidence/",
    "reports/kb_index/",
    "training/verl_grpo/reward_eval_report.json",
    "training/verl_grpo/runs/qwen3_4b_grpo_lora_amd/completion_evidence.md",
]

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
    "datasets",
    "runs",
    "wandb",
    "node_modules",
    "dist",
}

BINARY_EXTS = {
    ".pt", ".pth", ".safetensors", ".onnx", ".joblib", ".pkl",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg",
    ".pdf", ".pptx", ".docx", ".xlsx", ".zip", ".tar", ".gz",
    ".mp4", ".mov", ".avi", ".mp3", ".wav",
}

TEXT_EXTS = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini",
    ".cfg", ".sh", ".ps1", ".bat", ".html", ".css", ".js", ".ts",
    ".tsx", ".jsx", ".sql", ".csv"
}

def is_core(rel: str) -> bool:
    return any(rel == p or rel.startswith(p) for p in CORE_INCLUDE_PREFIXES)

def is_text(rel: str, path: Path) -> bool:
    if path.name in [".gitignore", ".dockerignore"]:
        return True
    return path.suffix.lower() in TEXT_EXTS

def main():
    core = []
    pdf_exclude = []
    review = []

    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue

        rel = path.relative_to(ROOT).as_posix()
        parts = set(path.relative_to(ROOT).parts)
        size_kb = path.stat().st_size / 1024

        if any(p in EXCLUDE_DIRS for p in parts):
            pdf_exclude.append((rel, "excluded directory", size_kb))
            continue

        if path.suffix.lower() in BINARY_EXTS:
            pdf_exclude.append((rel, "binary/artifact/image", size_kb))
            continue

        if is_core(rel):
            core.append((rel, size_kb))
            continue

        if is_text(rel, path):
            review.append((rel, "text file not in core include list", size_kb))
        else:
            pdf_exclude.append((rel, "non-text or unsupported", size_kb))

    out = []
    out.append("# Submission File Audit\n")
    out.append("## A. Core files to include in PDF\n")
    for rel, size in core:
        out.append(f"- `{rel}` ({size:.1f} KB)")

    out.append("\n## B. Text files to review manually\n")
    for rel, reason, size in review:
        out.append(f"- `{rel}` ({size:.1f} KB) — {reason}")

    out.append("\n## C. Files excluded from PDF\n")
    for rel, reason, size in pdf_exclude:
        out.append(f"- `{rel}` ({size:.1f} KB) — {reason}")

    Path("dist").mkdir(exist_ok=True)
    Path("dist/submission_file_audit.md").write_text("\n".join(out), encoding="utf-8")

    print("Created: dist/submission_file_audit.md")
    print(f"Core include files: {len(core)}")
    print(f"Manual review files: {len(review)}")
    print(f"PDF excluded files: {len(pdf_exclude)}")

if __name__ == "__main__":
    main()