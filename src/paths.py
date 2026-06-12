"""
paths.py — Centralized path configuration for InfraGraph AI.

New canonical layout
--------------------
runtime_state/    Live/generated runtime state (ingestion runs, absorption,
                  incidents, vector memory, global graph memory).
demo_assets/      Curated demo artifacts consumed by the Streamlit app
                  (GNN results, hero scenario, Qwen explanations, etc.).
model_artifacts/  Detector and model checkpoints (RF-DETR, GNN weights).
reports/          Evaluation reports, annotation QA, Hydra run logs.

Backward compatibility
----------------------
Every helper function checks the new canonical path first.  If it does not
exist *and* the legacy ``outputs/<subpath>`` path exists, the legacy path is
returned transparently.  This keeps the demo working whether or not
``scripts/migrate_outputs_structure.py --apply`` has been run.

New writes should always target the new canonical locations — the helpers
return the new path even when neither location exists, so mkdir calls on the
returned path land in the right place.
"""
from __future__ import annotations

from pathlib import Path

# ── Repository root ────────────────────────────────────────────────────────────
# src/paths.py lives at <repo>/src/paths.py → parent.parent == repo root
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# ── New canonical top-level directories ───────────────────────────────────────
RUNTIME_STATE_DIR:   Path = REPO_ROOT / "runtime_state"
DEMO_ASSETS_DIR:     Path = REPO_ROOT / "demo_assets"
MODEL_ARTIFACTS_DIR: Path = REPO_ROOT / "model_artifacts"
REPORTS_DIR:         Path = REPO_ROOT / "reports"

# ── Legacy directory (read-only backward compatibility) ────────────────────────
_OUTPUTS_DIR: Path = REPO_ROOT / "outputs"


# ── Internal helper ────────────────────────────────────────────────────────────

def _with_fallback(new: Path, legacy_rel: str) -> Path:
    """
    Return *new* if it exists; otherwise return ``outputs/<legacy_rel>`` if
    that exists; otherwise return *new* (write target).
    """
    if new.exists():
        return new
    legacy = _OUTPUTS_DIR / legacy_rel
    if legacy.exists():
        return legacy
    return new


# ── Public path helpers ────────────────────────────────────────────────────────

def runtime_path(*parts: str) -> Path:
    """
    Canonical path under ``runtime_state/``.

    Falls back to ``outputs/<parts>`` if the new path does not exist yet.
    Use for: live_ingestion, live_absorption, incident_runs, vector_memory,
    global_graph_memory.
    """
    new = RUNTIME_STATE_DIR.joinpath(*parts)
    return _with_fallback(new, "/".join(parts))


def demo_asset_path(*parts: str) -> Path:
    """
    Canonical path under ``demo_assets/``.

    Falls back to ``outputs/<parts>`` if the new path does not exist yet.
    Use for: demo_hero, enterprise_gnn_rca, gnn_rca, mlp_rca,
    qwen_explanation, annotation_overlays.
    """
    new = DEMO_ASSETS_DIR.joinpath(*parts)
    return _with_fallback(new, "/".join(parts))


def model_artifact_path(*parts: str) -> Path:
    """
    Canonical path under ``model_artifacts/``.

    Falls back to ``outputs/<parts>`` if the new path does not exist yet.
    Use for: rfdetr_v3, rfdetr_v3_smoke, trained_rca_models.
    """
    new = MODEL_ARTIFACTS_DIR.joinpath(*parts)
    return _with_fallback(new, "/".join(parts))


def report_path(*parts: str) -> Path:
    """
    Canonical path under ``reports/``.

    Falls back to ``outputs/<parts>`` if the new path does not exist yet.
    Use for: val_eval, v3_annotation_qa, hydra_runs.
    """
    new = REPORTS_DIR.joinpath(*parts)
    return _with_fallback(new, "/".join(parts))
