"""
calibration.py — Heuristic confidence calibration for Enterprise GNN RCA.

calibrate_confidence(raw_confidence, rca_source, impacted_diagrams,
                     top_candidates, evidence_summary) -> dict

Applies temperature-like scaling rules to produce a calibrated confidence
value and a calibration band. Purely rule-based — no LLM involved.
"""
from __future__ import annotations

_THRESHOLD: float = 0.75


def calibrate_confidence(
    raw_confidence: float,
    rca_source: str = "",
    impacted_diagrams: list[str] | None = None,
    top_candidates: list[dict] | None = None,
    evidence_summary: list[str] | None = None,
) -> dict:
    """
    Calibrate raw GNN/graph RCA confidence using heuristic rules.

    Returns
    -------
    dict with:
      raw_confidence        : float
      calibrated_confidence : float  (clamped to [0.05, 1.0])
      confidence_band       : "low" | "medium" | "high" | "very_high"
      calibration_method    : "heuristic_temperature_scaling"
      threshold_passed      : bool   (calibrated_confidence >= 0.75)
      threshold             : float  (0.75)
      explanation           : list[str]
    """
    impacted   = list(impacted_diagrams or [])
    candidates = list(top_candidates or [])
    evidence   = list(evidence_summary or [])
    src        = (rca_source or "").lower()

    explanation: list[str] = []
    cal = float(raw_confidence or 0.0)

    # Rule 1: Missing / zero → low
    if cal <= 0.0:
        cal = 0.10
        explanation.append("No confidence value provided — defaulting to 0.10 (low).")

    # Rule 2: Source quality modifier
    is_gnn      = "gnn" in src or "enterprise gnn" in src
    is_fallback = any(k in src for k in ("fallback", "error", "scenario_grounded"))

    if is_fallback and not is_gnn:
        if cal > 0.65:
            cal = 0.65
            explanation.append(
                "Capped at 0.65 — RCA is graph-grounded fallback, not Enterprise GNN RCA."
            )
        else:
            explanation.append("Fallback RCA source — confidence already within cap.")
    elif is_gnn:
        explanation.append("Enterprise GNN RCA source — full confidence range allowed.")
    else:
        explanation.append(f"RCA source: {rca_source or '(unknown)'}.")

    # Rule 3: Top candidate margin
    if len(candidates) >= 2:
        s1     = float(candidates[0].get("score", 0))
        s2     = float(candidates[1].get("score", 0))
        margin = s1 - s2
        if margin >= 0.20:
            boost = min(0.06, margin * 0.12)
            cal   = min(1.0, cal + boost)
            explanation.append(
                f"High candidate margin ({margin:.2f}) → boosted by {boost:.2f}."
            )
        elif margin < 0.05:
            cal = max(0.10, cal - 0.08)
            explanation.append(
                f"Low candidate margin ({margin:.2f}) — penalised 0.08."
            )
    elif len(candidates) == 1:
        explanation.append("Single GNN candidate — no margin comparison possible.")

    # Rule 4: Impacted diagram count
    n_imp = len(impacted)
    if n_imp == 0:
        cal = max(0.10, cal - 0.05)
        explanation.append("No impacted diagrams identified — penalised 0.05.")
    elif n_imp >= 3:
        explanation.append(
            f"{n_imp} diagrams impacted — wide blast radius; approval risk elevated."
        )

    # Rule 5: Evidence density
    n_ev = len(evidence)
    if n_ev < 2:
        cal = max(0.10, cal - 0.05)
        explanation.append(f"Sparse evidence ({n_ev} item(s)) — penalised 0.05.")
    elif n_ev >= 5:
        explanation.append(f"Rich evidence ({n_ev} item(s)) — no evidence penalty.")

    # Clamp
    cal = round(min(1.0, max(0.05, cal)), 4)

    # Band
    if cal >= 0.85:
        band = "very_high"
    elif cal >= 0.75:
        band = "high"
    elif cal >= 0.50:
        band = "medium"
    else:
        band = "low"

    return {
        "raw_confidence":        round(float(raw_confidence or 0.0), 4),
        "calibrated_confidence": cal,
        "confidence_band":       band,
        "calibration_method":    "heuristic_temperature_scaling",
        "threshold_passed":      cal >= _THRESHOLD,
        "threshold":             _THRESHOLD,
        "explanation":           explanation,
    }
