from __future__ import annotations

import numpy as np
import pandas as pd


def combine_model_and_rule_scores(
    scored_claims: pd.DataFrame,
    model_weight: float = 0.70,
    flag_rate: float = 0.05,
) -> pd.DataFrame:
    if not 0.0 <= model_weight <= 1.0:
        raise ValueError("model_weight must be between 0 and 1")
    if not 0 < flag_rate < 0.25:
        raise ValueError("flag_rate must be between 0 and 0.25")
    if scored_claims.empty:
        raise ValueError("scored_claims must not be empty")

    required = {"model_anomaly_score", "rule_score"}
    missing = sorted(required - set(scored_claims.columns))
    if missing:
        raise ValueError(f"Missing ensemble score columns: {missing}")

    scored = scored_claims.copy()
    model_scores = pd.to_numeric(scored["model_anomaly_score"], errors="coerce")
    rule_scores = pd.to_numeric(scored["rule_score"], errors="coerce")

    if model_scores.isna().any() or rule_scores.isna().any():
        raise ValueError("Ensemble score columns must contain numeric values")
    if not rule_scores.between(0.0, 1.0).all():
        raise ValueError("rule_score must be between 0 and 1")

    rule_weight = 1.0 - model_weight
    model_percentiles = model_scores.rank(method="average", pct=True)

    scored["model_score_percentile"] = model_percentiles.round(6)
    scored["ensemble_score"] = (
        model_weight * model_percentiles + rule_weight * rule_scores
    ).round(6)

    threshold = float(np.quantile(scored["ensemble_score"], 1 - flag_rate))
    scored["is_flagged"] = (scored["ensemble_score"] >= threshold).astype(int)
    scored["score_percentile"] = scored["ensemble_score"].rank(pct=True).round(6)

    # Preserve the existing evaluation contract while ranking by the ensemble.
    scored["anomaly_score"] = scored["ensemble_score"]
    return scored.sort_values("ensemble_score", ascending=False).reset_index(drop=True)