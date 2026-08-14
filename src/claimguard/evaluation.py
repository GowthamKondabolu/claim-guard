from __future__ import annotations

import math

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate_score_ranking(
    scored_claims: pd.DataFrame,
    score_column: str,
    top_k: int = 200,
) -> dict[str, float | int]:
    if "is_injected_anomaly" not in scored_claims:
        return {
            "evaluation_available": 0,
            "rows": len(scored_claims),
        }
    if score_column not in scored_claims:
        raise ValueError(f"Missing score column: {score_column}")

    labels = scored_claims["is_injected_anomaly"].astype(int)
    scores = scored_claims[score_column].astype(float)
    k = min(max(1, top_k), len(scored_claims))
    top = scored_claims.nlargest(k, score_column)
    positives = int(labels.sum())
    true_positives_at_k = int(top["is_injected_anomaly"].sum())

    roc_auc = roc_auc_score(labels, scores) if labels.nunique() > 1 else math.nan
    average_precision = average_precision_score(labels, scores) if positives else math.nan
    return {
        "evaluation_available": 1,
        "rows": len(scored_claims),
        "injected_anomalies": positives,
        "top_k": k,
        "true_positives_at_k": true_positives_at_k,
        "precision_at_k": round(true_positives_at_k / k, 6),
        "recall_at_k": round(true_positives_at_k / positives, 6) if positives else math.nan,
        "roc_auc": round(float(roc_auc), 6),
        "average_precision": round(float(average_precision), 6),
    }


def evaluate_injected_anomalies(
    scored_claims: pd.DataFrame,
    top_k: int = 200,
) -> dict[str, float | int]:
    metrics = evaluate_score_ranking(
        scored_claims,
        score_column="anomaly_score",
        top_k=top_k,
    )
    metrics["flagged_claims"] = int(scored_claims["is_flagged"].sum())
    return metrics


def compare_ranking_strategies(
    scored_claims: pd.DataFrame,
    top_k: int = 200,
) -> dict[str, dict[str, float | int]]:
    if "is_injected_anomaly" not in scored_claims:
        return {}

    score_columns = {
        "isolation_forest": "model_anomaly_score",
        "rules": "rule_score",
        "ensemble": "ensemble_score",
    }
    return {
        name: evaluate_score_ranking(
            scored_claims,
            score_column=score_column,
            top_k=top_k,
        )
        for name, score_column in score_columns.items()
    }