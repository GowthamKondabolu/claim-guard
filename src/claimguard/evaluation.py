from __future__ import annotations

import math

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate_injected_anomalies(scored_claims: pd.DataFrame, top_k: int = 200) -> dict[str, float | int]:
    if "is_injected_anomaly" not in scored_claims:
        return {
            "evaluation_available": 0,
            "rows": len(scored_claims),
            "flagged_claims": int(scored_claims["is_flagged"].sum()),
        }

    labels = scored_claims["is_injected_anomaly"].astype(int)
    scores = scored_claims["anomaly_score"].astype(float)
    k = min(max(1, top_k), len(scored_claims))
    top = scored_claims.nlargest(k, "anomaly_score")
    positives = int(labels.sum())
    true_positives_at_k = int(top["is_injected_anomaly"].sum())

    roc_auc = roc_auc_score(labels, scores) if labels.nunique() > 1 else math.nan
    average_precision = average_precision_score(labels, scores) if positives else math.nan
    return {
        "evaluation_available": 1,
        "rows": len(scored_claims),
        "injected_anomalies": positives,
        "flagged_claims": int(scored_claims["is_flagged"].sum()),
        "top_k": k,
        "true_positives_at_k": true_positives_at_k,
        "precision_at_k": round(true_positives_at_k / k, 6),
        "recall_at_k": round(true_positives_at_k / positives, 6) if positives else math.nan,
        "roc_auc": round(float(roc_auc), 6),
        "average_precision": round(float(average_precision), 6),
    }
