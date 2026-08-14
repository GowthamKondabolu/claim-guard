import pandas as pd
import pytest

from claimguard.evaluation import (
    compare_ranking_strategies,
    evaluate_injected_anomalies,
    evaluate_score_ranking,
)


def _scored_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "claim_id": ["A", "B", "C", "D"],
            "is_injected_anomaly": [1, 0, 1, 0],
            "model_anomaly_score": [0.9, 0.8, 0.2, 0.1],
            "rule_score": [0.9, 0.1, 0.8, 0.2],
            "ensemble_score": [0.95, 0.4, 0.85, 0.1],
            "anomaly_score": [0.95, 0.4, 0.85, 0.1],
            "is_flagged": [1, 0, 1, 0],
        }
    )


def test_comparison_reports_each_ranking_strategy() -> None:
    comparison = compare_ranking_strategies(_scored_frame(), top_k=2)

    assert set(comparison) == {"isolation_forest", "rules", "ensemble"}
    assert comparison["isolation_forest"]["precision_at_k"] == 0.5
    assert comparison["isolation_forest"]["recall_at_k"] == 0.5
    assert comparison["rules"]["precision_at_k"] == 1.0
    assert comparison["rules"]["recall_at_k"] == 1.0
    assert comparison["ensemble"]["precision_at_k"] == 1.0
    assert comparison["ensemble"]["recall_at_k"] == 1.0


def test_primary_evaluation_uses_ensemble_contract() -> None:
    metrics = evaluate_injected_anomalies(_scored_frame(), top_k=2)

    assert metrics["precision_at_k"] == 1.0
    assert metrics["recall_at_k"] == 1.0
    assert metrics["flagged_claims"] == 2


def test_comparison_is_unavailable_without_evaluation_labels() -> None:
    frame = _scored_frame().drop(columns=["is_injected_anomaly"])

    assert compare_ranking_strategies(frame, top_k=2) == {}


def test_score_evaluation_rejects_missing_score_column() -> None:
    frame = pd.DataFrame({"is_injected_anomaly": [0, 1]})

    with pytest.raises(ValueError, match="Missing score column"):
        evaluate_score_ranking(frame, score_column="missing_score")