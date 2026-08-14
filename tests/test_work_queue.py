import pandas as pd
import pytest

from claimguard.work_queue import (
    QUEUE_REQUIRED_COLUMNS,
    build_demo_work_queue,
    filter_work_queue,
    prepare_work_queue,
    summarize_work_queue,
)


def _queue_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "claim_id": ["A", "B", "C"],
            "claim_type": ["outpatient", "carrier", "pharmacy"],
            "paid_amount": [100.0, 200.0, 300.0],
            "model_score_percentile": [0.90, 0.40, 0.20],
            "rule_score": [0.30, 0.00, 0.25],
            "ensemble_score": [0.84, 0.36, 0.205],
            "is_flagged": [1, "false", True],
            "rule_reason_codes": [
                "['EXCESS_UNITS']",
                "[]",
                "POSSIBLE_DUPLICATE|EXCESS_UNITS",
            ],
            "is_injected_anomaly": [1, 0, 1],
        }
    )


def test_prepare_work_queue_parses_flags_and_reason_codes() -> None:
    queue = prepare_work_queue(_queue_frame())

    assert queue["claim_id"].tolist() == ["A", "B", "C"]
    assert queue["is_flagged"].tolist() == [True, False, True]
    assert queue.iloc[0]["rule_reason_codes"] == ["EXCESS_UNITS"]
    assert queue.iloc[1]["reason_codes_display"] == "MULTIVARIATE_OUTLIER"
    assert queue.iloc[2]["rule_reason_codes"] == [
        "POSSIBLE_DUPLICATE",
        "EXCESS_UNITS",
    ]


def test_filter_work_queue_combines_investigator_filters() -> None:
    queue = prepare_work_queue(_queue_frame())

    filtered = filter_work_queue(
        queue,
        claim_types=["pharmacy"],
        minimum_score=0.20,
        flagged_only=True,
        reason_codes=["POSSIBLE_DUPLICATE"],
    )

    assert filtered["claim_id"].tolist() == ["C"]


def test_summarize_work_queue_reports_review_capacity() -> None:
    queue = prepare_work_queue(_queue_frame())

    summary = summarize_work_queue(queue)

    assert summary["claims"] == 3
    assert summary["flagged_claims"] == 2
    assert summary["review_paid_amount"] == 400.0
    assert summary["injected_scenarios"] == 2
    assert summary["flagged_injected_scenarios"] == 2


def test_prepare_work_queue_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="Missing work-queue columns"):
        prepare_work_queue(pd.DataFrame({"claim_id": ["A"]}))


def test_demo_work_queue_is_ranked_and_reviewable() -> None:
    queue = build_demo_work_queue(rows=200, seed=7)

    assert len(queue) == 200
    assert set(QUEUE_REQUIRED_COLUMNS).issubset(queue.columns)
    assert queue["ensemble_score"].is_monotonic_decreasing
    assert queue["ensemble_score"].between(0.0, 1.0).all()
    assert 1 <= queue["is_flagged"].sum() < len(queue)