import pandas as pd
import pytest

from claimguard.models.ensemble import combine_model_and_rule_scores


def _score_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "claim_id": ["RULE_ONLY", "MODEL_ONLY", "BALANCED", "MIXED"],
            "model_anomaly_score": [0.1, 0.9, 0.4, 0.8],
            "rule_score": [1.0, 0.0, 0.5, 0.2],
        }
    )


def test_ensemble_combines_model_percentile_and_rule_score() -> None:
    scored = combine_model_and_rule_scores(
        _score_frame(),
        model_weight=0.70,
        flag_rate=0.20,
    )

    assert scored["claim_id"].tolist() == [
        "MODEL_ONLY",
        "MIXED",
        "BALANCED",
        "RULE_ONLY",
    ]
    assert scored.iloc[0]["ensemble_score"] == pytest.approx(0.70)
    assert scored.iloc[0]["model_score_percentile"] == 1.0
    assert scored["is_flagged"].sum() == 1
    assert scored["anomaly_score"].equals(scored["ensemble_score"])


def test_ensemble_can_rank_using_rules_only() -> None:
    scored = combine_model_and_rule_scores(
        _score_frame(),
        model_weight=0.0,
        flag_rate=0.20,
    )

    assert scored.iloc[0]["claim_id"] == "RULE_ONLY"
    assert scored.iloc[0]["ensemble_score"] == 1.0


@pytest.mark.parametrize(
    ("model_weight", "flag_rate"),
    [
        (-0.1, 0.05),
        (1.1, 0.05),
        (0.7, 0.0),
        (0.7, 0.25),
    ],
)
def test_ensemble_rejects_invalid_configuration(
    model_weight: float,
    flag_rate: float,
) -> None:
    with pytest.raises(ValueError):
        combine_model_and_rule_scores(
            _score_frame(),
            model_weight=model_weight,
            flag_rate=flag_rate,
        )


def test_ensemble_rejects_missing_score_columns() -> None:
    with pytest.raises(ValueError, match="Missing ensemble score columns"):
        combine_model_and_rule_scores(
            pd.DataFrame({"claim_id": ["CLM-1"]}),
        )


def test_ensemble_rejects_rule_scores_outside_unit_interval() -> None:
    frame = _score_frame()
    frame.loc[0, "rule_score"] = 1.2

    with pytest.raises(ValueError, match="rule_score must be between 0 and 1"):
        combine_model_and_rule_scores(frame)