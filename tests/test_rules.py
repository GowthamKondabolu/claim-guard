import pandas as pd
import pytest

from claimguard.models.rules import RuleThresholds, score_rule_baseline


def _rule_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "claim_id": ["ALL", "NONE", "PARTIAL"],
            "provider_paid_zscore": [4.0, 0.2, 3.0],
            "units": [25, 2, 5],
            "duplicate_indicator": [1, 0, 0],
            "provider_claim_count_30d": [50, 5, 39],
            "beneficiary_claim_count_30d": [13, 2, 12],
        }
    )


def test_rule_baseline_scores_all_triggered_rules() -> None:
    scored = score_rule_baseline(_rule_frame())

    row = scored.loc[scored["claim_id"] == "ALL"].iloc[0]
    assert row["rule_score"] == 1.0
    assert row["rule_reason_codes"] == [
        "HIGH_PROVIDER_RELATIVE_COST",
        "EXCESS_UNITS",
        "POSSIBLE_DUPLICATE",
        "HIGH_PROVIDER_UTILIZATION",
        "HIGH_BENEFICIARY_UTILIZATION",
    ]


def test_rule_baseline_returns_zero_when_no_rules_trigger() -> None:
    scored = score_rule_baseline(_rule_frame())

    row = scored.loc[scored["claim_id"] == "NONE"].iloc[0]
    assert row["rule_score"] == 0.0
    assert row["rule_reason_codes"] == []


def test_rule_thresholds_are_configurable() -> None:
    thresholds = RuleThresholds(
        provider_paid_zscore=10.0,
        units=100,
        provider_claim_count_30d=100,
        beneficiary_claim_count_30d=100,
    )

    scored = score_rule_baseline(_rule_frame(), thresholds)

    row = scored.loc[scored["claim_id"] == "ALL"].iloc[0]
    assert row["rule_score"] == 0.25
    assert row["rule_reason_codes"] == ["POSSIBLE_DUPLICATE"]


def test_rule_baseline_rejects_missing_features() -> None:
    with pytest.raises(ValueError, match="Missing rule features"):
        score_rule_baseline(pd.DataFrame({"units": [1]}))