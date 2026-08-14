from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RuleThresholds:
    provider_paid_zscore: float = 3.0
    units: int = 20
    provider_claim_count_30d: int = 40
    beneficiary_claim_count_30d: int = 12


RULE_WEIGHTS = {
    "HIGH_PROVIDER_RELATIVE_COST": 0.30,
    "EXCESS_UNITS": 0.20,
    "POSSIBLE_DUPLICATE": 0.25,
    "HIGH_PROVIDER_UTILIZATION": 0.15,
    "HIGH_BENEFICIARY_UTILIZATION": 0.10,
}

REQUIRED_RULE_FEATURES = [
    "provider_paid_zscore",
    "units",
    "duplicate_indicator",
    "provider_claim_count_30d",
    "beneficiary_claim_count_30d",
]


def score_rule_baseline(
    featured_claims: pd.DataFrame,
    thresholds: RuleThresholds | None = None,
) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_RULE_FEATURES) - set(featured_claims.columns))
    if missing:
        raise ValueError(f"Missing rule features: {missing}")

    if not np.isclose(sum(RULE_WEIGHTS.values()), 1.0):
        raise ValueError("Rule weights must sum to 1.0")

    thresholds = thresholds or RuleThresholds()
    scored = featured_claims.copy()

    provider_zscore = pd.to_numeric(scored["provider_paid_zscore"], errors="coerce").fillna(0)
    units = pd.to_numeric(scored["units"], errors="coerce").fillna(0)
    duplicate = pd.to_numeric(scored["duplicate_indicator"], errors="coerce").fillna(0)
    provider_count = pd.to_numeric(
        scored["provider_claim_count_30d"], errors="coerce"
    ).fillna(0)
    beneficiary_count = pd.to_numeric(
        scored["beneficiary_claim_count_30d"], errors="coerce"
    ).fillna(0)

    rule_masks = {
        "HIGH_PROVIDER_RELATIVE_COST": provider_zscore >= thresholds.provider_paid_zscore,
        "EXCESS_UNITS": units >= thresholds.units,
        "POSSIBLE_DUPLICATE": duplicate == 1,
        "HIGH_PROVIDER_UTILIZATION": (
            provider_count >= thresholds.provider_claim_count_30d
        ),
        "HIGH_BENEFICIARY_UTILIZATION": (
            beneficiary_count >= thresholds.beneficiary_claim_count_30d
        ),
    }

    rule_scores = np.zeros(len(scored), dtype=float)
    reason_codes: list[list[str]] = [[] for _ in range(len(scored))]

    for code, mask in rule_masks.items():
        triggered = mask.to_numpy(dtype=bool)
        rule_scores += triggered.astype(float) * RULE_WEIGHTS[code]
        for row_position in np.flatnonzero(triggered):
            reason_codes[int(row_position)].append(code)

    scored["rule_score"] = np.round(np.clip(rule_scores, 0.0, 1.0), 6)
    scored["rule_reason_codes"] = reason_codes
    return scored