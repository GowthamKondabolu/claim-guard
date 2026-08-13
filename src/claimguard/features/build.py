from __future__ import annotations

import numpy as np
import pandas as pd

from claimguard.data.validation import validate_claims

NUMERIC_FEATURES = [
    "paid_amount_log",
    "diagnosis_count",
    "procedure_count",
    "units",
    "length_of_stay",
    "provider_claim_count_30d",
    "beneficiary_claim_count_30d",
    "provider_paid_zscore",
    "duplicate_indicator",
    "weekend_service",
]
CATEGORICAL_FEATURES = ["claim_type"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _rolling_event_count(frame: pd.DataFrame, group_column: str, days: int = 30) -> pd.Series:
    result = pd.Series(index=frame.index, dtype="int64")
    window = np.timedelta64(days, "D")
    for _, group in frame.groupby(group_column, sort=False):
        ordered = group.sort_values("service_date")
        times = ordered["service_date"].to_numpy(dtype="datetime64[ns]")
        left = np.searchsorted(times, times - window, side="left")
        counts = np.arange(len(ordered)) - left + 1
        result.loc[ordered.index] = counts
    return result.astype(int)


def build_claim_features(claims: pd.DataFrame) -> pd.DataFrame:
    validate_claims(claims)
    featured = claims.copy()
    featured["service_date"] = pd.to_datetime(featured["service_date"], errors="raise")
    featured["paid_amount_log"] = np.log1p(featured["paid_amount"].astype(float))
    featured["weekend_service"] = (featured["service_date"].dt.dayofweek >= 5).astype(int)

    duplicate_fields = [
        "beneficiary_id",
        "provider_id",
        "claim_type",
        "service_date",
        "paid_amount",
        "diagnosis_count",
        "procedure_count",
        "units",
    ]
    featured["duplicate_indicator"] = featured.duplicated(
        subset=duplicate_fields, keep=False
    ).astype(int)
    featured["provider_claim_count_30d"] = _rolling_event_count(featured, "provider_id")
    featured["beneficiary_claim_count_30d"] = _rolling_event_count(featured, "beneficiary_id")

    provider_mean = featured.groupby("provider_id")["paid_amount"].transform("mean")
    provider_std = featured.groupby("provider_id")["paid_amount"].transform("std").replace(0, 1)
    featured["provider_paid_zscore"] = (
        (featured["paid_amount"] - provider_mean) / provider_std.fillna(1)
    ).fillna(0)
    return featured

