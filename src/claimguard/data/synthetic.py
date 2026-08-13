from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

CLAIM_TYPES = np.array(["carrier", "inpatient", "outpatient", "pharmacy"])


def generate_synthetic_claims(
    rows: int = 5_000,
    anomaly_rate: float = 0.04,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate privacy-safe claims and inject known anomalies for offline evaluation.

    The label is evaluation-only and must never be used as a model input.
    """
    if rows < 100:
        raise ValueError("rows must be at least 100")
    if not 0 < anomaly_rate < 0.25:
        raise ValueError("anomaly_rate must be between 0 and 0.25")

    rng = np.random.default_rng(seed)
    service_dates = pd.Timestamp(date(2025, 1, 1)) + pd.to_timedelta(
        rng.integers(0, 365, size=rows), unit="D"
    )
    claim_types = rng.choice(CLAIM_TYPES, size=rows, p=[0.42, 0.12, 0.34, 0.12])

    base_paid = {
        "carrier": 220.0,
        "inpatient": 8_500.0,
        "outpatient": 900.0,
        "pharmacy": 140.0,
    }
    paid_amount = np.array(
        [rng.lognormal(mean=np.log(base_paid[claim_type]), sigma=0.65) for claim_type in claim_types]
    )

    claims = pd.DataFrame(
        {
            "claim_id": [f"CLM-{index:08d}" for index in range(rows)],
            "beneficiary_id": [f"BEN-{value:06d}" for value in rng.integers(1, rows // 3, rows)],
            "provider_id": [f"PRV-{value:04d}" for value in rng.integers(1, max(80, rows // 15), rows)],
            "claim_type": claim_types,
            "service_date": service_dates,
            "paid_amount": paid_amount.round(2),
            "diagnosis_count": rng.poisson(2.4, rows).clip(1, 12),
            "procedure_count": rng.poisson(1.7, rows).clip(1, 10),
            "units": rng.poisson(1.3, rows).clip(1, 12),
            "length_of_stay": np.where(
                claim_types == "inpatient", rng.poisson(4.2, rows).clip(1, 30), 0
            ),
            "is_injected_anomaly": 0,
            "anomaly_scenario": "none",
        }
    )

    anomaly_count = max(1, round(rows * anomaly_rate))
    anomaly_indices = rng.choice(claims.index, size=anomaly_count, replace=False)
    scenarios = rng.choice(
        np.array(["high_paid", "excess_units", "duplicate_like", "utilization_spike"]),
        size=anomaly_count,
    )

    claims.loc[anomaly_indices, "is_injected_anomaly"] = 1
    claims.loc[anomaly_indices, "anomaly_scenario"] = scenarios

    for index, scenario in zip(anomaly_indices, scenarios, strict=True):
        if scenario == "high_paid":
            claims.loc[index, "paid_amount"] *= rng.uniform(8, 18)
        elif scenario == "excess_units":
            claims.loc[index, "units"] = int(rng.integers(30, 100))
        elif scenario == "duplicate_like":
            source_index = int(rng.integers(0, rows))
            fields = [
                "beneficiary_id",
                "provider_id",
                "claim_type",
                "service_date",
                "paid_amount",
                "diagnosis_count",
                "procedure_count",
                "units",
                "length_of_stay",
            ]
            claims.loc[index, fields] = claims.loc[source_index, fields].to_numpy()
        else:
            claims.loc[index, "paid_amount"] *= rng.uniform(3, 6)
            claims.loc[index, "procedure_count"] = int(rng.integers(12, 25))
            claims.loc[index, "diagnosis_count"] = int(rng.integers(12, 25))

    claims["paid_amount"] = claims["paid_amount"].astype(float).round(2)
    return claims.sort_values(["service_date", "claim_id"]).reset_index(drop=True)

