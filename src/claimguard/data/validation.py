from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "claim_id",
    "beneficiary_id",
    "provider_id",
    "claim_type",
    "service_date",
    "paid_amount",
    "diagnosis_count",
    "procedure_count",
    "units",
    "length_of_stay",
}


def validate_claims(claims: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(claims.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if claims.empty:
        raise ValueError("Claims dataset cannot be empty")
    if claims["claim_id"].isna().any() or claims["claim_id"].duplicated().any():
        raise ValueError("claim_id must be present and unique")
    if (claims["paid_amount"] < 0).any():
        raise ValueError("paid_amount cannot be negative")
    for column in ["diagnosis_count", "procedure_count", "units", "length_of_stay"]:
        if (claims[column] < 0).any():
            raise ValueError(f"{column} cannot be negative")

