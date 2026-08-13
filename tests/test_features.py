import pandas as pd

from claimguard.data.synthetic import generate_synthetic_claims
from claimguard.features.build import MODEL_FEATURES, build_claim_features


def test_feature_builder_creates_complete_model_matrix() -> None:
    claims = generate_synthetic_claims(rows=250, anomaly_rate=0.04, seed=7)
    featured = build_claim_features(claims)

    assert set(MODEL_FEATURES).issubset(featured.columns)
    assert featured[MODEL_FEATURES].isna().sum().sum() == 0
    assert featured["provider_claim_count_30d"].min() >= 1
    assert featured["beneficiary_claim_count_30d"].min() >= 1


def test_duplicate_indicator_detects_exact_duplicate_pattern() -> None:
    claims = generate_synthetic_claims(rows=150, anomaly_rate=0.02, seed=9)
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
    claims.loc[1, fields] = claims.loc[0, fields].to_numpy()
    featured = build_claim_features(claims)

    duplicate_rows = featured.loc[featured["claim_id"].isin([claims.loc[0, "claim_id"], claims.loc[1, "claim_id"]])]
    assert duplicate_rows["duplicate_indicator"].eq(1).all()


def test_service_date_is_normalized_to_datetime() -> None:
    claims = generate_synthetic_claims(rows=100, anomaly_rate=0.02, seed=10)
    claims["service_date"] = claims["service_date"].astype(str)
    featured = build_claim_features(claims)
    assert pd.api.types.is_datetime64_any_dtype(featured["service_date"])

