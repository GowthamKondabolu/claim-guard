from collections.abc import Generator

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from claimguard.features.build import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_claim_features,
)
from claimguard.features.spark_build import build_claim_features_spark


@pytest.fixture(scope="module")
def spark() -> Generator[SparkSession, None, None]:
    session = (
        SparkSession.builder.master("local[2]")
        .appName("claim-guard-tests")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def sample_claims() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_id": "C1",
                "beneficiary_id": "B1",
                "provider_id": "P1",
                "claim_type": "outpatient",
                "service_date": "2024-01-01",
                "paid_amount": 100.0,
                "diagnosis_count": 1,
                "procedure_count": 1,
                "units": 1,
                "length_of_stay": 0,
            },
            {
                "claim_id": "C2",
                "beneficiary_id": "B2",
                "provider_id": "P1",
                "claim_type": "outpatient",
                "service_date": "2024-01-15",
                "paid_amount": 200.0,
                "diagnosis_count": 2,
                "procedure_count": 1,
                "units": 1,
                "length_of_stay": 0,
            },
            {
                "claim_id": "C3",
                "beneficiary_id": "B3",
                "provider_id": "P1",
                "claim_type": "inpatient",
                "service_date": "2024-02-05",
                "paid_amount": 300.0,
                "diagnosis_count": 3,
                "procedure_count": 2,
                "units": 2,
                "length_of_stay": 3,
            },
            {
                "claim_id": "C4",
                "beneficiary_id": "B1",
                "provider_id": "P2",
                "claim_type": "outpatient",
                "service_date": "2024-01-20",
                "paid_amount": 150.0,
                "diagnosis_count": 1,
                "procedure_count": 1,
                "units": 1,
                "length_of_stay": 0,
            },
            {
                "claim_id": "C5",
                "beneficiary_id": "B1",
                "provider_id": "P2",
                "claim_type": "outpatient",
                "service_date": "2024-02-25",
                "paid_amount": 250.0,
                "diagnosis_count": 2,
                "procedure_count": 1,
                "units": 1,
                "length_of_stay": 0,
            },
        ]
    )


def test_spark_features_match_pandas_features(
    spark: SparkSession,
) -> None:
    claims = sample_claims()

    pandas_result = (
        build_claim_features(claims)
        .sort_values("claim_id")
        .reset_index(drop=True)
    )
    spark_result = (
        build_claim_features_spark(spark.createDataFrame(claims))
        .orderBy("claim_id")
        .toPandas()
    )

    np.testing.assert_allclose(
        spark_result[NUMERIC_FEATURES].to_numpy(dtype=float),
        pandas_result[NUMERIC_FEATURES].to_numpy(dtype=float),
        rtol=1e-7,
        atol=1e-7,
    )
    assert (
        spark_result[CATEGORICAL_FEATURES].to_numpy().tolist()
        == pandas_result[CATEGORICAL_FEATURES].to_numpy().tolist()
    )


def test_spark_window_excludes_claims_older_than_30_days(
    spark: SparkSession,
) -> None:
    result = (
        build_claim_features_spark(
            spark.createDataFrame(sample_claims())
        )
        .select(
            "claim_id",
            "provider_claim_count_30d",
            "beneficiary_claim_count_30d",
        )
        .orderBy("claim_id")
        .toPandas()
        .set_index("claim_id")
    )

    assert result.loc["C3", "provider_claim_count_30d"] == 2
    assert result.loc["C5", "beneficiary_claim_count_30d"] == 1


def test_spark_duplicate_indicator_marks_all_matches(
    spark: SparkSession,
) -> None:
    claims = sample_claims().iloc[[0, 0]].copy().reset_index(drop=True)
    claims["claim_id"] = ["D1", "D2"]

    indicators = (
        build_claim_features_spark(spark.createDataFrame(claims))
        .select("duplicate_indicator")
        .toPandas()["duplicate_indicator"]
    )

    assert indicators.eq(1).all()


def test_spark_builder_rejects_missing_columns(
    spark: SparkSession,
) -> None:
    claims = sample_claims().drop(columns=["provider_id"])

    with pytest.raises(ValueError, match="provider_id"):
        build_claim_features_spark(spark.createDataFrame(claims))