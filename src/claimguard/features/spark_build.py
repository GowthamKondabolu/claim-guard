from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from claimguard.data.validation import REQUIRED_COLUMNS

DUPLICATE_FIELDS = [
    "beneficiary_id",
    "provider_id",
    "claim_type",
    "service_date",
    "paid_amount",
    "diagnosis_count",
    "procedure_count",
    "units",
]

SECONDS_PER_DAY = 86_400


def _validate_required_columns(claims: DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(claims.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_claim_features_spark(
    claims: DataFrame,
    days: int = 30,
) -> DataFrame:
    """Build ClaimGuard model features with distributed Spark transformations."""
    if days <= 0:
        raise ValueError("days must be greater than zero")

    _validate_required_columns(claims)

    featured = (
        claims.withColumn("service_date", F.to_timestamp("service_date"))
        .withColumn("paid_amount", F.col("paid_amount").cast("double"))
        .withColumn("diagnosis_count", F.col("diagnosis_count").cast("long"))
        .withColumn("procedure_count", F.col("procedure_count").cast("long"))
        .withColumn("units", F.col("units").cast("long"))
        .withColumn("length_of_stay", F.col("length_of_stay").cast("long"))
        .withColumn("_service_epoch", F.col("service_date").cast("long"))
    )

    window_seconds = days * SECONDS_PER_DAY

    provider_window = (
        Window.partitionBy("provider_id")
        .orderBy("_service_epoch")
        .rangeBetween(-window_seconds, 0)
    )
    beneficiary_window = (
        Window.partitionBy("beneficiary_id")
        .orderBy("_service_epoch")
        .rangeBetween(-window_seconds, 0)
    )
    duplicate_window = Window.partitionBy(*DUPLICATE_FIELDS)
    provider_statistics = Window.partitionBy("provider_id")

    featured = (
        featured.withColumn(
            "paid_amount_log",
            F.log1p(F.col("paid_amount")),
        )
        .withColumn(
            "weekend_service",
            F.dayofweek("service_date").isin(1, 7).cast("int"),
        )
        .withColumn(
            "duplicate_indicator",
            (F.count(F.lit(1)).over(duplicate_window) > 1).cast("int"),
        )
        .withColumn(
            "provider_claim_count_30d",
            F.count(F.lit(1)).over(provider_window).cast("long"),
        )
        .withColumn(
            "beneficiary_claim_count_30d",
            F.count(F.lit(1)).over(beneficiary_window).cast("long"),
        )
        .withColumn(
            "_provider_paid_mean",
            F.avg("paid_amount").over(provider_statistics),
        )
        .withColumn(
            "_provider_paid_std",
            F.stddev_samp("paid_amount").over(provider_statistics),
        )
        .withColumn(
            "provider_paid_zscore",
            F.when(
                F.col("_provider_paid_std").isNull()
                | (F.col("_provider_paid_std") == 0),
                F.lit(0.0),
            ).otherwise(
                (
                    F.col("paid_amount")
                    - F.col("_provider_paid_mean")
                )
                / F.col("_provider_paid_std")
            ),
        )
    )

    return featured.drop(
        "_service_epoch",
        "_provider_paid_mean",
        "_provider_paid_std",
    )