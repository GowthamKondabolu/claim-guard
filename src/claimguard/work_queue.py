from __future__ import annotations

import ast
from numbers import Real

import pandas as pd

from claimguard.data.synthetic import generate_synthetic_claims
from claimguard.features.build import build_claim_features
from claimguard.models.anomaly import score_featured_claims, train_anomaly_model
from claimguard.models.ensemble import combine_model_and_rule_scores
from claimguard.models.rules import score_rule_baseline

QUEUE_REQUIRED_COLUMNS = [
    "claim_id",
    "claim_type",
    "paid_amount",
    "model_score_percentile",
    "rule_score",
    "ensemble_score",
    "is_flagged",
    "rule_reason_codes",
]

SCORE_COLUMNS = [
    "model_score_percentile",
    "rule_score",
    "ensemble_score",
]


def _parse_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Real) and not pd.isna(value):
        return bool(int(value))

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid review flag: {value!r}")


def _parse_reason_codes(value: object) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(code) for code in value if str(code).strip()]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    text = str(value).strip()
    if not text or text == "[]":
        return []

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None

    if isinstance(parsed, (list, tuple, set)):
        return [str(code) for code in parsed if str(code).strip()]

    return [
        code.strip()
        for code in text.replace("|", ",").split(",")
        if code.strip()
    ]


def prepare_work_queue(scored_claims: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(QUEUE_REQUIRED_COLUMNS) - set(scored_claims.columns))
    if missing:
        raise ValueError(f"Missing work-queue columns: {missing}")

    queue = scored_claims.copy()
    queue["claim_id"] = queue["claim_id"].astype(str)
    queue["claim_type"] = queue["claim_type"].astype(str)

    queue["paid_amount"] = pd.to_numeric(queue["paid_amount"], errors="raise")
    for column in SCORE_COLUMNS:
        queue[column] = pd.to_numeric(queue[column], errors="raise")
        if not queue[column].between(0.0, 1.0).all():
            raise ValueError(f"{column} must be between 0 and 1")

    queue["is_flagged"] = queue["is_flagged"].map(_parse_flag)
    queue["rule_reason_codes"] = queue["rule_reason_codes"].map(
        _parse_reason_codes
    )
    queue["reason_codes_display"] = queue["rule_reason_codes"].map(
        lambda codes: ", ".join(codes) if codes else "MULTIVARIATE_OUTLIER"
    )

    if "service_date" in queue:
        queue["service_date"] = pd.to_datetime(
            queue["service_date"],
            errors="coerce",
        )
    if "is_injected_anomaly" in queue:
        queue["is_injected_anomaly"] = pd.to_numeric(
            queue["is_injected_anomaly"],
            errors="coerce",
        ).fillna(0).astype(int)

    return queue.sort_values(
        "ensemble_score",
        ascending=False,
    ).reset_index(drop=True)


def build_demo_work_queue(
    rows: int = 2_000,
    seed: int = 42,
    anomaly_rate: float = 0.04,
    model_weight: float = 0.90,
    flag_rate: float = 0.05,
) -> pd.DataFrame:
    if rows < 100:
        raise ValueError("rows must be at least 100")

    claims = generate_synthetic_claims(
        rows=rows,
        anomaly_rate=anomaly_rate,
        seed=seed,
    )
    featured = build_claim_features(claims)
    artifact = train_anomaly_model(
        featured,
        contamination=flag_rate,
        seed=seed,
        ensemble_model_weight=model_weight,
    )
    model_scored = score_featured_claims(
        artifact,
        featured,
        flag_rate=flag_rate,
    )
    rule_scored = score_rule_baseline(model_scored)
    ensemble_scored = combine_model_and_rule_scores(
        rule_scored,
        model_weight=model_weight,
        flag_rate=flag_rate,
    )
    return prepare_work_queue(ensemble_scored)


def available_reason_codes(queue: pd.DataFrame) -> list[str]:
    codes = {
        code
        for reason_codes in queue["rule_reason_codes"]
        for code in reason_codes
    }
    return sorted(codes)


def filter_work_queue(
    queue: pd.DataFrame,
    claim_types: list[str] | None = None,
    minimum_score: float = 0.0,
    flagged_only: bool = True,
    reason_codes: list[str] | None = None,
) -> pd.DataFrame:
    filtered = queue.copy()

    if flagged_only:
        filtered = filtered[filtered["is_flagged"]]
    if claim_types:
        filtered = filtered[filtered["claim_type"].isin(claim_types)]

    filtered = filtered[filtered["ensemble_score"] >= minimum_score]

    if reason_codes:
        selected = set(reason_codes)
        filtered = filtered[
            filtered["rule_reason_codes"].map(
                lambda codes: bool(selected.intersection(codes))
            )
        ]

    return filtered.sort_values(
        "ensemble_score",
        ascending=False,
    ).reset_index(drop=True)


def summarize_work_queue(queue: pd.DataFrame) -> dict[str, float | int]:
    flagged = queue[queue["is_flagged"]]
    average_score = float(queue["ensemble_score"].mean()) if len(queue) else 0.0

    summary: dict[str, float | int] = {
        "claims": len(queue),
        "flagged_claims": int(queue["is_flagged"].sum()),
        "review_paid_amount": round(float(flagged["paid_amount"].sum()), 2),
        "average_ensemble_score": round(average_score, 6),
    }

    if "is_injected_anomaly" in queue:
        summary["injected_scenarios"] = int(
            queue["is_injected_anomaly"].sum()
        )
        summary["flagged_injected_scenarios"] = int(
            flagged["is_injected_anomaly"].sum()
        )

    return summary