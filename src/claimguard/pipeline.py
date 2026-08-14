from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from claimguard.config import Settings
from claimguard.data.synthetic import generate_synthetic_claims
from claimguard.evaluation import evaluate_injected_anomalies
from claimguard.features.build import build_claim_features
from claimguard.models.anomaly import save_artifact, score_featured_claims, train_anomaly_model
from claimguard.models.ensemble import combine_model_and_rule_scores
from claimguard.models.rules import score_rule_baseline


def run_claims_pipeline(
    claims: pd.DataFrame,
    settings: Settings,
) -> dict[str, float | int]:
    raw_path = Path(settings.raw_data_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    claims.to_csv(raw_path, index=False)

    featured = build_claim_features(claims)
    artifact = train_anomaly_model(
        featured,
        contamination=settings.model_contamination,
        seed=settings.seed,
    )
    model_scored = score_featured_claims(
        artifact,
        featured,
        flag_rate=settings.flag_rate,
    )
    rule_scored = score_rule_baseline(model_scored)
    scored = combine_model_and_rule_scores(
        rule_scored,
        model_weight=settings.ensemble_model_weight,
        flag_rate=settings.flag_rate,
    )
    metrics = evaluate_injected_anomalies(scored, top_k=settings.top_k)

    scored_path = Path(settings.scored_data_path)
    scored_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(scored_path, index=False)
    save_artifact(artifact, settings.model_path)

    metrics_path = Path(settings.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    return metrics


def run_training_pipeline(settings: Settings) -> dict[str, float | int]:
    claims = generate_synthetic_claims(
        rows=settings.synthetic_rows,
        anomaly_rate=settings.injected_anomaly_rate,
        seed=settings.seed,
    )
    return run_claims_pipeline(claims, settings)


def run_csv_training_pipeline(
    input_path: str | Path,
    settings: Settings,
) -> dict[str, float | int]:
    claims = pd.read_csv(input_path, low_memory=False)
    return run_claims_pipeline(claims, settings)
