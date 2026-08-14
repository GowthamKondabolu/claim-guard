from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from claimguard.api.schemas import ClaimScore, ScoreRequest, ScoreResponse
from claimguard.features.build import MODEL_FEATURES
from claimguard.models.anomaly import (
    ModelArtifact,
    load_artifact,
    model_score_percentiles,
)
from claimguard.models.rules import score_rule_baseline

DEFAULT_MODEL_PATH = "artifacts/isolation_forest.joblib"
app = FastAPI(
    title="ClaimGuard Scoring API",
    version="0.2.0",
    description="Ranks enriched healthcare claims using explainable ensemble scoring.",
)


@lru_cache(maxsize=1)
def get_model() -> ModelArtifact:
    path = Path(os.getenv("CLAIMGUARD_MODEL_PATH", DEFAULT_MODEL_PATH))
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}")
    return load_artifact(path)


@app.get("/health")
def health() -> dict[str, str]:
    try:
        artifact = get_model()
    except FileNotFoundError:
        return {"status": "degraded", "model": "not_loaded"}
    return {"status": "ok", "model": artifact.version}


@app.post("/v1/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    try:
        artifact = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if artifact.ensemble_threshold is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact does not contain an ensemble threshold",
        )

    frame = pd.DataFrame([claim.model_dump() for claim in request.claims])
    frame["paid_amount_log"] = np.log1p(frame["paid_amount"])

    raw_scores = -artifact.pipeline.decision_function(frame[MODEL_FEATURES])
    model_percentiles = model_score_percentiles(artifact, raw_scores)

    rule_scored = score_rule_baseline(frame)
    rule_scores = rule_scored["rule_score"].to_numpy(dtype=float)
    rule_weight = 1.0 - artifact.ensemble_model_weight
    ensemble_scores = (
        artifact.ensemble_model_weight * model_percentiles
        + rule_weight * rule_scores
    )

    results = [
        ClaimScore(
            claim_id=str(row["claim_id"]),
            model_anomaly_score=round(float(model_anomaly_score), 6),
            model_score_percentile=round(float(model_percentile), 6),
            rule_score=round(float(rule_score), 6),
            ensemble_score=round(float(ensemble_score), 6),
            anomaly_score=round(float(ensemble_score), 6),
            is_flagged=bool(ensemble_score >= artifact.ensemble_threshold),
            reason_codes=list(row["rule_reason_codes"])
            or ["MULTIVARIATE_OUTLIER"],
        )
        for (
            (_, row),
            model_anomaly_score,
            model_percentile,
            rule_score,
            ensemble_score,
        ) in zip(
            rule_scored.iterrows(),
            raw_scores,
            model_percentiles,
            rule_scores,
            ensemble_scores,
            strict=True,
        )
    ]
    return ScoreResponse(model_version=artifact.version, scores=results)