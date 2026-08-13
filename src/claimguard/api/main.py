from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from claimguard.api.schemas import ClaimScore, ScoreRequest, ScoreResponse
from claimguard.features.build import MODEL_FEATURES
from claimguard.models.anomaly import ModelArtifact, load_artifact

DEFAULT_MODEL_PATH = "artifacts/isolation_forest.joblib"
app = FastAPI(
    title="ClaimGuard Scoring API",
    version="0.1.0",
    description="Ranks enriched healthcare claims for payment-integrity review.",
)


@lru_cache(maxsize=1)
def get_model() -> ModelArtifact:
    path = Path(os.getenv("CLAIMGUARD_MODEL_PATH", DEFAULT_MODEL_PATH))
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}")
    return load_artifact(path)


def _reason_codes(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    if row["provider_paid_zscore"] >= 3:
        reasons.append("HIGH_PROVIDER_RELATIVE_COST")
    if row["units"] >= 20:
        reasons.append("EXCESS_UNITS")
    if row["duplicate_indicator"] == 1:
        reasons.append("POSSIBLE_DUPLICATE")
    if row["provider_claim_count_30d"] >= 40:
        reasons.append("HIGH_PROVIDER_UTILIZATION")
    if row["beneficiary_claim_count_30d"] >= 12:
        reasons.append("HIGH_BENEFICIARY_UTILIZATION")
    return reasons or ["MULTIVARIATE_OUTLIER"]


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

    frame = pd.DataFrame([claim.model_dump() for claim in request.claims])
    frame["paid_amount_log"] = np.log1p(frame["paid_amount"])
    scores = -artifact.pipeline.decision_function(frame[MODEL_FEATURES])
    # This API threshold follows the trained model's contamination decision boundary.
    predictions = artifact.pipeline.predict(frame[MODEL_FEATURES])
    results = [
        ClaimScore(
            claim_id=str(row["claim_id"]),
            anomaly_score=round(float(anomaly_score), 6),
            is_flagged=bool(prediction == -1),
            reason_codes=_reason_codes(row),
        )
        for (_, row), anomaly_score, prediction in zip(
            frame.iterrows(), scores, predictions, strict=True
        )
    ]
    return ScoreResponse(model_version=artifact.version, scores=results)

