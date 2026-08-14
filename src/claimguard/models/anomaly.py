from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from claimguard.features.build import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES


@dataclass
class ModelArtifact:
    pipeline: Pipeline
    feature_names: list[str]
    trained_at: str
    score_reference: list[float]
    ensemble_model_weight: float = 0.90
    ensemble_threshold: float | None = None
    version: str = "0.2.0"


def train_anomaly_model(
    featured_claims: pd.DataFrame,
    contamination: float = 0.05,
    seed: int = 42,
    ensemble_model_weight: float = 0.90,
) -> ModelArtifact:
    if not 0 < contamination < 0.25:
        raise ValueError("contamination must be between 0 and 0.25")
    if not 0.0 <= ensemble_model_weight <= 1.0:
        raise ValueError("ensemble_model_weight must be between 0 and 1")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                IsolationForest(
                    n_estimators=300,
                    contamination=contamination,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(featured_claims[MODEL_FEATURES])
    training_scores = -pipeline.decision_function(featured_claims[MODEL_FEATURES])

    return ModelArtifact(
        pipeline=pipeline,
        feature_names=MODEL_FEATURES,
        trained_at=datetime.now(UTC).isoformat(),
        score_reference=np.sort(training_scores).astype(float).tolist(),
        ensemble_model_weight=ensemble_model_weight,
    )


def model_score_percentiles(
    artifact: ModelArtifact,
    raw_scores: np.ndarray,
) -> np.ndarray:
    reference = np.asarray(artifact.score_reference, dtype=float)
    if reference.size == 0:
        raise ValueError("Model artifact has no score reference distribution")

    percentiles = np.searchsorted(
        reference,
        np.asarray(raw_scores, dtype=float),
        side="right",
    )
    return percentiles.astype(float) / reference.size


def score_featured_claims(
    artifact: ModelArtifact,
    featured_claims: pd.DataFrame,
    flag_rate: float = 0.05,
) -> pd.DataFrame:
    if not 0 < flag_rate < 0.25:
        raise ValueError("flag_rate must be between 0 and 0.25")

    scored = featured_claims.copy()
    raw_scores = -artifact.pipeline.decision_function(scored[artifact.feature_names])
    scored["model_anomaly_score"] = raw_scores

    threshold = float(np.quantile(raw_scores, 1 - flag_rate))
    scored["model_is_flagged"] = (
        scored["model_anomaly_score"] >= threshold
    ).astype(int)
    scored["model_score_percentile"] = np.round(
        model_score_percentiles(artifact, raw_scores),
        6,
    )
    return scored.sort_values(
        "model_anomaly_score",
        ascending=False,
    ).reset_index(drop=True)


def save_artifact(artifact: ModelArtifact, path: str | Path) -> None:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)


def load_artifact(path: str | Path) -> ModelArtifact:
    artifact = joblib.load(Path(path))
    if not isinstance(artifact, ModelArtifact):
        raise TypeError("Unexpected model artifact type")
    return artifact