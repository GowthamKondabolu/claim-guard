from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Settings:
    seed: int = 42
    synthetic_rows: int = 5_000
    injected_anomaly_rate: float = 0.04
    model_contamination: float = 0.05
    flag_rate: float = 0.05
    ensemble_model_weight: float = 0.70
    top_k: int = 200
    raw_data_path: str = "data/raw/synthetic_claims.csv"
    scored_data_path: str = "data/processed/scored_claims.csv"
    model_path: str = "artifacts/isolation_forest.joblib"
    metrics_path: str = "artifacts/metrics.json"


def load_settings(path: str | Path = "configs/base.yaml") -> Settings:
    config_path = Path(path)
    if not config_path.exists():
        return Settings()
    with config_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    return Settings(**payload)

