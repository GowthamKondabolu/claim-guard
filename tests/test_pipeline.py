import json
from pathlib import Path

from claimguard.config import Settings
from claimguard.models.anomaly import load_artifact
from claimguard.pipeline import run_training_pipeline


def test_training_pipeline_writes_artifacts(tmp_path: Path) -> None:
    settings = Settings(
        seed=11,
        synthetic_rows=500,
        injected_anomaly_rate=0.04,
        model_contamination=0.05,
        flag_rate=0.05,
        top_k=25,
        raw_data_path=str(tmp_path / "raw.csv"),
        scored_data_path=str(tmp_path / "scored.csv"),
        model_path=str(tmp_path / "model.joblib"),
        metrics_path=str(tmp_path / "metrics.json"),
    )

    metrics = run_training_pipeline(settings)

    assert Path(settings.raw_data_path).exists()
    assert Path(settings.scored_data_path).exists()
    assert Path(settings.model_path).exists()
    assert Path(settings.metrics_path).exists()
    assert metrics["rows"] == 500
    assert 0 <= metrics["precision_at_k"] <= 1
    assert 0 <= metrics["recall_at_k"] <= 1
    assert json.loads(Path(settings.metrics_path).read_text()) == metrics
    assert load_artifact(settings.model_path).version == "0.1.0"

