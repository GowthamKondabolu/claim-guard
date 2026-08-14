from pathlib import Path

from fastapi.testclient import TestClient

from claimguard.api.main import app, get_model
from claimguard.config import Settings
from claimguard.pipeline import run_training_pipeline


def test_scoring_api_returns_rankable_claim_score(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "model.joblib"
    run_training_pipeline(
        Settings(
            synthetic_rows=300,
            top_k=15,
            raw_data_path=str(tmp_path / "raw.csv"),
            scored_data_path=str(tmp_path / "scored.csv"),
            model_path=str(model_path),
            metrics_path=str(tmp_path / "metrics.json"),
        )
    )
    monkeypatch.setenv("CLAIMGUARD_MODEL_PATH", str(model_path))
    get_model.cache_clear()
    client = TestClient(app)

    response = client.post(
        "/v1/score",
        json={
            "claims": [
                {
                    "claim_id": "CLM-DEMO-001",
                    "claim_type": "outpatient",
                    "paid_amount": 32000,
                    "diagnosis_count": 8,
                    "procedure_count": 6,
                    "units": 55,
                    "length_of_stay": 0,
                    "provider_claim_count_30d": 60,
                    "beneficiary_claim_count_30d": 5,
                    "provider_paid_zscore": 6.2,
                    "duplicate_indicator": 0,
                    "weekend_service": 0,
                }
            ]
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "0.2.0"
    claim_score = body["scores"][0]
    assert claim_score["claim_id"] == "CLM-DEMO-001"
    assert claim_score["rule_score"] == 0.65
    assert 0.0 <= claim_score["model_score_percentile"] <= 1.0
    assert 0.0 <= claim_score["ensemble_score"] <= 1.0
    assert claim_score["anomaly_score"] == claim_score["ensemble_score"]
    assert "HIGH_PROVIDER_RELATIVE_COST" in claim_score["reason_codes"]
    assert "EXCESS_UNITS" in claim_score["reason_codes"]
    assert "HIGH_PROVIDER_UTILIZATION" in claim_score["reason_codes"]
    get_model.cache_clear()

