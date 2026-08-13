from pathlib import Path

import pandas as pd
import pytest

from claimguard.config import Settings
from claimguard.data.cms_synpuf import adapt_cms_synpuf, adapt_cms_synpuf_csv
from claimguard.pipeline import run_csv_training_pipeline


def test_adapts_inpatient_claim() -> None:
    source = pd.DataFrame(
        {
            "DESYNPUF_ID": ["BENE01"],
            "CLM_ID": ["1001"],
            "CLM_FROM_DT": ["20090102"],
            "PRVDR_NUM": ["HOSP01"],
            "CLM_PMT_AMT": ["1250.50"],
            "CLM_UTLZTN_DAY_CNT": ["3"],
            "ADMTNG_ICD9_DGNS_CD": ["25000"],
            "ICD9_DGNS_CD_1": ["4019"],
            "ICD9_PRCDR_CD_1": ["9904"],
            "HCPCS_CD_1": ["99223"],
        }
    )

    result = adapt_cms_synpuf(source, "inpatient")

    assert result.loc[0, "claim_id"] == "CMS-INPATIENT-1001"
    assert result.loc[0, "beneficiary_id"] == "CMS-BEN-BENE01"
    assert result.loc[0, "provider_id"] == "CMS-PRV-HOSP01"
    assert result.loc[0, "paid_amount"] == 1250.50
    assert result.loc[0, "diagnosis_count"] == 2
    assert result.loc[0, "procedure_count"] == 2
    assert result.loc[0, "length_of_stay"] == 3


def test_adapts_carrier_lines_and_sums_payments() -> None:
    source = pd.DataFrame(
        {
            "DESYNPUF_ID": ["BENE02"],
            "CLM_ID": ["2002"],
            "CLM_FROM_DT": ["20081215"],
            "PRF_PHYSN_NPI_1": ["1234567890"],
            "HCPCS_CD_1": ["99213"],
            "HCPCS_CD_2": ["80053"],
            "ICD9_DGNS_CD_1": ["2724"],
            "LINE_NCH_PMT_AMT_1": ["75.25"],
            "LINE_NCH_PMT_AMT_2": ["24.75"],
        }
    )

    result = adapt_cms_synpuf(source, "carrier")

    assert result.loc[0, "paid_amount"] == 100.0
    assert result.loc[0, "provider_id"] == "CMS-PRV-1234567890"
    assert result.loc[0, "procedure_count"] == 2
    assert result.loc[0, "units"] == 2


def test_adapts_prescription_drug_event_without_inventing_provider() -> None:
    source = pd.DataFrame(
        {
            "DESYNPUF_ID": ["BENE03"],
            "PDE_ID": ["3003"],
            "SRVC_DT": ["20100307"],
            "PROD_SRVC_ID": ["00011122233"],
            "QTY_DSPNSD_NUM": ["30"],
            "DAYS_SUPLY_NUM": ["30"],
            "TOT_RX_CST_AMT": ["89.40"],
        }
    )

    result = adapt_cms_synpuf(source, "pde")

    assert result.loc[0, "claim_type"] == "pharmacy"
    assert result.loc[0, "units"] == 30
    assert result.loc[0, "provider_id"].startswith("CMS-NO-PROVIDER-")


def test_collapses_repeated_segments_to_one_claim() -> None:
    source = pd.DataFrame(
        {
            "DESYNPUF_ID": ["BENE04", "BENE04"],
            "CLM_ID": ["4004", "4004"],
            "SEGMENT": ["1", "2"],
            "CLM_FROM_DT": ["20090101", "20090101"],
            "PRVDR_NUM": ["HOSP02", "HOSP02"],
            "CLM_PMT_AMT": ["900", "900"],
            "HCPCS_CD_1": ["71020", "71020"],
        }
    )

    result = adapt_cms_synpuf(source, "outpatient")

    assert len(result) == 1
    assert result.loc[0, "source_record_count"] == 2
    assert result.loc[0, "paid_amount"] == 900


def test_rejects_source_with_missing_required_identifiers() -> None:
    source = pd.DataFrame({"CLM_ID": ["5005"], "CLM_FROM_DT": ["20090101"]})
    with pytest.raises(ValueError, match="DESYNPUF_ID"):
        adapt_cms_synpuf(source, "outpatient")


def test_csv_adapter_and_unlabeled_training_path(tmp_path: Path) -> None:
    source_path = tmp_path / "outpatient.csv"
    canonical_path = tmp_path / "canonical.csv"
    rows = 120
    source = pd.DataFrame(
        {
            "DESYNPUF_ID": [f"BENE{index:04d}" for index in range(rows)],
            "CLM_ID": [f"CLM{index:04d}" for index in range(rows)],
            "CLM_FROM_DT": ["20090101"] * rows,
            "PRVDR_NUM": [f"PRV{index % 20:03d}" for index in range(rows)],
            "CLM_PMT_AMT": [str(100 + index * 3) for index in range(rows)],
            "ICD9_DGNS_CD_1": ["4019"] * rows,
            "HCPCS_CD_1": ["99213"] * rows,
        }
    )
    source.to_csv(source_path, index=False)
    canonical = adapt_cms_synpuf_csv(source_path, "outpatient", canonical_path)
    assert len(canonical) == rows

    settings = Settings(
        synthetic_rows=rows,
        top_k=10,
        raw_data_path=str(tmp_path / "raw.csv"),
        scored_data_path=str(tmp_path / "scored.csv"),
        model_path=str(tmp_path / "model.joblib"),
        metrics_path=str(tmp_path / "metrics.json"),
    )
    metrics = run_csv_training_pipeline(canonical_path, settings)

    assert metrics["evaluation_available"] == 0
    assert metrics["rows"] == rows
    assert metrics["flagged_claims"] == 6
