from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import pandas as pd

from claimguard.data.validation import validate_claims

CmsClaimType = Literal["inpatient", "outpatient", "carrier", "pharmacy"]
CLAIM_TYPE_ALIASES = {"pde": "pharmacy", "prescription": "pharmacy"}


def _normalize_claim_type(claim_type: str) -> CmsClaimType:
    normalized = CLAIM_TYPE_ALIASES.get(claim_type.lower().strip(), claim_type.lower().strip())
    supported = {"inpatient", "outpatient", "carrier", "pharmacy"}
    if normalized not in supported:
        raise ValueError(f"Unsupported CMS claim type: {claim_type}. Choose from {sorted(supported)}")
    return normalized  # type: ignore[return-value]


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [str(column).strip().upper() for column in normalized.columns]
    if normalized.columns.duplicated().any():
        duplicates = normalized.columns[normalized.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate columns after normalization: {duplicates}")
    return normalized


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"CMS source is missing required columns: {missing}")


def _clean_text(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    return cleaned.mask(cleaned.str.lower().isin({"", "nan", "none", "<na>"}))


def _number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _number_or_zero(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(0.0, index=frame.index)
    return _number(frame[column]).fillna(0.0)


def _ordered_prefix_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    columns = [column for column in frame.columns if column.startswith(prefix)]

    def suffix_number(column: str) -> int:
        match = re.search(r"(\d+)$", column)
        return int(match.group(1)) if match else 0

    return sorted(columns, key=suffix_number)


def _first_nonempty(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(pd.NA, index=frame.index, dtype="string")
    for column in columns:
        if column in frame:
            result = result.fillna(_clean_text(frame[column]))
    return result


def _count_populated_codes(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    present = [column for column in columns if column in frame]
    if not present:
        return pd.Series(0, index=frame.index, dtype="int64")
    populated = pd.concat([_clean_text(frame[column]).notna() for column in present], axis=1)
    return populated.sum(axis=1).astype(int)


def _parse_cms_date(series: pd.Series, column: str) -> pd.Series:
    cleaned = _clean_text(series).str.replace(r"\.0$", "", regex=True)
    parsed = pd.to_datetime(cleaned, format="%Y%m%d", errors="coerce")
    invalid = cleaned.notna() & parsed.isna()
    if invalid.any():
        examples = cleaned.loc[invalid].head(3).tolist()
        raise ValueError(f"Invalid YYYYMMDD values in {column}: {examples}")
    return parsed


def _source_identifiers(frame: pd.DataFrame, claim_type: CmsClaimType) -> tuple[pd.Series, pd.Series]:
    claim_id_column = "PDE_ID" if claim_type == "pharmacy" else "CLM_ID"
    date_column = "SRVC_DT" if claim_type == "pharmacy" else "CLM_FROM_DT"
    _require_columns(frame, ["DESYNPUF_ID", claim_id_column, date_column])

    raw_claim_id = _clean_text(frame[claim_id_column])
    beneficiary_id = _clean_text(frame["DESYNPUF_ID"])
    if raw_claim_id.isna().any() or beneficiary_id.isna().any():
        raise ValueError("CMS claim and beneficiary identifiers cannot be missing")
    return raw_claim_id, beneficiary_id


def _provider_identifier(
    frame: pd.DataFrame,
    claim_type: CmsClaimType,
    canonical_claim_id: pd.Series,
) -> pd.Series:
    if claim_type in {"inpatient", "outpatient"}:
        candidates = ["PRVDR_NUM", "AT_PHYSN_NPI", "OP_PHYSN_NPI", "OT_PHYSN_NPI"]
    elif claim_type == "carrier":
        candidates = _ordered_prefix_columns(frame, "PRF_PHYSN_NPI_")
        candidates += _ordered_prefix_columns(frame, "TAX_NUM_")
    else:
        candidates = []

    provider = _first_nonempty(frame, candidates)
    known = provider.notna()
    result = pd.Series(index=frame.index, dtype="string")
    result.loc[known] = "CMS-PRV-" + provider.loc[known]
    # PDE does not include a provider field. A unique fallback prevents false provider-level spikes.
    result.loc[~known] = "CMS-NO-PROVIDER-" + canonical_claim_id.loc[~known]
    return result


def _paid_amount(frame: pd.DataFrame, claim_type: CmsClaimType) -> pd.Series:
    if claim_type in {"inpatient", "outpatient"}:
        _require_columns(frame, ["CLM_PMT_AMT"])
        return _number(frame["CLM_PMT_AMT"]).fillna(0.0)
    if claim_type == "pharmacy":
        _require_columns(frame, ["TOT_RX_CST_AMT"])
        return _number(frame["TOT_RX_CST_AMT"]).fillna(0.0)

    payment_columns = _ordered_prefix_columns(frame, "LINE_NCH_PMT_AMT_")
    if not payment_columns:
        raise ValueError("Carrier source has no LINE_NCH_PMT_AMT_* columns")
    return frame[payment_columns].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)


def _diagnosis_count(frame: pd.DataFrame, claim_type: CmsClaimType) -> pd.Series:
    if claim_type == "pharmacy":
        return pd.Series(0, index=frame.index, dtype="int64")
    columns = _ordered_prefix_columns(frame, "ICD9_DGNS_CD_")
    columns += _ordered_prefix_columns(frame, "LINE_ICD9_DGNS_CD_")
    if "ADMTNG_ICD9_DGNS_CD" in frame:
        columns.append("ADMTNG_ICD9_DGNS_CD")
    return _count_populated_codes(frame, columns)


def _procedure_count(frame: pd.DataFrame, claim_type: CmsClaimType) -> pd.Series:
    if claim_type == "pharmacy":
        return _clean_text(frame["PROD_SRVC_ID"]).notna().astype(int) if "PROD_SRVC_ID" in frame else 0
    columns = _ordered_prefix_columns(frame, "ICD9_PRCDR_CD_")
    columns += _ordered_prefix_columns(frame, "HCPCS_CD_")
    return _count_populated_codes(frame, columns)


def _length_of_stay(frame: pd.DataFrame, claim_type: CmsClaimType) -> pd.Series:
    if claim_type != "inpatient":
        return pd.Series(0, index=frame.index, dtype="int64")
    if "CLM_UTLZTN_DAY_CNT" in frame:
        return _number(frame["CLM_UTLZTN_DAY_CNT"]).fillna(0).clip(lower=0).astype(int)
    _require_columns(frame, ["CLM_ADMSN_DT", "CLM_THRU_DT"])
    admission = _parse_cms_date(frame["CLM_ADMSN_DT"], "CLM_ADMSN_DT")
    discharge = _parse_cms_date(frame["CLM_THRU_DT"], "CLM_THRU_DT")
    return (discharge - admission).dt.days.add(1).fillna(0).clip(lower=0).astype(int)


def adapt_cms_synpuf(frame: pd.DataFrame, claim_type: str) -> pd.DataFrame:
    """Map a CMS DE-SynPUF claim/event frame to ClaimGuard's canonical claim schema."""
    normalized_type = _normalize_claim_type(claim_type)
    source = _normalize_columns(frame)
    raw_claim_id, raw_beneficiary_id = _source_identifiers(source, normalized_type)
    canonical_claim_id = "CMS-" + normalized_type.upper() + "-" + raw_claim_id
    date_column = "SRVC_DT" if normalized_type == "pharmacy" else "CLM_FROM_DT"

    procedures = _procedure_count(source, normalized_type)
    canonical = pd.DataFrame(
        {
            "claim_id": canonical_claim_id,
            "source_claim_id": raw_claim_id,
            "beneficiary_id": "CMS-BEN-" + raw_beneficiary_id,
            "claim_type": normalized_type,
            "service_date": _parse_cms_date(source[date_column], date_column),
            "paid_amount": _paid_amount(source, normalized_type).clip(lower=0).round(2),
            "diagnosis_count": _diagnosis_count(source, normalized_type),
            "procedure_count": procedures,
            "units": (
                _number_or_zero(source, "QTY_DSPNSD_NUM").clip(lower=0).astype(int)
                if normalized_type == "pharmacy"
                else procedures.clip(lower=0).astype(int)
            ),
            "length_of_stay": _length_of_stay(source, normalized_type),
            "source_file_type": normalized_type,
        }
    )
    canonical["provider_id"] = _provider_identifier(
        source, normalized_type, canonical["claim_id"]
    )

    # SEGMENT can yield repeated claim IDs. Collapse conservatively to one claim-level record.
    canonical["source_record_count"] = 1
    collapsed = (
        canonical.groupby("claim_id", as_index=False, sort=False)
        .agg(
            source_claim_id=("source_claim_id", "first"),
            beneficiary_id=("beneficiary_id", "first"),
            provider_id=("provider_id", "first"),
            claim_type=("claim_type", "first"),
            service_date=("service_date", "min"),
            paid_amount=("paid_amount", "max"),
            diagnosis_count=("diagnosis_count", "max"),
            procedure_count=("procedure_count", "max"),
            units=("units", "max"),
            length_of_stay=("length_of_stay", "max"),
            source_file_type=("source_file_type", "first"),
            source_record_count=("source_record_count", "sum"),
        )
        .reset_index(drop=True)
    )
    validate_claims(collapsed)
    return collapsed


def adapt_cms_synpuf_csv(
    input_path: str | Path,
    claim_type: str,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    source = pd.read_csv(input_path, dtype=str, low_memory=False)
    canonical = adapt_cms_synpuf(source, claim_type)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        canonical.to_csv(destination, index=False)
    return canonical
