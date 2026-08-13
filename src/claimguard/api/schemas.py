from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EnrichedClaim(BaseModel):
    claim_id: str = Field(min_length=1)
    claim_type: Literal["carrier", "inpatient", "outpatient", "pharmacy"]
    paid_amount: float = Field(ge=0)
    diagnosis_count: int = Field(ge=0)
    procedure_count: int = Field(ge=0)
    units: int = Field(ge=0)
    length_of_stay: int = Field(ge=0)
    provider_claim_count_30d: int = Field(ge=1)
    beneficiary_claim_count_30d: int = Field(ge=1)
    provider_paid_zscore: float
    duplicate_indicator: int = Field(ge=0, le=1)
    weekend_service: int = Field(ge=0, le=1)


class ScoreRequest(BaseModel):
    claims: list[EnrichedClaim] = Field(min_length=1, max_length=500)


class ClaimScore(BaseModel):
    claim_id: str
    anomaly_score: float
    is_flagged: bool
    reason_codes: list[str]


class ScoreResponse(BaseModel):
    model_version: str
    scores: list[ClaimScore]

