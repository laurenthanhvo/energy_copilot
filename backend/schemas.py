from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    site_id: str = Field(..., examples=["alpha"])
    horizon_hours: int = Field(24, ge=6, le=168)
    battery_mwh: float = Field(20.0, gt=0)
    max_charge_mw: float = Field(5.0, gt=0)
    max_discharge_mw: float = Field(5.0, gt=0)
    soc0_mwh: float = Field(10.0, ge=0)
    emissions_weight: float = Field(0.02, ge=0)
    peak_demand_weight: float = Field(15.0, ge=0)


class AskRequest(BaseModel):
    question: str