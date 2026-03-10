from __future__ import annotations

import re

DEFAULT_SITE_ID = "alpha"
DEFAULT_LOOKBACK_HOURS = 168
DEFAULT_HORIZON_HOURS = 24

SITE_PATTERN = re.compile(r"\b(alpha|bravo|charlie|delta)\b", re.IGNORECASE)
HOURS_PATTERN = re.compile(r"(\d+)\s*hours?", re.IGNORECASE)
DAYS_PATTERN = re.compile(r"(\d+)\s*days?", re.IGNORECASE)


def _extract_site(question: str) -> str:
    match = SITE_PATTERN.search(question)
    return match.group(1).lower() if match else DEFAULT_SITE_ID


def _extract_hours(question: str, default_hours: int) -> int:
    hour_match = HOURS_PATTERN.search(question)
    if hour_match:
        return int(hour_match.group(1))
    day_match = DAYS_PATTERN.search(question)
    if day_match:
        return int(day_match.group(1)) * 24
    return default_hours


def parse_question(question: str) -> dict:
    q = question.strip().lower()
    site_id = _extract_site(q)

    if any(word in q for word in ["forecast", "predict", "prediction"]):
        return {
            "intent": "forecast",
            "site_id": site_id,
            "horizon_hours": _extract_hours(q, DEFAULT_HORIZON_HOURS),
            "lookback_hours": DEFAULT_LOOKBACK_HOURS,
        }

    if any(word in q for word in ["anomaly", "anomalies", "spike", "outlier"]):
        return {
            "intent": "anomalies",
            "site_id": site_id,
            "horizon_hours": DEFAULT_HORIZON_HOURS,
            "lookback_hours": _extract_hours(q, DEFAULT_LOOKBACK_HOURS),
        }

    if any(word in q for word in ["recommend", "dispatch", "optimize", "optimization"]):
        return {
            "intent": "recommendation",
            "site_id": site_id,
            "horizon_hours": _extract_hours(q, DEFAULT_HORIZON_HOURS),
            "lookback_hours": DEFAULT_LOOKBACK_HOURS,
        }

    return {
        "intent": "kpis",
        "site_id": site_id,
    }