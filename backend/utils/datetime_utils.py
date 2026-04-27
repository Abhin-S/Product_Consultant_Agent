from __future__ import annotations

from datetime import datetime, timedelta, timezone


IST = timezone(timedelta(hours=5, minutes=30))


def now_ist() -> datetime:
    return datetime.now(IST)


def to_ist_datetime(value: datetime | None) -> datetime | None:
    if not isinstance(value, datetime):
        return None

    normalized = value
    if normalized.tzinfo is None:
        # SQLite commonly returns naive datetimes. In this project we persist in IST,
        # so treat naive values as IST-naive to avoid adding +5:30 twice.
        normalized = normalized.replace(tzinfo=IST)

    return normalized.astimezone(IST)


def to_ist_iso(value: datetime | None) -> str | None:
    converted = to_ist_datetime(value)
    if converted is None:
        return None
    return converted.isoformat()