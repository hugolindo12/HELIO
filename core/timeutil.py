"""
HEILO timezone utilities.
Internal storage: always UTC (aware).
Display: convert to configured user timezone.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional, Union

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore

# Default display timezone (Brazil)
DEFAULT_USER_TZ = "America/Sao_Paulo"

_FALLBACK_OFFSETS = {
    "UTC": timezone.utc,
    "America/Sao_Paulo": timezone(timedelta(hours=-3), "BRT"),
    "America/Manaus": timezone(timedelta(hours=-4), "AMT"),
    "America/Belem": timezone(timedelta(hours=-3), "BRT"),
    "America/Fortaleza": timezone(timedelta(hours=-3), "BRT"),
    "America/New_York": timezone(timedelta(hours=-5), "EST"),
    "America/Los_Angeles": timezone(timedelta(hours=-8), "PST"),
    "Europe/Lisbon": timezone(timedelta(hours=0), "WET"),
    "Europe/London": timezone(timedelta(hours=0), "GMT"),
    "Europe/Paris": timezone(timedelta(hours=1), "CET"),
    "Asia/Tokyo": timezone(timedelta(hours=9), "JST"),
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def ensure_aware(dt: datetime) -> datetime:
    """Treat naive datetimes as UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def parse_iso(value: str) -> datetime:
    """Parse ISO string; Z suffix supported; naive assumed UTC."""
    s = (value or "").strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    return ensure_aware(dt)


def get_zone(name: Optional[str] = None):
    tz_name = name or DEFAULT_USER_TZ
    if ZoneInfo is not None:
        try:
            return ZoneInfo(tz_name)
        except Exception:
            pass
    return _FALLBACK_OFFSETS.get(tz_name, timezone.utc)


def to_user_tz(dt: Union[datetime, str], user_tz: Optional[str] = None) -> datetime:
    if isinstance(dt, str):
        dt = parse_iso(dt)
    else:
        dt = ensure_aware(dt)
    return dt.astimezone(get_zone(user_tz))


def format_user(
    dt: Union[datetime, str, None],
    user_tz: Optional[str] = None,
    fmt: str = "%Y-%m-%d %H:%M:%S %Z",
) -> str:
    """Format a UTC (or aware) moment for display in the user timezone."""
    if dt is None:
        return ""
    try:
        local = to_user_tz(dt, user_tz)
        return local.strftime(fmt)
    except Exception:
        return str(dt)


def list_common_timezones() -> list:
    return [
        "UTC",
        "America/Sao_Paulo",
        "America/Manaus",
        "America/Belem",
        "America/Fortaleza",
        "America/New_York",
        "America/Los_Angeles",
        "Europe/Lisbon",
        "Europe/London",
        "Europe/Paris",
        "Asia/Tokyo",
    ]
