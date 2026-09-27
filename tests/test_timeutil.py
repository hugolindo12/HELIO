from datetime import datetime, timezone
from heilo.core import timeutil


def test_utc_now_aware():
    dt = timeutil.utc_now()
    assert dt.tzinfo is not None
    assert dt.utcoffset().total_seconds() == 0


def test_parse_iso_z():
    dt = timeutil.parse_iso("2026-09-26T23:00:00Z")
    assert dt.tzinfo is not None


def test_parse_iso_naive_assumed_utc():
    dt = timeutil.parse_iso("2026-09-26T23:00:00")
    assert dt.tzinfo is not None


def test_format_user_sao_paulo():
    utc = datetime(2026, 9, 26, 23, 0, 0, tzinfo=timezone.utc)
    s = timeutil.format_user(utc, "America/Sao_Paulo", fmt="%Y-%m-%d %H:%M")
    assert s == "2026-09-26 20:00"


def test_to_user_tz_tokyo():
    utc = datetime(2026, 9, 26, 23, 0, 0, tzinfo=timezone.utc)
    local = timeutil.to_user_tz(utc, "Asia/Tokyo")
    assert local.hour == 8  # 23 UTC -> 8 JST next day
