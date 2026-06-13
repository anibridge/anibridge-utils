"""Tests for shared datetime helpers."""

from datetime import UTC, datetime, timedelta, timezone

from anibridge.utils.datetime import normalize_local_datetime


def test_normalize_local_datetime_none_passthrough() -> None:
    """None should pass through unchanged."""
    assert normalize_local_datetime(None) is None


def test_normalize_local_datetime_localizes_naive_values() -> None:
    """Naive datetimes should get localized without shifting wall clock time."""
    local_tz = timezone(timedelta(hours=2))
    value = datetime(2026, 1, 1, 8, 30)

    normalized = normalize_local_datetime(value, local_tz=local_tz)

    assert normalized is not None
    assert normalized.tzinfo == local_tz
    assert normalized.hour == 8
    assert normalized.minute == 30


def test_normalize_local_datetime_defaults_naive_values_to_utc() -> None:
    """Naive datetimes should default to UTC when no timezone is provided."""
    value = datetime(2026, 1, 1, 8, 30)

    normalized = normalize_local_datetime(value)

    assert normalized is not None
    assert normalized.tzinfo == UTC
    assert normalized.hour == 8
    assert normalized.minute == 30


def test_normalize_local_datetime_converts_aware_values() -> None:
    """Aware datetimes should be converted into the local timezone."""
    local_tz = timezone(timedelta(hours=-5))
    value = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    normalized = normalize_local_datetime(value, local_tz=local_tz)

    assert normalized is not None
    assert normalized.utcoffset() == timedelta(hours=-5)
    assert normalized.hour == 7
