"""Shared datetime helpers for AniBridge packages."""

from datetime import UTC, datetime, tzinfo

__all__ = ["normalize_local_datetime"]


def normalize_local_datetime(
    value: datetime | None, *, local_tz: tzinfo | None = None
) -> datetime | None:
    """Return a timezone-aware datetime normalized to local timezone.

    Args:
        value (datetime | None): Datetime to normalize.
        local_tz (tzinfo | None): Local timezone override for deterministic tests.

    Returns:
        datetime | None: Timezone-aware datetime in local timezone, or `None`.
    """
    if value is None:
        return None

    resolved_local_tz = local_tz or UTC
    if value.tzinfo is None:
        return value.replace(tzinfo=resolved_local_tz)
    return value.astimezone(resolved_local_tz)
