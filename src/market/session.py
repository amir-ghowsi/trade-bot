"""
Market Session and Time Handling Utilities.
Provides time range validation and session-timezone projections (e.g. America/New_York)
while guaranteeing that internal storage and processing remain strictly in UTC.
"""

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.core.clock import ensure_utc, to_timezone
from src.core.exceptions import DataValidationError, TimezoneError


def validate_time_range(start_time: datetime, end_time: datetime) -> None:
    """
    Validate that a query time interval is timezone-aware UTC and chronologically valid.
    """
    utc_start = ensure_utc(start_time)
    utc_end = ensure_utc(end_time)

    if utc_end < utc_start:
        raise DataValidationError(
            f"Query end_time ({utc_end.isoformat()}) cannot be earlier than start_time ({utc_start.isoformat()})"
        )


def to_session_time(dt: datetime, tz_name: str = "America/New_York") -> datetime:
    """
    Project a UTC datetime to session timezone (default: America/New_York) for session date inspection.
    Internal processing must continue to use UTC.
    """
    return to_timezone(dt, tz_name)


def get_session_date(dt: datetime, tz_name: str = "America/New_York") -> str:
    """
    Extract the session date string ('YYYY-MM-DD') based on session timezone projection.
    """
    session_dt = to_session_time(dt, tz_name)
    return session_dt.strftime("%Y-%m-%d")
