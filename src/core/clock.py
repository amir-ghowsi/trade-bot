"""
Clock Abstraction and Timezone-Aware Datetime Utilities.
Enforces strict timezone awareness (UTC internally) and prevents naive datetime usage.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Optional

from src.core.exceptions import CausalityViolationError, TimezoneError


def ensure_utc(dt: datetime) -> datetime:
    """
    Ensure datetime is timezone-aware and converted to UTC.
    Rejects naive datetimes strictly.
    """
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise TimezoneError("Naive datetimes are strictly forbidden. Datetime must be timezone-aware.")
    return dt.astimezone(timezone.utc)


def to_timezone(dt: datetime, tz_name: str) -> datetime:
    """
    Convert a timezone-aware datetime to a specific timezone (e.g. America/New_York).
    """
    utc_dt = ensure_utc(dt)
    try:
        target_tz = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError as err:
        raise TimezoneError(f"Unrecognized timezone identifier: '{tz_name}'") from err
    return utc_dt.astimezone(target_tz)


class IClock(ABC):
    """Abstract interface for system time provider."""

    @abstractmethod
    def now(self) -> datetime:
        """Return the current time as a timezone-aware UTC datetime."""
        pass

    @property
    @abstractmethod
    def is_simulated(self) -> bool:
        """Indicate whether the clock is running in simulation mode."""
        pass


class LiveClock(IClock):
    """
    Production wall-clock time provider.
    Always returns timezone.utc aware current datetime.
    """

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    @property
    def is_simulated(self) -> bool:
        return False


class SimulatedClock(IClock):
    """
    Deterministic simulated clock for backtesting.
    Enforces monotonic progression (time cannot flow backward).
    """

    def __init__(self, initial_time: Optional[datetime] = None) -> None:
        if initial_time is not None:
            self._current_time = ensure_utc(initial_time)
        else:
            self._current_time = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self._current_time

    @property
    def is_simulated(self) -> bool:
        return True

    def set_time(self, new_time: datetime) -> None:
        """Set initial time. Must be timezone-aware."""
        self._current_time = ensure_utc(new_time)

    def advance_to(self, new_time: datetime) -> None:
        """
        Advance clock to new_time.
        Strictly raises CausalityViolationError if new_time < current_time.
        """
        utc_new = ensure_utc(new_time)
        if utc_new < self._current_time:
            raise CausalityViolationError(
                f"Clock cannot move backwards: current={self._current_time.isoformat()} -> new={utc_new.isoformat()}"
            )
        self._current_time = utc_new
