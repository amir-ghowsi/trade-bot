"""
Unit Tests for src/core/clock.py
Validates timezone awareness, monotonic progression, and causality enforcement.
"""

from datetime import datetime, timezone
import unittest
from zoneinfo import ZoneInfo

from src.core.clock import LiveClock, SimulatedClock, ensure_utc, to_timezone
from src.core.exceptions import CausalityViolationError, TimezoneError


class TestClock(unittest.TestCase):
    """Test suite for Clock abstraction and datetime utilities."""

    def test_ensure_utc_with_timezone_aware(self) -> None:
        aware_dt = datetime(2023, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = ensure_utc(aware_dt)
        self.assertEqual(result, aware_dt)
        self.assertEqual(result.tzinfo, timezone.utc)

    def test_ensure_utc_rejects_naive_datetime(self) -> None:
        naive_dt = datetime(2023, 6, 15, 12, 0, 0)
        with self.assertRaises(TimezoneError):
            ensure_utc(naive_dt)

    def test_to_timezone_conversion(self) -> None:
        utc_dt = datetime(2023, 6, 15, 13, 30, 0, tzinfo=timezone.utc)
        ny_dt = to_timezone(utc_dt, "America/New_York")
        # In June (EDT), UTC is 4 hours ahead of NY
        self.assertEqual(ny_dt.hour, 9)
        self.assertEqual(ny_dt.minute, 30)
        self.assertEqual(str(ny_dt.tzinfo), "America/New_York")

    def test_to_timezone_invalid_timezone(self) -> None:
        utc_dt = datetime(2023, 6, 15, 13, 30, 0, tzinfo=timezone.utc)
        with self.assertRaises(TimezoneError):
            to_timezone(utc_dt, "Invalid/Nonexistent_Timezone")

    def test_live_clock_is_timezone_aware_utc(self) -> None:
        clock = LiveClock()
        self.assertFalse(clock.is_simulated)
        now = clock.now()
        self.assertIsNotNone(now.tzinfo)
        self.assertEqual(now.tzinfo, timezone.utc)

    def test_simulated_clock_initialization(self) -> None:
        init_time = datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        clock = SimulatedClock(initial_time=init_time)
        self.assertTrue(clock.is_simulated)
        self.assertEqual(clock.now(), init_time)

    def test_simulated_clock_monotonic_forward(self) -> None:
        t1 = datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2023, 1, 1, 10, 5, 0, tzinfo=timezone.utc)
        clock = SimulatedClock(initial_time=t1)
        clock.advance_to(t2)
        self.assertEqual(clock.now(), t2)

    def test_simulated_clock_advance_equal_time_permitted(self) -> None:
        t1 = datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        clock = SimulatedClock(initial_time=t1)
        clock.advance_to(t1)
        self.assertEqual(clock.now(), t1)

    def test_simulated_clock_causality_violation_on_backward_advance(self) -> None:
        t1 = datetime(2023, 1, 1, 10, 5, 0, tzinfo=timezone.utc)
        t_earlier = datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        clock = SimulatedClock(initial_time=t1)
        with self.assertRaises(CausalityViolationError):
            clock.advance_to(t_earlier)

    def test_simulated_clock_rejects_naive_time(self) -> None:
        naive_dt = datetime(2023, 1, 1, 10, 0, 0)
        with self.assertRaises(TimezoneError):
            SimulatedClock(initial_time=naive_dt)

        clock = SimulatedClock()
        with self.assertRaises(TimezoneError):
            clock.advance_to(naive_dt)


if __name__ == "__main__":
    unittest.main()
