"""
Unit Tests for Phase 4 - Strategy Detectors.
Verifies all locked mathematical strategy rules, determinism, causality,
symbol validation, and immutability invariants.
"""

from datetime import datetime, timedelta, timezone
import unittest

from src.core.constants import (
    CanonicalSymbol,
    Direction,
    Timeframe,
)
from src.core.exceptions import DataValidationError
from src.core.types import Candle, GapEvent, SpikeEvent, StructuralLevel
from src.strategy.average_range import (
    calculate_average_m5_range,
    calculate_candle_range,
)
from src.strategy.direction import (
    detect_candle_direction,
    is_bearish,
    is_bullish,
    is_neutral,
)
from src.strategy.exceptions import (
    DetectorError,
    InvalidStructureError,
)
from src.strategy.gap import (
    detect_gap_in_triplet,
    detect_gaps_in_sequence,
)
from src.strategy.spike import (
    generate_spike_id,
    SpikeDetector,
)
from src.strategy.structure import (
    detect_first_bottom,
    detect_first_top,
)


class TestStrategyDetectors(unittest.TestCase):
    """Complete test suite for Phase 4 strategy detectors."""

    def setUp(self) -> None:
        self.symbol = CanonicalSymbol.DOW_JONES
        self.tf_m5 = Timeframe.M5
        self.base_time = datetime(2023, 10, 2, 10, 0, 0, tzinfo=timezone.utc)

    def _make_candle(
        self,
        minute_offset: int,
        open_: float,
        high: float,
        low: float,
        close: float,
        spread: float = 1.0,
        volume: float = 100.0,
        timeframe: Timeframe = Timeframe.M5,
        symbol: CanonicalSymbol = CanonicalSymbol.DOW_JONES,
    ) -> Candle:
        open_time = self.base_time + timedelta(minutes=minute_offset)
        duration = 5 if timeframe == Timeframe.M5 else 1
        close_time = open_time + timedelta(minutes=duration)
        return Candle(
            symbol=symbol,
            timeframe=timeframe,
            open_time=open_time,
            close_time=close_time,
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            spread=spread,
        )

    # =========================================================================
    # 1. CANDLE DIRECTION TESTS
    # =========================================================================

    def test_candle_direction_bullish(self) -> None:
        c = self._make_candle(0, open_=34000.0, high=34050.0, low=33990.0, close=34040.0)
        self.assertEqual(detect_candle_direction(c), Direction.BULLISH)
        self.assertTrue(is_bullish(c))
        self.assertFalse(is_bearish(c))
        self.assertFalse(is_neutral(c))

    def test_candle_direction_bearish(self) -> None:
        c = self._make_candle(0, open_=34040.0, high=34050.0, low=33980.0, close=33990.0)
        self.assertEqual(detect_candle_direction(c), Direction.BEARISH)
        self.assertFalse(is_bullish(c))
        self.assertTrue(is_bearish(c))
        self.assertFalse(is_neutral(c))

    def test_candle_direction_neutral(self) -> None:
        c = self._make_candle(0, open_=34000.0, high=34030.0, low=33970.0, close=34000.0)
        self.assertEqual(detect_candle_direction(c), Direction.NEUTRAL)
        self.assertFalse(is_bullish(c))
        self.assertFalse(is_bearish(c))
        self.assertTrue(is_neutral(c))

    # =========================================================================
    # 2. BULLISH GAP TESTS
    # =========================================================================

    def test_bullish_gap_valid(self) -> None:
        # High(C1) < Low(C3)
        c1 = self._make_candle(0, open_=34000.0, high=34020.0, low=33990.0, close=34015.0)
        c2 = self._make_candle(5, open_=34020.0, high=34060.0, low=34018.0, close=34055.0)
        c3 = self._make_candle(10, open_=34055.0, high=34100.0, low=34030.0, close=34090.0)

        # High(C1) = 34020.0 < Low(C3) = 34030.0 -> valid bullish gap of 10.0
        gap = detect_gap_in_triplet(c1, c2, c3)
        self.assertIsNotNone(gap)
        self.assertEqual(gap.direction, Direction.BULLISH)
        self.assertEqual(gap.c1_high, 34020.0)
        self.assertEqual(gap.c3_low, 34030.0)
        self.assertAlmostEqual(gap.gap_size, 10.0)

    def test_bullish_gap_invalid_overlap(self) -> None:
        # High(C1) > Low(C3)
        c1 = self._make_candle(0, open_=34000.0, high=34040.0, low=33990.0, close=34015.0)
        c2 = self._make_candle(5, open_=34015.0, high=34060.0, low=34010.0, close=34055.0)
        c3 = self._make_candle(10, open_=34055.0, high=34100.0, low=34030.0, close=34090.0)
        # High(C1) = 34040.0 > Low(C3) = 34030.0 -> Overlap, no gap
        gap = detect_gap_in_triplet(c1, c2, c3)
        self.assertIsNone(gap)

    def test_bullish_gap_boundary_equality_is_not_gap(self) -> None:
        # High(C1) == Low(C3) -> Boundary condition, NOT a gap
        c1 = self._make_candle(0, open_=34000.0, high=34030.0, low=33990.0, close=34015.0)
        c2 = self._make_candle(5, open_=34015.0, high=34060.0, low=34010.0, close=34055.0)
        c3 = self._make_candle(10, open_=34055.0, high=34100.0, low=34030.0, close=34090.0)
        # High(C1) = 34030.0 == Low(C3) = 34030.0
        gap = detect_gap_in_triplet(c1, c2, c3)
        self.assertIsNone(gap)

    # =========================================================================
    # 3. BEARISH GAP TESTS
    # =========================================================================

    def test_bearish_gap_valid(self) -> None:
        # Low(C1) > High(C3)
        c1 = self._make_candle(0, open_=34100.0, high=34110.0, low=34070.0, close=34075.0)
        c2 = self._make_candle(5, open_=34075.0, high=34080.0, low=34030.0, close=34035.0)
        c3 = self._make_candle(10, open_=34035.0, high=34060.0, low=34000.0, close=34005.0)
        # Low(C1) = 34070.0 > High(C3) = 34060.0 -> valid bearish gap of 10.0
        gap = detect_gap_in_triplet(c1, c2, c3)
        self.assertIsNotNone(gap)
        self.assertEqual(gap.direction, Direction.BEARISH)
        self.assertEqual(gap.c1_low, 34070.0)
        self.assertEqual(gap.c3_high, 34060.0)
        self.assertAlmostEqual(gap.gap_size, 10.0)

    def test_bearish_gap_invalid_overlap(self) -> None:
        # Low(C1) < High(C3)
        c1 = self._make_candle(0, open_=34100.0, high=34110.0, low=34050.0, close=34055.0)
        c2 = self._make_candle(5, open_=34055.0, high=34080.0, low=34030.0, close=34035.0)
        c3 = self._make_candle(10, open_=34035.0, high=34060.0, low=34000.0, close=34005.0)
        # Low(C1) = 34050.0 < High(C3) = 34060.0 -> Overlap
        gap = detect_gap_in_triplet(c1, c2, c3)
        self.assertIsNone(gap)

    def test_bearish_gap_boundary_equality_is_not_gap(self) -> None:
        # Low(C1) == High(C3)
        c1 = self._make_candle(0, open_=34100.0, high=34110.0, low=34060.0, close=34065.0)
        c2 = self._make_candle(5, open_=34065.0, high=34070.0, low=34020.0, close=34025.0)
        c3 = self._make_candle(10, open_=34025.0, high=34060.0, low=33990.0, close=34000.0)
        # Low(C1) = 34060.0 == High(C3) = 34060.0
        gap = detect_gap_in_triplet(c1, c2, c3)
        self.assertIsNone(gap)

    # =========================================================================
    # 4. BULLISH SPIKE TESTS
    # =========================================================================

    def test_bullish_spike_exactly_3_candles_with_gap(self) -> None:
        c1 = self._make_candle(0, open_=34000.0, high=34020.0, low=33990.0, close=34015.0)  # Bullish
        c2 = self._make_candle(5, open_=34020.0, high=34060.0, low=34018.0, close=34055.0)  # Bullish
        c3 = self._make_candle(10, open_=34055.0, high=34100.0, low=34030.0, close=34090.0)  # Bullish, c3.low > c1.high (gap)

        spike = SpikeDetector.evaluate_candidate_run([c1, c2, c3])
        self.assertIsNotNone(spike)
        self.assertEqual(spike.direction, Direction.BULLISH)
        self.assertEqual(spike.candle_count, 3)
        self.assertEqual(spike.gap_count, 1)
        self.assertEqual(spike.low, 33990.0)
        self.assertEqual(spike.high, 34100.0)
        self.assertEqual(spike.symbol, self.symbol)

    def test_bullish_spike_more_than_3_candles_with_gap(self) -> None:
        c1 = self._make_candle(0, open_=34000.0, high=34020.0, low=33990.0, close=34015.0)
        c2 = self._make_candle(5, open_=34020.0, high=34060.0, low=34018.0, close=34055.0)
        c3 = self._make_candle(10, open_=34055.0, high=34100.0, low=34030.0, close=34090.0)
        c4 = self._make_candle(15, open_=34090.0, high=34130.0, low=34085.0, close=34125.0)

        spike = SpikeDetector.evaluate_candidate_run([c1, c2, c3, c4])
        self.assertIsNotNone(spike)
        self.assertEqual(spike.candle_count, 4)
        self.assertEqual(spike.gap_count, 2)  # c1<c3 and c2<c4
        self.assertEqual(spike.high, 34130.0)

    def test_bullish_spike_less_than_3_candles_rejected(self) -> None:
        c1 = self._make_candle(0, open_=34000.0, high=34020.0, low=33990.0, close=34015.0)
        c2 = self._make_candle(5, open_=34020.0, high=34060.0, low=34018.0, close=34055.0)

        spike = SpikeDetector.evaluate_candidate_run([c1, c2])
        self.assertIsNone(spike)

    def test_bullish_spike_no_gap_rejected(self) -> None:
        # 3 bullish candles, but High(C1) >= Low(C3) (no gap)
        c1 = self._make_candle(0, open_=34000.0, high=34050.0, low=33990.0, close=34040.0)
        c2 = self._make_candle(5, open_=34040.0, high=34070.0, low=34035.0, close=34065.0)
        c3 = self._make_candle(10, open_=34065.0, high=34090.0, low=34045.0, close=34085.0)
        # High(C1) = 34050.0 >= Low(C3) = 34045.0 -> No gap

        spike = SpikeDetector.evaluate_candidate_run([c1, c2, c3])
        self.assertIsNone(spike)

    def test_bullish_spike_neutral_candle_interruption_rejected(self) -> None:
        c1 = self._make_candle(0, open_=34000.0, high=34020.0, low=33990.0, close=34015.0)
        c2 = self._make_candle(5, open_=34020.0, high=34040.0, low=34010.0, close=34020.0)  # NEUTRAL
        c3 = self._make_candle(10, open_=34020.0, high=34080.0, low=34015.0, close=34075.0)

        spike = SpikeDetector.evaluate_candidate_run([c1, c2, c3])
        self.assertIsNone(spike)

    def test_bullish_spike_bearish_candle_interruption_rejected(self) -> None:
        c1 = self._make_candle(0, open_=34000.0, high=34020.0, low=33990.0, close=34015.0)
        c2 = self._make_candle(5, open_=34020.0, high=34030.0, low=34005.0, close=34010.0)  # BEARISH
        c3 = self._make_candle(10, open_=34010.0, high=34080.0, low=34005.0, close=34075.0)

        spike = SpikeDetector.evaluate_candidate_run([c1, c2, c3])
        self.assertIsNone(spike)

    # =========================================================================
    # 5. BEARISH SPIKE TESTS
    # =========================================================================

    def test_bearish_spike_exactly_3_candles_with_gap(self) -> None:
        c1 = self._make_candle(0, open_=34100.0, high=34110.0, low=34070.0, close=34075.0)  # Bearish
        c2 = self._make_candle(5, open_=34075.0, high=34080.0, low=34030.0, close=34035.0)  # Bearish
        c3 = self._make_candle(10, open_=34035.0, high=34060.0, low=34000.0, close=34005.0)  # Bearish, c3.high < c1.low (gap)

        spike = SpikeDetector.evaluate_candidate_run([c1, c2, c3])
        self.assertIsNotNone(spike)
        self.assertEqual(spike.direction, Direction.BEARISH)
        self.assertEqual(spike.candle_count, 3)
        self.assertEqual(spike.gap_count, 1)
        self.assertEqual(spike.high, 34110.0)
        self.assertEqual(spike.low, 34000.0)

    def test_bearish_spike_more_than_3_candles_with_gap(self) -> None:
        c1 = self._make_candle(0, open_=34100.0, high=34110.0, low=34070.0, close=34075.0)
        c2 = self._make_candle(5, open_=34075.0, high=34080.0, low=34030.0, close=34035.0)
        c3 = self._make_candle(10, open_=34035.0, high=34060.0, low=34000.0, close=34005.0)
        c4 = self._make_candle(15, open_=34005.0, high=34015.0, low=33960.0, close=33965.0)

        spike = SpikeDetector.evaluate_candidate_run([c1, c2, c3, c4])
        self.assertIsNotNone(spike)
        self.assertEqual(spike.candle_count, 4)
        self.assertEqual(spike.gap_count, 2)
        self.assertEqual(spike.low, 33960.0)

    def test_bearish_spike_less_than_3_candles_rejected(self) -> None:
        c1 = self._make_candle(0, open_=34100.0, high=34110.0, low=34070.0, close=34075.0)
        c2 = self._make_candle(5, open_=34075.0, high=34080.0, low=34030.0, close=34035.0)

        spike = SpikeDetector.evaluate_candidate_run([c1, c2])
        self.assertIsNone(spike)

    def test_bearish_spike_no_gap_rejected(self) -> None:
        # 3 bearish candles, but Low(C1) <= High(C3)
        c1 = self._make_candle(0, open_=34100.0, high=34110.0, low=34050.0, close=34060.0)
        c2 = self._make_candle(5, open_=34060.0, high=34070.0, low=34020.0, close=34030.0)
        c3 = self._make_candle(10, open_=34030.0, high=34060.0, low=33990.0, close=34000.0)
        # Low(C1) = 34050.0 <= High(C3) = 34060.0

        spike = SpikeDetector.evaluate_candidate_run([c1, c2, c3])
        self.assertIsNone(spike)

    def test_bearish_spike_neutral_candle_interruption_rejected(self) -> None:
        c1 = self._make_candle(0, open_=34100.0, high=34110.0, low=34070.0, close=34075.0)
        c2 = self._make_candle(5, open_=34075.0, high=34080.0, low=34050.0, close=34075.0)  # NEUTRAL
        c3 = self._make_candle(10, open_=34075.0, high=34080.0, low=34000.0, close=34010.0)

        spike = SpikeDetector.evaluate_candidate_run([c1, c2, c3])
        self.assertIsNone(spike)

    def test_bearish_spike_bullish_candle_interruption_rejected(self) -> None:
        c1 = self._make_candle(0, open_=34100.0, high=34110.0, low=34070.0, close=34075.0)
        c2 = self._make_candle(5, open_=34075.0, high=34095.0, low=34070.0, close=34090.0)  # BULLISH
        c3 = self._make_candle(10, open_=34090.0, high=34100.0, low=34000.0, close=34010.0)

        spike = SpikeDetector.evaluate_candidate_run([c1, c2, c3])
        self.assertIsNone(spike)

    # =========================================================================
    # 6. FIRST BOTTOM TESTS (BLK-02)
    # =========================================================================

    def test_first_bottom_confirmation_and_immutability(self) -> None:
        # 1. Form a valid Bullish Spike
        c1 = self._make_candle(0, open_=34000.0, high=34020.0, low=33990.0, close=34015.0)
        c2 = self._make_candle(5, open_=34020.0, high=34060.0, low=34018.0, close=34055.0)
        c3 = self._make_candle(10, open_=34055.0, high=34100.0, low=34030.0, close=34090.0)
        spike = SpikeDetector.evaluate_candidate_run([c1, c2, c3])
        self.assertIsNotNone(spike)

        # 2. Initial Bearish Correction candle
        # Correction candle: Open=34085, High=34090, Low=34040, Close=34050 (BEARISH)
        corr1 = self._make_candle(15, open_=34085.0, high=34090.0, low=34040.0, close=34050.0)

        # 3. Candle that breaks the correction candle high (BLK-02)
        # High=34095 > corr1.high (34090)
        confirm_c = self._make_candle(20, open_=34050.0, high=34095.0, low=34045.0, close=34080.0)

        level = detect_first_bottom(spike, [corr1, confirm_c])
        self.assertIsNotNone(level)
        self.assertEqual(level.direction, Direction.BULLISH)
        self.assertEqual(level.level_type, "BOTTOM")
        self.assertEqual(level.price, 34040.0)  # First Bottom Low
        self.assertTrue(level.is_locked)
        self.assertEqual(level.formation_time, confirm_c.close_time)

        # 4. Immutability verification: later candles trading lower do NOT change the locked level
        later_c = self._make_candle(25, open_=34080.0, high=34085.0, low=34010.0, close=34020.0)
        # Evaluate again with additional subsequent candle
        level_after = detect_first_bottom(spike, [corr1, confirm_c, later_c])
        self.assertIsNotNone(level_after)
        self.assertEqual(level_after.price, 34040.0)  # Remained strictly 34040.0

    def test_first_bottom_multiple_correction_candles(self) -> None:
        c1 = self._make_candle(0, open_=34000.0, high=34020.0, low=33990.0, close=34015.0)
        c2 = self._make_candle(5, open_=34020.0, high=34060.0, low=34018.0, close=34055.0)
        c3 = self._make_candle(10, open_=34055.0, high=34100.0, low=34030.0, close=34090.0)
        spike = SpikeDetector.evaluate_candidate_run([c1, c2, c3])

        # Correction candle 1: Low 34050
        corr1 = self._make_candle(15, open_=34090.0, high=34092.0, low=34050.0, close=34060.0)
        # Correction candle 2: Low 34030, High 34065
        corr2 = self._make_candle(20, open_=34060.0, high=34065.0, low=34030.0, close=34035.0)
        # Confirming candle breaks corr2.high (34065): High 34075
        confirm_c = self._make_candle(25, open_=34035.0, high=34075.0, low=34032.0, close=34070.0)

        level = detect_first_bottom(spike, [corr1, corr2, confirm_c])
        self.assertIsNotNone(level)
        self.assertEqual(level.price, 34030.0)  # corr2.low

    def test_first_bottom_low_is_first_bottom_candle_low_not_previous_lower_candle(self) -> None:
        c1 = self._make_candle(0, open_=34000.0, high=34020.0, low=33990.0, close=34015.0)
        c2 = self._make_candle(5, open_=34020.0, high=34060.0, low=34018.0, close=34055.0)
        c3 = self._make_candle(10, open_=34055.0, high=34100.0, low=34030.0, close=34090.0)
        spike = SpikeDetector.evaluate_candidate_run([c1, c2, c3])

        # Correction candle 1 has a lower low: 34020
        corr1 = self._make_candle(15, open_=34090.0, high=34092.0, low=34020.0, close=34030.0)
        # Correction candle 2 is the First Bottom candle (bearish): Low 34035, High 34060
        corr2 = self._make_candle(20, open_=34050.0, high=34060.0, low=34035.0, close=34040.0)
        # Confirming candle breaks corr2.high (34060): High 34070
        confirm_c = self._make_candle(25, open_=34040.0, high=34070.0, low=34038.0, close=34065.0)

        level = detect_first_bottom(spike, [corr1, corr2, confirm_c])
        self.assertIsNotNone(level)
        # Must be corr2.low (34035.0), NOT min of all correction candles (34020.0)
        self.assertEqual(level.price, 34035.0)

    def test_first_bottom_no_correction(self) -> None:
        c1 = self._make_candle(0, open_=34000.0, high=34020.0, low=33990.0, close=34015.0)
        c2 = self._make_candle(5, open_=34020.0, high=34060.0, low=34018.0, close=34055.0)
        c3 = self._make_candle(10, open_=34055.0, high=34100.0, low=34030.0, close=34090.0)
        spike = SpikeDetector.evaluate_candidate_run([c1, c2, c3])

        # Next candle is bullish, no bearish correction occurs
        c4 = self._make_candle(15, open_=34090.0, high=34120.0, low=34085.0, close=34115.0)
        level = detect_first_bottom(spike, [c4])
        self.assertIsNone(level)

    def test_first_bottom_correction_without_confirmation_break(self) -> None:
        c1 = self._make_candle(0, open_=34000.0, high=34020.0, low=33990.0, close=34015.0)
        c2 = self._make_candle(5, open_=34020.0, high=34060.0, low=34018.0, close=34055.0)
        c3 = self._make_candle(10, open_=34055.0, high=34100.0, low=34030.0, close=34090.0)
        spike = SpikeDetector.evaluate_candidate_run([c1, c2, c3])

        # Bearish correction candle: High 34090
        corr1 = self._make_candle(15, open_=34085.0, high=34090.0, low=34040.0, close=34050.0)
        # Subsequent candle does not break corr1.high: High 34070 <= 34090
        subsequent = self._make_candle(20, open_=34050.0, high=34070.0, low=34045.0, close=34060.0)
        level = detect_first_bottom(spike, [corr1, subsequent])
        self.assertIsNone(level)

    # =========================================================================
    # 7. FIRST TOP TESTS (BLK-03)
    # =========================================================================

    def test_first_top_confirmation_and_immutability(self) -> None:
        # 1. Bearish Spike
        c1 = self._make_candle(0, open_=34100.0, high=34110.0, low=34070.0, close=34075.0)
        c2 = self._make_candle(5, open_=34075.0, high=34080.0, low=34030.0, close=34035.0)
        c3 = self._make_candle(10, open_=34035.0, high=34060.0, low=34000.0, close=34005.0)
        spike = SpikeDetector.evaluate_candidate_run([c1, c2, c3])
        self.assertIsNotNone(spike)

        # 2. Initial Bullish Correction candle
        corr1 = self._make_candle(15, open_=34005.0, high=34050.0, low=34002.0, close=34045.0)

        # 3. Confirming candle breaks correction candle low (BLK-03): Low 33995 < corr1.low (34002)
        confirm_c = self._make_candle(20, open_=34045.0, high=34048.0, low=33995.0, close=34000.0)

        level = detect_first_top(spike, [corr1, confirm_c])
        self.assertIsNotNone(level)
        self.assertEqual(level.direction, Direction.BEARISH)
        self.assertEqual(level.level_type, "TOP")
        self.assertEqual(level.price, 34050.0)  # First Top High
        self.assertTrue(level.is_locked)
        self.assertEqual(level.formation_time, confirm_c.close_time)

        # 4. Immutability verification: later candles trading higher do NOT change the locked level
        later_c = self._make_candle(25, open_=34000.0, high=34090.0, low=33998.0, close=34085.0)
        level_after = detect_first_top(spike, [corr1, confirm_c, later_c])
        self.assertIsNotNone(level_after)
        self.assertEqual(level_after.price, 34050.0)  # Remained strictly 34050.0

    def test_first_top_high_is_first_top_candle_high_not_previous_higher_candle(self) -> None:
        c1 = self._make_candle(0, open_=34100.0, high=34110.0, low=34070.0, close=34075.0)
        c2 = self._make_candle(5, open_=34075.0, high=34080.0, low=34030.0, close=34035.0)
        c3 = self._make_candle(10, open_=34035.0, high=34060.0, low=34000.0, close=34005.0)
        spike = SpikeDetector.evaluate_candidate_run([c1, c2, c3])

        # Correction candle 1 has a higher high: 34070
        corr1 = self._make_candle(15, open_=34005.0, high=34070.0, low=34002.0, close=34065.0)
        # Correction candle 2 is the First Top candle (bullish): High 34055, Low 34020
        corr2 = self._make_candle(20, open_=34030.0, high=34055.0, low=34020.0, close=34050.0)
        # Confirming candle breaks corr2.low (34020): Low 34015
        confirm_c = self._make_candle(25, open_=34050.0, high=34052.0, low=34015.0, close=34018.0)

        level = detect_first_top(spike, [corr1, corr2, confirm_c])
        self.assertIsNotNone(level)
        # Must be corr2.high (34055.0), NOT max of all correction candles (34070.0)
        self.assertEqual(level.price, 34055.0)

    def test_first_top_no_correction(self) -> None:
        c1 = self._make_candle(0, open_=34100.0, high=34110.0, low=34070.0, close=34075.0)
        c2 = self._make_candle(5, open_=34075.0, high=34080.0, low=34030.0, close=34035.0)
        c3 = self._make_candle(10, open_=34035.0, high=34060.0, low=34000.0, close=34005.0)
        spike = SpikeDetector.evaluate_candidate_run([c1, c2, c3])

        # Next candle continues bearish, no bullish correction occurs
        c4 = self._make_candle(15, open_=34005.0, high=34008.0, low=33960.0, close=33970.0)
        level = detect_first_top(spike, [c4])
        self.assertIsNone(level)

    def test_first_top_correction_without_confirmation_break(self) -> None:
        c1 = self._make_candle(0, open_=34100.0, high=34110.0, low=34070.0, close=34075.0)
        c2 = self._make_candle(5, open_=34075.0, high=34080.0, low=34030.0, close=34035.0)
        c3 = self._make_candle(10, open_=34035.0, high=34060.0, low=34000.0, close=34005.0)
        spike = SpikeDetector.evaluate_candidate_run([c1, c2, c3])

        # Bullish correction candle: Low 34002
        corr1 = self._make_candle(15, open_=34005.0, high=34050.0, low=34002.0, close=34045.0)
        # Subsequent candle does not break corr1.low: Low 34010 >= 34002
        subsequent = self._make_candle(20, open_=34045.0, high=34055.0, low=34010.0, close=34030.0)
        level = detect_first_top(spike, [corr1, subsequent])
        self.assertIsNone(level)

    # =========================================================================
    # 8. AVERAGE M5 RANGE TESTS
    # =========================================================================

    def test_average_m5_range_exact_mathematics(self) -> None:
        # 3 candles with known ranges: 40.0, 50.0, 60.0
        # Expected average = (40 + 50 + 60) / 3 = 50.0
        c1 = self._make_candle(0, open_=34000.0, high=34040.0, low=34000.0, close=34030.0)  # range = 40
        c2 = self._make_candle(5, open_=34030.0, high=34070.0, low=34020.0, close=34060.0)  # range = 50
        c3 = self._make_candle(10, open_=34060.0, high=34110.0, low=34050.0, close=34100.0)  # range = 60

        avg = calculate_average_m5_range([c1, c2, c3])
        self.assertAlmostEqual(avg, 50.0)

    def test_average_m5_range_tokyo_oceania_exclusion(self) -> None:
        # Candle 1 (Tokyo/Oceania excluded): range = 100
        # Candle 2 (valid): range = 40
        # Candle 3 (valid): range = 60
        # Expected average = (40 + 60) / 2 = 50.0
        c1 = self._make_candle(0, open_=34000.0, high=34100.0, low=34000.0, close=34080.0)
        c2 = self._make_candle(5, open_=34030.0, high=34070.0, low=34030.0, close=34060.0)
        c3 = self._make_candle(10, open_=34060.0, high=34120.0, low=34060.0, close=34100.0)

        # Filter excluding c1
        def mock_tokyo_filter(c: Candle) -> bool:
            return c.open_time == c1.open_time

        avg = calculate_average_m5_range([c1, c2, c3], is_tokyo_oceania_filter=mock_tokyo_filter)
        self.assertAlmostEqual(avg, 50.0)

    def test_average_m5_range_rejects_non_m5_candles(self) -> None:
        c1 = self._make_candle(0, open_=34000.0, high=34020.0, low=34000.0, close=34015.0, timeframe=Timeframe.M1)
        with self.assertRaises(DetectorError) as ctx:
            calculate_average_m5_range([c1])
        self.assertIn("must be M5", str(ctx.exception))

    def test_average_m5_range_no_atr_pure_range(self) -> None:
        # Gap between previous close and current open does NOT affect range (unlike ATR)
        # Previous close = 34000. Current open = 34050, high = 34060, low = 34045, close = 34055.
        # ATR True Range would be max(H-L, |H-Cp|, |L-Cp|) = 34060 - 34000 = 60.
        # Locked Candle Range is strictly High - Low = 34060 - 34045 = 15.
        c = self._make_candle(0, open_=34050.0, high=34060.0, low=34045.0, close=34055.0)
        candle_range = calculate_candle_range(c)
        self.assertEqual(candle_range, 15.0)
        self.assertNotEqual(candle_range, 60.0)  # Proves ATR is not used

    def test_average_m5_range_zero_valid_candles_raises_error(self) -> None:
        # If input list is empty or all candles are filtered out by session exclusion
        c1 = self._make_candle(0, open_=34000.0, high=34040.0, low=34000.0, close=34030.0)
        with self.assertRaises(DetectorError) as ctx:
            calculate_average_m5_range([], is_tokyo_oceania_filter=lambda _: True)
        self.assertIn("empty candle sequence", str(ctx.exception).lower())

        with self.assertRaises(DetectorError) as ctx2:
            calculate_average_m5_range([c1], is_tokyo_oceania_filter=lambda _: True)
        self.assertIn("zero valid m5 candles", str(ctx2.exception).lower())

    # =========================================================================
    # 9. DETERMINISM TESTS
    # =========================================================================

    def test_detector_determinism_identical_results(self) -> None:
        c1 = self._make_candle(0, open_=34000.0, high=34020.0, low=33990.0, close=34015.0)
        c2 = self._make_candle(5, open_=34020.0, high=34060.0, low=34018.0, close=34055.0)
        c3 = self._make_candle(10, open_=34055.0, high=34100.0, low=34030.0, close=34090.0)

        # Run multiple times
        res1 = SpikeDetector.evaluate_candidate_run([c1, c2, c3])
        res2 = SpikeDetector.evaluate_candidate_run([c1, c2, c3])

        self.assertIsNotNone(res1)
        self.assertIsNotNone(res2)
        self.assertEqual(res1.spike_id, res2.spike_id)
        self.assertEqual(res1.high, res2.high)
        self.assertEqual(res1.low, res2.low)
        self.assertEqual(res1.candle_count, res2.candle_count)

    # =========================================================================
    # 10. CAUSALITY TESTS
    # =========================================================================

    def test_causality_future_data_does_not_affect_past(self) -> None:
        # Past spike
        c1 = self._make_candle(0, open_=34000.0, high=34020.0, low=33990.0, close=34015.0)
        c2 = self._make_candle(5, open_=34020.0, high=34060.0, low=34018.0, close=34055.0)
        c3 = self._make_candle(10, open_=34055.0, high=34100.0, low=34030.0, close=34090.0)

        spikes_without_future = SpikeDetector.find_spikes([c1, c2, c3])
        self.assertEqual(len(spikes_without_future), 1)

        # Add wildly fluctuating future candles
        c4 = self._make_candle(15, open_=34090.0, high=35000.0, low=32000.0, close=32500.0)
        c5 = self._make_candle(20, open_=32500.0, high=32600.0, low=31000.0, close=31100.0)

        spikes_with_future = SpikeDetector.find_spikes([c1, c2, c3, c4, c5])
        self.assertEqual(len(spikes_with_future), 1)
        self.assertEqual(spikes_without_future[0].spike_id, spikes_with_future[0].spike_id)
        self.assertEqual(spikes_without_future[0].high, spikes_with_future[0].high)

    # =========================================================================
    # 11. CANONICAL SYMBOLS ONLY TESTS
    # =========================================================================

    def test_canonical_symbols_accepted(self) -> None:
        c_dj = self._make_candle(0, 34000, 34010, 33990, 34005, symbol=CanonicalSymbol.DOW_JONES)
        c_nas = self._make_candle(0, 15000, 15010, 14990, 15005, symbol=CanonicalSymbol.NASDAQ)
        self.assertEqual(detect_candle_direction(c_dj), Direction.BULLISH)
        self.assertEqual(detect_candle_direction(c_nas), Direction.BULLISH)

    def test_broker_symbols_rejected(self) -> None:
        with self.assertRaises(DataValidationError):
            # Attempt to instantiate Candle with raw broker symbol string
            Candle(
                symbol="US100_i",  # type: ignore
                timeframe=Timeframe.M5,
                open_time=self.base_time,
                close_time=self.base_time + timedelta(minutes=5),
                open=34000.0,
                high=34010.0,
                low=33990.0,
                close=34005.0,
                volume=10.0,
                spread=1.0,
            )


if __name__ == "__main__":
    unittest.main()
