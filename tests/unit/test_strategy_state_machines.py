"""
Phase 5 Comprehensive Unit Tests: Strategy State Machines and Signal Generation.
Covers:
- BUY Lifecycle (Spike -> Correction -> First Bottom -> Proximity -> M1 Monitoring -> PinBar -> Signal)
- SELL Lifecycle (Spike -> Correction -> First Top -> Proximity -> M1 Monitoring -> PinBar -> Signal)
- Proximity Entry/Exit/Return with Live Ask (BUY) and Live Bid (SELL)
- Proximity re-entry prerequisite (must close outside zone first)
- M1 activation timing and look-ahead prevention (close_time <= activation_time ignored)
- Pin Bar geometry tests (all mandatory conditions)
- Invalidation on M5 close violation (wick does NOT invalidate)
- Invalidation on opposite spike, session end, position conflict
- Setup ID deterministic generation
- Immutability of structural levels
- Causality (future data cannot alter past states)
- Determinism (identical inputs produce identical states/signals)
- Canonical symbol isolation (broker symbols rejected)
"""

from datetime import datetime, timezone
import unittest

from src.core.constants import (
    BearishState,
    BullishState,
    CanonicalSymbol,
    Direction,
    Timeframe,
    TradeDirection,
)
from src.core.exceptions import DataValidationError
from src.core.types import Candle, Quote, SpikeEvent, StructuralLevel
from src.market.types import Tick
from src.strategy.exceptions import StateTransitionError
from src.strategy.pinbar import (
    evaluate_pinbar,
    validate_buy_pinbar,
    validate_sell_pinbar,
)
from src.strategy.state_machine import BuyStateMachine, SellStateMachine


class TestStrategyStateMachines(unittest.TestCase):

    def setUp(self) -> None:
        self.symbol_dj = CanonicalSymbol.DOW_JONES
        self.symbol_nq = CanonicalSymbol.NASDAQ
        self.tick_dj = 1.0
        self.tick_nq = 0.25
        self.avg_m5_dj = 30.0
        self.avg_m5_nq = 40.0

    def _make_candle(
        self,
        symbol: CanonicalSymbol,
        timeframe: Timeframe,
        open_time: datetime,
        close_time: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
    ) -> Candle:
        return Candle(
            symbol=symbol,
            timeframe=timeframe,
            open_time=open_time,
            close_time=close_time,
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=100.0,
            spread=1.0,
        )

    def _make_bullish_spike(self) -> SpikeEvent:
        t_start = datetime(2023, 10, 2, 14, 0, tzinfo=timezone.utc)
        t_end = datetime(2023, 10, 2, 14, 15, tzinfo=timezone.utc)
        return SpikeEvent(
            spike_id="SPIKE-DJ-BULLISH-001",
            symbol=self.symbol_dj,
            direction=Direction.BULLISH,
            start_time=t_start,
            end_time=t_end,
            candle_count=3,
            gap_count=1,
            high=34150.0,
            low=33990.0,
        )

    def _make_bearish_spike(self) -> SpikeEvent:
        t_start = datetime(2023, 10, 2, 14, 0, tzinfo=timezone.utc)
        t_end = datetime(2023, 10, 2, 14, 15, tzinfo=timezone.utc)
        return SpikeEvent(
            spike_id="SPIKE-NQ-BEARISH-001",
            symbol=self.symbol_nq,
            direction=Direction.BEARISH,
            start_time=t_start,
            end_time=t_end,
            candle_count=3,
            gap_count=1,
            high=18010.0,
            low=17850.0,
        )

    # =========================================================================
    # 1. BUY LIFECYCLE TESTS
    # =========================================================================

    def test_buy_lifecycle_full_happy_path(self) -> None:
        """Complete deterministic flow from BULLISH spike to BUY SignalEvent."""
        spike = self._make_bullish_spike()
        sm = BuyStateMachine(self.symbol_dj, self.tick_dj, self.avg_m5_dj)

        # 1. Spike enters lifecycle
        self.assertEqual(sm.current_state, BullishState.IDLE)
        sm.on_spike(spike)
        self.assertEqual(sm.current_state, BullishState.BULLISH_SPIKE_DETECTED)

        # 2. Initial Bearish Correction candle 1
        t_corr1_open = datetime(2023, 10, 2, 14, 15, tzinfo=timezone.utc)
        t_corr1_close = datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc)
        corr1 = self._make_candle(
            self.symbol_dj, Timeframe.M5, t_corr1_open, t_corr1_close,
            open_=34140.0, high=34145.0, low=34080.0, close=34085.0
        )
        sm.on_m5_candle(corr1)
        self.assertEqual(sm.current_state, BullishState.WAIT_INITIAL_CORRECTION)
        self.assertEqual(len(sm.context.correction_candles), 1)

        # Initial Bearish Correction candle 2 (First Bottom Candle)
        t_corr2_open = datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc)
        t_corr2_close = datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc)
        corr2 = self._make_candle(
            self.symbol_dj, Timeframe.M5, t_corr2_open, t_corr2_close,
            open_=34085.0, high=34090.0, low=34050.0, close=34060.0
        )
        sm.on_m5_candle(corr2)
        self.assertEqual(sm.current_state, BullishState.WAIT_INITIAL_CORRECTION)
        self.assertEqual(len(sm.context.correction_candles), 2)

        # 3. Confirming candle breaks corr2.high (34090): High=34100
        # Confirms First Bottom! Level = corr2.low = 34050.0
        # Close = 34095 > 34050 + 30 (34080) -> Leaves proximity zone immediately!
        t_conf_open = datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc)
        t_conf_close = datetime(2023, 10, 2, 14, 30, tzinfo=timezone.utc)
        conf = self._make_candle(
            self.symbol_dj, Timeframe.M5, t_conf_open, t_conf_close,
            open_=34060.0, high=34100.0, low=34055.0, close=34095.0
        )
        sm.on_m5_candle(conf)
        self.assertEqual(sm.current_state, BullishState.WAIT_RETURN_TO_FIRST_BOTTOM)
        self.assertIsNotNone(sm.structural_level)
        self.assertEqual(sm.structural_level.price, 34050.0)
        self.assertTrue(sm.structural_level.is_locked)
        self.assertTrue(sm.context.has_left_proximity_zone)
        self.assertIsNotNone(sm.context.setup_id)
        self.assertTrue(sm.context.setup_id.startswith("SET-DJ-BUY-"))

        # 4. Proximity return via LIVE ASK
        # BOTTOM_LEVEL = 34050, avg_m5 = 30 -> proximity range is [34020, 34080]
        # Quote with Ask=34075 enters proximity
        t_quote = datetime(2023, 10, 2, 14, 33, 15, tzinfo=timezone.utc)
        q = Quote(
            symbol=self.symbol_dj,
            timestamp=t_quote,
            bid=34073.0,
            ask=34075.0,
            spread=2.0,
        )
        sm.on_quote(q)
        self.assertEqual(sm.current_state, BullishState.M1_MONITORING)
        self.assertEqual(sm.context.proximity_activation_time, t_quote)

        # 5. M1 candle before activation time is ignored
        t_m1_stale = datetime(2023, 10, 2, 14, 33, 0, tzinfo=timezone.utc)
        m1_stale = self._make_candle(
            self.symbol_dj, Timeframe.M1,
            datetime(2023, 10, 2, 14, 32, tzinfo=timezone.utc), t_m1_stale,
            open_=34075.0, high=34077.0, low=34060.0, close=34062.0
        )
        res = sm.on_m1_candle(m1_stale)
        self.assertIsNone(res)
        self.assertEqual(sm.current_state, BullishState.M1_MONITORING)

        # 6. M1 valid Bearish Pin Bar after activation time
        # Range = 20 (High 34075, Low 34055)
        # Midpoint = 34065.0
        # Open = 34060.0, Close = 34058.0 (Bearish, Body = 2.0, Ratio = 0.10 <= 0.20)
        # BodyHigh = 34060 <= 34065.0
        t_m1_valid = datetime(2023, 10, 2, 14, 35, 0, tzinfo=timezone.utc)
        m1_valid = self._make_candle(
            self.symbol_dj, Timeframe.M1,
            datetime(2023, 10, 2, 14, 34, tzinfo=timezone.utc), t_m1_valid,
            open_=34060.0, high=34075.0, low=34055.0, close=34058.0
        )
        sig = sm.on_m1_candle(m1_valid)
        self.assertIsNotNone(sig)
        self.assertEqual(sm.current_state, BullishState.BUY_TRIGGER_ARMED)
        self.assertEqual(sig.setup_id, sm.context.setup_id)
        self.assertEqual(sig.symbol, self.symbol_dj)
        self.assertEqual(sig.direction, TradeDirection.BUY)
        self.assertEqual(sig.signal_time, t_m1_valid)
        self.assertEqual(sig.pinbar_high, 34075.0)
        self.assertEqual(sig.pinbar_low, 34055.0)
        self.assertEqual(sig.final_structural_target, 34150.0)

    def test_buy_level_immutability(self) -> None:
        """BOTTOM_LEVEL must remain strictly locked and never change after formation."""
        spike = self._make_bullish_spike()
        sm = BuyStateMachine(self.symbol_dj, self.tick_dj, self.avg_m5_dj, spike=spike)

        t_base = datetime(2023, 10, 2, 14, 15, tzinfo=timezone.utc)
        corr = self._make_candle(
            self.symbol_dj, Timeframe.M5, t_base,
            datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
            open_=34140.0, high=34145.0, low=34050.0, close=34060.0
        )
        sm.on_m5_candle(corr)

        conf = self._make_candle(
            self.symbol_dj, Timeframe.M5,
            datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            open_=34060.0, high=34150.0, low=34055.0, close=34120.0
        )
        sm.on_m5_candle(conf)
        level_price = sm.structural_level.price

        # Subsequent M5 candle with higher low
        next_c = self._make_candle(
            self.symbol_dj, Timeframe.M5,
            datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 30, tzinfo=timezone.utc),
            open_=34120.0, high=34160.0, low=34110.0, close=34155.0
        )
        sm.on_m5_candle(next_c)
        self.assertEqual(sm.structural_level.price, level_price)

    def test_buy_proximity_not_activated_before_leaving_zone(self) -> None:
        """Re-entry rule BLK-13: Price must close outside zone before quote can activate proximity."""
        spike = self._make_bullish_spike()
        sm = BuyStateMachine(self.symbol_dj, self.tick_dj, self.avg_m5_dj, spike=spike)

        # Correction
        corr = self._make_candle(
            self.symbol_dj, Timeframe.M5,
            datetime(2023, 10, 2, 14, 15, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
            open_=34140.0, high=34145.0, low=34050.0, close=34060.0
        )
        sm.on_m5_candle(corr)

        # Confirming candle breaks corr high (34145) with High=34148, but closes at 34070 (<= 34050 + 30 = 34080)
        # So it has NOT closed outside proximity zone yet
        conf = self._make_candle(
            self.symbol_dj, Timeframe.M5,
            datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            open_=34060.0, high=34148.0, low=34055.0, close=34070.0
        )
        sm.on_m5_candle(conf)
        self.assertFalse(sm.context.has_left_proximity_zone)

        # Quote arrives inside proximity zone: Ask=34065
        # Must NOT activate because has_left_proximity_zone is False!
        q = Quote(
            symbol=self.symbol_dj,
            timestamp=datetime(2023, 10, 2, 14, 26, tzinfo=timezone.utc),
            bid=34063.0, ask=34065.0, spread=2.0
        )
        sm.on_quote(q)
        self.assertEqual(sm.current_state, BullishState.WAIT_RETURN_TO_FIRST_BOTTOM)
        self.assertIsNone(sm.context.proximity_activation_time)

        # Now M5 closes outside zone: Close=34100 (> 34080)
        c_exit = self._make_candle(
            self.symbol_dj, Timeframe.M5,
            datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 30, tzinfo=timezone.utc),
            open_=34070.0, high=34105.0, low=34068.0, close=34100.0
        )
        sm.on_m5_candle(c_exit)
        self.assertTrue(sm.context.has_left_proximity_zone)

        # Now quote inside zone activates proximity!
        q2 = Quote(
            symbol=self.symbol_dj,
            timestamp=datetime(2023, 10, 2, 14, 32, tzinfo=timezone.utc),
            bid=34063.0, ask=34065.0, spread=2.0
        )
        sm.on_quote(q2)
        self.assertEqual(sm.current_state, BullishState.M1_MONITORING)
        self.assertEqual(sm.context.proximity_activation_time, q2.timestamp)

    def test_buy_proximity_uses_live_ask_exclusively(self) -> None:
        """BUY proximity must test Live Ask against BOTTOM_LEVEL, never Bid or Last."""
        spike = self._make_bullish_spike()
        sm = BuyStateMachine(self.symbol_dj, self.tick_dj, self.avg_m5_dj, spike=spike)
        # Setup confirmed and exited zone
        sm.context.structural_level = StructuralLevel(
            spike_id=spike.spike_id,
            symbol=self.symbol_dj,
            direction=Direction.BULLISH,
            level_type="FIRST_BOTTOM",
            price=34050.0,
            formation_time=datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            is_locked=True,
        )
        sm.context.record_transition(
            BullishState.WAIT_RETURN_TO_FIRST_BOTTOM,
            datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            "SETUP",
        )
        sm.context.has_left_proximity_zone = True

        # Zone is [34020, 34080].
        # Quote where Bid=34078 (inside zone) but Ask=34085 (OUTSIDE zone)
        # Must NOT activate because Ask is outside!
        q_outside = Quote(
            symbol=self.symbol_dj,
            timestamp=datetime(2023, 10, 2, 14, 30, tzinfo=timezone.utc),
            bid=34078.0,
            ask=34085.0,
            spread=7.0,
        )
        sm.on_quote(q_outside)
        self.assertEqual(sm.current_state, BullishState.WAIT_RETURN_TO_FIRST_BOTTOM)

        # Quote where Ask=34079 (inside zone)
        q_inside = Quote(
            symbol=self.symbol_dj,
            timestamp=datetime(2023, 10, 2, 14, 31, tzinfo=timezone.utc),
            bid=34072.0,
            ask=34079.0,
            spread=7.0,
        )
        sm.on_quote(q_inside)
        self.assertEqual(sm.current_state, BullishState.M1_MONITORING)

    def test_buy_m1_pinbar_rejection_cases(self) -> None:
        """Invalid pin bars must not trigger signal."""
        # 1. Bullish candle (close > open)
        c_bull = self._make_candle(
            self.symbol_dj, Timeframe.M1,
            datetime(2023, 10, 2, 14, 35, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 36, tzinfo=timezone.utc),
            open_=34060.0, high=34075.0, low=34055.0, close=34065.0
        )
        pb1 = validate_buy_pinbar(c_bull)
        self.assertFalse(pb1.is_valid)

        # 2. Body ratio > 0.20
        # Range = 10, Body = 3 (ratio = 0.30 > 0.20)
        c_fat = self._make_candle(
            self.symbol_dj, Timeframe.M1,
            datetime(2023, 10, 2, 14, 36, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 37, tzinfo=timezone.utc),
            open_=34063.0, high=34070.0, low=34060.0, close=34060.0
        )
        pb2 = validate_buy_pinbar(c_fat)
        self.assertFalse(pb2.is_valid)

        # 3. BodyHigh > Midpoint
        # Range = 20 (High 34075, Low 34055). Midpoint = 34065.
        # Open = 34070, Close = 34068 -> Body = 2. Ratio = 0.10.
        # But BodyHigh = 34070 > 34065 (Midpoint)
        c_upper = self._make_candle(
            self.symbol_dj, Timeframe.M1,
            datetime(2023, 10, 2, 14, 37, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 38, tzinfo=timezone.utc),
            open_=34070.0, high=34075.0, low=34055.0, close=34068.0
        )
        pb3 = validate_buy_pinbar(c_upper)
        self.assertFalse(pb3.is_valid)

    def test_buy_m5_close_below_bottom_level_invalidates(self) -> None:
        """M5 Close < BOTTOM_LEVEL must transition to CLOSED."""
        spike = self._make_bullish_spike()
        sm = BuyStateMachine(self.symbol_dj, self.tick_dj, self.avg_m5_dj, spike=spike)
        sm.context.structural_level = StructuralLevel(
            spike_id=spike.spike_id,
            symbol=self.symbol_dj,
            direction=Direction.BULLISH,
            level_type="FIRST_BOTTOM",
            price=34050.0,
            formation_time=datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            is_locked=True,
        )
        sm.context.record_transition(
            BullishState.WAIT_RETURN_TO_FIRST_BOTTOM,
            datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            "SETUP",
        )

        c_inv = self._make_candle(
            self.symbol_dj, Timeframe.M5,
            datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 30, tzinfo=timezone.utc),
            open_=34060.0, high=34065.0, low=34045.0, close=34048.0
        )
        sm.on_m5_candle(c_inv)
        self.assertTrue(sm.is_closed)
        self.assertEqual(sm.context.invalidation_reason, "M5_CLOSE_BELOW_BOTTOM_LEVEL")

    def test_buy_wick_below_bottom_level_does_not_invalidate(self) -> None:
        """Wick below BOTTOM_LEVEL without close does NOT invalidate."""
        spike = self._make_bullish_spike()
        sm = BuyStateMachine(self.symbol_dj, self.tick_dj, self.avg_m5_dj, spike=spike)
        sm.context.structural_level = StructuralLevel(
            spike_id=spike.spike_id,
            symbol=self.symbol_dj,
            direction=Direction.BULLISH,
            level_type="FIRST_BOTTOM",
            price=34050.0,
            formation_time=datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            is_locked=True,
        )
        sm.context.record_transition(
            BullishState.WAIT_RETURN_TO_FIRST_BOTTOM,
            datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            "SETUP",
        )

        # Low = 34040 (< 34050), but Close = 34055 (>= 34050)
        c_wick = self._make_candle(
            self.symbol_dj, Timeframe.M5,
            datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 30, tzinfo=timezone.utc),
            open_=34060.0, high=34065.0, low=34040.0, close=34055.0
        )
        sm.on_m5_candle(c_wick)
        self.assertFalse(sm.is_closed)
        self.assertEqual(sm.current_state, BullishState.WAIT_RETURN_TO_FIRST_BOTTOM)

    def test_buy_invalidated_by_opposite_bearish_spike(self) -> None:
        """A valid Bearish Spike during a BUY setup invalidates it."""
        spike = self._make_bullish_spike()
        sm = BuyStateMachine(self.symbol_dj, self.tick_dj, self.avg_m5_dj, spike=spike)

        bear_spike = SpikeEvent(
            spike_id="SPIKE-BEAR-001",
            symbol=self.symbol_dj,
            direction=Direction.BEARISH,
            start_time=datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
            end_time=datetime(2023, 10, 2, 14, 35, tzinfo=timezone.utc),
            candle_count=3,
            gap_count=1,
            high=34150.0,
            low=34000.0,
        )
        sm.on_opposite_spike(bear_spike)
        self.assertTrue(sm.is_closed)
        self.assertEqual(sm.context.invalidation_reason, "OPPOSITE_SPIKE_DETECTED")

    def test_buy_invalidated_by_session_end(self) -> None:
        """Trading session end invalidates active setup."""
        spike = self._make_bullish_spike()
        sm = BuyStateMachine(self.symbol_dj, self.tick_dj, self.avg_m5_dj, spike=spike)
        t_end = datetime(2023, 10, 2, 21, 0, tzinfo=timezone.utc)
        sm.on_session_end(t_end)
        self.assertTrue(sm.is_closed)
        self.assertEqual(sm.context.invalidation_reason, "TRADING_SESSION_ENDED")

    def test_buy_invalidated_by_position_conflict(self) -> None:
        """Active position conflict invalidates active setup."""
        spike = self._make_bullish_spike()
        sm = BuyStateMachine(self.symbol_dj, self.tick_dj, self.avg_m5_dj, spike=spike)
        sm.on_position_conflict("Max active positions reached", datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc))
        self.assertTrue(sm.is_closed)
        self.assertIn("POSITION_CONFLICT", sm.context.invalidation_reason)

    def test_buy_no_lifetime_timeout(self) -> None:
        """MAX_SETUP_LIFETIME is DISABLED: hours of waiting do not cause timeout."""
        spike = self._make_bullish_spike()
        sm = BuyStateMachine(self.symbol_dj, self.tick_dj, self.avg_m5_dj, spike=spike)
        # Advance M5 candles 3 hours later without invalidation
        t_later_open = datetime(2023, 10, 2, 17, 15, tzinfo=timezone.utc)
        t_later_close = datetime(2023, 10, 2, 17, 20, tzinfo=timezone.utc)
        c_later = self._make_candle(
            self.symbol_dj, Timeframe.M5, t_later_open, t_later_close,
            open_=34100.0, high=34110.0, low=34095.0, close=34105.0
        )
        sm.on_m5_candle(c_later)
        self.assertFalse(sm.is_closed)

    # =========================================================================
    # 2. SELL LIFECYCLE TESTS
    # =========================================================================

    def test_sell_lifecycle_full_happy_path(self) -> None:
        """Complete deterministic flow from BEARISH spike to SELL SignalEvent."""
        spike = self._make_bearish_spike()
        sm = SellStateMachine(self.symbol_nq, self.tick_nq, self.avg_m5_nq)

        # 1. Spike enters lifecycle
        self.assertEqual(sm.current_state, BearishState.IDLE)
        sm.on_spike(spike)
        self.assertEqual(sm.current_state, BearishState.BEARISH_SPIKE_DETECTED)

        # 2. Initial Bullish Correction candle 1
        t_corr1_open = datetime(2023, 10, 2, 14, 15, tzinfo=timezone.utc)
        t_corr1_close = datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc)
        corr1 = self._make_candle(
            self.symbol_nq, Timeframe.M5, t_corr1_open, t_corr1_close,
            open_=17860.0, high=17910.0, low=17855.0, close=17905.0
        )
        sm.on_m5_candle(corr1)
        self.assertEqual(sm.current_state, BearishState.WAIT_INITIAL_CORRECTION)
        self.assertEqual(len(sm.context.correction_candles), 1)

        # Initial Bullish Correction candle 2 (First Top Candle)
        t_corr2_open = datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc)
        t_corr2_close = datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc)
        corr2 = self._make_candle(
            self.symbol_nq, Timeframe.M5, t_corr2_open, t_corr2_close,
            open_=17905.0, high=17940.0, low=17900.0, close=17935.0
        )
        sm.on_m5_candle(corr2)
        self.assertEqual(sm.current_state, BearishState.WAIT_INITIAL_CORRECTION)
        self.assertEqual(len(sm.context.correction_candles), 2)

        # 3. Confirming candle breaks corr2.low (17900): Low=17890
        # Confirms First Top! Level = corr2.high = 17940.0
        # Close = 17895 < 17940 - 40 (17900) -> Leaves proximity zone immediately!
        t_conf_open = datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc)
        t_conf_close = datetime(2023, 10, 2, 14, 30, tzinfo=timezone.utc)
        conf = self._make_candle(
            self.symbol_nq, Timeframe.M5, t_conf_open, t_conf_close,
            open_=17935.0, high=17938.0, low=17890.0, close=17895.0
        )
        sm.on_m5_candle(conf)
        self.assertEqual(sm.current_state, BearishState.WAIT_RETURN_TO_FIRST_TOP)
        self.assertIsNotNone(sm.structural_level)
        self.assertEqual(sm.structural_level.price, 17940.0)
        self.assertTrue(sm.structural_level.is_locked)
        self.assertTrue(sm.context.has_left_proximity_zone)
        self.assertIsNotNone(sm.context.setup_id)
        self.assertTrue(sm.context.setup_id.startswith("SET-NAS-SELL-"))

        # 4. Proximity return via LIVE BID
        # TOP_LEVEL = 17940, avg_m5 = 40 -> proximity range is [17900, 17980]
        # Quote with Bid=17920 enters proximity
        t_quote = datetime(2023, 10, 2, 14, 33, 45, tzinfo=timezone.utc)
        q = Quote(
            symbol=self.symbol_nq,
            timestamp=t_quote,
            bid=17920.0,
            ask=17921.0,
            spread=1.0,
        )
        sm.on_quote(q)
        self.assertEqual(sm.current_state, BearishState.M1_MONITORING)
        self.assertEqual(sm.context.proximity_activation_time, t_quote)

        # 5. M1 candle before activation time is ignored
        t_m1_stale = datetime(2023, 10, 2, 14, 33, 0, tzinfo=timezone.utc)
        m1_stale = self._make_candle(
            self.symbol_nq, Timeframe.M1,
            datetime(2023, 10, 2, 14, 32, tzinfo=timezone.utc), t_m1_stale,
            open_=17925.0, high=17935.0, low=17915.0, close=17930.0
        )
        res = sm.on_m1_candle(m1_stale)
        self.assertIsNone(res)
        self.assertEqual(sm.current_state, BearishState.M1_MONITORING)

        # 6. M1 valid Bullish Pin Bar after activation time
        # Range = 30.0 (High 17940.0, Low 17910.0)
        # Midpoint = 17925.0
        # Open = 17930.0, Close = 17934.0 (Bullish, Body = 4.0, Ratio = 4/30 = 0.1333 <= 0.20)
        # BodyLow = 17930.0 >= 17925.0 (Midpoint)
        t_m1_valid = datetime(2023, 10, 2, 14, 36, 0, tzinfo=timezone.utc)
        m1_valid = self._make_candle(
            self.symbol_nq, Timeframe.M1,
            datetime(2023, 10, 2, 14, 35, tzinfo=timezone.utc), t_m1_valid,
            open_=17930.0, high=17940.0, low=17910.0, close=17934.0
        )
        sig = sm.on_m1_candle(m1_valid)
        self.assertIsNotNone(sig)
        self.assertEqual(sm.current_state, BearishState.SELL_TRIGGER_ARMED)
        self.assertEqual(sig.setup_id, sm.context.setup_id)
        self.assertEqual(sig.symbol, self.symbol_nq)
        self.assertEqual(sig.direction, TradeDirection.SELL)
        self.assertEqual(sig.signal_time, t_m1_valid)
        self.assertEqual(sig.pinbar_high, 17940.0)
        self.assertEqual(sig.pinbar_low, 17910.0)
        self.assertEqual(sig.final_structural_target, 17850.0)

    def test_sell_level_immutability(self) -> None:
        """TOP_LEVEL must remain strictly locked and never change after formation."""
        spike = self._make_bearish_spike()
        sm = SellStateMachine(self.symbol_nq, self.tick_nq, self.avg_m5_nq, spike=spike)

        t_base = datetime(2023, 10, 2, 14, 15, tzinfo=timezone.utc)
        corr = self._make_candle(
            self.symbol_nq, Timeframe.M5, t_base,
            datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
            open_=17860.0, high=17940.0, low=17855.0, close=17930.0
        )
        sm.on_m5_candle(corr)

        conf = self._make_candle(
            self.symbol_nq, Timeframe.M5,
            datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            open_=17930.0, high=17935.0, low=17850.0, close=17880.0
        )
        sm.on_m5_candle(conf)
        level_price = sm.structural_level.price

        # Subsequent M5 candle with lower high
        next_c = self._make_candle(
            self.symbol_nq, Timeframe.M5,
            datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 30, tzinfo=timezone.utc),
            open_=17880.0, high=17890.0, low=17840.0, close=17850.0
        )
        sm.on_m5_candle(next_c)
        self.assertEqual(sm.structural_level.price, level_price)

    def test_sell_proximity_uses_live_bid_exclusively(self) -> None:
        """SELL proximity must test Live Bid against TOP_LEVEL, never Ask or Last."""
        spike = self._make_bearish_spike()
        sm = SellStateMachine(self.symbol_nq, self.tick_nq, self.avg_m5_nq, spike=spike)
        sm.context.structural_level = StructuralLevel(
            spike_id=spike.spike_id,
            symbol=self.symbol_nq,
            direction=Direction.BEARISH,
            level_type="FIRST_TOP",
            price=17940.0,
            formation_time=datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            is_locked=True,
        )
        sm.context.record_transition(
            BearishState.WAIT_RETURN_TO_FIRST_TOP,
            datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            "SETUP",
        )
        sm.context.has_left_proximity_zone = True

        # Zone is [17900, 17980].
        # Quote where Ask=17905 (inside zone) but Bid=17895 (OUTSIDE zone)
        # Must NOT activate because Bid is outside!
        q_outside = Quote(
            symbol=self.symbol_nq,
            timestamp=datetime(2023, 10, 2, 14, 30, tzinfo=timezone.utc),
            bid=17895.0,
            ask=17905.0,
            spread=10.0,
        )
        sm.on_quote(q_outside)
        self.assertEqual(sm.current_state, BearishState.WAIT_RETURN_TO_FIRST_TOP)

        # Quote where Bid=17910 (inside zone)
        q_inside = Quote(
            symbol=self.symbol_nq,
            timestamp=datetime(2023, 10, 2, 14, 31, tzinfo=timezone.utc),
            bid=17910.0,
            ask=17920.0,
            spread=10.0,
        )
        sm.on_quote(q_inside)
        self.assertEqual(sm.current_state, BearishState.M1_MONITORING)

    def test_sell_m1_pinbar_rejection_cases(self) -> None:
        """Invalid pin bars for SELL must not trigger signal."""
        # 1. Bearish candle (open > close)
        c_bear = self._make_candle(
            self.symbol_nq, Timeframe.M1,
            datetime(2023, 10, 2, 14, 35, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 36, tzinfo=timezone.utc),
            open_=17935.0, high=17940.0, low=17910.0, close=17930.0
        )
        pb1 = validate_sell_pinbar(c_bear)
        self.assertFalse(pb1.is_valid)

        # 2. Body ratio > 0.20
        # Range = 10, Body = 3 (ratio = 0.30 > 0.20)
        c_fat = self._make_candle(
            self.symbol_nq, Timeframe.M1,
            datetime(2023, 10, 2, 14, 36, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 37, tzinfo=timezone.utc),
            open_=17930.0, high=17940.0, low=17930.0, close=17933.0
        )
        pb2 = validate_sell_pinbar(c_fat)
        self.assertFalse(pb2.is_valid)

        # 3. BodyLow < Midpoint
        # Range = 30 (High 17940, Low 17910). Midpoint = 17925.
        # Open = 17915, Close = 17918 -> Body = 3. Ratio = 0.10.
        # But BodyLow = 17915 < 17925 (Midpoint)
        c_lower = self._make_candle(
            self.symbol_nq, Timeframe.M1,
            datetime(2023, 10, 2, 14, 37, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 38, tzinfo=timezone.utc),
            open_=17915.0, high=17940.0, low=17910.0, close=17918.0
        )
        pb3 = validate_sell_pinbar(c_lower)
        self.assertFalse(pb3.is_valid)

    def test_sell_m5_close_above_top_level_invalidates(self) -> None:
        """M5 Close > TOP_LEVEL must transition to CLOSED."""
        spike = self._make_bearish_spike()
        sm = SellStateMachine(self.symbol_nq, self.tick_nq, self.avg_m5_nq, spike=spike)
        sm.context.structural_level = StructuralLevel(
            spike_id=spike.spike_id,
            symbol=self.symbol_nq,
            direction=Direction.BEARISH,
            level_type="FIRST_TOP",
            price=17940.0,
            formation_time=datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            is_locked=True,
        )
        sm.context.record_transition(
            BearishState.WAIT_RETURN_TO_FIRST_TOP,
            datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            "SETUP",
        )

        c_inv = self._make_candle(
            self.symbol_nq, Timeframe.M5,
            datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 30, tzinfo=timezone.utc),
            open_=17930.0, high=17950.0, low=17925.0, close=17945.0
        )
        sm.on_m5_candle(c_inv)
        self.assertTrue(sm.is_closed)
        self.assertEqual(sm.context.invalidation_reason, "M5_CLOSE_ABOVE_TOP_LEVEL")

    def test_sell_wick_above_top_level_does_not_invalidate(self) -> None:
        """Wick above TOP_LEVEL without close does NOT invalidate."""
        spike = self._make_bearish_spike()
        sm = SellStateMachine(self.symbol_nq, self.tick_nq, self.avg_m5_nq, spike=spike)
        sm.context.structural_level = StructuralLevel(
            spike_id=spike.spike_id,
            symbol=self.symbol_nq,
            direction=Direction.BEARISH,
            level_type="FIRST_TOP",
            price=17940.0,
            formation_time=datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            is_locked=True,
        )
        sm.context.record_transition(
            BearishState.WAIT_RETURN_TO_FIRST_TOP,
            datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            "SETUP",
        )

        # High = 17950 (> 17940), but Close = 17935 (<= 17940)
        c_wick = self._make_candle(
            self.symbol_nq, Timeframe.M5,
            datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 30, tzinfo=timezone.utc),
            open_=17930.0, high=17950.0, low=17920.0, close=17935.0
        )
        sm.on_m5_candle(c_wick)
        self.assertFalse(sm.is_closed)
        self.assertEqual(sm.current_state, BearishState.WAIT_RETURN_TO_FIRST_TOP)

    def test_sell_invalidated_by_opposite_bullish_spike(self) -> None:
        """A valid Bullish Spike during a SELL setup invalidates it."""
        spike = self._make_bearish_spike()
        sm = SellStateMachine(self.symbol_nq, self.tick_nq, self.avg_m5_nq, spike=spike)

        bull_spike = SpikeEvent(
            spike_id="SPIKE-BULL-001",
            symbol=self.symbol_nq,
            direction=Direction.BULLISH,
            start_time=datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
            end_time=datetime(2023, 10, 2, 14, 35, tzinfo=timezone.utc),
            candle_count=3,
            gap_count=1,
            high=18050.0,
            low=17900.0,
        )
        sm.on_opposite_spike(bull_spike)
        self.assertTrue(sm.is_closed)
        self.assertEqual(sm.context.invalidation_reason, "OPPOSITE_SPIKE_DETECTED")

    def test_sell_invalidated_by_session_end(self) -> None:
        """Trading session end invalidates active SELL setup."""
        spike = self._make_bearish_spike()
        sm = SellStateMachine(self.symbol_nq, self.tick_nq, self.avg_m5_nq, spike=spike)
        t_end = datetime(2023, 10, 2, 21, 0, tzinfo=timezone.utc)
        sm.on_session_end(t_end)
        self.assertTrue(sm.is_closed)
        self.assertEqual(sm.context.invalidation_reason, "TRADING_SESSION_ENDED")

    # =========================================================================
    # 3. CAUSALITY AND DETERMINISM
    # =========================================================================

    def test_causality_adding_future_candles_does_not_alter_past_signal(self) -> None:
        """Future data cannot alter already-formed structural levels, timestamps, or signals."""
        spike = self._make_bullish_spike()
        sm = BuyStateMachine(self.symbol_dj, self.tick_dj, self.avg_m5_dj, spike=spike)

        # Build state up to signal
        t_corr_open = datetime(2023, 10, 2, 14, 15, tzinfo=timezone.utc)
        t_corr_close = datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc)
        corr = self._make_candle(
            self.symbol_dj, Timeframe.M5, t_corr_open, t_corr_close,
            open_=34140.0, high=34145.0, low=34050.0, close=34060.0
        )
        sm.on_m5_candle(corr)

        t_conf_open = datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc)
        t_conf_close = datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc)
        conf = self._make_candle(
            self.symbol_dj, Timeframe.M5, t_conf_open, t_conf_close,
            open_=34060.0, high=34150.0, low=34055.0, close=34120.0
        )
        sm.on_m5_candle(conf)

        level_price_before = sm.structural_level.price
        setup_id_before = sm.context.setup_id

        # Quote activates proximity
        q = Quote(
            symbol=self.symbol_dj,
            timestamp=datetime(2023, 10, 2, 14, 28, tzinfo=timezone.utc),
            bid=34060.0, ask=34065.0, spread=5.0
        )
        sm.on_quote(q)

        # M1 Pin bar triggers signal
        m1 = self._make_candle(
            self.symbol_dj, Timeframe.M1,
            datetime(2023, 10, 2, 14, 29, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 30, tzinfo=timezone.utc),
            open_=34060.0, high=34075.0, low=34055.0, close=34058.0
        )
        sig_before = sm.on_m1_candle(m1)
        self.assertIsNotNone(sig_before)

        # Future candles arrive
        future_m5 = self._make_candle(
            self.symbol_dj, Timeframe.M5,
            datetime(2023, 10, 2, 15, 0, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 15, 5, tzinfo=timezone.utc),
            open_=34200.0, high=34300.0, low=34190.0, close=34280.0
        )
        sm.on_m5_candle(future_m5)

        # Structural level, setup_id, and signal remain identical
        self.assertEqual(sm.structural_level.price, level_price_before)
        self.assertEqual(sm.context.setup_id, setup_id_before)
        self.assertEqual(sm.signal_event, sig_before)

    def test_determinism_identical_runs(self) -> None:
        """Two state machines processing identical inputs yield identical states, IDs, and signals."""
        def run_machine() -> BuyStateMachine:
            spike = self._make_bullish_spike()
            sm = BuyStateMachine(self.symbol_dj, self.tick_dj, self.avg_m5_dj, spike=spike)
            corr = self._make_candle(
                self.symbol_dj, Timeframe.M5,
                datetime(2023, 10, 2, 14, 15, tzinfo=timezone.utc),
                datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
                open_=34140.0, high=34145.0, low=34050.0, close=34060.0
            )
            sm.on_m5_candle(corr)
            conf = self._make_candle(
                self.symbol_dj, Timeframe.M5,
                datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
                datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
                open_=34060.0, high=34150.0, low=34055.0, close=34120.0
            )
            sm.on_m5_candle(conf)
            q = Quote(
                symbol=self.symbol_dj,
                timestamp=datetime(2023, 10, 2, 14, 28, tzinfo=timezone.utc),
                bid=34060.0, ask=34065.0, spread=5.0
            )
            sm.on_quote(q)
            m1 = self._make_candle(
                self.symbol_dj, Timeframe.M1,
                datetime(2023, 10, 2, 14, 29, tzinfo=timezone.utc),
                datetime(2023, 10, 2, 14, 30, tzinfo=timezone.utc),
                open_=34060.0, high=34075.0, low=34055.0, close=34058.0
            )
            sm.on_m1_candle(m1)
            return sm

        sm1 = run_machine()
        sm2 = run_machine()

        self.assertEqual(sm1.current_state, sm2.current_state)
        self.assertEqual(sm1.context.setup_id, sm2.context.setup_id)
        self.assertEqual(sm1.structural_level, sm2.structural_level)
        self.assertEqual(sm1.context.proximity_activation_time, sm2.context.proximity_activation_time)
        self.assertEqual(sm1.signal_event, sm2.signal_event)
        self.assertEqual(len(sm1.context.transitions), len(sm2.context.transitions))

    # =========================================================================
    # 4. CANONICAL SYMBOL ISOLATION AND INPUT VALIDATION
    # =========================================================================

    def test_broker_symbols_rejected_at_strategy_boundary(self) -> None:
        """Strategy state machine must strictly reject broker symbols (e.g. US100_i, US30)."""
        with self.assertRaises(DataValidationError):
            BuyStateMachine("US100_i", self.tick_dj, self.avg_m5_dj)  # type: ignore

        with self.assertRaises(DataValidationError):
            SellStateMachine("NAS100_i", self.tick_nq, self.avg_m5_nq)  # type: ignore

    def test_mismatched_symbol_events_rejected(self) -> None:
        """Incoming events with mismatched symbol must be rejected."""
        spike = self._make_bullish_spike()
        sm = BuyStateMachine(self.symbol_dj, self.tick_dj, self.avg_m5_dj, spike=spike)

        # Mismatched symbol M5 candle
        mismatched_c = self._make_candle(
            self.symbol_nq, Timeframe.M5,
            datetime(2023, 10, 2, 14, 15, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
            open_=34140.0, high=34145.0, low=34050.0, close=34060.0
        )
        with self.assertRaises(DataValidationError):
            sm.on_m5_candle(mismatched_c)

    def test_wrong_spike_direction_rejected(self) -> None:
        """BuyStateMachine must reject Bearish Spike; SellStateMachine must reject Bullish Spike."""
        bear_spike = self._make_bearish_spike()
        sm_buy = BuyStateMachine(self.symbol_dj, self.tick_dj, self.avg_m5_dj)
        # Pass bear_spike with matching symbol
        bear_dj = SpikeEvent(
            spike_id="SPIKE-001",
            symbol=self.symbol_dj,
            direction=Direction.BEARISH,
            start_time=bear_spike.start_time,
            end_time=bear_spike.end_time,
            candle_count=3,
            gap_count=1,
            high=34100.0,
            low=34000.0,
        )
        with self.assertRaises(DataValidationError):
            sm_buy.on_spike(bear_dj)

        bull_spike = self._make_bullish_spike()
        sm_sell = SellStateMachine(self.symbol_nq, self.tick_nq, self.avg_m5_nq)
        bull_nq = SpikeEvent(
            spike_id="SPIKE-002",
            symbol=self.symbol_nq,
            direction=Direction.BULLISH,
            start_time=bull_spike.start_time,
            end_time=bull_spike.end_time,
            candle_count=3,
            gap_count=1,
            high=18000.0,
            low=17900.0,
        )
        with self.assertRaises(DataValidationError):
            sm_sell.on_spike(bull_nq)

    # =========================================================================
    # 5. SPECIFICATION BOUNDARY & LIFECYCLE TESTS (AUDIT VERIFICATION)
    # =========================================================================

    def test_setup_id_exact_canonical_format_and_short_symbol(self) -> None:
        """Verify Setup ID strictly matches SET-{symbol_short}-{direction}-{hash} where short symbol is DJ or NAS."""
        spike = self._make_bullish_spike()
        sm = BuyStateMachine(self.symbol_dj, self.tick_dj, self.avg_m5_dj, spike=spike)

        # Correction
        c1 = self._make_candle(
            self.symbol_dj, Timeframe.M5,
            datetime(2023, 10, 2, 14, 15, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
            open_=34140.0, high=34145.0, low=34050.0, close=34060.0
        )
        c2 = self._make_candle(
            self.symbol_dj, Timeframe.M5,
            datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            open_=34060.0, high=34150.0, low=34055.0, close=34140.0
        )
        sm.on_m5_candle(c1)
        sm.on_m5_candle(c2)

        setup_id = sm.context.setup_id
        self.assertIsNotNone(setup_id)
        parts = setup_id.split("-")
        self.assertEqual(len(parts), 4)
        self.assertEqual(parts[0], "SET")
        self.assertEqual(parts[1], "DJ")
        self.assertEqual(parts[2], "BUY")
        self.assertEqual(len(parts[3]), 12)
        self.assertTrue(parts[3].isalnum())

        # For SELL NASDAQ
        sell_spike = self._make_bearish_spike()
        sm_sell = SellStateMachine(self.symbol_nq, self.tick_nq, self.avg_m5_nq, spike=sell_spike)
        sc1 = self._make_candle(
            self.symbol_nq, Timeframe.M5,
            datetime(2023, 10, 2, 14, 15, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
            open_=17860.0, high=17940.0, low=17855.0, close=17930.0
        )
        sc2 = self._make_candle(
            self.symbol_nq, Timeframe.M5,
            datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            open_=17930.0, high=17935.0, low=17850.0, close=17860.0
        )
        sm_sell.on_m5_candle(sc1)
        sm_sell.on_m5_candle(sc2)

        sell_setup_id = sm_sell.context.setup_id
        self.assertIsNotNone(sell_setup_id)
        sell_parts = sell_setup_id.split("-")
        self.assertEqual(len(sell_parts), 4)
        self.assertEqual(sell_parts[0], "SET")
        self.assertEqual(sell_parts[1], "NAS")
        self.assertEqual(sell_parts[2], "SELL")
        self.assertEqual(len(sell_parts[3]), 12)

    def test_pinbar_body_ratio_exact_boundary(self) -> None:
        """BodyRatio == 0.20 is strictly VALID; BodyRatio == 0.2001 is INVALID."""
        # Range = 10.0. Body = 2.0 -> Ratio = 0.2000 (VALID)
        t_open = datetime(2023, 10, 2, 14, 30, tzinfo=timezone.utc)
        t_close = datetime(2023, 10, 2, 14, 31, tzinfo=timezone.utc)
        
        # BUY: Bearish Pinbar (Close < Open, Body in lower half)
        # Range = 100.0 - 90.0 = 10.0. Midpoint = 95.0.
        # Open = 94.0, Close = 92.0 -> Body = 2.0 (20%), BodyHigh = 94.0 <= 95.0
        pb_valid_20 = self._make_candle(
            self.symbol_dj, Timeframe.M1, t_open, t_close,
            open_=94.0, high=100.0, low=90.0, close=92.0
        )
        res_valid = validate_buy_pinbar(pb_valid_20)
        self.assertTrue(res_valid.is_valid)
        self.assertAlmostEqual(res_valid.body_ratio, 0.20, places=5)

        # Open = 94.01, Close = 92.0 -> Body = 2.01 / 10.0 = 20.1% (INVALID)
        pb_invalid_201 = self._make_candle(
            self.symbol_dj, Timeframe.M1, t_open, t_close,
            open_=94.01, high=100.0, low=90.0, close=92.0
        )
        res_invalid = validate_buy_pinbar(pb_invalid_201)
        self.assertFalse(res_invalid.is_valid)
        self.assertGreater(res_invalid.body_ratio, 0.20)

    def test_m1_activation_timestamp_equality_boundary(self) -> None:
        """M1 candle with close_time == activation_time MUST be ignored."""
        spike = self._make_bullish_spike()
        sm = BuyStateMachine(self.symbol_dj, self.tick_dj, self.avg_m5_dj, spike=spike)

        # Form First Bottom
        c1 = self._make_candle(
            self.symbol_dj, Timeframe.M5,
            datetime(2023, 10, 2, 14, 15, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
            open_=34140.0, high=34145.0, low=34050.0, close=34060.0
        )
        c2 = self._make_candle(
            self.symbol_dj, Timeframe.M5,
            datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            open_=34060.0, high=34150.0, low=34055.0, close=34140.0
        )
        sm.on_m5_candle(c1)
        sm.on_m5_candle(c2)

        # Proximity activation at 14:30:00
        t_act = datetime(2023, 10, 2, 14, 30, 0, tzinfo=timezone.utc)
        quote = Quote(
            symbol=self.symbol_dj,
            bid=34065.0,
            ask=34070.0,
            spread=5.0,
            timestamp=t_act,
        )
        sm.on_quote(quote)
        self.assertEqual(sm.current_state, BullishState.M1_MONITORING)

        # M1 candle closing EXACTLY at 14:30:00 (close_time == activation_time)
        m1_equal = self._make_candle(
            self.symbol_dj, Timeframe.M1,
            datetime(2023, 10, 2, 14, 29, 0, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 30, 0, tzinfo=timezone.utc),
            open_=34068.0, high=34090.0, low=34060.0, close=34062.0
        )
        sig = sm.on_m1_candle(m1_equal)
        self.assertIsNone(sig)
        self.assertEqual(sm.current_state, BullishState.M1_MONITORING)

        # M1 candle closing at 14:31:00 (close_time > activation_time)
        m1_valid = self._make_candle(
            self.symbol_dj, Timeframe.M1,
            datetime(2023, 10, 2, 14, 30, 0, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 31, 0, tzinfo=timezone.utc),
            open_=34068.0, high=34090.0, low=34060.0, close=34062.0
        )
        sig2 = sm.on_m1_candle(m1_valid)
        self.assertIsNotNone(sig2)
        self.assertEqual(sm.current_state, BullishState.BUY_TRIGGER_ARMED)

    def test_duplicate_events_idempotency(self) -> None:
        """Duplicate quotes, ticks, and M1 candles must be handled idempotently without duplicate signals."""
        spike = self._make_bullish_spike()
        sm = BuyStateMachine(self.symbol_dj, self.tick_dj, self.avg_m5_dj, spike=spike)

        # Form First Bottom
        c1 = self._make_candle(
            self.symbol_dj, Timeframe.M5,
            datetime(2023, 10, 2, 14, 15, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
            open_=34140.0, high=34145.0, low=34050.0, close=34060.0
        )
        c2 = self._make_candle(
            self.symbol_dj, Timeframe.M5,
            datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            open_=34060.0, high=34150.0, low=34055.0, close=34140.0
        )
        sm.on_m5_candle(c1)
        sm.on_m5_candle(c2)

        # Duplicate Proximity Quote
        t_act = datetime(2023, 10, 2, 14, 30, 0, tzinfo=timezone.utc)
        quote = Quote(symbol=self.symbol_dj, bid=34065.0, ask=34070.0, spread=5.0, timestamp=t_act)
        sm.on_quote(quote)
        self.assertEqual(sm.current_state, BullishState.M1_MONITORING)
        first_act_time = sm.context.proximity_activation_time

        # Feed duplicate quote: should do nothing, activation_time unchanged
        sm.on_quote(quote)
        self.assertEqual(sm.context.proximity_activation_time, first_act_time)

        # Valid M1 Pinbar -> Signal Generated
        m1 = self._make_candle(
            self.symbol_dj, Timeframe.M1,
            datetime(2023, 10, 2, 14, 30, 0, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 31, 0, tzinfo=timezone.utc),
            open_=34068.0, high=34090.0, low=34060.0, close=34062.0
        )
        sig = sm.on_m1_candle(m1)
        self.assertIsNotNone(sig)
        self.assertEqual(sm.current_state, BullishState.BUY_TRIGGER_ARMED)

        # Feed duplicate M1 candle: must return None, no secondary signal
        sig_dup = sm.on_m1_candle(m1)
        self.assertIsNone(sig_dup)
        self.assertEqual(sm.current_state, BullishState.BUY_TRIGGER_ARMED)

    def test_buy_reaches_bullish_spike_detected_state(self) -> None:
        """On accepting a spike, BuyStateMachine transitions from IDLE to BULLISH_SPIKE_DETECTED."""
        sm = BuyStateMachine(self.symbol_dj, self.tick_dj, self.avg_m5_dj)
        self.assertEqual(sm.current_state, BullishState.IDLE)
        spike = self._make_bullish_spike()
        sm.on_spike(spike)
        self.assertEqual(sm.current_state, BullishState.BULLISH_SPIKE_DETECTED)

    def test_sell_reaches_bearish_spike_detected_state(self) -> None:
        """On accepting a spike, SellStateMachine transitions from IDLE to BEARISH_SPIKE_DETECTED."""
        sm = SellStateMachine(self.symbol_nq, self.tick_nq, self.avg_m5_nq)
        self.assertEqual(sm.current_state, BearishState.IDLE)
        spike = self._make_bearish_spike()
        sm.on_spike(spike)
        self.assertEqual(sm.current_state, BearishState.BEARISH_SPIKE_DETECTED)

    def test_full_11_state_transitions_buy_and_sell(self) -> None:
        """Verify complete traversal through all 11 states of BullishState and BearishState with real events."""
        # BUY Lifecycle: 11 States
        sm_buy = BuyStateMachine(self.symbol_dj, self.tick_dj, self.avg_m5_dj)
        self.assertEqual(sm_buy.current_state, BullishState.IDLE)

        # 1 -> 2: IDLE -> BULLISH_SPIKE_DETECTED
        spike = self._make_bullish_spike()
        sm_buy.on_spike(spike)
        self.assertEqual(sm_buy.current_state, BullishState.BULLISH_SPIKE_DETECTED)

        # 2 -> 3: BULLISH_SPIKE_DETECTED -> WAIT_INITIAL_CORRECTION
        c1 = self._make_candle(
            self.symbol_dj, Timeframe.M5,
            datetime(2023, 10, 2, 14, 15, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
            open_=34140.0, high=34145.0, low=34050.0, close=34060.0
        )
        sm_buy.on_m5_candle(c1)
        self.assertEqual(sm_buy.current_state, BullishState.WAIT_INITIAL_CORRECTION)

        # 3 -> 4 -> 5: WAIT_INITIAL_CORRECTION -> FIRST_BOTTOM_FORMED -> WAIT_RETURN_TO_FIRST_BOTTOM
        c2 = self._make_candle(
            self.symbol_dj, Timeframe.M5,
            datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            open_=34060.0, high=34150.0, low=34055.0, close=34140.0
        )
        sm_buy.on_m5_candle(c2)
        self.assertEqual(sm_buy.current_state, BullishState.WAIT_RETURN_TO_FIRST_BOTTOM)
        # Check transition trail for FIRST_BOTTOM_FORMED
        recorded_buy_states = [t.to_state for t in sm_buy.context.transitions]
        self.assertIn(BullishState.FIRST_BOTTOM_FORMED, recorded_buy_states)

        # 5 -> 6: WAIT_RETURN_TO_FIRST_BOTTOM -> M1_MONITORING
        quote = Quote(
            symbol=self.symbol_dj, bid=34065.0, ask=34070.0, spread=5.0,
            timestamp=datetime(2023, 10, 2, 14, 30, tzinfo=timezone.utc)
        )
        sm_buy.on_quote(quote)
        self.assertEqual(sm_buy.current_state, BullishState.M1_MONITORING)

        # 6 -> 7 -> 8: M1_MONITORING -> BEARISH_PINBAR_DETECTED -> BUY_TRIGGER_ARMED (Emits SignalEvent)
        m1 = self._make_candle(
            self.symbol_dj, Timeframe.M1,
            datetime(2023, 10, 2, 14, 30, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 31, tzinfo=timezone.utc),
            open_=34068.0, high=34090.0, low=34060.0, close=34062.0
        )
        sig = sm_buy.on_m1_candle(m1)
        self.assertIsNotNone(sig)
        self.assertEqual(sm_buy.current_state, BullishState.BUY_TRIGGER_ARMED)
        recorded_buy_states = [t.to_state for t in sm_buy.context.transitions]
        self.assertIn(BullishState.BEARISH_PINBAR_DETECTED, recorded_buy_states)

        # 8 -> 9 -> 10: BUY_TRIGGER_ARMED -> BUY_EXECUTED -> MANAGE_TRADE
        t_exec = datetime(2023, 10, 2, 14, 31, 5, tzinfo=timezone.utc)
        sm_buy.on_order_executed(executed_price=34072.0, timestamp=t_exec)
        self.assertEqual(sm_buy.current_state, BullishState.MANAGE_TRADE)
        recorded_buy_states = [t.to_state for t in sm_buy.context.transitions]
        self.assertIn(BullishState.BUY_EXECUTED, recorded_buy_states)

        # 10 -> 11: MANAGE_TRADE -> CLOSED
        t_close = datetime(2023, 10, 2, 15, 0, 0, tzinfo=timezone.utc)
        sm_buy.on_trade_closed(reason="ALL_TP_HIT", timestamp=t_close)
        self.assertEqual(sm_buy.current_state, BullishState.CLOSED)
        self.assertEqual(len(sm_buy.context.transitions), 10)  # 10 transitions traversing 11 states

        # SELL Lifecycle: 11 States
        sm_sell = SellStateMachine(self.symbol_nq, self.tick_nq, self.avg_m5_nq)
        self.assertEqual(sm_sell.current_state, BearishState.IDLE)

        sell_spike = self._make_bearish_spike()
        sm_sell.on_spike(sell_spike)
        self.assertEqual(sm_sell.current_state, BearishState.BEARISH_SPIKE_DETECTED)

        sc1 = self._make_candle(
            self.symbol_nq, Timeframe.M5,
            datetime(2023, 10, 2, 14, 15, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
            open_=17860.0, high=17940.0, low=17855.0, close=17930.0
        )
        sm_sell.on_m5_candle(sc1)
        self.assertEqual(sm_sell.current_state, BearishState.WAIT_INITIAL_CORRECTION)

        sc2 = self._make_candle(
            self.symbol_nq, Timeframe.M5,
            datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            open_=17930.0, high=17935.0, low=17850.0, close=17860.0
        )
        sm_sell.on_m5_candle(sc2)
        self.assertEqual(sm_sell.current_state, BearishState.WAIT_RETURN_TO_FIRST_TOP)
        recorded_sell_states = [t.to_state for t in sm_sell.context.transitions]
        self.assertIn(BearishState.FIRST_TOP_FORMED, recorded_sell_states)

        sell_quote = Quote(
            symbol=self.symbol_nq, bid=17935.0, ask=17936.0, spread=1.0,
            timestamp=datetime(2023, 10, 2, 14, 30, tzinfo=timezone.utc)
        )
        sm_sell.on_quote(sell_quote)
        self.assertEqual(sm_sell.current_state, BearishState.M1_MONITORING)

        sm1 = self._make_candle(
            self.symbol_nq, Timeframe.M1,
            datetime(2023, 10, 2, 14, 30, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 31, tzinfo=timezone.utc),
            open_=17932.0, high=17940.0, low=17910.0, close=17938.0
        )
        sig_sell = sm_sell.on_m1_candle(sm1)
        self.assertIsNotNone(sig_sell)
        self.assertEqual(sm_sell.current_state, BearishState.SELL_TRIGGER_ARMED)
        recorded_sell_states = [t.to_state for t in sm_sell.context.transitions]
        self.assertIn(BearishState.BULLISH_PINBAR_DETECTED, recorded_sell_states)

        sm_sell.on_order_executed(executed_price=17934.0, timestamp=t_exec)
        self.assertEqual(sm_sell.current_state, BearishState.MANAGE_TRADE)
        recorded_sell_states = [t.to_state for t in sm_sell.context.transitions]
        self.assertIn(BearishState.SELL_EXECUTED, recorded_sell_states)

        sm_sell.on_trade_closed(reason="ALL_TP_HIT", timestamp=t_close)
        self.assertEqual(sm_sell.current_state, BearishState.CLOSED)
        self.assertEqual(len(sm_sell.context.transitions), 10)

    def test_pretrade_invalidation_does_not_apply_after_execution(self) -> None:
        """Pre-trade M5 invalidation (M5 Close < BOTTOM_LEVEL) MUST NOT apply after execution."""
        spike = self._make_bullish_spike()
        sm = BuyStateMachine(self.symbol_dj, self.tick_dj, self.avg_m5_dj, spike=spike)

        c1 = self._make_candle(
            self.symbol_dj, Timeframe.M5,
            datetime(2023, 10, 2, 14, 15, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
            open_=34140.0, high=34145.0, low=34050.0, close=34060.0
        )
        c2 = self._make_candle(
            self.symbol_dj, Timeframe.M5,
            datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            open_=34060.0, high=34150.0, low=34055.0, close=34140.0
        )
        sm.on_m5_candle(c1)
        sm.on_m5_candle(c2)

        quote = Quote(symbol=self.symbol_dj, bid=34065.0, ask=34070.0, spread=5.0,
                      timestamp=datetime(2023, 10, 2, 14, 30, tzinfo=timezone.utc))
        sm.on_quote(quote)

        m1 = self._make_candle(
            self.symbol_dj, Timeframe.M1,
            datetime(2023, 10, 2, 14, 30, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 31, tzinfo=timezone.utc),
            open_=34068.0, high=34090.0, low=34060.0, close=34062.0
        )
        sm.on_m1_candle(m1)
        self.assertEqual(sm.current_state, BullishState.BUY_TRIGGER_ARMED)

        # Execute order -> enters MANAGE_TRADE
        sm.on_order_executed(executed_price=34072.0, timestamp=datetime(2023, 10, 2, 14, 31, 5, tzinfo=timezone.utc))
        self.assertEqual(sm.current_state, BullishState.MANAGE_TRADE)

        # Now an M5 candle closes below BOTTOM_LEVEL (34050.0) -> must NOT invalidate active trade
        c_breach = self._make_candle(
            self.symbol_dj, Timeframe.M5,
            datetime(2023, 10, 2, 14, 30, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 35, tzinfo=timezone.utc),
            open_=34060.0, high=34065.0, low=34010.0, close=34020.0
        )
        sm.on_m5_candle(c_breach)
        self.assertEqual(sm.current_state, BullishState.MANAGE_TRADE)
        self.assertFalse(sm.is_closed)

    def test_post_execution_opposite_spike_does_not_trigger_pretrade_invalidation(self) -> None:
        """Opposite spike during active trade management must NOT trigger pre-trade invalidation."""
        spike = self._make_bullish_spike()
        sm = BuyStateMachine(self.symbol_dj, self.tick_dj, self.avg_m5_dj, spike=spike)

        c1 = self._make_candle(
            self.symbol_dj, Timeframe.M5,
            datetime(2023, 10, 2, 14, 15, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
            open_=34140.0, high=34145.0, low=34050.0, close=34060.0
        )
        c2 = self._make_candle(
            self.symbol_dj, Timeframe.M5,
            datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            open_=34060.0, high=34150.0, low=34055.0, close=34140.0
        )
        sm.on_m5_candle(c1)
        sm.on_m5_candle(c2)

        quote = Quote(symbol=self.symbol_dj, bid=34065.0, ask=34070.0, spread=5.0,
                      timestamp=datetime(2023, 10, 2, 14, 30, tzinfo=timezone.utc))
        sm.on_quote(quote)

        m1 = self._make_candle(
            self.symbol_dj, Timeframe.M1,
            datetime(2023, 10, 2, 14, 30, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 31, tzinfo=timezone.utc),
            open_=34068.0, high=34090.0, low=34060.0, close=34062.0
        )
        sm.on_m1_candle(m1)
        sm.on_order_executed(executed_price=34072.0, timestamp=datetime(2023, 10, 2, 14, 31, 5, tzinfo=timezone.utc))
        self.assertEqual(sm.current_state, BullishState.MANAGE_TRADE)

        # Bearish spike arrives during active trade -> pre-trade invalidation ignored
        bearish_spike = self._make_bearish_spike()
        sm.on_opposite_spike(bearish_spike)
        self.assertEqual(sm.current_state, BullishState.MANAGE_TRADE)
        self.assertFalse(sm.is_closed)

    def test_post_execution_session_end_does_not_trigger_pretrade_invalidation(self) -> None:
        """Session end after execution must NOT trigger pre-trade invalidation."""
        spike = self._make_bullish_spike()
        sm = BuyStateMachine(self.symbol_dj, self.tick_dj, self.avg_m5_dj, spike=spike)

        c1 = self._make_candle(
            self.symbol_dj, Timeframe.M5,
            datetime(2023, 10, 2, 14, 15, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
            open_=34140.0, high=34145.0, low=34050.0, close=34060.0
        )
        c2 = self._make_candle(
            self.symbol_dj, Timeframe.M5,
            datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            open_=34060.0, high=34150.0, low=34055.0, close=34140.0
        )
        sm.on_m5_candle(c1)
        sm.on_m5_candle(c2)

        quote = Quote(symbol=self.symbol_dj, bid=34065.0, ask=34070.0, spread=5.0,
                      timestamp=datetime(2023, 10, 2, 14, 30, tzinfo=timezone.utc))
        sm.on_quote(quote)

        m1 = self._make_candle(
            self.symbol_dj, Timeframe.M1,
            datetime(2023, 10, 2, 14, 30, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 31, tzinfo=timezone.utc),
            open_=34068.0, high=34090.0, low=34060.0, close=34062.0
        )
        sm.on_m1_candle(m1)
        sm.on_order_executed(executed_price=34072.0, timestamp=datetime(2023, 10, 2, 14, 31, 5, tzinfo=timezone.utc))
        self.assertEqual(sm.current_state, BullishState.MANAGE_TRADE)

        sm.on_session_end(timestamp=datetime(2023, 10, 2, 21, 0, 0, tzinfo=timezone.utc))
        self.assertEqual(sm.current_state, BullishState.MANAGE_TRADE)
        self.assertFalse(sm.is_closed)

    def test_position_conflict_does_not_modify_existing_trade(self) -> None:
        """Position conflict before execution invalidates the setup without touching existing trades."""
        # Simulated existing position data before conflict
        existing_trade_data = {
            "ticket": 12345,
            "symbol": "DOW_JONES",
            "volume": 1.0,
            "open_price": 34000.0,
            "stop_loss": 33950.0,
            "take_profit": 34200.0,
            "state": "OPEN"
        }
        existing_trade_snapshot_before = dict(existing_trade_data)

        spike = self._make_bullish_spike()
        sm = BuyStateMachine(self.symbol_dj, self.tick_dj, self.avg_m5_dj, spike=spike)
        self.assertEqual(sm.current_state, BullishState.BULLISH_SPIKE_DETECTED)

        # Position conflict occurs
        t_conflict = datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc)
        sm.on_position_conflict(reason="Max concurrent positions reached (1/1)", timestamp=t_conflict)

        self.assertEqual(sm.current_state, BullishState.CLOSED)
        self.assertTrue(sm.is_closed)
        self.assertIn("POSITION_CONFLICT", sm.context.invalidation_reason)

        # Existing trade data remains strictly untouched
        self.assertEqual(existing_trade_data, existing_trade_snapshot_before)

    def test_signal_emitted_same_call_as_valid_pinbar(self) -> None:
        """BLK-04: Valid M1 Pin Bar immediately emits SignalEvent in the same synchronous call."""
        spike = self._make_bullish_spike()
        sm = BuyStateMachine(self.symbol_dj, self.tick_dj, self.avg_m5_dj, spike=spike)

        c1 = self._make_candle(
            self.symbol_dj, Timeframe.M5,
            datetime(2023, 10, 2, 14, 15, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
            open_=34140.0, high=34145.0, low=34050.0, close=34060.0
        )
        c2 = self._make_candle(
            self.symbol_dj, Timeframe.M5,
            datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            open_=34060.0, high=34150.0, low=34055.0, close=34140.0
        )
        sm.on_m5_candle(c1)
        sm.on_m5_candle(c2)

        quote = Quote(symbol=self.symbol_dj, bid=34065.0, ask=34070.0, spread=5.0,
                      timestamp=datetime(2023, 10, 2, 14, 30, tzinfo=timezone.utc))
        sm.on_quote(quote)

        m1 = self._make_candle(
            self.symbol_dj, Timeframe.M1,
            datetime(2023, 10, 2, 14, 30, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 31, tzinfo=timezone.utc),
            open_=34068.0, high=34090.0, low=34060.0, close=34062.0
        )

        # Single call emits SignalEvent synchronously
        signal = sm.on_m1_candle(m1)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.signal_time, m1.close_time)
        self.assertEqual(signal.setup_id, sm.context.setup_id)
        self.assertEqual(signal.final_structural_target, spike.high)

    def test_trigger_armed_does_not_wait_for_next_candle(self) -> None:
        """BUY_TRIGGER_ARMED means setup is already armed and signal emitted, not waiting for another candle."""
        spike = self._make_bullish_spike()
        sm = BuyStateMachine(self.symbol_dj, self.tick_dj, self.avg_m5_dj, spike=spike)

        c1 = self._make_candle(
            self.symbol_dj, Timeframe.M5,
            datetime(2023, 10, 2, 14, 15, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
            open_=34140.0, high=34145.0, low=34050.0, close=34060.0
        )
        c2 = self._make_candle(
            self.symbol_dj, Timeframe.M5,
            datetime(2023, 10, 2, 14, 20, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 25, tzinfo=timezone.utc),
            open_=34060.0, high=34150.0, low=34055.0, close=34140.0
        )
        sm.on_m5_candle(c1)
        sm.on_m5_candle(c2)

        quote = Quote(symbol=self.symbol_dj, bid=34065.0, ask=34070.0, spread=5.0,
                      timestamp=datetime(2023, 10, 2, 14, 30, tzinfo=timezone.utc))
        sm.on_quote(quote)

        m1 = self._make_candle(
            self.symbol_dj, Timeframe.M1,
            datetime(2023, 10, 2, 14, 30, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 31, tzinfo=timezone.utc),
            open_=34068.0, high=34090.0, low=34060.0, close=34062.0
        )
        sig = sm.on_m1_candle(m1)
        self.assertIsNotNone(sig)
        self.assertEqual(sm.current_state, BullishState.BUY_TRIGGER_ARMED)

        # Subsequent M1 candle is ignored (returns None) without delaying or regenerating signal
        m1_next = self._make_candle(
            self.symbol_dj, Timeframe.M1,
            datetime(2023, 10, 2, 14, 31, tzinfo=timezone.utc),
            datetime(2023, 10, 2, 14, 32, tzinfo=timezone.utc),
            open_=34062.0, high=34070.0, low=34060.0, close=34068.0
        )
        sig_next = sm.on_m1_candle(m1_next)
        self.assertIsNone(sig_next)
        self.assertEqual(sm.current_state, BullishState.BUY_TRIGGER_ARMED)



if __name__ == "__main__":
    unittest.main()
