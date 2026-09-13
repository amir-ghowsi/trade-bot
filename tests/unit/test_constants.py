"""
Unit Tests for src/core/constants.py
Validates completeness, invariants, and strict adherence to Phase 0 Specification.
"""

import unittest

from src.core.constants import (
    BearishState,
    BullishState,
    CanonicalSymbol,
    DEFAULT_BACKTEST_COMMISSION_PER_LOT,
    DEFAULT_BACKTEST_SLIPPAGE_POINTS,
    Direction,
    INTERNAL_TIMEZONE,
    MAX_ACTIVE_POSITIONS_PER_SYMBOL,
    MAX_GLOBAL_ACTIVE_POSITIONS,
    MIN_GAP_COUNT,
    MIN_SPIKE_CANDLES,
    OrderStatus,
    OrderType,
    PARTIAL_CLOSE_STRUCTURAL_TARGET_RATIO,
    PARTIAL_CLOSE_TP1_RATIO,
    PARTIAL_CLOSE_TP2_RATIO,
    PARTIAL_CLOSE_TP3_RATIO,
    PINBAR_MAX_BODY_RATIO,
    PINBAR_MIDPOINT_RATIO,
    PositionStatus,
    RecoveryStatus,
    TARGET_RISK_FRACTION,
    Timeframe,
    TRADE_RECORD_EXACT_FIELD_COUNT,
    TradeDirection,
    TRADING_DAY_TIMEZONE,
    TradingSession,
)


class TestConstants(unittest.TestCase):
    """Test suite verifying all core constants and specifications."""

    def test_timeframe_constants(self) -> None:
        self.assertEqual(Timeframe.M1.value, "M1")
        self.assertEqual(Timeframe.M5.value, "M5")
        self.assertEqual(len(Timeframe), 2)

    def test_direction_constants(self) -> None:
        self.assertEqual(Direction.BULLISH.value, "BULLISH")
        self.assertEqual(Direction.BEARISH.value, "BEARISH")
        self.assertEqual(Direction.NEUTRAL.value, "NEUTRAL")
        self.assertEqual(len(Direction), 3)

    def test_trade_direction_constants(self) -> None:
        self.assertEqual(TradeDirection.BUY.value, "BUY")
        self.assertEqual(TradeDirection.SELL.value, "SELL")
        self.assertEqual(len(TradeDirection), 2)

    def test_canonical_symbols(self) -> None:
        self.assertEqual(CanonicalSymbol.DOW_JONES.value, "DOW_JONES")
        self.assertEqual(CanonicalSymbol.NASDAQ.value, "NASDAQ")
        self.assertEqual(len(CanonicalSymbol), 2)

    def test_bullish_state_machine_exact_11_states(self) -> None:
        expected_states = [
            "IDLE",
            "BULLISH_SPIKE_DETECTED",
            "WAIT_INITIAL_CORRECTION",
            "FIRST_BOTTOM_FORMED",
            "WAIT_RETURN_TO_FIRST_BOTTOM",
            "M1_MONITORING",
            "BEARISH_PINBAR_DETECTED",
            "BUY_TRIGGER_ARMED",
            "BUY_EXECUTED",
            "MANAGE_TRADE",
            "CLOSED",
        ]
        actual_states = [s.value for s in BullishState]
        self.assertEqual(len(BullishState), 11, "BullishState must contain exactly 11 states")
        self.assertEqual(actual_states, expected_states)

    def test_bearish_state_machine_exact_11_states(self) -> None:
        expected_states = [
            "IDLE",
            "BEARISH_SPIKE_DETECTED",
            "WAIT_INITIAL_CORRECTION",
            "FIRST_TOP_FORMED",
            "WAIT_RETURN_TO_FIRST_TOP",
            "M1_MONITORING",
            "BULLISH_PINBAR_DETECTED",
            "SELL_TRIGGER_ARMED",
            "SELL_EXECUTED",
            "MANAGE_TRADE",
            "CLOSED",
        ]
        actual_states = [s.value for s in BearishState]
        self.assertEqual(len(BearishState), 11, "BearishState must contain exactly 11 states")
        self.assertEqual(actual_states, expected_states)

    def test_locked_numeric_strategy_rules(self) -> None:
        # Risk target 1%
        self.assertEqual(TARGET_RISK_FRACTION, 0.01)

        # Spike criteria
        self.assertEqual(MIN_SPIKE_CANDLES, 3)
        self.assertEqual(MIN_GAP_COUNT, 1)

        # Pin bar geometry
        self.assertEqual(PINBAR_MAX_BODY_RATIO, 0.20)
        self.assertEqual(PINBAR_MIDPOINT_RATIO, 0.50)

        # Partial close locked distribution (25% each)
        self.assertEqual(PARTIAL_CLOSE_TP1_RATIO, 0.25)
        self.assertEqual(PARTIAL_CLOSE_TP2_RATIO, 0.25)
        self.assertEqual(PARTIAL_CLOSE_TP3_RATIO, 0.25)
        self.assertEqual(PARTIAL_CLOSE_STRUCTURAL_TARGET_RATIO, 0.25)
        self.assertEqual(
            PARTIAL_CLOSE_TP1_RATIO
            + PARTIAL_CLOSE_TP2_RATIO
            + PARTIAL_CLOSE_TP3_RATIO
            + PARTIAL_CLOSE_STRUCTURAL_TARGET_RATIO,
            1.00,
        )

        # Timezones
        self.assertEqual(INTERNAL_TIMEZONE, "UTC")
        self.assertEqual(TRADING_DAY_TIMEZONE, "America/New_York")

        # Concurrency
        self.assertEqual(MAX_ACTIVE_POSITIONS_PER_SYMBOL, 1)
        self.assertEqual(MAX_GLOBAL_ACTIVE_POSITIONS, 2)

        # Exact TradeRecord field invariant count
        self.assertEqual(TRADE_RECORD_EXACT_FIELD_COUNT, 35)

        # Backtest defaults
        self.assertEqual(DEFAULT_BACKTEST_COMMISSION_PER_LOT, 0.0)
        self.assertEqual(DEFAULT_BACKTEST_SLIPPAGE_POINTS, 2.0)


if __name__ == "__main__":
    unittest.main()
