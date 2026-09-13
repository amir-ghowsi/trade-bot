"""
Unit Tests for Market Layer (Phase 3).
Validates broker-agnostic market data transport, validation invariants, causal ordering,
spread provenance, and in-memory market data provider behavior.
"""

from datetime import datetime, timezone
import unittest

from src.core.constants import CanonicalSymbol, Direction, Timeframe
from src.core.exceptions import (
    DataValidationError,
    MissingDataError,
    TimezoneError,
)
from src.core.types import Candle, Quote, SymbolSpecification
from src.market.exceptions import (
    InvalidSequenceError,
    SpreadProvenanceError,
    SymbolMismatchError,
    TimeframeMismatchError,
)
from src.market.in_memory_provider import InMemoryMarketDataProvider
from src.market.provenance import (
    assert_execution_spread_not_from_candle,
    build_spread_audit_record,
    extract_execution_spread,
)
from src.market.session import (
    get_session_date,
    to_session_time,
    validate_time_range,
)
from src.market.types import (
    MarketFeedType,
    SpreadProvenance,
    Tick,
)
from src.market.validator import MarketDataValidator


class TestMarketLayer(unittest.TestCase):
    """Test suite for Phase 3 Market Layer."""

    def setUp(self) -> None:
        self.t1 = datetime(2023, 10, 1, 14, 0, 0, tzinfo=timezone.utc)
        self.t2 = datetime(2023, 10, 1, 14, 1, 0, tzinfo=timezone.utc)
        self.t3 = datetime(2023, 10, 1, 14, 2, 0, tzinfo=timezone.utc)
        self.t5_1 = datetime(2023, 10, 1, 14, 0, 0, tzinfo=timezone.utc)
        self.t5_2 = datetime(2023, 10, 1, 14, 5, 0, tzinfo=timezone.utc)

    # =========================================================================
    # 1. Candle Validation & Contracts
    # =========================================================================

    def test_valid_m1_candle(self) -> None:
        candle = Candle(
            symbol=CanonicalSymbol.DOW_JONES,
            timeframe=Timeframe.M1,
            open_time=self.t1,
            close_time=self.t2,
            open=34000.0,
            high=34020.0,
            low=33990.0,
            close=34015.0,
            volume=150.0,
            spread=2.0,
        )
        MarketDataValidator.validate_candle(candle, expected_symbol=CanonicalSymbol.DOW_JONES, expected_timeframe=Timeframe.M1)
        self.assertEqual(candle.timeframe, Timeframe.M1)
        self.assertEqual(candle.direction, Direction.BULLISH)
        self.assertEqual(candle.range, 30.0)
        self.assertEqual(candle.body, 15.0)

    def test_valid_m5_candle(self) -> None:
        candle = Candle(
            symbol=CanonicalSymbol.NASDAQ,
            timeframe=Timeframe.M5,
            open_time=self.t5_1,
            close_time=self.t5_2,
            open=15000.0,
            high=15050.0,
            low=14980.0,
            close=14990.0,
            volume=500.0,
            spread=1.5,
        )
        MarketDataValidator.validate_candle(candle, expected_symbol=CanonicalSymbol.NASDAQ, expected_timeframe=Timeframe.M5)
        self.assertEqual(candle.timeframe, Timeframe.M5)
        self.assertEqual(candle.direction, Direction.BEARISH)

    def test_candle_rejects_naive_timestamp(self) -> None:
        naive_dt = datetime(2023, 10, 1, 14, 0, 0)
        with self.assertRaises(DataValidationError):
            Candle(
                symbol=CanonicalSymbol.DOW_JONES,
                timeframe=Timeframe.M1,
                open_time=naive_dt,
                close_time=self.t2,
                open=34000.0,
                high=34020.0,
                low=33990.0,
                close=34015.0,
                volume=100.0,
                spread=1.0,
            )

    def test_candle_rejects_invalid_time_ordering(self) -> None:
        with self.assertRaises(DataValidationError):
            Candle(
                symbol=CanonicalSymbol.DOW_JONES,
                timeframe=Timeframe.M1,
                open_time=self.t2,
                close_time=self.t1,  # close before open
                open=34000.0,
                high=34020.0,
                low=33990.0,
                close=34015.0,
                volume=100.0,
                spread=1.0,
            )

    def test_candle_rejects_invalid_ohlc(self) -> None:
        # High < Low
        with self.assertRaises(DataValidationError):
            Candle(
                symbol=CanonicalSymbol.DOW_JONES,
                timeframe=Timeframe.M1,
                open_time=self.t1,
                close_time=self.t2,
                open=34000.0,
                high=33980.0,
                low=34020.0,
                close=34000.0,
                volume=10.0,
                spread=1.0,
            )
        # High < Open
        with self.assertRaises(DataValidationError):
            Candle(
                symbol=CanonicalSymbol.DOW_JONES,
                timeframe=Timeframe.M1,
                open_time=self.t1,
                close_time=self.t2,
                open=34050.0,
                high=34020.0,
                low=33980.0,
                close=34000.0,
                volume=10.0,
                spread=1.0,
            )
        # Low > Close
        with self.assertRaises(DataValidationError):
            Candle(
                symbol=CanonicalSymbol.DOW_JONES,
                timeframe=Timeframe.M1,
                open_time=self.t1,
                close_time=self.t2,
                open=34000.0,
                high=34020.0,
                low=33990.0,
                close=33950.0,
                volume=10.0,
                spread=1.0,
            )

    def test_candle_rejects_negative_volume_and_spread(self) -> None:
        with self.assertRaises(DataValidationError):
            Candle(
                symbol=CanonicalSymbol.DOW_JONES,
                timeframe=Timeframe.M1,
                open_time=self.t1,
                close_time=self.t2,
                open=34000.0,
                high=34010.0,
                low=33990.0,
                close=34005.0,
                volume=-5.0,
                spread=1.0,
            )
        with self.assertRaises(DataValidationError):
            Candle(
                symbol=CanonicalSymbol.DOW_JONES,
                timeframe=Timeframe.M1,
                open_time=self.t1,
                close_time=self.t2,
                open=34000.0,
                high=34010.0,
                low=33990.0,
                close=34005.0,
                volume=5.0,
                spread=-1.0,
            )

    # =========================================================================
    # 2. Quote Validation & Spread Contract
    # =========================================================================

    def test_valid_quote(self) -> None:
        q = Quote(
            symbol=CanonicalSymbol.DOW_JONES,
            timestamp=self.t1,
            bid=34000.0,
            ask=34002.0,
            spread=2.0,
        )
        MarketDataValidator.validate_quote(q, expected_symbol=CanonicalSymbol.DOW_JONES)
        self.assertEqual(q.spread, 2.0)

    def test_quote_rejects_invalid_bid_ask(self) -> None:
        # Negative bid
        with self.assertRaises(DataValidationError):
            Quote(
                symbol=CanonicalSymbol.DOW_JONES,
                timestamp=self.t1,
                bid=-10.0,
                ask=34002.0,
                spread=2.0,
            )
        # Ask < Bid (crossed market)
        with self.assertRaises(DataValidationError):
            Quote(
                symbol=CanonicalSymbol.DOW_JONES,
                timestamp=self.t1,
                bid=34005.0,
                ask=34000.0,
                spread=2.0,
            )

    def test_quote_rejects_spread_inconsistency(self) -> None:
        with self.assertRaises(DataValidationError):
            Quote(
                symbol=CanonicalSymbol.DOW_JONES,
                timestamp=self.t1,
                bid=34000.0,
                ask=34002.0,
                spread=5.0,  # Reported 5.0, actual ask-bid is 2.0
            )

    def test_quote_rejects_naive_timestamp(self) -> None:
        with self.assertRaises(DataValidationError):
            Quote(
                symbol=CanonicalSymbol.DOW_JONES,
                timestamp=datetime(2023, 10, 1, 14, 0, 0),  # Naive
                bid=34000.0,
                ask=34002.0,
                spread=2.0,
            )

    # =========================================================================
    # 3. Tick Model & Sequences
    # =========================================================================

    def test_valid_tick(self) -> None:
        tick = Tick(
            symbol=CanonicalSymbol.NASDAQ,
            timestamp=self.t1,
            bid=15000.0,
            ask=15000.5,
            last=15000.25,
            volume=10.0,
        )
        MarketDataValidator.validate_tick(tick, expected_symbol=CanonicalSymbol.NASDAQ)
        self.assertEqual(tick.spread, 0.5)

    def test_tick_rejects_invalid_prices(self) -> None:
        with self.assertRaises(DataValidationError):
            Tick(
                symbol=CanonicalSymbol.NASDAQ,
                timestamp=self.t1,
                bid=15001.0,
                ask=15000.0,  # ask < bid
            )
        with self.assertRaises(DataValidationError):
            Tick(
                symbol=CanonicalSymbol.NASDAQ,
                timestamp=datetime(2023, 10, 1, 14, 0, 0),  # naive
                bid=15000.0,
                ask=15000.5,
            )

    def test_tick_sequence_ordering(self) -> None:
        t_seq = [
            Tick(symbol=CanonicalSymbol.NASDAQ, timestamp=self.t1, bid=15000.0, ask=15000.5),
            Tick(symbol=CanonicalSymbol.NASDAQ, timestamp=self.t1, bid=15000.25, ask=15000.75),  # same timestamp allowed
            Tick(symbol=CanonicalSymbol.NASDAQ, timestamp=self.t2, bid=15001.0, ask=15001.5),
        ]
        MarketDataValidator.validate_tick_sequence(t_seq, expected_symbol=CanonicalSymbol.NASDAQ)

        # Backward tick timestamp rejected
        backward_seq = [
            Tick(symbol=CanonicalSymbol.NASDAQ, timestamp=self.t2, bid=15001.0, ask=15001.5),
            Tick(symbol=CanonicalSymbol.NASDAQ, timestamp=self.t1, bid=15000.0, ask=15000.5),
        ]
        with self.assertRaises(InvalidSequenceError):
            MarketDataValidator.validate_tick_sequence(backward_seq)

    # =========================================================================
    # 4. Causal Ordering & Candle Sequences
    # =========================================================================

    def test_candle_sequence_chronological_ordering(self) -> None:
        c1 = Candle(
            symbol=CanonicalSymbol.DOW_JONES,
            timeframe=Timeframe.M1,
            open_time=self.t1,
            close_time=self.t2,
            open=34000.0,
            high=34010.0,
            low=33990.0,
            close=34005.0,
            volume=50.0,
            spread=1.0,
        )
        c2 = Candle(
            symbol=CanonicalSymbol.DOW_JONES,
            timeframe=Timeframe.M1,
            open_time=self.t2,
            close_time=self.t3,
            open=34005.0,
            high=34015.0,
            low=34000.0,
            close=34010.0,
            volume=60.0,
            spread=1.0,
        )
        MarketDataValidator.validate_candle_sequence([c1, c2])

    def test_candle_sequence_rejects_backward_timestamps(self) -> None:
        c1 = Candle(
            symbol=CanonicalSymbol.DOW_JONES,
            timeframe=Timeframe.M1,
            open_time=self.t1,
            close_time=self.t2,
            open=34000.0,
            high=34010.0,
            low=33990.0,
            close=34005.0,
            volume=50.0,
            spread=1.0,
        )
        c2 = Candle(
            symbol=CanonicalSymbol.DOW_JONES,
            timeframe=Timeframe.M1,
            open_time=self.t2,
            close_time=self.t3,
            open=34005.0,
            high=34015.0,
            low=34000.0,
            close=34010.0,
            volume=60.0,
            spread=1.0,
        )
        with self.assertRaises(InvalidSequenceError):
            MarketDataValidator.validate_candle_sequence([c2, c1])

    def test_candle_sequence_rejects_duplicate_timestamps(self) -> None:
        c1 = Candle(
            symbol=CanonicalSymbol.DOW_JONES,
            timeframe=Timeframe.M1,
            open_time=self.t1,
            close_time=self.t2,
            open=34000.0,
            high=34010.0,
            low=33990.0,
            close=34005.0,
            volume=50.0,
            spread=1.0,
        )
        c2 = Candle(
            symbol=CanonicalSymbol.DOW_JONES,
            timeframe=Timeframe.M1,
            open_time=self.t1,  # Duplicate open_time
            close_time=self.t2,
            open=34005.0,
            high=34015.0,
            low=34000.0,
            close=34010.0,
            volume=60.0,
            spread=1.0,
        )
        with self.assertRaises(InvalidSequenceError):
            MarketDataValidator.validate_candle_sequence([c1, c2])

    def test_candle_sequence_rejects_overlapping_intervals(self) -> None:
        t_overlap = datetime(2023, 10, 1, 14, 0, 30, tzinfo=timezone.utc)
        c1 = Candle(
            symbol=CanonicalSymbol.DOW_JONES,
            timeframe=Timeframe.M1,
            open_time=self.t1,
            close_time=self.t2,
            open=34000.0,
            high=34010.0,
            low=33990.0,
            close=34005.0,
            volume=50.0,
            spread=1.0,
        )
        c2 = Candle(
            symbol=CanonicalSymbol.DOW_JONES,
            timeframe=Timeframe.M1,
            open_time=t_overlap,  # Starts before c1 closes
            close_time=self.t3,
            open=34005.0,
            high=34015.0,
            low=34000.0,
            close=34010.0,
            volume=60.0,
            spread=1.0,
        )
        with self.assertRaises(InvalidSequenceError):
            MarketDataValidator.validate_candle_sequence([c1, c2])

    def test_candle_sequence_rejects_mismatched_symbols_or_timeframes(self) -> None:
        c1 = Candle(
            symbol=CanonicalSymbol.DOW_JONES,
            timeframe=Timeframe.M1,
            open_time=self.t1,
            close_time=self.t2,
            open=34000.0,
            high=34010.0,
            low=33990.0,
            close=34005.0,
            volume=50.0,
            spread=1.0,
        )
        c2_diff_sym = Candle(
            symbol=CanonicalSymbol.NASDAQ,
            timeframe=Timeframe.M1,
            open_time=self.t2,
            close_time=self.t3,
            open=15000.0,
            high=15010.0,
            low=14990.0,
            close=15005.0,
            volume=50.0,
            spread=1.0,
        )
        with self.assertRaises(SymbolMismatchError):
            MarketDataValidator.validate_candle_sequence([c1, c2_diff_sym])

        c2_diff_tf = Candle(
            symbol=CanonicalSymbol.DOW_JONES,
            timeframe=Timeframe.M5,
            open_time=self.t2,
            close_time=self.t5_2,
            open=34005.0,
            high=34020.0,
            low=34000.0,
            close=34015.0,
            volume=100.0,
            spread=1.0,
        )
        with self.assertRaises(TimeframeMismatchError):
            MarketDataValidator.validate_candle_sequence([c1, c2_diff_tf])

    # =========================================================================
    # 5. Symbol Specification Validation
    # =========================================================================

    def test_valid_symbol_specification(self) -> None:
        spec = SymbolSpecification(
            canonical_symbol=CanonicalSymbol.DOW_JONES,
            broker_symbol="US100_i",
            tick_size=1.0,
            tick_value=1.0,
            contract_size=1.0,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            point=1.0,
        )
        MarketDataValidator.validate_symbol_specification(spec, expected_symbol=CanonicalSymbol.DOW_JONES)

    def test_symbol_specification_rejects_invalid_parameters(self) -> None:
        # Invalid tick_size <= 0
        with self.assertRaises(DataValidationError):
            SymbolSpecification(
                canonical_symbol=CanonicalSymbol.DOW_JONES,
                broker_symbol="US100_i",
                tick_size=0.0,
                tick_value=1.0,
                contract_size=1.0,
                volume_min=0.01,
                volume_max=100.0,
                volume_step=0.01,
                point=1.0,
            )
        # Invalid volume limits
        with self.assertRaises(DataValidationError):
            SymbolSpecification(
                canonical_symbol=CanonicalSymbol.DOW_JONES,
                broker_symbol="US100_i",
                tick_size=1.0,
                tick_value=1.0,
                contract_size=1.0,
                volume_min=10.0,
                volume_max=5.0,  # max < min
                volume_step=0.01,
                point=1.0,
            )

    # =========================================================================
    # 6. Spread Provenance Contracts
    # =========================================================================

    def test_spread_provenance_and_execution_spread(self) -> None:
        q = Quote(
            symbol=CanonicalSymbol.DOW_JONES,
            timestamp=self.t1,
            bid=34000.0,
            ask=34002.5,
            spread=2.5,
        )
        exec_spread = extract_execution_spread(q)
        self.assertEqual(exec_spread, 2.5)

        # Candle spread cannot be used as execution spread
        c = Candle(
            symbol=CanonicalSymbol.DOW_JONES,
            timeframe=Timeframe.M1,
            open_time=self.t1,
            close_time=self.t2,
            open=34000.0,
            high=34010.0,
            low=33990.0,
            close=34005.0,
            volume=50.0,
            spread=1.0,
        )
        with self.assertRaises(SpreadProvenanceError):
            assert_execution_spread_not_from_candle(c)

        with self.assertRaises(SpreadProvenanceError):
            extract_execution_spread(c)  # type: ignore[arg-type]

        audit = build_spread_audit_record(q)
        self.assertEqual(audit["execution_spread"], 2.5)
        self.assertEqual(audit["provenance"], SpreadProvenance.LIVE_QUOTE_BID_ASK.value)

    # =========================================================================
    # 7. Time and Session Handling
    # =========================================================================

    def test_time_range_validation(self) -> None:
        validate_time_range(self.t1, self.t2)
        with self.assertRaises(DataValidationError):
            validate_time_range(self.t2, self.t1)  # end before start

    def test_session_timezone_projection(self) -> None:
        # 14:00 UTC is 10:00 AM EDT (America/New_York)
        ny_dt = to_session_time(self.t1, "America/New_York")
        self.assertEqual(ny_dt.hour, 10)
        self.assertEqual(get_session_date(self.t1), "2023-10-01")

        # Internal timestamp remains UTC
        self.assertEqual(self.t1.tzinfo, timezone.utc)

    # =========================================================================
    # 8. In-Memory Market Data Provider
    # =========================================================================

    def test_in_memory_provider_candles(self) -> None:
        provider = InMemoryMarketDataProvider()

        c1 = Candle(
            symbol=CanonicalSymbol.DOW_JONES,
            timeframe=Timeframe.M1,
            open_time=self.t1,
            close_time=self.t2,
            open=34000.0,
            high=34010.0,
            low=33990.0,
            close=34005.0,
            volume=50.0,
            spread=1.0,
        )
        c2 = Candle(
            symbol=CanonicalSymbol.DOW_JONES,
            timeframe=Timeframe.M1,
            open_time=self.t2,
            close_time=self.t3,
            open=34005.0,
            high=34015.0,
            low=34000.0,
            close=34010.0,
            volume=60.0,
            spread=1.0,
        )
        provider.add_candles([c1, c2])

        # Query range covering both
        res = provider.get_historical_candles(CanonicalSymbol.DOW_JONES, Timeframe.M1, self.t1, self.t3)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0], c1)
        self.assertEqual(res[1], c2)

        # Query latest
        latest = provider.get_latest_candle(CanonicalSymbol.DOW_JONES, Timeframe.M1)
        self.assertEqual(latest, c2)

        # Query empty timeframe
        self.assertIsNone(provider.get_latest_candle(CanonicalSymbol.DOW_JONES, Timeframe.M5))
        self.assertEqual(provider.get_historical_candles(CanonicalSymbol.DOW_JONES, Timeframe.M5, self.t1, self.t3), [])

    def test_in_memory_provider_ticks(self) -> None:
        provider = InMemoryMarketDataProvider()
        tick1 = Tick(symbol=CanonicalSymbol.NASDAQ, timestamp=self.t1, bid=15000.0, ask=15000.5)
        tick2 = Tick(symbol=CanonicalSymbol.NASDAQ, timestamp=self.t2, bid=15001.0, ask=15001.5)
        provider.add_ticks([tick1, tick2])

        ticks = provider.get_historical_ticks(CanonicalSymbol.NASDAQ, self.t1, self.t2)
        self.assertEqual(len(ticks), 2)
        self.assertEqual(provider.get_latest_tick(CanonicalSymbol.NASDAQ), tick2)

        # Unknown symbol
        self.assertIsNone(provider.get_latest_tick(CanonicalSymbol.DOW_JONES))

    def test_in_memory_provider_quote_and_specification(self) -> None:
        provider = InMemoryMarketDataProvider()

        # Missing quote raises MissingDataError
        with self.assertRaises(MissingDataError):
            provider.get_live_quote(CanonicalSymbol.DOW_JONES)

        q = Quote(symbol=CanonicalSymbol.DOW_JONES, timestamp=self.t1, bid=34000.0, ask=34002.0, spread=2.0)
        provider.set_quote(q)
        self.assertEqual(provider.get_live_quote(CanonicalSymbol.DOW_JONES), q)

        # Missing spec raises MissingDataError
        with self.assertRaises(MissingDataError):
            provider.get_specification(CanonicalSymbol.DOW_JONES)

        spec = SymbolSpecification(
            canonical_symbol=CanonicalSymbol.DOW_JONES,
            broker_symbol="US100_i",
            tick_size=1.0,
            tick_value=1.0,
            contract_size=1.0,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            point=1.0,
        )
        provider.set_specification(spec)
        self.assertEqual(provider.get_specification(CanonicalSymbol.DOW_JONES), spec)


if __name__ == "__main__":
    unittest.main()
