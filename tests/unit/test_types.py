"""
Unit Tests for src/core/types.py
Validates frozen immutability, data contracts, field count invariants,
timezone validation, and deterministic Setup ID generation.
"""

from dataclasses import FrozenInstanceError, fields
from datetime import datetime, timezone
import unittest

from src.core.constants import (
    CanonicalSymbol,
    Direction,
    PositionStatus,
    Timeframe,
    TRADE_RECORD_EXACT_FIELD_COUNT,
    TradeDirection,
)
from src.core.exceptions import (
    DataValidationError,
    LevelImmutabilityViolationError,
)
from src.core.types import (
    AccountInfo,
    Candle,
    GapEvent,
    OrderRequest,
    OrderResult,
    PinBarEvent,
    Quote,
    SignalEvent,
    SpikeEvent,
    StructuralLevel,
    SymbolSpecification,
    TradeRecord,
    generate_setup_id,
)


class TestDomainTypes(unittest.TestCase):
    """Comprehensive test suite for Phase 2 domain types and data contracts."""

    def setUp(self) -> None:
        self.t_open = datetime(2023, 10, 1, 14, 0, 0, tzinfo=timezone.utc)
        self.t_close = datetime(2023, 10, 1, 14, 5, 0, tzinfo=timezone.utc)

    # --- Candle Tests ---
    def test_candle_valid_creation_and_properties(self) -> None:
        candle = Candle(
            symbol=CanonicalSymbol.DOW_JONES,
            timeframe=Timeframe.M5,
            open_time=self.t_open,
            close_time=self.t_close,
            open=34000.0,
            high=34050.0,
            low=33980.0,
            close=34030.0,
            volume=150.0,
            spread=2.0,
        )
        self.assertEqual(candle.symbol, CanonicalSymbol.DOW_JONES)
        self.assertEqual(candle.direction, Direction.BULLISH)
        self.assertAlmostEqual(candle.range, 70.0)
        self.assertAlmostEqual(candle.body, 30.0)
        self.assertEqual(candle.body_high, 34030.0)
        self.assertEqual(candle.body_low, 34000.0)

    def test_candle_immutability(self) -> None:
        candle = Candle(
            symbol=CanonicalSymbol.NASDAQ,
            timeframe=Timeframe.M1,
            open_time=self.t_open,
            close_time=self.t_close,
            open=15000.0,
            high=15020.0,
            low=14990.0,
            close=15010.0,
            volume=50.0,
            spread=1.0,
        )
        with self.assertRaises(FrozenInstanceError):
            candle.close = 15050.0  # type: ignore[misc]

    def test_candle_rejects_naive_timestamp(self) -> None:
        naive_dt = datetime(2023, 10, 1, 14, 0, 0)
        with self.assertRaises(DataValidationError):
            Candle(
                symbol=CanonicalSymbol.DOW_JONES,
                timeframe=Timeframe.M5,
                open_time=naive_dt,
                close_time=self.t_close,
                open=34000.0,
                high=34050.0,
                low=33980.0,
                close=34030.0,
                volume=150.0,
                spread=2.0,
            )

    def test_candle_rejects_inverted_high_low(self) -> None:
        with self.assertRaises(DataValidationError):
            Candle(
                symbol=CanonicalSymbol.DOW_JONES,
                timeframe=Timeframe.M5,
                open_time=self.t_open,
                close_time=self.t_close,
                open=34000.0,
                high=33900.0,  # High < Low
                low=34050.0,
                close=34000.0,
                volume=100.0,
                spread=1.0,
            )

    # --- Quote Tests ---
    def test_quote_valid_and_immutable(self) -> None:
        q = Quote(
            symbol=CanonicalSymbol.NASDAQ,
            timestamp=self.t_open,
            bid=15100.0,
            ask=15101.5,
            spread=1.5,
        )
        self.assertEqual(q.symbol, CanonicalSymbol.NASDAQ)
        self.assertEqual(q.spread, 1.5)
        with self.assertRaises(FrozenInstanceError):
            q.bid = 15105.0  # type: ignore[misc]

    def test_quote_rejects_negative_or_inverted_prices(self) -> None:
        with self.assertRaises(DataValidationError):
            Quote(
                symbol=CanonicalSymbol.NASDAQ,
                timestamp=self.t_open,
                bid=15100.0,
                ask=15090.0,  # ask < bid
                spread=1.0,
            )

    # --- SymbolSpecification Tests ---
    def test_symbol_specification_separation(self) -> None:
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
        self.assertEqual(spec.canonical_symbol, CanonicalSymbol.DOW_JONES)
        self.assertEqual(spec.broker_symbol, "US100_i")
        with self.assertRaises(FrozenInstanceError):
            spec.broker_symbol = "MODIFIED"  # type: ignore[misc]

    def test_symbol_specification_rejects_invalid_values(self) -> None:
        with self.assertRaises(DataValidationError):
            SymbolSpecification(
                canonical_symbol=CanonicalSymbol.DOW_JONES,
                broker_symbol="",  # empty broker symbol
                tick_size=1.0,
                tick_value=1.0,
                contract_size=1.0,
                volume_min=0.01,
                volume_max=100.0,
                volume_step=0.01,
                point=1.0,
            )

    # --- AccountInfo Tests ---
    def test_account_info_valid(self) -> None:
        acct = AccountInfo(
            balance=100000.0,
            equity=100500.0,
            margin=2500.0,
            free_margin=98000.0,
            currency="USD",
        )
        self.assertEqual(acct.currency, "USD")
        self.assertEqual(acct.balance, 100000.0)

    # --- GapEvent Tests ---
    def test_bullish_gap_event_valid(self) -> None:
        t1 = self.t_open
        t3 = self.t_close
        gap = GapEvent(
            direction=Direction.BULLISH,
            c1_high=34000.0,
            c1_low=33950.0,
            c3_high=34150.0,
            c3_low=34020.0,  # c3_low > c1_high
            c1_time=t1,
            c3_time=t3,
        )
        self.assertEqual(gap.direction, Direction.BULLISH)
        self.assertAlmostEqual(gap.gap_size, 20.0)

    def test_bullish_gap_event_rejects_invalid_gap(self) -> None:
        with self.assertRaises(DataValidationError):
            GapEvent(
                direction=Direction.BULLISH,
                c1_high=34000.0,
                c1_low=33950.0,
                c3_high=34150.0,
                c3_low=33990.0,  # c3_low not > c1_high
                c1_time=self.t_open,
                c3_time=self.t_close,
            )

    # --- SpikeEvent Tests ---
    def test_spike_event_valid(self) -> None:
        spike = SpikeEvent(
            spike_id="SPIKE-DOW-001",
            symbol=CanonicalSymbol.DOW_JONES,
            direction=Direction.BULLISH,
            start_time=self.t_open,
            end_time=self.t_close,
            high=34200.0,
            low=34000.0,
            candle_count=4,
            gap_count=2,
        )
        self.assertEqual(spike.candle_count, 4)
        self.assertEqual(spike.gap_count, 2)

    def test_spike_event_rejects_below_minimum_candles(self) -> None:
        with self.assertRaises(DataValidationError):
            SpikeEvent(
                spike_id="SPIKE-002",
                symbol=CanonicalSymbol.DOW_JONES,
                direction=Direction.BULLISH,
                start_time=self.t_open,
                end_time=self.t_close,
                high=34200.0,
                low=34000.0,
                candle_count=2,  # min is 3
                gap_count=1,
            )

    # --- StructuralLevel Tests ---
    def test_structural_level_locked_immutability(self) -> None:
        level = StructuralLevel(
            spike_id="SPIKE-001",
            symbol=CanonicalSymbol.NASDAQ,
            direction=Direction.BEARISH,
            level_type="FIRST_TOP",
            price=15250.0,
            formation_time=self.t_close,
            is_locked=True,
        )
        self.assertTrue(level.is_locked)
        with self.assertRaises(FrozenInstanceError):
            level.price = 15300.0  # type: ignore[misc]

    def test_structural_level_rejects_unlocked(self) -> None:
        with self.assertRaises(LevelImmutabilityViolationError):
            StructuralLevel(
                spike_id="SPIKE-001",
                symbol=CanonicalSymbol.NASDAQ,
                direction=Direction.BEARISH,
                level_type="FIRST_TOP",
                price=15250.0,
                formation_time=self.t_close,
                is_locked=False,
            )

    # --- PinBarEvent Tests ---
    def test_pinbar_event_validation(self) -> None:
        pin = PinBarEvent(
            symbol=CanonicalSymbol.DOW_JONES,
            timeframe=Timeframe.M1,
            close_time=self.t_close,
            open=34010.0,
            high=34060.0,
            low=34000.0,
            close=34008.0,
            range=60.0,
            body=2.0,
            body_ratio=2.0 / 60.0,
            midpoint=34030.0,
            is_valid=True,
            direction=Direction.BEARISH,
        )
        self.assertTrue(pin.is_valid)
        self.assertLess(pin.body_ratio, 0.20)

    # --- SignalEvent & OrderRequest Tests ---
    def test_signal_and_order_request_contracts(self) -> None:
        setup_id = "SET-DJ-BUY-ABC123456789"
        signal = SignalEvent(
            setup_id=setup_id,
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.BUY,
            signal_time=self.t_close,
            pinbar_high=34050.0,
            pinbar_low=34000.0,
            final_structural_target=34200.0,
        )
        self.assertEqual(signal.setup_id, setup_id)

        req = OrderRequest(
            setup_id=setup_id,
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.BUY,
            volume=0.5,
            entry_price=34052.0,
            stop_loss=33998.0,
            take_profits=(34080.0, 34110.0, 34150.0),
            final_target=34200.0,
            magic_number=123456,
            deviation=10,
        )
        self.assertEqual(req.volume, 0.5)
        self.assertEqual(req.magic_number, 123456)

    # --- OrderResult Tests ---
    def test_order_result_contract(self) -> None:
        res = OrderResult(
            success=True,
            order_ticket=1001,
            deal_ticket=2001,
            position_ticket=3001,
            executed_price=34052.0,
            executed_volume=0.5,
        )
        self.assertTrue(res.success)
        self.assertEqual(res.order_ticket, 1001)
        self.assertIsNone(res.error_message)

    # --- TradeRecord Invariant 35 Fields Test ---
    def test_trade_record_exact_field_count(self) -> None:
        actual_fields = len(fields(TradeRecord))
        self.assertEqual(
            actual_fields,
            TRADE_RECORD_EXACT_FIELD_COUNT,
            f"TradeRecord must have exactly {TRADE_RECORD_EXACT_FIELD_COUNT} fields, got {actual_fields}",
        )
        self.assertEqual(actual_fields, 35)

    def test_trade_record_exact_field_names_and_order(self) -> None:
        expected_fields = [
            "symbol",
            "date",
            "session",
            "spike_direction",
            "spike_time",
            "spike_high",
            "spike_low",
            "spike_candle_count",
            "gap_count",
            "first_level_price",
            "average_m5_range",
            "proximity_distance",
            "pinbar_open",
            "pinbar_high",
            "pinbar_low",
            "pinbar_close",
            "pinbar_body",
            "pinbar_range",
            "pinbar_body_ratio",
            "entry_price",
            "initial_sl",
            "tp1",
            "tp2",
            "tp3",
            "final_target",
            "spread",
            "risk_percent",
            "lot_size",
            "order_ticket",
            "deal_ticket",
            "position_ticket",
            "result",
            "r_multiple",
            "mfe",
            "mae",
        ]
        actual_field_names = [f.name for f in fields(TradeRecord)]
        self.assertEqual(actual_field_names, expected_fields)

    def test_trade_record_creation_and_immutability(self) -> None:
        rec = TradeRecord(
            symbol="DOW_JONES",
            date="2023-10-01",
            session="LONDON",
            spike_direction="BULLISH",
            spike_time="2023-10-01T14:00:00Z",
            spike_high=34100.0,
            spike_low=34000.0,
            spike_candle_count=3,
            gap_count=1,
            first_level_price=34050.0,
            average_m5_range=25.0,
            proximity_distance=2.5,
            pinbar_open=34045.0,
            pinbar_high=34060.0,
            pinbar_low=34040.0,
            pinbar_close=34048.0,
            pinbar_body=3.0,
            pinbar_range=20.0,
            pinbar_body_ratio=0.15,
            entry_price=34062.0,
            initial_sl=34036.0,
            tp1=34088.0,
            tp2=34114.0,
            tp3=34140.0,
            final_target=34200.0,
            spread=2.0,
            risk_percent=1.0,
            lot_size=0.5,
            order_ticket=1001,
            deal_ticket=2001,
            position_ticket=3001,
            result="WIN",
            r_multiple=3.5,
            mfe=4.0,
            mae=0.2,
        )
        self.assertEqual(rec.symbol, "DOW_JONES")
        self.assertEqual(rec.lot_size, 0.5)
        self.assertEqual(rec.result, "WIN")
        with self.assertRaises(FrozenInstanceError):
            rec.result = "LOSS"  # type: ignore[misc]

    # --- Setup ID Determinism Tests ---
    def test_setup_id_determinism_and_independence(self) -> None:
        t1 = datetime(2023, 10, 1, 14, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2023, 10, 1, 14, 15, 0, tzinfo=timezone.utc)
        t_level = datetime(2023, 10, 1, 14, 20, 0, tzinfo=timezone.utc)

        id1 = generate_setup_id(
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.BUY,
            spike_start_time=t1,
            spike_end_time=t2,
            level_formed_time=t_level,
            structural_price=34050.123,
            tick_size=1.0,
        )
        id2 = generate_setup_id(
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.BUY,
            spike_start_time=t1,
            spike_end_time=t2,
            level_formed_time=t_level,
            structural_price=34050.123,
            tick_size=1.0,
        )
        self.assertEqual(id1, id2, "Setup ID must be purely deterministic for identical inputs")
        self.assertTrue(id1.startswith("SET-DJ-BUY-"))
        self.assertEqual(len(id1.split("-")[-1]), 12, "Hash component must be 12 characters")

    def test_setup_id_format_and_symbol_prefix(self) -> None:
        t1 = datetime(2023, 10, 1, 14, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2023, 10, 1, 14, 15, 0, tzinfo=timezone.utc)
        t_level = datetime(2023, 10, 1, 14, 20, 0, tzinfo=timezone.utc)

        id_dow = generate_setup_id(
            symbol="DOW_JONES",
            direction="BUY",
            spike_start_time=t1,
            spike_end_time=t2,
            level_formed_time=t_level,
            structural_price=34000.0,
            tick_size=1.0,
        )
        id_nas = generate_setup_id(
            symbol="NASDAQ",
            direction="SELL",
            spike_start_time=t1,
            spike_end_time=t2,
            level_formed_time=t_level,
            structural_price=15000.0,
            tick_size=0.25,
        )
        self.assertTrue(id_dow.startswith("SET-DJ-BUY-"))
        self.assertTrue(id_nas.startswith("SET-NAS-SELL-"))
        self.assertNotEqual(id_dow, id_nas)

    def test_setup_id_rejects_broker_symbols(self) -> None:
        t1 = datetime(2023, 10, 1, 14, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2023, 10, 1, 14, 15, 0, tzinfo=timezone.utc)
        t_level = datetime(2023, 10, 1, 14, 20, 0, tzinfo=timezone.utc)

        with self.assertRaises(DataValidationError):
            generate_setup_id(
                symbol="US100_i",  # Broker symbol, forbidden as setup identity
                direction="BUY",
                spike_start_time=t1,
                spike_end_time=t2,
                level_formed_time=t_level,
                structural_price=34000.0,
                tick_size=1.0,
            )
        with self.assertRaises(DataValidationError):
            generate_setup_id(
                symbol="NAS100_i",  # Broker symbol, forbidden as setup identity
                direction="SELL",
                spike_start_time=t1,
                spike_end_time=t2,
                level_formed_time=t_level,
                structural_price=15000.0,
                tick_size=0.25,
            )

    def test_setup_id_explicit_tick_size_enforcement(self) -> None:
        t1 = datetime(2023, 10, 1, 14, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2023, 10, 1, 14, 15, 0, tzinfo=timezone.utc)
        t_level = datetime(2023, 10, 1, 14, 20, 0, tzinfo=timezone.utc)

        # 1. Calling without tick_size raises TypeError (no default parameter)
        with self.assertRaises(TypeError):
            generate_setup_id(  # type: ignore[call-arg]
                symbol=CanonicalSymbol.DOW_JONES,
                direction=TradeDirection.BUY,
                spike_start_time=t1,
                spike_end_time=t2,
                level_formed_time=t_level,
                structural_price=34000.0,
            )

        # 2. tick_size <= 0 rejected
        with self.assertRaises(DataValidationError):
            generate_setup_id(
                symbol=CanonicalSymbol.DOW_JONES,
                direction=TradeDirection.BUY,
                spike_start_time=t1,
                spike_end_time=t2,
                level_formed_time=t_level,
                structural_price=34000.0,
                tick_size=0.0,
            )
        with self.assertRaises(DataValidationError):
            generate_setup_id(
                symbol=CanonicalSymbol.DOW_JONES,
                direction=TradeDirection.BUY,
                spike_start_time=t1,
                spike_end_time=t2,
                level_formed_time=t_level,
                structural_price=34000.0,
                tick_size=-0.01,
            )

        # 3. Different valid tick sizes can produce different normalized prices & IDs
        id_tick1 = generate_setup_id(
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.BUY,
            spike_start_time=t1,
            spike_end_time=t2,
            level_formed_time=t_level,
            structural_price=34000.4,
            tick_size=1.0,
        )
        id_tick_fine = generate_setup_id(
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.BUY,
            spike_start_time=t1,
            spike_end_time=t2,
            level_formed_time=t_level,
            structural_price=34000.4,
            tick_size=0.25,
        )
        self.assertNotEqual(id_tick1, id_tick_fine)

    def test_setup_id_trade_direction_enforcement(self) -> None:
        t1 = datetime(2023, 10, 1, 14, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2023, 10, 1, 14, 15, 0, tzinfo=timezone.utc)
        t_level = datetime(2023, 10, 1, 14, 20, 0, tzinfo=timezone.utc)

        # 1. TradeDirection.BUY works
        id_buy = generate_setup_id(
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.BUY,
            spike_start_time=t1,
            spike_end_time=t2,
            level_formed_time=t_level,
            structural_price=34000.0,
            tick_size=1.0,
        )
        self.assertIn("-BUY-", id_buy)

        # 2. TradeDirection.SELL works
        id_sell = generate_setup_id(
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.SELL,
            spike_start_time=t1,
            spike_end_time=t2,
            level_formed_time=t_level,
            structural_price=34000.0,
            tick_size=1.0,
        )
        self.assertIn("-SELL-", id_sell)

        # 3. String "BUY" works
        id_str_buy = generate_setup_id(
            symbol=CanonicalSymbol.DOW_JONES,
            direction="BUY",
            spike_start_time=t1,
            spike_end_time=t2,
            level_formed_time=t_level,
            structural_price=34000.0,
            tick_size=1.0,
        )
        self.assertEqual(id_buy, id_str_buy)

        # 4. String "SELL" works
        id_str_sell = generate_setup_id(
            symbol=CanonicalSymbol.DOW_JONES,
            direction="SELL",
            spike_start_time=t1,
            spike_end_time=t2,
            level_formed_time=t_level,
            structural_price=34000.0,
            tick_size=1.0,
        )
        self.assertEqual(id_sell, id_str_sell)

        # 5. Direction.BULLISH is rejected
        with self.assertRaises(DataValidationError):
            generate_setup_id(
                symbol=CanonicalSymbol.DOW_JONES,
                direction=Direction.BULLISH,  # type: ignore[arg-type]
                spike_start_time=t1,
                spike_end_time=t2,
                level_formed_time=t_level,
                structural_price=34000.0,
                tick_size=1.0,
            )

        # 6. Direction.BEARISH is rejected
        with self.assertRaises(DataValidationError):
            generate_setup_id(
                symbol=CanonicalSymbol.DOW_JONES,
                direction=Direction.BEARISH,  # type: ignore[arg-type]
                spike_start_time=t1,
                spike_end_time=t2,
                level_formed_time=t_level,
                structural_price=34000.0,
                tick_size=1.0,
            )

        # 7. Direction.NEUTRAL is rejected
        with self.assertRaises(DataValidationError):
            generate_setup_id(
                symbol=CanonicalSymbol.DOW_JONES,
                direction=Direction.NEUTRAL,  # type: ignore[arg-type]
                spike_start_time=t1,
                spike_end_time=t2,
                level_formed_time=t_level,
                structural_price=34000.0,
                tick_size=1.0,
            )

        # 8. "BULLISH" is rejected
        with self.assertRaises(DataValidationError):
            generate_setup_id(
                symbol=CanonicalSymbol.DOW_JONES,
                direction="BULLISH",
                spike_start_time=t1,
                spike_end_time=t2,
                level_formed_time=t_level,
                structural_price=34000.0,
                tick_size=1.0,
            )

        # 9. "BEARISH" is rejected
        with self.assertRaises(DataValidationError):
            generate_setup_id(
                symbol=CanonicalSymbol.DOW_JONES,
                direction="BEARISH",
                spike_start_time=t1,
                spike_end_time=t2,
                level_formed_time=t_level,
                structural_price=34000.0,
                tick_size=1.0,
            )

        # 10. "NEUTRAL" is rejected
        with self.assertRaises(DataValidationError):
            generate_setup_id(
                symbol=CanonicalSymbol.DOW_JONES,
                direction="NEUTRAL",
                spike_start_time=t1,
                spike_end_time=t2,
                level_formed_time=t_level,
                structural_price=34000.0,
                tick_size=1.0,
            )

    def test_setup_id_sensitivity_to_each_input(self) -> None:
        t1 = datetime(2023, 10, 1, 14, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2023, 10, 1, 14, 15, 0, tzinfo=timezone.utc)
        t_level = datetime(2023, 10, 1, 14, 20, 0, tzinfo=timezone.utc)
        base_price = 34000.0

        base_id = generate_setup_id(
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.BUY,
            spike_start_time=t1,
            spike_end_time=t2,
            level_formed_time=t_level,
            structural_price=base_price,
            tick_size=1.0,
            version="v1.0",
        )

        # 1. Sensitivity to version
        diff_version = generate_setup_id(
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.BUY,
            spike_start_time=t1,
            spike_end_time=t2,
            level_formed_time=t_level,
            structural_price=base_price,
            tick_size=1.0,
            version="v1.1",
        )
        self.assertNotEqual(base_id, diff_version)

        # 2. Sensitivity to symbol
        diff_symbol = generate_setup_id(
            symbol=CanonicalSymbol.NASDAQ,
            direction=TradeDirection.BUY,
            spike_start_time=t1,
            spike_end_time=t2,
            level_formed_time=t_level,
            structural_price=base_price,
            tick_size=1.0,
        )
        self.assertNotEqual(base_id, diff_symbol)

        # 3. Sensitivity to direction
        diff_direction = generate_setup_id(
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.SELL,
            spike_start_time=t1,
            spike_end_time=t2,
            level_formed_time=t_level,
            structural_price=base_price,
            tick_size=1.0,
        )
        self.assertNotEqual(base_id, diff_direction)

        # 4. Sensitivity to spike_start_time
        diff_start = generate_setup_id(
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.BUY,
            spike_start_time=datetime(2023, 10, 1, 14, 5, 0, tzinfo=timezone.utc),
            spike_end_time=t2,
            level_formed_time=t_level,
            structural_price=base_price,
            tick_size=1.0,
        )
        self.assertNotEqual(base_id, diff_start)

        # 5. Sensitivity to spike_end_time
        diff_end = generate_setup_id(
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.BUY,
            spike_start_time=t1,
            spike_end_time=datetime(2023, 10, 1, 14, 18, 0, tzinfo=timezone.utc),
            level_formed_time=t_level,
            structural_price=base_price,
            tick_size=1.0,
        )
        self.assertNotEqual(base_id, diff_end)

        # 6. Sensitivity to level_formed_time
        diff_level = generate_setup_id(
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.BUY,
            spike_start_time=t1,
            spike_end_time=t2,
            level_formed_time=datetime(2023, 10, 1, 14, 25, 0, tzinfo=timezone.utc),
            structural_price=base_price,
            tick_size=1.0,
        )
        self.assertNotEqual(base_id, diff_level)

        # 7. Sensitivity to structural_price
        diff_price = generate_setup_id(
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.BUY,
            spike_start_time=t1,
            spike_end_time=t2,
            level_formed_time=t_level,
            structural_price=base_price + 10.0,
            tick_size=1.0,
        )
        self.assertNotEqual(base_id, diff_price)


if __name__ == "__main__":
    unittest.main()
