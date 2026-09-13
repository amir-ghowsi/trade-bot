"""
Unit Tests for src/adapters/simulated_broker.py
Validates deterministic execution, account state, position constraints, partial closing,
numeric safety (boolean/NaN/Inf rejection), tickets, commissions, and quote semantics.

Strictly covers all 28 required Phase 7 test scenarios:
1. Correct canonical-to-broker mapping (US100_i, NAS100_i)
2. No US30 fallback (unregistered raises DataValidationError)
3. No wrong NAS100 mapping (unregistered raises DataValidationError)
4. BUY uses Ask
5. SELL uses Bid
6. BUY close uses Bid
7. SELL close uses Ask
8. Spread = Ask - Bid
9. Max one position per symbol
10. Max two positions globally
11. Rejected order leaves existing state intact (rejection isolation)
12. Deterministic ticket generation (Order 100001+, Deal 200001+, Position 300001+)
13. Identical input sequence -> identical results
14. NaN rejection
15. Infinity rejection
16. bool rejection
17. Invalid symbol rejection
18. Missing quote rejection
19. Volume min rejection
20. Volume max rejection
21. Volume step rejection
22. Disconnected broker rejection
23. Partial close
24. Full close
25. Closed position cannot be closed again
26. Position modification
27. Commission accounting
28. Slippage parameter & base quote price execution
"""

from datetime import datetime, timezone
import math
from typing import Any, Dict
import unittest

from src.adapters.simulated_broker import SimulatedBroker
from src.adapters.types import BrokerDeal, BrokerOrder, BrokerPosition
from src.core.constants import (
    CanonicalSymbol,
    OrderStatus,
    PositionStatus,
    TradeDirection,
)
from src.core.exceptions import DataValidationError
from src.core.types import AccountInfo, OrderRequest, Quote, SymbolSpecification


class TestSimulatedBroker(unittest.TestCase):
    """Test suite for the deterministic SimulatedBroker implementation."""

    def setUp(self) -> None:
        """Create fresh broker instance and standard market quotes for tests."""
        self.dj_spec = SymbolSpecification(
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
        self.nas_spec = SymbolSpecification(
            canonical_symbol=CanonicalSymbol.NASDAQ,
            broker_symbol="NAS100_i",
            tick_size=0.25,
            tick_value=0.25,
            contract_size=1.0,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            point=0.25,
        )
        self.broker = SimulatedBroker(
            initial_balance=100000.0,
            currency="USD",
            symbol_specs={
                CanonicalSymbol.DOW_JONES: self.dj_spec,
                CanonicalSymbol.NASDAQ: self.nas_spec,
            },
        )
        self.t0 = datetime(2026, 9, 13, 14, 30, 0, tzinfo=timezone.utc)
        self.t1 = datetime(2026, 9, 13, 14, 35, 0, tzinfo=timezone.utc)

        # Standard quotes: Dow Jones & Nasdaq
        self.quote_dj = Quote(
            symbol=CanonicalSymbol.DOW_JONES,
            timestamp=self.t0,
            bid=34000.0,
            ask=34002.0,
            spread=2.0,
        )
        self.quote_nas = Quote(
            symbol=CanonicalSymbol.NASDAQ,
            timestamp=self.t0,
            bid=15000.0,
            ask=15001.0,
            spread=1.0,
        )
        self.broker.set_quote(self.quote_dj)
        self.broker.set_quote(self.quote_nas)

    # -----------------------------------------------------------------
    # 1. Correct CanonicalSymbol specification query
    # -----------------------------------------------------------------
    def test_01_symbol_specification_mapping_canonical(self) -> None:
        """Verify CanonicalSymbol specification queries return injected specs."""
        spec_dj = self.broker.get_symbol_specification(CanonicalSymbol.DOW_JONES)
        self.assertEqual(spec_dj["tick_size"], 1.0)
        self.assertEqual(spec_dj["tick_value"], 1.0)
        self.assertEqual(spec_dj["spread"], 2.0)

        spec_nas = self.broker.get_symbol_specification(CanonicalSymbol.NASDAQ)
        self.assertEqual(spec_nas["tick_size"], 0.25)
        self.assertEqual(spec_nas["tick_value"], 0.25)
        self.assertEqual(spec_nas["spread"], 1.0)

    # -----------------------------------------------------------------
    # 2. Rejects non-canonical broker symbol US30 / US100_i
    # -----------------------------------------------------------------
    def test_02_symbol_mapping_rejects_unregistered_us30(self) -> None:
        """Broker symbols must raise DataValidationError on strategy-facing APIs."""
        with self.assertRaises(DataValidationError):
            self.broker.get_symbol_specification("US30")

        with self.assertRaises(DataValidationError):
            self.broker.get_live_quote("US30")

        with self.assertRaises(DataValidationError):
            self.broker.get_symbol_specification("US100_i")

    # -----------------------------------------------------------------
    # 3. Rejects non-canonical broker symbol NAS100 / NAS100_i
    # -----------------------------------------------------------------
    def test_03_symbol_mapping_rejects_unregistered_nas100(self) -> None:
        """Broker symbols must raise DataValidationError on strategy-facing APIs."""
        with self.assertRaises(DataValidationError):
            self.broker.get_symbol_specification("NAS100")

        with self.assertRaises(DataValidationError):
            self.broker.get_live_quote("NAS100")

        with self.assertRaises(DataValidationError):
            self.broker.get_symbol_specification("NAS100_i")

    # -----------------------------------------------------------------
    # 4. BUY uses Ask
    # -----------------------------------------------------------------
    def test_04_buy_executes_at_ask_price(self) -> None:
        """BUY market orders must execute strictly at live Ask price."""
        req = OrderRequest(
            setup_id="SET-DJ-BUY-1111",
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.BUY,
            volume=1.5,
            entry_price=34000.0,
            stop_loss=33950.0,
            take_profits=(34050.0,),
            final_target=34150.0,
            magic_number=1001,
            deviation=0,
        )
        res = self.broker.send_order(req)
        self.assertTrue(res["success"])
        self.assertEqual(res["executed_price"], 34002.0)  # quote_dj.ask is 34002.0
        self.assertEqual(res["executed_volume"], 1.5)

        pos = self.broker.get_position_by_ticket(res["position_ticket"])
        self.assertIsNotNone(pos)
        self.assertEqual(pos["entry_price"], 34002.0)

    # -----------------------------------------------------------------
    # 5. SELL uses Bid
    # -----------------------------------------------------------------
    def test_05_sell_executes_at_bid_price(self) -> None:
        """SELL market orders must execute strictly at live Bid price."""
        req = OrderRequest(
            setup_id="SET-NAS-SELL-2222",
            symbol=CanonicalSymbol.NASDAQ,
            direction=TradeDirection.SELL,
            volume=2.0,
            entry_price=15005.0,
            stop_loss=15050.0,
            take_profits=(),
            final_target=14900.0,
            magic_number=1001,
            deviation=0,
        )
        res = self.broker.send_order(req)
        self.assertTrue(res["success"])
        self.assertEqual(res["executed_price"], 15000.0)  # quote_nas.bid is 15000.0
        self.assertEqual(res["executed_volume"], 2.0)

        pos = self.broker.get_position_by_ticket(res["position_ticket"])
        self.assertIsNotNone(pos)
        self.assertEqual(pos["entry_price"], 15000.0)

    # -----------------------------------------------------------------
    # 6. BUY close uses Bid
    # -----------------------------------------------------------------
    def test_06_buy_close_executes_at_bid_price(self) -> None:
        """Closing a BUY position must execute strictly at live Bid price."""
        req = OrderRequest(
            setup_id="SET-DJ-BUY-1",
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.BUY,
            volume=1.0,
            entry_price=34002.0,
            stop_loss=33950.0,
            take_profits=(),
            final_target=34150.0,
            magic_number=1001,
            deviation=0,
        )
        res = self.broker.send_order(req)
        ticket = res["position_ticket"]

        # Market moves: new quote bid=34050, ask=34052
        quote_new = Quote(
            symbol=CanonicalSymbol.DOW_JONES,
            timestamp=self.t1,
            bid=34050.0,
            ask=34052.0,
            spread=2.0,
        )
        self.broker.set_quote(quote_new)

        close_res = self.broker.close_position(ticket)
        self.assertTrue(close_res["success"])
        self.assertEqual(close_res["close_price"], 34050.0)  # closed at live Bid

    # -----------------------------------------------------------------
    # 7. SELL close uses Ask
    # -----------------------------------------------------------------
    def test_07_sell_close_executes_at_ask_price(self) -> None:
        """Closing a SELL position must execute strictly at live Ask price."""
        req = OrderRequest(
            setup_id="SET-NAS-SELL-1",
            symbol=CanonicalSymbol.NASDAQ,
            direction=TradeDirection.SELL,
            volume=1.0,
            entry_price=15000.0,
            stop_loss=15050.0,
            take_profits=(),
            final_target=14900.0,
            magic_number=1001,
            deviation=0,
        )
        res = self.broker.send_order(req)
        ticket = res["position_ticket"]

        # Market moves: new quote bid=14950, ask=14951
        quote_new = Quote(
            symbol=CanonicalSymbol.NASDAQ,
            timestamp=self.t1,
            bid=14950.0,
            ask=14951.0,
            spread=1.0,
        )
        self.broker.set_quote(quote_new)

        close_res = self.broker.close_position(ticket)
        self.assertTrue(close_res["success"])
        self.assertEqual(close_res["close_price"], 14951.0)  # closed at live Ask

    # -----------------------------------------------------------------
    # 8. Spread = Ask - Bid
    # -----------------------------------------------------------------
    def test_08_spread_equals_ask_minus_bid(self) -> None:
        """Spread must always equal ask - bid."""
        q_dj = self.broker.get_live_quote(CanonicalSymbol.DOW_JONES)
        self.assertEqual(q_dj["spread"], round(q_dj["ask"] - q_dj["bid"], 8))
        self.assertEqual(q_dj["spread"], 2.0)

        q_nas = self.broker.get_live_quote(CanonicalSymbol.NASDAQ)
        self.assertEqual(q_nas["spread"], round(q_nas["ask"] - q_nas["bid"], 8))
        self.assertEqual(q_nas["spread"], 1.0)

    # -----------------------------------------------------------------
    # 9. Max one position per symbol
    # -----------------------------------------------------------------
    def test_09_position_constraint_one_per_symbol(self) -> None:
        """Maximum 1 active position per symbol constraint."""
        req1 = OrderRequest(
            setup_id="SET-DJ-BUY-1",
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.BUY,
            volume=1.0,
            entry_price=34002.0,
            stop_loss=33950.0,
            take_profits=(),
            final_target=34150.0,
            magic_number=1001,
            deviation=0,
        )
        res1 = self.broker.send_order(req1)
        self.assertTrue(res1["success"])

        req2 = OrderRequest(
            setup_id="SET-DJ-BUY-2",
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.BUY,
            volume=1.0,
            entry_price=34002.0,
            stop_loss=33950.0,
            take_profits=(),
            final_target=34150.0,
            magic_number=1001,
            deviation=0,
        )
        res2 = self.broker.send_order(req2)
        self.assertFalse(res2["success"])
        self.assertIn("Maximum active positions reached for symbol", res2["error_message"] or "")

    # -----------------------------------------------------------------
    # 10. Max two positions globally
    # -----------------------------------------------------------------
    def test_10_position_constraint_max_two_globally(self) -> None:
        """Maximum 2 total active positions across all symbols."""
        req_dj = OrderRequest(
            setup_id="SET-DJ-BUY-1",
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.BUY,
            volume=1.0,
            entry_price=34002.0,
            stop_loss=33950.0,
            take_profits=(),
            final_target=34150.0,
            magic_number=1001,
            deviation=0,
        )
        res_dj = self.broker.send_order(req_dj)
        self.assertTrue(res_dj["success"])

        req_nas = OrderRequest(
            setup_id="SET-NAS-SELL-2",
            symbol=CanonicalSymbol.NASDAQ,
            direction=TradeDirection.SELL,
            volume=2.0,
            entry_price=15000.0,
            stop_loss=15050.0,
            take_profits=(),
            final_target=14900.0,
            magic_number=1001,
            deviation=0,
        )
        res_nas = self.broker.send_order(req_nas)
        self.assertTrue(res_nas["success"])
        self.assertEqual(len(self.broker.get_active_positions()), 2)

        # Third order must be rejected by global position limit
        req_third = OrderRequest(
            setup_id="SET-DJ-BUY-3",
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.BUY,
            volume=1.0,
            entry_price=34002.0,
            stop_loss=33950.0,
            take_profits=(),
            final_target=34150.0,
            magic_number=1001,
            deviation=0,
        )
        res_third = self.broker.send_order(req_third)
        self.assertFalse(res_third["success"])
        self.assertIn("Maximum active positions reached", res_third["error_message"] or "")
        self.assertEqual(len(self.broker.get_active_positions()), 2)

    # -----------------------------------------------------------------
    # 11. Rejection isolation (existing state preserved)
    # -----------------------------------------------------------------
    def test_11_rejected_order_isolation_state_preservation(self) -> None:
        """When an order is rejected, existing positions, balance, and deals remain completely intact."""
        # Open 1 valid position
        req1 = OrderRequest(
            setup_id="SET-DJ-BUY-1",
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.BUY,
            volume=1.0,
            entry_price=34002.0,
            stop_loss=33950.0,
            take_profits=(),
            final_target=34150.0,
            magic_number=1001,
            deviation=0,
        )
        res1 = self.broker.send_order(req1)
        self.assertTrue(res1["success"])
        initial_balance = self.broker.get_account_info()["balance"]
        initial_positions = self.broker.get_active_positions()
        initial_deals = list(self.broker.get_all_deals())

        # Attempt invalid orders (e.g. limit breach, invalid volume, boolean type, corrupted symbol)
        invalid_orders = [
            # Max positions limit breach
            {
                "setup_id": "SET-DJ-BUY-FAIL-1",
                "symbol": CanonicalSymbol.DOW_JONES,
                "direction": TradeDirection.BUY,
                "volume": 1.0,
                "entry_price": 34002.0,
                "stop_loss": 33950.0,
                "final_target": 34150.0,
                "magic_number": 1001,
            },
            # Boolean in volume
            {
                "setup_id": "SET-NAS-BUY-FAIL-2",
                "symbol": CanonicalSymbol.NASDAQ,
                "direction": TradeDirection.BUY,
                "volume": True,
                "entry_price": 15001.0,
                "stop_loss": 14950.0,
                "final_target": 15100.0,
                "magic_number": 1001,
            },
            # Volume step violation
            {
                "setup_id": "SET-NAS-BUY-FAIL-3",
                "symbol": CanonicalSymbol.NASDAQ,
                "direction": TradeDirection.BUY,
                "volume": 1.005,
                "entry_price": 15001.0,
                "stop_loss": 14950.0,
                "final_target": 15100.0,
                "magic_number": 1001,
            },
        ]

        for bad_req in invalid_orders:
            bad_res = self.broker.send_order(bad_req)
            self.assertFalse(bad_res["success"])
            self.assertIsNone(bad_res["order_ticket"])
            self.assertIsNone(bad_res["deal_ticket"])
            self.assertIsNone(bad_res["position_ticket"])

        # State must remain 100% identical to before rejection attempts
        self.assertEqual(self.broker.get_account_info()["balance"], initial_balance)
        self.assertEqual(self.broker.get_active_positions(), initial_positions)
        self.assertEqual(self.broker.get_all_deals(), initial_deals)

        # CRITICAL REJECTION DETERMINISM:
        # A valid order submitted AFTER rejected orders receives the exact next monotonic ticket (100002)
        # proving rejected orders consumed ZERO tickets.
        req2 = OrderRequest(
            setup_id="SET-NAS-BUY-2",
            symbol=CanonicalSymbol.NASDAQ,
            direction=TradeDirection.BUY,
            volume=1.0,
            entry_price=15001.0,
            stop_loss=14950.0,
            take_profits=(),
            final_target=15100.0,
            magic_number=1001,
            deviation=0,
        )
        res2 = self.broker.send_order(req2)
        self.assertTrue(res2["success"])
        self.assertEqual(res2["order_ticket"], 100002)
        self.assertEqual(res2["deal_ticket"], 200002)
        self.assertEqual(res2["position_ticket"], 300002)

    # -----------------------------------------------------------------
    # 12. Deterministic ticket generation
    # -----------------------------------------------------------------
    def test_12_deterministic_monotonic_tickets(self) -> None:
        """Order (100001+), Deal (200001+), and Position (300001+) tickets must increment deterministically."""
        req1 = OrderRequest(
            setup_id="SET-DJ-BUY-1",
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.BUY,
            volume=1.0,
            entry_price=34002.0,
            stop_loss=33950.0,
            take_profits=(),
            final_target=34150.0,
            magic_number=1001,
            deviation=0,
        )
        res1 = self.broker.send_order(req1)
        self.assertEqual(res1["order_ticket"], 100001)
        self.assertEqual(res1["deal_ticket"], 200001)
        self.assertEqual(res1["position_ticket"], 300001)

        req2 = OrderRequest(
            setup_id="SET-NAS-SELL-2",
            symbol=CanonicalSymbol.NASDAQ,
            direction=TradeDirection.SELL,
            volume=1.0,
            entry_price=15000.0,
            stop_loss=15050.0,
            take_profits=(),
            final_target=14900.0,
            magic_number=1001,
            deviation=0,
        )
        res2 = self.broker.send_order(req2)
        self.assertEqual(res2["order_ticket"], 100002)
        self.assertEqual(res2["deal_ticket"], 200002)
        self.assertEqual(res2["position_ticket"], 300002)

    # -----------------------------------------------------------------
    # 13. Identical input sequence -> identical results
    # -----------------------------------------------------------------
    def test_13_identical_runs_yield_identical_state(self) -> None:
        """Two separate simulated broker instances processing identical events yield identical state."""
        specs = {
            CanonicalSymbol.DOW_JONES: self.dj_spec,
            CanonicalSymbol.NASDAQ: self.nas_spec,
        }
        b1 = SimulatedBroker(initial_balance=100000.0, commission_per_lot=0.0, symbol_specs=specs)
        b2 = SimulatedBroker(initial_balance=100000.0, commission_per_lot=0.0, symbol_specs=specs)

        b1.set_quote(self.quote_dj)
        b2.set_quote(self.quote_dj)

        req = OrderRequest(
            setup_id="SET-DJ-BUY-DETERMINISM",
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.BUY,
            volume=1.0,
            entry_price=34002.0,
            stop_loss=33950.0,
            take_profits=(34050.0, 34100.0),
            final_target=34150.0,
            magic_number=1001,
            deviation=0,
        )

        res1 = b1.send_order(req)
        res2 = b2.send_order(req)

        self.assertEqual(res1, res2)
        self.assertEqual(b1.get_account_info(), b2.get_account_info())
        self.assertEqual(
            b1.get_position_by_ticket(res1["position_ticket"]),
            b2.get_position_by_ticket(res2["position_ticket"]),
        )

    # -----------------------------------------------------------------
    # 14. NaN rejection
    # -----------------------------------------------------------------
    def test_14_numeric_safety_rejects_nan(self) -> None:
        """NaN must be rejected across all numeric fields."""
        base_req = {
            "setup_id": "SET-DJ-BUY-1234567890AB",
            "symbol": CanonicalSymbol.DOW_JONES,
            "direction": TradeDirection.BUY,
            "volume": 1.0,
            "entry_price": 34002.0,
            "stop_loss": 33950.0,
            "final_target": 34150.0,
            "magic_number": 1001,
        }
        for f in ["volume", "entry_price", "stop_loss", "final_target"]:
            bad_req = dict(base_req)
            bad_req[f] = float("nan")
            is_valid, err = self.broker.check_order(bad_req)
            self.assertFalse(is_valid)
            self.assertIn("finite", err or "")

    # -----------------------------------------------------------------
    # 15. Infinity rejection
    # -----------------------------------------------------------------
    def test_15_numeric_safety_rejects_infinity(self) -> None:
        """+Inf and -Inf must be rejected across all numeric fields."""
        base_req = {
            "setup_id": "SET-DJ-BUY-1234567890AB",
            "symbol": CanonicalSymbol.DOW_JONES,
            "direction": TradeDirection.BUY,
            "volume": 1.0,
            "entry_price": 34002.0,
            "stop_loss": 33950.0,
            "final_target": 34150.0,
            "magic_number": 1001,
        }
        for f in ["volume", "entry_price", "stop_loss", "final_target"]:
            for inf_val in [float("inf"), float("-inf")]:
                bad_req = dict(base_req)
                bad_req[f] = inf_val
                is_valid, err = self.broker.check_order(bad_req)
                self.assertFalse(is_valid)
                self.assertIn("finite", err or "")

    # -----------------------------------------------------------------
    # 16. Boolean rejection
    # -----------------------------------------------------------------
    def test_16_numeric_safety_rejects_booleans(self) -> None:
        """Booleans (True/False) must be strictly rejected on all numeric fields."""
        base_req = {
            "setup_id": "SET-DJ-BUY-1234567890AB",
            "symbol": CanonicalSymbol.DOW_JONES,
            "direction": TradeDirection.BUY,
            "volume": 1.0,
            "entry_price": 34002.0,
            "stop_loss": 33950.0,
            "take_profits": (34050.0,),
            "final_target": 34150.0,
            "magic_number": 1001,
            "deviation": 5,
        }
        for f in ["volume", "entry_price", "stop_loss", "final_target", "magic_number", "deviation"]:
            for b in [True, False]:
                bad_req = dict(base_req)
                bad_req[f] = b
                is_valid, err = self.broker.check_order(bad_req)
                self.assertFalse(is_valid)

    # -----------------------------------------------------------------
    # 17. Invalid symbol rejection
    # -----------------------------------------------------------------
    def test_17_invalid_symbol_rejection(self) -> None:
        """Invalid or unregistered symbol representation must be rejected."""
        base_req = {
            "setup_id": "SET-FAIL-SYM",
            "symbol": "INVALID_SYMBOL_XYZ",
            "direction": TradeDirection.BUY,
            "volume": 1.0,
            "entry_price": 100.0,
            "stop_loss": 90.0,
            "final_target": 110.0,
            "magic_number": 1001,
        }
        is_valid, err = self.broker.check_order(base_req)
        self.assertFalse(is_valid)
        self.assertIn("Invalid symbol", err or "")

    # -----------------------------------------------------------------
    # 18. Missing quote rejection
    # -----------------------------------------------------------------
    def test_18_missing_quote_rejection(self) -> None:
        """Missing market quote must reject check_order and send_order."""
        fresh_broker = SimulatedBroker(
            symbol_specs={CanonicalSymbol.DOW_JONES: self.dj_spec}
        )  # Spec registered, but no quotes set
        req = {
            "setup_id": "SET-DJ-BUY-1",
            "symbol": CanonicalSymbol.DOW_JONES,
            "direction": TradeDirection.BUY,
            "volume": 1.0,
            "entry_price": 34002.0,
            "stop_loss": 33950.0,
            "final_target": 34150.0,
            "magic_number": 1001,
        }
        is_valid, err = fresh_broker.check_order(req)
        self.assertFalse(is_valid)
        self.assertIn("No live market quote", err or "")

        res = fresh_broker.send_order(req)
        self.assertFalse(res["success"])
        self.assertIn("No live market quote", res["error_message"] or "")

    # -----------------------------------------------------------------
    # 19. Volume min rejection
    # -----------------------------------------------------------------
    def test_19_volume_min_rejection(self) -> None:
        """Volume below broker minimum must be rejected."""
        req = {
            "setup_id": "SET-DJ-BUY-MIN",
            "symbol": CanonicalSymbol.DOW_JONES,
            "direction": TradeDirection.BUY,
            "volume": 0.001,  # below 0.01 min
            "entry_price": 34002.0,
            "stop_loss": 33950.0,
            "final_target": 34150.0,
            "magic_number": 1001,
        }
        is_valid, err = self.broker.check_order(req)
        self.assertFalse(is_valid)
        self.assertIn("below broker minimum", err or "")

    # -----------------------------------------------------------------
    # 20. Volume max rejection
    # -----------------------------------------------------------------
    def test_20_volume_max_rejection(self) -> None:
        """Volume exceeding broker maximum must be rejected."""
        req = {
            "setup_id": "SET-DJ-BUY-MAX",
            "symbol": CanonicalSymbol.DOW_JONES,
            "direction": TradeDirection.BUY,
            "volume": 150.0,  # above 100.0 max
            "entry_price": 34002.0,
            "stop_loss": 33950.0,
            "final_target": 34150.0,
            "magic_number": 1001,
        }
        is_valid, err = self.broker.check_order(req)
        self.assertFalse(is_valid)
        self.assertIn("exceeds broker maximum", err or "")

    # -----------------------------------------------------------------
    # 21. Volume step rejection
    # -----------------------------------------------------------------
    def test_21_volume_step_rejection(self) -> None:
        """Volume not aligned to volume_step must be rejected."""
        req = {
            "setup_id": "SET-DJ-BUY-STEP",
            "symbol": CanonicalSymbol.DOW_JONES,
            "direction": TradeDirection.BUY,
            "volume": 1.005,  # step is 0.01
            "entry_price": 34002.0,
            "stop_loss": 33950.0,
            "final_target": 34150.0,
            "magic_number": 1001,
        }
        is_valid, err = self.broker.check_order(req)
        self.assertFalse(is_valid)
        self.assertIn("volume_step", err or "")

    # -----------------------------------------------------------------
    # 22. Disconnected broker rejection
    # -----------------------------------------------------------------
    def test_22_disconnected_broker_rejection(self) -> None:
        """Disconnected broker must reject check_order and send_order."""
        self.broker.shutdown()
        self.assertFalse(self.broker.is_connected())

        req = {
            "setup_id": "SET-DJ-BUY-DISC",
            "symbol": CanonicalSymbol.DOW_JONES,
            "direction": TradeDirection.BUY,
            "volume": 1.0,
            "entry_price": 34002.0,
            "stop_loss": 33950.0,
            "final_target": 34150.0,
            "magic_number": 1001,
        }
        is_valid, err = self.broker.check_order(req)
        self.assertFalse(is_valid)
        self.assertIn("not connected", err or "")

        res = self.broker.send_order(req)
        self.assertFalse(res["success"])

    # -----------------------------------------------------------------
    # 23. Partial close
    # -----------------------------------------------------------------
    def test_23_partial_close(self) -> None:
        """Partial close must update remaining volume and record realized PnL."""
        req = OrderRequest(
            setup_id="SET-DJ-BUY-PARTIAL",
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.BUY,
            volume=1.0,
            entry_price=34002.0,
            stop_loss=33950.0,
            take_profits=(),
            final_target=34150.0,
            magic_number=1001,
            deviation=0,
        )
        res = self.broker.send_order(req)
        ticket = res["position_ticket"]

        # Move market up to 34050 bid
        self.broker.set_quote(
            Quote(
                symbol=CanonicalSymbol.DOW_JONES,
                timestamp=self.t1,
                bid=34050.0,
                ask=34052.0,
                spread=2.0,
            )
        )

        close_res = self.broker.close_position_partial(ticket, volume=0.4)
        self.assertTrue(close_res["success"])
        self.assertEqual(close_res["closed_volume"], 0.4)
        self.assertEqual(close_res["remaining_volume"], 0.6)
        self.assertEqual(close_res["close_price"], 34050.0)

        # Gross profit: (34050 - 34002) * 1.0 * 0.4 = 48 * 0.4 = 19.2 USD
        self.assertEqual(close_res["profit"], 19.2)

        pos = self.broker.get_position_by_ticket(ticket)
        self.assertEqual(pos["volume"], 0.6)
        self.assertEqual(pos["closed_volume"], 0.4)
        self.assertEqual(pos["status"], PositionStatus.PARTIALLY_CLOSED.value)

    # -----------------------------------------------------------------
    # 24. Full close
    # -----------------------------------------------------------------
    def test_24_full_close(self) -> None:
        """Full close sets position status to CLOSED and volume to 0."""
        req = OrderRequest(
            setup_id="SET-DJ-BUY-FULL",
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.BUY,
            volume=1.0,
            entry_price=34002.0,
            stop_loss=33950.0,
            take_profits=(),
            final_target=34150.0,
            magic_number=1001,
            deviation=0,
        )
        res = self.broker.send_order(req)
        ticket = res["position_ticket"]

        close_res = self.broker.close_position(ticket)
        self.assertTrue(close_res["success"])
        self.assertEqual(close_res["closed_volume"], 1.0)
        self.assertEqual(close_res["remaining_volume"], 0.0)

        pos = self.broker.get_position_by_ticket(ticket)
        self.assertEqual(pos["status"], PositionStatus.CLOSED.value)
        self.assertEqual(pos["volume"], 0.0)
        self.assertIsNotNone(pos["close_time"])

    # -----------------------------------------------------------------
    # 25. Closed position cannot be closed again
    # -----------------------------------------------------------------
    def test_25_cannot_close_already_closed_position(self) -> None:
        """Attempting to close an already closed position must return failure."""
        req = OrderRequest(
            setup_id="SET-DJ-BUY-1",
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.BUY,
            volume=1.0,
            entry_price=34002.0,
            stop_loss=33950.0,
            take_profits=(),
            final_target=34150.0,
            magic_number=1001,
            deviation=0,
        )
        res = self.broker.send_order(req)
        ticket = res["position_ticket"]
        self.broker.close_position(ticket)

        # Second close attempt
        close2 = self.broker.close_position(ticket)
        self.assertFalse(close2["success"])
        self.assertIn("already closed", close2["error_message"] or "")

    # -----------------------------------------------------------------
    # 26. Position modification
    # -----------------------------------------------------------------
    def test_26_modify_position(self) -> None:
        """Modify protective stop loss on an open position without conflating take_profit with final_target."""
        req = OrderRequest(
            setup_id="SET-DJ-BUY-1",
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.BUY,
            volume=1.0,
            entry_price=34002.0,
            stop_loss=33950.0,
            take_profits=(),
            final_target=34150.0,
            magic_number=1001,
            deviation=0,
        )
        res = self.broker.send_order(req)
        ticket = res["position_ticket"]

        ok = self.broker.modify_position(ticket, stop_loss=33980.0, take_profit=34200.0)
        self.assertTrue(ok)
        pos = self.broker.get_position_by_ticket(ticket)
        self.assertEqual(pos["stop_loss"], 33980.0)
        # final_target is strategy-level target and must not be mutated by broker take_profit
        self.assertEqual(pos["final_target"], 34150.0)

        # Reject booleans, non-finite, and negative prices
        self.assertFalse(self.broker.modify_position(ticket, stop_loss=True))  # type: ignore
        self.assertFalse(self.broker.modify_position(ticket, stop_loss=float("nan")))
        self.assertFalse(self.broker.modify_position(ticket, stop_loss=-100.0))

    # -----------------------------------------------------------------
    # 27. Commission accounting
    # -----------------------------------------------------------------
    def test_27_commission_accounting(self) -> None:
        """Opening and closing commissions are accurately deducted from balance."""
        broker_comm = SimulatedBroker(
            initial_balance=100000.0,
            commission_per_lot=5.0,  # 5 USD per lot
            symbol_specs={CanonicalSymbol.DOW_JONES: self.dj_spec},
        )
        broker_comm.set_quote(self.quote_dj)

        # Open 2.0 lots -> 2 * 5 = 10 USD opening commission
        req = OrderRequest(
            setup_id="SET-DJ-BUY-COMM",
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.BUY,
            volume=2.0,
            entry_price=34002.0,
            stop_loss=33950.0,
            take_profits=(),
            final_target=34150.0,
            magic_number=1001,
            deviation=0,
        )
        res = broker_comm.send_order(req)
        self.assertTrue(res["success"])

        # Balance immediately reflects opening commission: 100000 - 10 = 99990.0
        acct = broker_comm.get_account_info()
        self.assertEqual(acct["balance"], 99990.0)

        # Close 2.0 lots at same quote (bid 34000)
        # Gross pnl = (34000 - 34002) * 2 * 1.0 = -4.0 USD
        # Exit commission = 2 * 5 = 10.0 USD
        # Net pnl = -4.0 - 10.0 = -14.0 USD
        close_res = broker_comm.close_position(res["position_ticket"])
        self.assertTrue(close_res["success"])

        acct2 = broker_comm.get_account_info()
        self.assertEqual(acct2["balance"], 99976.0)  # 99990 - 14 = 99976.0

    # -----------------------------------------------------------------
    # 28. Slippage parameter & base quote price execution
    # -----------------------------------------------------------------
    def test_28_slippage_parameter_and_no_invented_formula(self) -> None:
        """
        Verify slippage_points property and validation, and verify execution strictly at Ask/Bid
        without invented directional execution adjustments.
        """
        broker_slip = SimulatedBroker(
            initial_balance=100000.0,
            slippage_points=2.0,
            symbol_specs={
                CanonicalSymbol.DOW_JONES: self.dj_spec,
                CanonicalSymbol.NASDAQ: self.nas_spec,
            },
        )
        self.assertEqual(broker_slip.slippage_points, 2.0)
        broker_slip.set_quote(self.quote_dj)
        broker_slip.set_quote(self.quote_nas)

        # BUY executes directly at live ask (34002.0)
        req_dj = OrderRequest(
            setup_id="SET-DJ-BUY-SLIP",
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.BUY,
            volume=1.0,
            entry_price=34002.0,
            stop_loss=33950.0,
            take_profits=(),
            final_target=34150.0,
            magic_number=1001,
            deviation=0,
        )
        res_dj = broker_slip.send_order(req_dj)
        self.assertEqual(res_dj["executed_price"], 34002.0)

        # SELL executes directly at live bid (15000.0)
        req_nas = OrderRequest(
            setup_id="SET-NAS-SELL-SLIP",
            symbol=CanonicalSymbol.NASDAQ,
            direction=TradeDirection.SELL,
            volume=1.0,
            entry_price=15000.0,
            stop_loss=15050.0,
            take_profits=(),
            final_target=14900.0,
            magic_number=1001,
            deviation=0,
        )
        res_nas = broker_slip.send_order(req_nas)
        self.assertEqual(res_nas["executed_price"], 15000.0)

    # -----------------------------------------------------------------
    # 29. Partial close volume step and minimum validation
    # -----------------------------------------------------------------
    def test_29_partial_close_volume_validation(self) -> None:
        """Partial close must validate volume against volume_min and volume_step."""
        req = OrderRequest(
            setup_id="SET-DJ-BUY-PCV",
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.BUY,
            volume=1.0,
            entry_price=34002.0,
            stop_loss=33950.0,
            take_profits=(),
            final_target=34150.0,
            magic_number=1001,
            deviation=0,
        )
        res = self.broker.send_order(req)
        ticket = res["position_ticket"]

        # Below volume_min (0.01)
        r_min = self.broker.close_position_partial(ticket, volume=0.005)
        self.assertFalse(r_min["success"])
        self.assertIn("below broker minimum", r_min["error_message"] or "")

        # Volume step mismatch (e.g. 0.015 with step 0.01)
        r_step = self.broker.close_position_partial(ticket, volume=0.015)
        self.assertFalse(r_step["success"])
        self.assertIn("multiple of volume_step", r_step["error_message"] or "")

        # Invalid volume types
        r_bool = self.broker.close_position_partial(ticket, volume=True)  # type: ignore
        self.assertFalse(r_bool["success"])
        r_nan = self.broker.close_position_partial(ticket, volume=float("nan"))
        self.assertFalse(r_nan["success"])

        # Exceeds volume
        r_exceed = self.broker.close_position_partial(ticket, volume=1.5)
        self.assertFalse(r_exceed["success"])

        # Position volume must remain intact after failed partial closes
        pos = self.broker.get_position_by_ticket(ticket)
        self.assertEqual(pos["volume"], 1.0)
        self.assertEqual(pos["closed_volume"], 0.0)

    # -----------------------------------------------------------------
    # 30. Zero commission default and unsimulated margin semantics
    # -----------------------------------------------------------------
    def test_30_zero_commission_default_and_unsimulated_margin(self) -> None:
        """Default commission is 0.0, and margin calculation is unsimulated (0.0)."""
        broker_default = SimulatedBroker(
            initial_balance=50000.0,
            symbol_specs={CanonicalSymbol.DOW_JONES: self.dj_spec},
        )
        self.assertEqual(broker_default.commission_per_lot, 0.0)
        broker_default.set_quote(self.quote_dj)

        req = OrderRequest(
            setup_id="SET-DJ-BUY-DEF",
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.BUY,
            volume=5.0,
            entry_price=34002.0,
            stop_loss=33950.0,
            take_profits=(),
            final_target=34150.0,
            magic_number=1001,
            deviation=0,
        )
        res = broker_default.send_order(req)
        self.assertTrue(res["success"])

        # Balance remains unchanged with zero commission
        acct = broker_default.get_account_info()
        self.assertEqual(acct["balance"], 50000.0)
        self.assertEqual(acct["margin"], 0.0)
        self.assertEqual(acct["free_margin"], acct["equity"])

        # Close position
        close_res = broker_default.close_position(res["position_ticket"])
        self.assertTrue(close_res["success"])
        self.assertEqual(close_res["closed_volume"], 5.0)

    # -----------------------------------------------------------------
    # 31. Multiple rejections followed by valid order produces identical tickets to clean run
    # -----------------------------------------------------------------
    def test_31_multiple_rejections_preserve_clean_ticket_sequence(self) -> None:
        """Submitting multiple invalid orders does not advance ticket sequences compared to a clean run."""
        specs = {
            CanonicalSymbol.DOW_JONES: self.dj_spec,
            CanonicalSymbol.NASDAQ: self.nas_spec,
        }
        b_clean = SimulatedBroker(initial_balance=100000.0, symbol_specs=specs)
        b_noisy = SimulatedBroker(initial_balance=100000.0, symbol_specs=specs)

        b_clean.set_quote(self.quote_dj)
        b_noisy.set_quote(self.quote_dj)

        # In noisy broker, spam 5 invalid orders
        bad_reqs = [
            {"setup_id": "BAD-1", "symbol": "INVALID", "direction": TradeDirection.BUY, "volume": 1.0, "entry_price": 34002.0, "stop_loss": 33950.0, "final_target": 34150.0, "magic_number": 1001},
            {"setup_id": "BAD-2", "symbol": CanonicalSymbol.DOW_JONES, "direction": TradeDirection.BUY, "volume": -1.0, "entry_price": 34002.0, "stop_loss": 33950.0, "final_target": 34150.0, "magic_number": 1001},
            {"setup_id": "BAD-3", "symbol": CanonicalSymbol.DOW_JONES, "direction": TradeDirection.BUY, "volume": float("nan"), "entry_price": 34002.0, "stop_loss": 33950.0, "final_target": 34150.0, "magic_number": 1001},
            {"setup_id": "BAD-4", "symbol": CanonicalSymbol.DOW_JONES, "direction": TradeDirection.BUY, "volume": 1.0, "entry_price": 34002.0, "stop_loss": float("inf"), "final_target": 34150.0, "magic_number": 1001},
            {"setup_id": "BAD-5", "symbol": CanonicalSymbol.DOW_JONES, "direction": TradeDirection.BUY, "volume": 0.005, "entry_price": 34002.0, "stop_loss": 33950.0, "final_target": 34150.0, "magic_number": 1001},
        ]
        for bad in bad_reqs:
            res_bad = b_noisy.send_order(bad)
            self.assertFalse(res_bad["success"])
            self.assertIsNone(res_bad["order_ticket"])

        # Now send valid order to both brokers
        valid_req = OrderRequest(
            setup_id="SET-DJ-BUY-1",
            symbol=CanonicalSymbol.DOW_JONES,
            direction=TradeDirection.BUY,
            volume=1.0,
            entry_price=34002.0,
            stop_loss=33950.0,
            take_profits=(),
            final_target=34150.0,
            magic_number=1001,
            deviation=0,
        )

        res_clean = b_clean.send_order(valid_req)
        res_noisy = b_noisy.send_order(valid_req)

        # Tickets must match exactly
        self.assertEqual(res_clean["order_ticket"], res_noisy["order_ticket"])
        self.assertEqual(res_clean["deal_ticket"], res_noisy["deal_ticket"])
        self.assertEqual(res_clean["position_ticket"], res_noisy["position_ticket"])
        self.assertEqual(res_clean["order_ticket"], 100001)
        self.assertEqual(res_clean["deal_ticket"], 200001)
        self.assertEqual(res_clean["position_ticket"], 300001)

    # -----------------------------------------------------------------
    # 32. Missing magic number rejected (no silent fallback)
    # -----------------------------------------------------------------
    def test_32_missing_magic_number_rejected(self) -> None:
        """Missing or None magic_number in order request must be strictly rejected."""
        req_missing_magic = {
            "setup_id": "SET-DJ-BUY-NOMAGIC",
            "symbol": CanonicalSymbol.DOW_JONES,
            "direction": TradeDirection.BUY,
            "volume": 1.0,
            "entry_price": 34002.0,
            "stop_loss": 33950.0,
            "final_target": 34150.0,
            # magic_number explicitly omitted
        }
        is_valid, err = self.broker.check_order(req_missing_magic)
        self.assertFalse(is_valid)
        self.assertIn("magic_number is required", err or "")

        res = self.broker.send_order(req_missing_magic)
        self.assertFalse(res["success"])
        self.assertIn("magic_number is required", res["error_message"] or "")

        # None magic_number
        req_none_magic = dict(req_missing_magic)
        req_none_magic["magic_number"] = None
        is_valid_none, err_none = self.broker.check_order(req_none_magic)
        self.assertFalse(is_valid_none)
        self.assertIn("magic_number is required", err_none or "")

    # -----------------------------------------------------------------
    # 33. Default slippage_points matches constant
    # -----------------------------------------------------------------
    def test_33_default_slippage_points_matches_constant(self) -> None:
        """SimulatedBroker default slippage_points must equal DEFAULT_BACKTEST_SLIPPAGE_POINTS (2.0)."""
        from src.core.constants import DEFAULT_BACKTEST_SLIPPAGE_POINTS
        default_broker = SimulatedBroker()
        self.assertEqual(default_broker.slippage_points, DEFAULT_BACKTEST_SLIPPAGE_POINTS)
        self.assertEqual(default_broker.slippage_points, 2.0)


if __name__ == "__main__":
    unittest.main()
