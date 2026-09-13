"""
Unit Tests for src/adapters/broker_protocol.py
Validates the abstract base protocol and confirms independence from MetaTrader5.
"""

from typing import Any, Dict, List, Optional, Tuple
import unittest

from src.adapters.broker_protocol import IBrokerAdapter


class DummyConformingBroker(IBrokerAdapter):
    """Minimal conforming implementation to verify abstract contract enforcement."""

    def initialize(self) -> bool:
        return True

    def shutdown(self) -> None:
        pass

    def is_connected(self) -> bool:
        return True

    def get_account_info(self) -> Dict[str, Any]:
        return {
            "balance": 100000.0,
            "equity": 100000.0,
            "margin": 0.0,
            "free_margin": 100000.0,
            "currency": "USD",
        }

    def get_symbol_specification(self, broker_symbol: str) -> Dict[str, Any]:
        return {
            "tick_size": 1.0,
            "tick_value": 1.0,
            "contract_size": 1.0,
            "volume_min": 0.01,
            "volume_max": 100.0,
            "volume_step": 0.01,
            "point": 1.0,
            "spread": 2.0,
        }

    def get_live_quote(self, broker_symbol: str) -> Dict[str, Any]:
        return {"bid": 34000.0, "ask": 34002.0, "spread": 2.0, "timestamp": 1600000000}

    def check_order(self, order_request: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        return True, None

    def send_order(self, order_request: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": True,
            "order_ticket": 101,
            "deal_ticket": 201,
            "position_ticket": 301,
            "executed_price": 34002.0,
            "executed_volume": 1.0,
            "error_message": None,
        }

    def modify_position(
        self, position_ticket: int, stop_loss: float, take_profit: Optional[float] = None
    ) -> bool:
        return True

    def close_position_partial(
        self, position_ticket: int, volume: float, deviation: int
    ) -> Dict[str, Any]:
        return {"success": True, "closed_volume": volume, "remaining_volume": 0.0}

    def get_active_positions(self, magic_number: int) -> List[Dict[str, Any]]:
        return []

    def get_position_by_ticket(self, position_ticket: int) -> Optional[Dict[str, Any]]:
        return None


class TestBrokerProtocol(unittest.TestCase):
    """Test suite verifying contract enforcement of IBrokerAdapter."""

    def test_cannot_instantiate_abstract_protocol(self) -> None:
        with self.assertRaises(TypeError):
            IBrokerAdapter()  # type: ignore[abstract]

    def test_conforming_broker_satisfies_all_abstract_methods(self) -> None:
        broker = DummyConformingBroker()
        self.assertTrue(broker.initialize())
        self.assertTrue(broker.is_connected())

        acct = broker.get_account_info()
        self.assertEqual(acct["balance"], 100000.0)
        self.assertEqual(acct["currency"], "USD")

        quote = broker.get_live_quote("DJI")
        self.assertEqual(quote["bid"], 34000.0)
        self.assertEqual(quote["ask"], 34002.0)

        chk_ok, chk_err = broker.check_order({})
        self.assertTrue(chk_ok)
        self.assertIsNone(chk_err)

        order_res = broker.send_order({})
        self.assertTrue(order_res["success"])
        self.assertEqual(order_res["order_ticket"], 101)
        self.assertEqual(order_res["deal_ticket"], 201)
        self.assertEqual(order_res["position_ticket"], 301)


if __name__ == "__main__":
    unittest.main()
