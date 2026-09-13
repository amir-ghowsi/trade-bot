"""
Abstract Broker Protocol Interface.
Defines the architectural contract for broker communication.
Completely decoupled from any specific broker SDK (Zero MetaTrader5 dependency in Core).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from src.core.constants import CanonicalSymbol


class IBrokerAdapter(ABC):
    """
    Abstract Interface Contract for Broker Adapters.
    Must be implemented by MT5Adapter (live/demo) or SimulatedBroker (backtest).
    Strategy-facing APIs operate strictly on CanonicalSymbol.
    """

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize connection to the broker terminal or simulator."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Close connection and release terminal resources cleanly."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if terminal connection is active and healthy."""
        pass

    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """
        Query account balance, equity, margin and free margin.
        Returns dictionary with keys: balance, equity, margin, free_margin, currency.
        """
        pass

    @abstractmethod
    def get_symbol_specification(self, symbol: CanonicalSymbol) -> Dict[str, Any]:
        """
        Query specifications for an instrument using its CanonicalSymbol.
        Returns dictionary with keys: tick_size, tick_value, contract_size,
        volume_min, volume_max, volume_step, point, spread.
        """
        pass

    @abstractmethod
    def get_live_quote(self, symbol: CanonicalSymbol) -> Dict[str, Any]:
        """
        Query current best bid and ask quotes for an instrument using its CanonicalSymbol.
        Returns dictionary with keys: bid, ask, spread, timestamp.
        """
        pass

    @abstractmethod
    def check_order(self, order_request: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Perform pre-flight order and parameter validation.
        Validates connection, symbol specification, direction, numeric safety, volume bounds,
        volume step alignment, quote availability, and position constraints.
        Returns (is_valid, error_message).
        """
        pass

    @abstractmethod
    def send_order(self, order_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit a market execution order.
        Returns dictionary with keys: success, order_ticket, deal_ticket, position_ticket,
        executed_price, executed_volume, error_message.
        """
        pass

    @abstractmethod
    def modify_position(
        self, position_ticket: int, stop_loss: float, take_profit: Optional[float] = None
    ) -> bool:
        """
        Modify protective stops on an existing open position ticket.
        """
        pass

    @abstractmethod
    def close_position_partial(
        self, position_ticket: int, volume: float, deviation: int
    ) -> Dict[str, Any]:
        """
        Execute partial volume close on an existing open position ticket.
        """
        pass

    @abstractmethod
    def get_active_positions(self, magic_number: int) -> List[Dict[str, Any]]:
        """
        Query all open positions filtered by magic number.
        """
        pass

    @abstractmethod
    def get_position_by_ticket(self, position_ticket: int) -> Optional[Dict[str, Any]]:
        """
        Query status and properties of a specific position ticket.
        """
        pass
