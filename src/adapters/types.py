"""
Broker Adapter Domain Types and Data Contracts.
Defines in-memory models for simulated broker orders, deals, and positions.
Decoupled strictly from MetaTrader5.
"""

from dataclasses import dataclass, field
from datetime import datetime
import math
from typing import Any, Dict, Optional, Tuple, Union

from src.core.constants import CanonicalSymbol, OrderStatus, PositionStatus, TradeDirection
from src.core.exceptions import DataValidationError


def _validate_utc(dt: datetime, field_name: str) -> None:
    """Validate that a datetime is timezone-aware UTC."""
    if not isinstance(dt, datetime):
        raise DataValidationError(f"{field_name} must be a datetime instance, got {type(dt).__name__}")
    if dt.tzinfo is None or dt.utcoffset() is None or dt.utcoffset().total_seconds() != 0:
        raise DataValidationError(f"{field_name} must be timezone-aware UTC, got {dt}")


@dataclass
class BrokerPosition:
    """
    Mutable in-memory broker position representation for SimulatedBroker.
    Tracks active lifecycle, protective stops, remaining volume, and realized PnL.
    """
    position_ticket: int
    order_ticket: int
    deal_ticket: int
    setup_id: str
    symbol: CanonicalSymbol
    broker_symbol: str
    direction: TradeDirection
    volume: float
    entry_price: float
    stop_loss: float
    take_profits: Tuple[float, ...]
    final_target: float
    magic_number: int
    open_time: datetime
    status: PositionStatus = PositionStatus.OPEN
    close_time: Optional[datetime] = None
    close_price: Optional[float] = None
    closed_volume: float = 0.0
    realized_profit: float = 0.0
    commission: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.position_ticket, bool) or not isinstance(self.position_ticket, int) or self.position_ticket <= 0:
            raise DataValidationError("position_ticket must be a positive integer")
        if isinstance(self.order_ticket, bool) or not isinstance(self.order_ticket, int) or self.order_ticket <= 0:
            raise DataValidationError("order_ticket must be a positive integer")
        if isinstance(self.deal_ticket, bool) or not isinstance(self.deal_ticket, int) or self.deal_ticket <= 0:
            raise DataValidationError("deal_ticket must be a positive integer")
        if not self.setup_id or not isinstance(self.setup_id, str):
            raise DataValidationError("setup_id must be a non-empty string")
        if not isinstance(self.symbol, CanonicalSymbol):
            raise DataValidationError(f"symbol must be a CanonicalSymbol, got {self.symbol}")
        if not isinstance(self.direction, TradeDirection):
            raise DataValidationError(f"direction must be a TradeDirection, got {self.direction}")
        if isinstance(self.volume, bool) or not isinstance(self.volume, (int, float)) or self.volume < 0 or not math.isfinite(self.volume):
            raise DataValidationError("volume must be a non-negative finite number")
        if isinstance(self.entry_price, bool) or not isinstance(self.entry_price, (int, float)) or self.entry_price <= 0 or not math.isfinite(self.entry_price):
            raise DataValidationError("entry_price must be a strictly positive finite number")
        if isinstance(self.stop_loss, bool) or not isinstance(self.stop_loss, (int, float)) or self.stop_loss <= 0 or not math.isfinite(self.stop_loss):
            raise DataValidationError("stop_loss must be a strictly positive finite number")
        if isinstance(self.final_target, bool) or not isinstance(self.final_target, (int, float)) or self.final_target <= 0 or not math.isfinite(self.final_target):
            raise DataValidationError("final_target must be a strictly positive finite number")
        if isinstance(self.magic_number, bool) or not isinstance(self.magic_number, int) or self.magic_number <= 0:
            raise DataValidationError("magic_number must be a positive integer")
        if isinstance(self.closed_volume, bool) or not isinstance(self.closed_volume, (int, float)) or self.closed_volume < 0 or not math.isfinite(self.closed_volume):
            raise DataValidationError("closed_volume must be a non-negative finite number")
        if isinstance(self.realized_profit, bool) or not isinstance(self.realized_profit, (int, float)) or not math.isfinite(self.realized_profit):
            raise DataValidationError("realized_profit must be a finite number")
        if isinstance(self.commission, bool) or not isinstance(self.commission, (int, float)) or self.commission < 0 or not math.isfinite(self.commission):
            raise DataValidationError("commission must be a non-negative finite number")
        if self.close_price is not None:
            if isinstance(self.close_price, bool) or not isinstance(self.close_price, (int, float)) or self.close_price <= 0 or not math.isfinite(self.close_price):
                raise DataValidationError("close_price must be a strictly positive finite number")
        _validate_utc(self.open_time, "open_time")
        if self.close_time is not None:
            _validate_utc(self.close_time, "close_time")

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to standard dictionary contract."""
        return {
            "position_ticket": self.position_ticket,
            "order_ticket": self.order_ticket,
            "deal_ticket": self.deal_ticket,
            "setup_id": self.setup_id,
            "symbol": self.symbol.value,
            "broker_symbol": self.broker_symbol,
            "direction": self.direction.value,
            "volume": self.volume,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profits": list(self.take_profits),
            "final_target": self.final_target,
            "magic_number": self.magic_number,
            "open_time": self.open_time.isoformat(),
            "status": self.status.value,
            "close_time": self.close_time.isoformat() if self.close_time else None,
            "close_price": self.close_price,
            "closed_volume": self.closed_volume,
            "realized_profit": round(self.realized_profit, 2),
            "commission": round(self.commission, 2),
        }


@dataclass(frozen=True)
class BrokerDeal:
    """
    Immutable deal record generated upon order execution or position close.
    """
    deal_ticket: int
    order_ticket: int
    position_ticket: int
    symbol: CanonicalSymbol
    broker_symbol: str
    direction: TradeDirection
    volume: float
    price: float
    commission: float
    timestamp: datetime
    entry_or_exit: str  # "IN" or "OUT"
    profit: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.deal_ticket, bool) or not isinstance(self.deal_ticket, int) or self.deal_ticket <= 0:
            raise DataValidationError("deal_ticket must be a positive integer")
        if isinstance(self.order_ticket, bool) or not isinstance(self.order_ticket, int) or self.order_ticket <= 0:
            raise DataValidationError("order_ticket must be a positive integer")
        if isinstance(self.position_ticket, bool) or not isinstance(self.position_ticket, int) or self.position_ticket <= 0:
            raise DataValidationError("position_ticket must be a positive integer")
        if not isinstance(self.symbol, CanonicalSymbol):
            raise DataValidationError(f"symbol must be a CanonicalSymbol, got {self.symbol}")
        if not isinstance(self.direction, TradeDirection):
            raise DataValidationError(f"direction must be a TradeDirection, got {self.direction}")
        if isinstance(self.volume, bool) or not isinstance(self.volume, (int, float)) or self.volume <= 0 or not math.isfinite(self.volume):
            raise DataValidationError("volume must be a strictly positive finite number")
        if isinstance(self.price, bool) or not isinstance(self.price, (int, float)) or self.price <= 0 or not math.isfinite(self.price):
            raise DataValidationError("price must be a strictly positive finite number")
        if isinstance(self.commission, bool) or not isinstance(self.commission, (int, float)) or self.commission < 0 or not math.isfinite(self.commission):
            raise DataValidationError("commission must be a non-negative finite number")
        if isinstance(self.profit, bool) or not isinstance(self.profit, (int, float)) or not math.isfinite(self.profit):
            raise DataValidationError("profit must be a finite number")
        _validate_utc(self.timestamp, "timestamp")
        if self.entry_or_exit not in ("IN", "OUT", "INOUT"):
            raise DataValidationError(f"entry_or_exit must be 'IN', 'OUT', or 'INOUT', got {self.entry_or_exit}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert deal to standard dictionary contract."""
        return {
            "deal_ticket": self.deal_ticket,
            "order_ticket": self.order_ticket,
            "position_ticket": self.position_ticket,
            "symbol": self.symbol.value,
            "broker_symbol": self.broker_symbol,
            "direction": self.direction.value,
            "volume": self.volume,
            "price": self.price,
            "commission": round(self.commission, 2),
            "timestamp": self.timestamp.isoformat(),
            "entry_or_exit": self.entry_or_exit,
            "profit": round(self.profit, 2),
        }


@dataclass(frozen=True)
class BrokerOrder:
    """
    Immutable historical audit record of an order submitted to the broker.
    """
    order_ticket: int
    setup_id: str
    symbol: CanonicalSymbol
    broker_symbol: str
    direction: TradeDirection
    volume: float
    requested_price: float
    stop_loss: float
    take_profits: Tuple[float, ...]
    final_target: float
    magic_number: int
    status: OrderStatus
    timestamp: datetime
    executed_price: Optional[float] = None
    executed_volume: Optional[float] = None
    deal_ticket: Optional[int] = None
    position_ticket: Optional[int] = None
    rejection_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.order_ticket, bool) or not isinstance(self.order_ticket, int) or self.order_ticket <= 0:
            raise DataValidationError("order_ticket must be a positive integer")
        if not self.setup_id or not isinstance(self.setup_id, str):
            raise DataValidationError("setup_id must be a non-empty string")
        if not isinstance(self.symbol, CanonicalSymbol):
            raise DataValidationError(f"symbol must be a CanonicalSymbol, got {self.symbol}")
        if not isinstance(self.direction, TradeDirection):
            raise DataValidationError(f"direction must be a TradeDirection, got {self.direction}")
        if isinstance(self.volume, bool) or not isinstance(self.volume, (int, float)) or self.volume <= 0 or not math.isfinite(self.volume):
            raise DataValidationError("volume must be a strictly positive finite number")
        if isinstance(self.requested_price, bool) or not isinstance(self.requested_price, (int, float)) or self.requested_price <= 0 or not math.isfinite(self.requested_price):
            raise DataValidationError("requested_price must be a strictly positive finite number")
        if isinstance(self.stop_loss, bool) or not isinstance(self.stop_loss, (int, float)) or self.stop_loss <= 0 or not math.isfinite(self.stop_loss):
            raise DataValidationError("stop_loss must be a strictly positive finite number")
        if isinstance(self.final_target, bool) or not isinstance(self.final_target, (int, float)) or self.final_target <= 0 or not math.isfinite(self.final_target):
            raise DataValidationError("final_target must be a strictly positive finite number")
        if isinstance(self.magic_number, bool) or not isinstance(self.magic_number, int) or self.magic_number <= 0:
            raise DataValidationError("magic_number must be a positive integer")
        if not isinstance(self.status, OrderStatus):
            raise DataValidationError(f"status must be an OrderStatus, got {self.status}")
        if self.executed_price is not None:
            if isinstance(self.executed_price, bool) or not isinstance(self.executed_price, (int, float)) or self.executed_price <= 0 or not math.isfinite(self.executed_price):
                raise DataValidationError("executed_price must be a strictly positive finite number")
        if self.executed_volume is not None:
            if isinstance(self.executed_volume, bool) or not isinstance(self.executed_volume, (int, float)) or self.executed_volume < 0 or not math.isfinite(self.executed_volume):
                raise DataValidationError("executed_volume must be a non-negative finite number")
        if self.deal_ticket is not None:
            if isinstance(self.deal_ticket, bool) or not isinstance(self.deal_ticket, int) or self.deal_ticket <= 0:
                raise DataValidationError("deal_ticket must be a positive integer")
        if self.position_ticket is not None:
            if isinstance(self.position_ticket, bool) or not isinstance(self.position_ticket, int) or self.position_ticket <= 0:
                raise DataValidationError("position_ticket must be a positive integer")
        _validate_utc(self.timestamp, "timestamp")

    def to_dict(self) -> Dict[str, Any]:
        """Convert order record to standard dictionary contract."""
        return {
            "order_ticket": self.order_ticket,
            "setup_id": self.setup_id,
            "symbol": self.symbol.value,
            "broker_symbol": self.broker_symbol,
            "direction": self.direction.value,
            "volume": self.volume,
            "requested_price": self.requested_price,
            "stop_loss": self.stop_loss,
            "take_profits": list(self.take_profits),
            "final_target": self.final_target,
            "magic_number": self.magic_number,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "executed_price": self.executed_price,
            "executed_volume": self.executed_volume,
            "deal_ticket": self.deal_ticket,
            "position_ticket": self.position_ticket,
            "rejection_reason": self.rejection_reason,
        }
