"""
Deterministic In-Memory Simulated Broker Implementation.
Conforms strictly to IBrokerAdapter protocol.
Zero external network or MetaTrader5 dependencies.
Maintains 100% deterministic order, deal, and position lifecycles.
"""

from decimal import Decimal
import math
from typing import Any, Dict, List, Optional, Tuple, Union

from src.adapters.broker_protocol import IBrokerAdapter
from src.adapters.types import BrokerDeal, BrokerOrder, BrokerPosition
from src.core.constants import (
    CanonicalSymbol,
    DEFAULT_BACKTEST_COMMISSION_PER_LOT,
    DEFAULT_BACKTEST_SLIPPAGE_POINTS,
    MAX_ACTIVE_POSITIONS_PER_SYMBOL,
    MAX_GLOBAL_ACTIVE_POSITIONS,
    OrderStatus,
    PositionStatus,
    TradeDirection,
)
from src.core.exceptions import (
    BrokerConnectionError,
    DataValidationError,
    OrderExecutionError,
    OrderValidationError,
    PositionNotFoundError,
)
from src.core.types import AccountInfo, OrderRequest, OrderResult, Quote, SymbolSpecification


class SimulatedBroker(IBrokerAdapter):
    """
    Deterministic In-Memory Simulated Broker.
    Provides execution simulation for backtesting and verification.
    """

    def __init__(
        self,
        initial_balance: float = 100000.0,
        currency: str = "USD",
        commission_per_lot: float = DEFAULT_BACKTEST_COMMISSION_PER_LOT,
        slippage_points: float = DEFAULT_BACKTEST_SLIPPAGE_POINTS,
        symbol_specs: Optional[Dict[CanonicalSymbol, SymbolSpecification]] = None,
    ) -> None:
        if isinstance(initial_balance, bool) or not isinstance(initial_balance, (int, float)) or initial_balance <= 0 or not math.isfinite(initial_balance):
            raise DataValidationError("initial_balance must be a strictly positive finite number")
        if not isinstance(currency, str) or not currency.strip():
            raise DataValidationError("currency must be a non-empty string")
        if isinstance(commission_per_lot, bool) or not isinstance(commission_per_lot, (int, float)) or commission_per_lot < 0 or not math.isfinite(commission_per_lot):
            raise DataValidationError("commission_per_lot must be a non-negative finite number")
        if isinstance(slippage_points, bool) or not isinstance(slippage_points, (int, float)) or slippage_points < 0 or not math.isfinite(slippage_points):
            raise DataValidationError("slippage_points must be a non-negative finite number")

        self._balance: float = float(initial_balance)
        self._initial_balance: float = float(initial_balance)
        self._currency: str = currency.strip()
        self._commission_per_lot: float = float(commission_per_lot)
        self._slippage_points: float = float(slippage_points)

        self._is_connected: bool = True

        # Deterministic sequence counters
        self._next_order_ticket: int = 100001
        self._next_deal_ticket: int = 200001
        self._next_position_ticket: int = 300001

        # In-memory storage
        self._quotes: Dict[CanonicalSymbol, Quote] = {}
        self._symbol_specs: Dict[CanonicalSymbol, SymbolSpecification] = {}
        self._positions: Dict[int, BrokerPosition] = {}
        self._deals: Dict[int, BrokerDeal] = {}
        self._orders: Dict[int, BrokerOrder] = {}

        # Initialize symbol specifications (injected from outside)
        if symbol_specs:
            for sym, spec in symbol_specs.items():
                self.register_symbol_specification(spec)

    @property
    def commission_per_lot(self) -> float:
        """Configured commission per lot (USD)."""
        return self._commission_per_lot

    @property
    def slippage_points(self) -> float:
        """Configured execution slippage in points."""
        return self._slippage_points

    def initialize(self) -> bool:
        """Initialize connection to simulated broker."""
        self._is_connected = True
        return True

    def shutdown(self) -> None:
        """Shutdown connection to simulated broker."""
        self._is_connected = False

    def is_connected(self) -> bool:
        """Check if simulated broker is connected."""
        return self._is_connected

    def register_symbol_specification(self, spec: SymbolSpecification) -> None:
        """Register instrument contract specification."""
        if not isinstance(spec, SymbolSpecification):
            raise DataValidationError(f"spec must be a SymbolSpecification instance, got {type(spec).__name__}")
        self._symbol_specs[spec.canonical_symbol] = spec

    def set_quote(self, quote: Quote) -> None:
        """Update the latest top-of-book market quote for an instrument."""
        if not isinstance(quote, Quote):
            raise DataValidationError(f"quote must be a Quote instance, got {type(quote).__name__}")
        self._quotes[quote.symbol] = quote

    def _resolve_canonical_symbol(self, symbol_repr: Union[CanonicalSymbol, str]) -> CanonicalSymbol:
        """
        Validate and resolve symbol to CanonicalSymbol.
        Rejects broker-specific symbols and non-canonical identifiers.
        """
        if isinstance(symbol_repr, CanonicalSymbol):
            return symbol_repr
        if isinstance(symbol_repr, str):
            clean = symbol_repr.strip().upper()
            if clean in (CanonicalSymbol.DOW_JONES.value, CanonicalSymbol.DOW_JONES.name):
                return CanonicalSymbol.DOW_JONES
            if clean in (CanonicalSymbol.NASDAQ.value, CanonicalSymbol.NASDAQ.name):
                return CanonicalSymbol.NASDAQ
        raise DataValidationError(f"Invalid canonical symbol: '{symbol_repr}'. Broker-specific symbols are not accepted.")

    def get_symbol_specification(self, symbol: Union[CanonicalSymbol, str]) -> Dict[str, Any]:
        """Query specifications for an instrument using its CanonicalSymbol."""
        canonical = self._resolve_canonical_symbol(symbol)
        if canonical not in self._symbol_specs:
            raise DataValidationError(f"No specification registered for canonical symbol: {canonical.value}")
        spec = self._symbol_specs[canonical]
        quote = self._quotes.get(canonical)
        spread = quote.spread if quote else 0.0
        return {
            "tick_size": spec.tick_size,
            "tick_value": spec.tick_value,
            "contract_size": spec.contract_size,
            "volume_min": spec.volume_min,
            "volume_max": spec.volume_max,
            "volume_step": spec.volume_step,
            "point": spec.point,
            "spread": spread,
        }

    def get_live_quote(self, symbol: Union[CanonicalSymbol, str]) -> Dict[str, Any]:
        """Query current best bid and ask quotes for an instrument using its CanonicalSymbol."""
        canonical = self._resolve_canonical_symbol(symbol)
        if canonical not in self._quotes:
            raise DataValidationError(f"No quote available for canonical symbol: {canonical.value}")
        quote = self._quotes[canonical]
        return {
            "bid": quote.bid,
            "ask": quote.ask,
            "spread": quote.spread,
            "timestamp": quote.timestamp.isoformat(),
        }

    def get_account_info(self) -> Dict[str, Any]:
        """
        Query account balance, equity, margin and free margin.
        Returns dictionary conforming to IBrokerAdapter contract.
        
        PHASE 7 SIMULATION SIMPLIFICATION:
        Margin model is not specified by the core project spec;
        margin is 0.0 and free_margin equals equity.
        """
        unrealized_pnl = 0.0
        for pos in self._positions.values():
            if pos.status in (PositionStatus.OPEN, PositionStatus.PARTIALLY_CLOSED):
                quote = self._quotes.get(pos.symbol)
                spec = self._symbol_specs.get(pos.symbol)
                if quote and spec:
                    if pos.direction == TradeDirection.BUY:
                        pnl = (quote.bid - pos.entry_price) / spec.tick_size * spec.tick_value * pos.volume
                    else:
                        pnl = (pos.entry_price - quote.ask) / spec.tick_size * spec.tick_value * pos.volume
                    unrealized_pnl += pnl

        equity = round(self._balance + unrealized_pnl, 2)
        balance = round(self._balance, 2)

        return {
            "balance": balance,
            "equity": equity,
            "margin": 0.0,
            "free_margin": equity,
            "currency": self._currency,
        }

    def get_account_info_typed(self) -> AccountInfo:
        """Query typed AccountInfo snapshot."""
        info = self.get_account_info()
        return AccountInfo(
            balance=info["balance"],
            equity=info["equity"],
            margin=info["margin"],
            free_margin=info["free_margin"],
            currency=info["currency"],
        )

    def _generate_order_ticket(self) -> int:
        """Deterministic monotonic order ticket generation."""
        ticket = self._next_order_ticket
        self._next_order_ticket += 1
        return ticket

    def _generate_deal_ticket(self) -> int:
        """Deterministic monotonic deal ticket generation."""
        ticket = self._next_deal_ticket
        self._next_deal_ticket += 1
        return ticket

    def _generate_position_ticket(self) -> int:
        """Deterministic monotonic position ticket generation."""
        ticket = self._next_position_ticket
        self._next_position_ticket += 1
        return ticket

    def _parse_order_request(
        self, order_request: Union[OrderRequest, Dict[str, Any]]
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Parse and perform low-level validation on incoming order request data.
        Returns (is_valid, error_message, parsed_dict).
        """
        if isinstance(order_request, OrderRequest):
            req_dict: Dict[str, Any] = {
                "setup_id": order_request.setup_id,
                "symbol": order_request.symbol,
                "direction": order_request.direction,
                "volume": order_request.volume,
                "entry_price": order_request.entry_price,
                "stop_loss": order_request.stop_loss,
                "take_profits": order_request.take_profits,
                "final_target": order_request.final_target,
                "magic_number": order_request.magic_number,
                "deviation": order_request.deviation,
            }
        elif isinstance(order_request, dict):
            req_dict = dict(order_request)
        else:
            return False, f"order_request must be an OrderRequest or dict, got {type(order_request).__name__}", None

        # 1. Setup ID
        setup_id = req_dict.get("setup_id", "")
        if not setup_id or not isinstance(setup_id, str):
            return False, "setup_id must be a non-empty string", None

        # 2. Symbol resolution
        raw_symbol = req_dict.get("symbol")
        try:
            canonical_symbol = self._resolve_canonical_symbol(raw_symbol)
        except DataValidationError as err:
            return False, f"Invalid symbol: {err}", None

        if canonical_symbol not in self._symbol_specs:
            return False, f"No symbol specification registered for {canonical_symbol.value}", None

        # 3. Direction
        raw_dir = req_dict.get("direction")
        if isinstance(raw_dir, TradeDirection):
            direction = raw_dir
        elif isinstance(raw_dir, str):
            dir_str = raw_dir.strip().upper()
            if dir_str == TradeDirection.BUY.value:
                direction = TradeDirection.BUY
            elif dir_str == TradeDirection.SELL.value:
                direction = TradeDirection.SELL
            else:
                return False, f"direction must be BUY or SELL, got {raw_dir}", None
        else:
            return False, f"direction must be TradeDirection or str ('BUY'/'SELL'), got {type(raw_dir).__name__}", None

        # 4. Numeric fields with strict boolean rejection
        numeric_fields = {
            "volume": req_dict.get("volume"),
            "entry_price": req_dict.get("entry_price"),
            "stop_loss": req_dict.get("stop_loss"),
            "final_target": req_dict.get("final_target"),
        }

        for f_name, f_val in numeric_fields.items():
            if isinstance(f_val, bool) or not isinstance(f_val, (int, float)):
                return False, f"{f_name} must be a numeric value, got {type(f_val).__name__}", None
            if not math.isfinite(f_val):
                return False, f"{f_name} must be finite (no NaN or Inf), got {f_val}", None
            if f_val <= 0:
                return False, f"{f_name} must be strictly positive (> 0), got {f_val}", None

        volume = float(numeric_fields["volume"])
        entry_price = float(numeric_fields["entry_price"])
        stop_loss = float(numeric_fields["stop_loss"])
        final_target = float(numeric_fields["final_target"])

        # 5. Magic number & deviation (magic_number must be explicitly provided by the caller/configuration)
        if "magic_number" not in req_dict or req_dict["magic_number"] is None:
            return False, "magic_number is required and cannot be omitted", None
        magic_number = req_dict["magic_number"]
        if isinstance(magic_number, bool) or not isinstance(magic_number, int) or magic_number <= 0:
            return False, "magic_number must be a positive integer", None

        # Deviation: optional from caller, kept explicitly inert without execution price adjustment
        raw_deviation = req_dict.get("deviation", 0)
        if raw_deviation is None:
            deviation = 0
        elif isinstance(raw_deviation, bool) or not isinstance(raw_deviation, int) or raw_deviation < 0:
            return False, "deviation must be a non-negative integer", None
        else:
            deviation = raw_deviation

        # 6. Take profits tuple
        raw_tps = req_dict.get("take_profits", ())
        if isinstance(raw_tps, (list, tuple)):
            tp_list = []
            for tp in raw_tps:
                if isinstance(tp, bool) or not isinstance(tp, (int, float)) or not math.isfinite(tp) or tp <= 0:
                    return False, f"take_profit element must be a positive finite number, got {tp}", None
                tp_list.append(float(tp))
            take_profits = tuple(tp_list)
        else:
            return False, "take_profits must be a list or tuple of numbers", None

        parsed = {
            "setup_id": setup_id,
            "symbol": canonical_symbol,
            "direction": direction,
            "volume": volume,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profits": take_profits,
            "final_target": final_target,
            "magic_number": magic_number,
            "deviation": deviation,
        }
        return True, None, parsed

    def check_order(self, order_request: Union[OrderRequest, Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
        """
        Perform pre-flight validation on order parameters, quote availability, and position limits.
        Returns (is_valid, error_message).
        """
        if not self._is_connected:
            return False, "Broker is not connected"

        is_valid, err_msg, parsed = self._parse_order_request(order_request)
        if not is_valid or parsed is None:
            return False, err_msg

        canonical_symbol = parsed["symbol"]
        volume = parsed["volume"]
        spec = self._symbol_specs[canonical_symbol]

        # 1. Volume constraints
        if volume < spec.volume_min:
            return False, f"Volume {volume} is below broker minimum {spec.volume_min}"
        if volume > spec.volume_max:
            return False, f"Volume {volume} exceeds broker maximum {spec.volume_max}"

        # Volume step alignment check using Decimal to prevent float inaccuracies
        v_dec = Decimal(str(volume))
        step_dec = Decimal(str(spec.volume_step))
        if (v_dec % step_dec) != Decimal("0"):
            return False, f"Volume {volume} is not an exact multiple of volume_step {spec.volume_step}"

        # 2. Quote availability check
        quote = self._quotes.get(canonical_symbol)
        if quote is None:
            return False, f"No live market quote available for symbol {canonical_symbol.value}"

        # 3. Position & Concurrency Constraints (Locked Specification)
        # Max 1 active position per symbol
        active_sym_positions = [
            p for p in self._positions.values()
            if p.symbol == canonical_symbol and p.status in (PositionStatus.OPEN, PositionStatus.PARTIALLY_CLOSED)
        ]
        if len(active_sym_positions) >= MAX_ACTIVE_POSITIONS_PER_SYMBOL:
            return False, f"Maximum active positions reached for symbol {canonical_symbol.value} (max {MAX_ACTIVE_POSITIONS_PER_SYMBOL})"

        # Max 2 global active positions total
        total_active_positions = [
            p for p in self._positions.values()
            if p.status in (PositionStatus.OPEN, PositionStatus.PARTIALLY_CLOSED)
        ]
        if len(total_active_positions) >= MAX_GLOBAL_ACTIVE_POSITIONS:
            return False, f"Maximum global active positions reached (max {MAX_GLOBAL_ACTIVE_POSITIONS})"

        return True, None

    def send_order(self, order_request: Union[OrderRequest, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Submit a market execution order.
        
        Zero state mutation on rejection:
        Order ticket, deal ticket, and position ticket counters are ONLY incremented
        after check_order passes successfully. Rejection consumes NO tickets.
        """
        is_valid, err_msg = self.check_order(order_request)
        if not is_valid:
            return {
                "success": False,
                "order_ticket": None,
                "deal_ticket": None,
                "position_ticket": None,
                "executed_price": 0.0,
                "executed_volume": 0.0,
                "error_message": err_msg,
            }

        # Valid order: parse and generate tickets deterministically
        _, _, parsed = self._parse_order_request(order_request)
        if parsed is None:
            return {
                "success": False,
                "order_ticket": None,
                "deal_ticket": None,
                "position_ticket": None,
                "executed_price": 0.0,
                "executed_volume": 0.0,
                "error_message": "Failed to parse order request",
            }

        canonical_symbol = parsed["symbol"]
        direction = parsed["direction"]
        volume = parsed["volume"]
        spec = self._symbol_specs[canonical_symbol]
        quote = self._quotes[canonical_symbol]

        # Execution Price according to locked quote semantics:
        # BUY executes at live Ask
        # SELL executes at live Bid
        # (slippage_points is not applied as directional slippage is unspecified)
        if direction == TradeDirection.BUY:
            executed_price = quote.ask
        else:
            executed_price = quote.bid

        commission = round(volume * self._commission_per_lot, 2)

        order_ticket = self._generate_order_ticket()
        deal_ticket = self._generate_deal_ticket()
        position_ticket = self._generate_position_ticket()

        # Record Order
        self._orders[order_ticket] = BrokerOrder(
            order_ticket=order_ticket,
            setup_id=parsed["setup_id"],
            symbol=canonical_symbol,
            broker_symbol=spec.broker_symbol,
            direction=direction,
            volume=volume,
            requested_price=parsed["entry_price"],
            stop_loss=parsed["stop_loss"],
            take_profits=parsed["take_profits"],
            final_target=parsed["final_target"],
            magic_number=parsed["magic_number"],
            status=OrderStatus.FILLED,
            timestamp=quote.timestamp,
            executed_price=executed_price,
            executed_volume=volume,
            deal_ticket=deal_ticket,
            position_ticket=position_ticket,
        )

        # Record Deal (Entry deal "IN")
        self._deals[deal_ticket] = BrokerDeal(
            deal_ticket=deal_ticket,
            order_ticket=order_ticket,
            position_ticket=position_ticket,
            symbol=canonical_symbol,
            broker_symbol=spec.broker_symbol,
            direction=direction,
            volume=volume,
            price=executed_price,
            commission=commission,
            timestamp=quote.timestamp,
            entry_or_exit="IN",
            profit=0.0,
        )

        # Record Position
        self._positions[position_ticket] = BrokerPosition(
            position_ticket=position_ticket,
            order_ticket=order_ticket,
            deal_ticket=deal_ticket,
            setup_id=parsed["setup_id"],
            symbol=canonical_symbol,
            broker_symbol=spec.broker_symbol,
            direction=direction,
            volume=volume,
            entry_price=executed_price,
            stop_loss=parsed["stop_loss"],
            take_profits=parsed["take_profits"],
            final_target=parsed["final_target"],
            magic_number=parsed["magic_number"],
            open_time=quote.timestamp,
            status=PositionStatus.OPEN,
            commission=commission,
        )

        # Deduct entry commission from account balance (if configured)
        self._balance = round(self._balance - commission, 2)

        return {
            "success": True,
            "order_ticket": order_ticket,
            "deal_ticket": deal_ticket,
            "position_ticket": position_ticket,
            "executed_price": executed_price,
            "executed_volume": volume,
            "error_message": None,
        }

    def modify_position(
        self, position_ticket: int, stop_loss: float, take_profit: Optional[float] = None
    ) -> bool:
        """
        Modify protective stop loss on an existing open position ticket.
        
        NOTE ON TAKE_PROFIT:
        In the core strategy, multi-targets (TP1, TP2, TP3, Structural Target) are managed
        in software and distinct from a single broker take_profit.
        take_profit parameter is NOT mapped to final_target to prevent semantic conflation.
        """
        if not self._is_connected:
            return False

        if isinstance(position_ticket, bool) or not isinstance(position_ticket, int):
            return False

        pos = self._positions.get(position_ticket)
        if pos is None or pos.status == PositionStatus.CLOSED:
            return False

        if isinstance(stop_loss, bool) or not isinstance(stop_loss, (int, float)) or stop_loss <= 0 or not math.isfinite(stop_loss):
            return False

        if take_profit is not None:
            if isinstance(take_profit, bool) or not isinstance(take_profit, (int, float)) or take_profit <= 0 or not math.isfinite(take_profit):
                return False

        pos.stop_loss = float(stop_loss)
        # Structural final_target is intentionally not mutated by broker take_profit

        return True

    def close_position_partial(
        self, position_ticket: int, volume: float, deviation: int = 0
    ) -> Dict[str, Any]:
        """
        Execute partial volume close on an existing open position ticket.
        If volume matches the position volume, executes a full close.
        Validates volume bounds and volume_step alignment against registered SymbolSpecification.
        """
        if not self._is_connected:
            return {
                "success": False,
                "deal_ticket": None,
                "closed_volume": 0.0,
                "remaining_volume": 0.0,
                "close_price": 0.0,
                "error_message": "Broker is not connected",
            }

        if isinstance(position_ticket, bool) or not isinstance(position_ticket, int):
            return {
                "success": False,
                "deal_ticket": None,
                "closed_volume": 0.0,
                "remaining_volume": 0.0,
                "close_price": 0.0,
                "error_message": f"position_ticket must be an integer, got {type(position_ticket).__name__}",
            }

        pos = self._positions.get(position_ticket)
        if pos is None or pos.status == PositionStatus.CLOSED:
            return {
                "success": False,
                "deal_ticket": None,
                "closed_volume": 0.0,
                "remaining_volume": 0.0,
                "close_price": 0.0,
                "error_message": f"Position ticket {position_ticket} not found or already closed",
            }

        if isinstance(volume, bool) or not isinstance(volume, (int, float)) or volume <= 0 or not math.isfinite(volume):
            return {
                "success": False,
                "deal_ticket": None,
                "closed_volume": 0.0,
                "remaining_volume": pos.volume,
                "close_price": 0.0,
                "error_message": f"volume must be a strictly positive finite number, got {volume}",
            }

        spec = self._symbol_specs.get(pos.symbol)
        if spec is None:
            return {
                "success": False,
                "deal_ticket": None,
                "closed_volume": 0.0,
                "remaining_volume": pos.volume,
                "close_price": 0.0,
                "error_message": f"No symbol specification registered for {pos.symbol.value}",
            }

        # Validate volume bounds & volume_step for partial close
        # (Full close of remaining volume is always allowed even if remaining is equal to pos.volume)
        is_full_close = abs(volume - pos.volume) < 1e-9
        if not is_full_close:
            if volume < spec.volume_min:
                return {
                    "success": False,
                    "deal_ticket": None,
                    "closed_volume": 0.0,
                    "remaining_volume": pos.volume,
                    "close_price": 0.0,
                    "error_message": f"Close volume {volume} is below broker minimum {spec.volume_min}",
                }
            v_dec = Decimal(str(volume))
            step_dec = Decimal(str(spec.volume_step))
            if (v_dec % step_dec) != Decimal("0"):
                return {
                    "success": False,
                    "deal_ticket": None,
                    "closed_volume": 0.0,
                    "remaining_volume": pos.volume,
                    "close_price": 0.0,
                    "error_message": f"Close volume {volume} is not an exact multiple of volume_step {spec.volume_step}",
                }

        # Normalize close volume against remaining volume
        if volume > pos.volume + 1e-9:
            return {
                "success": False,
                "deal_ticket": None,
                "closed_volume": 0.0,
                "remaining_volume": pos.volume,
                "close_price": 0.0,
                "error_message": f"Close volume ({volume}) exceeds active position volume ({pos.volume})",
            }

        if is_full_close:
            close_vol = pos.volume
        else:
            close_vol = round(float(volume), 8)

        quote = self._quotes.get(pos.symbol)
        if quote is None:
            return {
                "success": False,
                "deal_ticket": None,
                "closed_volume": 0.0,
                "remaining_volume": pos.volume,
                "close_price": 0.0,
                "error_message": f"No live quote available for symbol {pos.symbol.value}",
            }

        # Closing price semantics:
        # BUY position closes at live Bid
        # SELL position closes at live Ask
        if pos.direction == TradeDirection.BUY:
            close_price = quote.bid
            gross_pnl = (close_price - pos.entry_price) / spec.tick_size * spec.tick_value * close_vol
        else:
            close_price = quote.ask
            gross_pnl = (pos.entry_price - close_price) / spec.tick_size * spec.tick_value * close_vol

        commission = round(close_vol * self._commission_per_lot, 2)
        net_pnl = round(gross_pnl - commission, 2)

        deal_ticket = self._generate_deal_ticket()

        # Record Deal (Exit deal "OUT")
        self._deals[deal_ticket] = BrokerDeal(
            deal_ticket=deal_ticket,
            order_ticket=pos.order_ticket,
            position_ticket=pos.position_ticket,
            symbol=pos.symbol,
            broker_symbol=pos.broker_symbol,
            direction=pos.direction,
            volume=close_vol,
            price=close_price,
            commission=commission,
            timestamp=quote.timestamp,
            entry_or_exit="OUT",
            profit=round(gross_pnl, 2),
        )

        # Update position
        pos.closed_volume = round(pos.closed_volume + close_vol, 8)
        pos.volume = round(pos.volume - close_vol, 8)
        pos.realized_profit = round(pos.realized_profit + gross_pnl, 2)
        pos.commission = round(pos.commission + commission, 2)

        if pos.volume <= 1e-9:
            pos.volume = 0.0
            pos.status = PositionStatus.CLOSED
            pos.close_time = quote.timestamp
            pos.close_price = close_price
        else:
            pos.status = PositionStatus.PARTIALLY_CLOSED

        # Realize PnL into account balance
        self._balance = round(self._balance + net_pnl, 2)

        return {
            "success": True,
            "deal_ticket": deal_ticket,
            "closed_volume": close_vol,
            "remaining_volume": pos.volume,
            "close_price": close_price,
            "profit": round(gross_pnl, 2),
            "error_message": None,
        }

    def close_position(self, position_ticket: int, deviation: int = 0) -> Dict[str, Any]:
        """
        Execute full close on an existing open position ticket.
        Convenience primitive delegating to close_position_partial with full volume.
        """
        pos = self._positions.get(position_ticket)
        if pos is None:
            return {
                "success": False,
                "deal_ticket": None,
                "closed_volume": 0.0,
                "remaining_volume": 0.0,
                "close_price": 0.0,
                "error_message": f"Position ticket {position_ticket} not found",
            }
        return self.close_position_partial(position_ticket, pos.volume, deviation)

    def get_active_positions(self, magic_number: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Query all open/partially-closed positions optionally filtered by magic number.
        """
        return [
            pos.to_dict()
            for pos in self._positions.values()
            if pos.status in (PositionStatus.OPEN, PositionStatus.PARTIALLY_CLOSED)
            and (magic_number is None or pos.magic_number == magic_number)
        ]

    def get_position_by_ticket(self, position_ticket: int) -> Optional[Dict[str, Any]]:
        """
        Query status and properties of a specific position ticket.
        """
        pos = self._positions.get(position_ticket)
        return pos.to_dict() if pos else None

    def get_all_positions(self) -> List[BrokerPosition]:
        """Internal helper returning all position domain objects."""
        return list(self._positions.values())

    def get_all_deals(self) -> List[BrokerDeal]:
        """Internal helper returning all deal domain objects."""
        return list(self._deals.values())

    def get_all_orders(self) -> List[BrokerOrder]:
        """Internal helper returning all order domain objects."""
        return list(self._orders.values())
