"""
Domain Types and Data Contracts for the Trading System.
Defines immutable (frozen) dataclasses, validation invariants, and deterministic ID generators.
Decoupled strictly from any broker SDK (zero MT5 imports).
"""

from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple, Union

from src.core.clock import ensure_utc
from src.core.constants import (
    CanonicalSymbol,
    Direction,
    MIN_GAP_COUNT,
    MIN_SPIKE_CANDLES,
    PINBAR_MAX_BODY_RATIO,
    PINBAR_MIDPOINT_RATIO,
    PositionStatus,
    SPECIFICATION_VERSION,
    Timeframe,
    TRADE_RECORD_EXACT_FIELD_COUNT,
    TradeDirection,
)
from src.core.exceptions import DataValidationError, LevelImmutabilityViolationError


def _validate_utc_datetime(dt: datetime, field_name: str) -> None:
    """Validate that a datetime is timezone-aware and set to UTC."""
    if not isinstance(dt, datetime):
        raise DataValidationError(f"{field_name} must be a datetime instance, got {type(dt).__name__}")
    if dt.tzinfo is None or dt.tzinfo != timezone.utc:
        raise DataValidationError(
            f"{field_name} must be timezone-aware UTC, got tzinfo={dt.tzinfo}"
        )


def _normalize_price(price: float, tick_size: float) -> float:
    """
    Locked Structural Level Price Normalization Formula:
    Normalized_Level_Price = floor(fixed_level_price / symbol_tick_size + 0.5) * symbol_tick_size
    """
    if tick_size <= 0:
        raise DataValidationError("tick_size must be strictly positive.")
    return round(math.floor(price / tick_size + 0.5) * tick_size, 8)


def generate_setup_id(
    symbol: Union[CanonicalSymbol, str],
    direction: Union[TradeDirection, str],
    spike_start_time: datetime,
    spike_end_time: datetime,
    level_formed_time: datetime,
    structural_price: float,
    tick_size: float,
    version: str = SPECIFICATION_VERSION,
) -> str:
    """
    Generate the canonical deterministic Setup ID.
    Must strictly use CanonicalSymbol (never broker symbol), TradeDirection, and UTC timestamps.
    Format: SET-{symbol_short}-{direction}-{hash}
    """
    if tick_size <= 0:
        raise DataValidationError("tick_size must be strictly positive.")

    _validate_utc_datetime(spike_start_time, "spike_start_time")
    _validate_utc_datetime(spike_end_time, "spike_end_time")
    _validate_utc_datetime(level_formed_time, "level_formed_time")

    if spike_end_time < spike_start_time:
        raise DataValidationError("spike_end_time cannot be earlier than spike_start_time")

    canonical_val = symbol.value if isinstance(symbol, CanonicalSymbol) else str(symbol).strip()
    if canonical_val not in (CanonicalSymbol.DOW_JONES.value, CanonicalSymbol.NASDAQ.value):
        raise DataValidationError(f"Invalid canonical symbol for setup ID: {canonical_val}")

    if isinstance(direction, Direction):
        raise DataValidationError(
            f"direction must be a TradeDirection (BUY or SELL), got Direction.{direction.name}"
        )
    elif isinstance(direction, TradeDirection):
        dir_val = direction.value
    elif isinstance(direction, str):
        dir_val = direction.strip().upper()
        if dir_val not in (TradeDirection.BUY.value, TradeDirection.SELL.value):
            raise DataValidationError(
                f"Invalid trade direction for setup ID: {direction}. Must be BUY or SELL."
            )
    else:
        raise DataValidationError(
            f"direction must be TradeDirection or str ('BUY'/'SELL'), got {type(direction).__name__}"
        )

    norm_price = _normalize_price(structural_price, tick_size)

    # Deterministic canonical identity string construction
    canonical_string = (
        f"{version}|{canonical_val}|{dir_val}|"
        f"{spike_start_time.isoformat()}|{spike_end_time.isoformat()}|"
        f"{level_formed_time.isoformat()}|{norm_price:.8f}"
    )

    hash_digest = hashlib.sha256(canonical_string.encode("utf-8")).hexdigest()[:12].upper()
    symbol_short = "DJ" if canonical_val == CanonicalSymbol.DOW_JONES.value else "NAS"
    return f"SET-{symbol_short}-{dir_val}-{hash_digest}"


@dataclass(frozen=True)
class Candle:
    """
    Immutable representation of an OHLCV candle.
    All timestamps are strictly timezone-aware UTC.
    """
    symbol: CanonicalSymbol
    timeframe: Timeframe
    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    spread: float

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, CanonicalSymbol):
            raise DataValidationError(f"symbol must be a CanonicalSymbol, got {self.symbol}")
        if not isinstance(self.timeframe, Timeframe):
            raise DataValidationError(f"timeframe must be a Timeframe, got {self.timeframe}")
        _validate_utc_datetime(self.open_time, "open_time")
        _validate_utc_datetime(self.close_time, "close_time")

        if self.close_time <= self.open_time:
            raise DataValidationError(
                f"close_time ({self.close_time}) must be strictly greater than open_time ({self.open_time})"
            )

        if not (self.high >= self.low):
            raise DataValidationError(f"Candle high ({self.high}) cannot be less than low ({self.low})")
        if not (self.high >= self.open and self.high >= self.close):
            raise DataValidationError(f"Candle high ({self.high}) must be >= open ({self.open}) and close ({self.close})")
        if not (self.low <= self.open and self.low <= self.close):
            raise DataValidationError(f"Candle low ({self.low}) must be <= open ({self.open}) and close ({self.close})")
        if self.volume < 0:
            raise DataValidationError(f"Candle volume ({self.volume}) cannot be negative")
        if self.spread < 0:
            raise DataValidationError(f"Candle spread ({self.spread}) cannot be negative")

    @property
    def direction(self) -> Direction:
        if self.close > self.open:
            return Direction.BULLISH
        elif self.close < self.open:
            return Direction.BEARISH
        return Direction.NEUTRAL

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def body_high(self) -> float:
        return max(self.open, self.close)

    @property
    def body_low(self) -> float:
        return min(self.open, self.close)


@dataclass(frozen=True)
class Quote:
    """
    Immutable representation of a live top-of-book market quote.
    """
    symbol: CanonicalSymbol
    timestamp: datetime
    bid: float
    ask: float
    spread: float

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, CanonicalSymbol):
            raise DataValidationError(f"symbol must be a CanonicalSymbol, got {self.symbol}")
        _validate_utc_datetime(self.timestamp, "timestamp")
        if self.bid <= 0 or self.ask <= 0:
            raise DataValidationError(f"Bid ({self.bid}) and ask ({self.ask}) must be strictly positive")
        if self.ask < self.bid:
            raise DataValidationError(f"Ask ({self.ask}) cannot be strictly less than bid ({self.bid})")
        if self.spread < 0:
            raise DataValidationError(f"Spread ({self.spread}) cannot be negative")
        
        expected_spread = round(self.ask - self.bid, 8)
        if not math.isclose(self.spread, expected_spread, abs_tol=1e-7):
            raise DataValidationError(
                f"Spread ({self.spread}) does not match ask - bid ({expected_spread})"
            )


@dataclass(frozen=True)
class SymbolSpecification:
    """
    Instrument contract specifications from broker.
    Maintains clean separation between canonical_symbol and broker_symbol.
    """
    canonical_symbol: CanonicalSymbol
    broker_symbol: str
    tick_size: float
    tick_value: float
    contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    point: float

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_symbol, CanonicalSymbol):
            raise DataValidationError(f"canonical_symbol must be a CanonicalSymbol, got {self.canonical_symbol}")
        if not isinstance(self.broker_symbol, str) or not self.broker_symbol.strip():
            raise DataValidationError("broker_symbol must be a non-empty string")
        if self.tick_size <= 0:
            raise DataValidationError("tick_size must be strictly positive")
        if self.tick_value <= 0:
            raise DataValidationError("tick_value must be strictly positive")
        if self.contract_size <= 0:
            raise DataValidationError("contract_size must be strictly positive")
        if self.volume_min <= 0:
            raise DataValidationError("volume_min must be strictly positive")
        if self.volume_max < self.volume_min:
            raise DataValidationError("volume_max cannot be less than volume_min")
        if self.volume_step <= 0:
            raise DataValidationError("volume_step must be strictly positive")
        if self.point <= 0:
            raise DataValidationError("point must be strictly positive")


@dataclass(frozen=True)
class AccountInfo:
    """
    Snapshot of broker account balance, equity, and margins.
    """
    balance: float
    equity: float
    margin: float
    free_margin: float
    currency: str

    def __post_init__(self) -> None:
        if not isinstance(self.currency, str) or not self.currency.strip():
            raise DataValidationError("Account currency must be a non-empty string")
        if self.margin < 0:
            raise DataValidationError("Margin cannot be negative")


@dataclass(frozen=True)
class GapEvent:
    """
    Represents an identified 3-candle Fair Value Gap.
    """
    direction: Direction
    c1_high: float
    c1_low: float
    c3_high: float
    c3_low: float
    c1_time: datetime
    c3_time: datetime

    def __post_init__(self) -> None:
        if self.direction not in (Direction.BULLISH, Direction.BEARISH):
            raise DataValidationError(f"Gap direction must be BULLISH or BEARISH, got {self.direction}")
        _validate_utc_datetime(self.c1_time, "c1_time")
        _validate_utc_datetime(self.c3_time, "c3_time")
        if self.c3_time <= self.c1_time:
            raise DataValidationError("c3_time must be strictly after c1_time")

        if self.direction == Direction.BULLISH:
            if not (self.c3_low > self.c1_high):
                raise DataValidationError(
                    f"Bullish gap invalid: c3_low ({self.c3_low}) must exceed c1_high ({self.c1_high})"
                )
        elif self.direction == Direction.BEARISH:
            if not (self.c1_low > self.c3_high):
                raise DataValidationError(
                    f"Bearish gap invalid: c1_low ({self.c1_low}) must exceed c3_high ({self.c3_high})"
                )

    @property
    def gap_size(self) -> float:
        if self.direction == Direction.BULLISH:
            return self.c3_low - self.c1_high
        return self.c1_low - self.c3_high


@dataclass(frozen=True)
class SpikeEvent:
    """
    Represents a verified momentum spike meeting min candle and gap criteria.
    """
    spike_id: str
    symbol: CanonicalSymbol
    direction: Direction
    start_time: datetime
    end_time: datetime
    high: float
    low: float
    candle_count: int
    gap_count: int

    def __post_init__(self) -> None:
        if not self.spike_id or not isinstance(self.spike_id, str):
            raise DataValidationError("spike_id must be a non-empty string")
        if not isinstance(self.symbol, CanonicalSymbol):
            raise DataValidationError(f"symbol must be a CanonicalSymbol, got {self.symbol}")
        if self.direction not in (Direction.BULLISH, Direction.BEARISH):
            raise DataValidationError(f"direction must be BULLISH or BEARISH, got {self.direction}")
        _validate_utc_datetime(self.start_time, "start_time")
        _validate_utc_datetime(self.end_time, "end_time")
        if self.end_time < self.start_time:
            raise DataValidationError("end_time cannot be earlier than start_time")
        if self.high < self.low:
            raise DataValidationError("high cannot be less than low")
        if self.candle_count < MIN_SPIKE_CANDLES:
            raise DataValidationError(
                f"candle_count ({self.candle_count}) below minimum required ({MIN_SPIKE_CANDLES})"
            )
        if self.gap_count < MIN_GAP_COUNT:
            raise DataValidationError(
                f"gap_count ({self.gap_count}) below minimum required ({MIN_GAP_COUNT})"
            )


@dataclass(frozen=True)
class StructuralLevel:
    """
    Immutable structural reference level (First Bottom / First Top).
    Once locked, its price and timestamp must never be mutated or recalculated.
    """
    spike_id: str
    symbol: CanonicalSymbol
    direction: Direction
    level_type: str
    price: float
    formation_time: datetime
    is_locked: bool = True

    def __post_init__(self) -> None:
        if not self.spike_id or not isinstance(self.spike_id, str):
            raise DataValidationError("spike_id must be a non-empty string")
        if not isinstance(self.symbol, CanonicalSymbol):
            raise DataValidationError(f"symbol must be a CanonicalSymbol, got {self.symbol}")
        if self.direction not in (Direction.BULLISH, Direction.BEARISH):
            raise DataValidationError(f"direction must be BULLISH or BEARISH, got {self.direction}")
        if not self.level_type or not isinstance(self.level_type, str):
            raise DataValidationError("level_type must be a non-empty string")
        if self.price <= 0:
            raise DataValidationError("Structural price must be strictly positive")
        _validate_utc_datetime(self.formation_time, "formation_time")
        if not self.is_locked:
            raise LevelImmutabilityViolationError("StructuralLevel must be locked upon formation")


@dataclass(frozen=True)
class PinBarEvent:
    """
    Represents an M1 Pin Bar validation event conforming to locked geometry.
    Body ratio <= 0.20, open and close on opposite side of midpoint.
    """
    symbol: CanonicalSymbol
    timeframe: Timeframe
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    range: float
    body: float
    body_ratio: float
    midpoint: float
    is_valid: bool
    direction: Direction

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, CanonicalSymbol):
            raise DataValidationError(f"symbol must be a CanonicalSymbol, got {self.symbol}")
        if self.timeframe != Timeframe.M1:
            raise DataValidationError(f"PinBar must form on M1 timeframe, got {self.timeframe}")
        _validate_utc_datetime(self.close_time, "close_time")
        if self.range < 0 or self.body < 0:
            raise DataValidationError("PinBar range and body must be non-negative")
        if self.body_ratio < 0:
            raise DataValidationError("body_ratio cannot be negative")


@dataclass(frozen=True)
class SignalEvent:
    """
    Signal generated upon Pin Bar trigger activation.
    """
    setup_id: str
    symbol: CanonicalSymbol
    direction: TradeDirection
    signal_time: datetime
    pinbar_high: float
    pinbar_low: float
    final_structural_target: float

    def __post_init__(self) -> None:
        if not self.setup_id or not isinstance(self.setup_id, str):
            raise DataValidationError("setup_id must be a non-empty string")
        if not isinstance(self.symbol, CanonicalSymbol):
            raise DataValidationError(f"symbol must be a CanonicalSymbol, got {self.symbol}")
        if not isinstance(self.direction, TradeDirection):
            raise DataValidationError(f"direction must be a TradeDirection, got {self.direction}")
        _validate_utc_datetime(self.signal_time, "signal_time")
        if self.pinbar_high < self.pinbar_low:
            raise DataValidationError("pinbar_high cannot be less than pinbar_low")
        if self.final_structural_target <= 0:
            raise DataValidationError("final_structural_target must be strictly positive")


@dataclass(frozen=True)
class OrderRequest:
    """
    Instruction sent to the broker adapter to execute a trade.
    """
    setup_id: str
    symbol: CanonicalSymbol
    direction: TradeDirection
    volume: float
    entry_price: float
    stop_loss: float
    take_profits: Tuple[float, ...]
    final_target: float
    magic_number: int
    deviation: int

    def __post_init__(self) -> None:
        if not self.setup_id:
            raise DataValidationError("setup_id cannot be empty")
        if not isinstance(self.symbol, CanonicalSymbol):
            raise DataValidationError(f"symbol must be a CanonicalSymbol, got {self.symbol}")
        if not isinstance(self.direction, TradeDirection):
            raise DataValidationError(f"direction must be a TradeDirection, got {self.direction}")
        if self.volume <= 0:
            raise DataValidationError("volume must be strictly positive")
        if self.entry_price <= 0 or self.stop_loss <= 0 or self.final_target <= 0:
            raise DataValidationError("Prices (entry, SL, final_target) must be strictly positive")
        if self.magic_number <= 0:
            raise DataValidationError("magic_number must be strictly positive")
        if self.deviation < 0:
            raise DataValidationError("deviation cannot be negative")


@dataclass(frozen=True)
class OrderResult:
    """
    Result returned by the broker adapter after order execution attempt.
    """
    success: bool
    order_ticket: Optional[int]
    deal_ticket: Optional[int]
    position_ticket: Optional[int]
    executed_price: float
    executed_volume: float
    error_message: Optional[str] = None


@dataclass(frozen=True)
class TradeRecord:
    """
    Canonical record of an executed trade containing EXACTLY 35 fields in locked order.
    Represents the full structural trade audit and performance outcome.
    """
    symbol: str
    date: str
    session: str
    spike_direction: str
    spike_time: str
    spike_high: float
    spike_low: float
    spike_candle_count: int
    gap_count: int
    first_level_price: float
    average_m5_range: float
    proximity_distance: float
    pinbar_open: float
    pinbar_high: float
    pinbar_low: float
    pinbar_close: float
    pinbar_body: float
    pinbar_range: float
    pinbar_body_ratio: float
    entry_price: float
    initial_sl: float
    tp1: float
    tp2: float
    tp3: float
    final_target: float
    spread: float
    risk_percent: float
    lot_size: float
    order_ticket: int
    deal_ticket: int
    position_ticket: int
    result: str
    r_multiple: float
    mfe: float
    mae: float

    def __post_init__(self) -> None:
        # Strict invariant validation on field count
        actual_fields = len(fields(self))
        if actual_fields != TRADE_RECORD_EXACT_FIELD_COUNT:
            raise DataValidationError(
                f"TradeRecord must have exactly {TRADE_RECORD_EXACT_FIELD_COUNT} fields, got {actual_fields}"
            )

        if not self.symbol:
            raise DataValidationError("TradeRecord symbol cannot be empty")
        if not self.date:
            raise DataValidationError("TradeRecord date cannot be empty")
        if not self.session:
            raise DataValidationError("TradeRecord session cannot be empty")
        if not self.spike_direction:
            raise DataValidationError("TradeRecord spike_direction cannot be empty")
        if not self.spike_time:
            raise DataValidationError("TradeRecord spike_time cannot be empty")

        if self.spike_high < self.spike_low:
            raise DataValidationError("spike_high cannot be less than spike_low")
        if self.pinbar_high < self.pinbar_low:
            raise DataValidationError("pinbar_high cannot be less than pinbar_low")
        if self.lot_size <= 0:
            raise DataValidationError("lot_size must be strictly positive")
        if self.risk_percent <= 0:
            raise DataValidationError("risk_percent must be strictly positive")
        if not self.result:
            raise DataValidationError("TradeRecord result cannot be empty")
