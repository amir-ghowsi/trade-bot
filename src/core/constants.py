"""
Core Constants and Enumerations.
Strictly derived from Phase 0 Locked Specification.
No third-party dependencies, purely standard library.
"""

from enum import Enum, unique


@unique
class Timeframe(str, Enum):
    """Supported timeframe identifiers."""
    M1 = "M1"
    M5 = "M5"


@unique
class Direction(str, Enum):
    """Candle and market structure movement directions."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@unique
class TradeDirection(str, Enum):
    """Trade and order entry execution directions."""
    BUY = "BUY"
    SELL = "SELL"


@unique
class TradingSession(str, Enum):
    """Trading sessions for filtering trade entries."""
    LONDON = "LONDON"
    NEW_YORK = "NEW_YORK"
    ASIAN = "ASIAN"
    OFF_HOURS = "OFF_HOURS"


@unique
class CanonicalSymbol(str, Enum):
    """System-canonical instrument identifiers."""
    DOW_JONES = "DOW_JONES"
    NASDAQ = "NASDAQ"


@unique
class OrderType(str, Enum):
    """Order type classification."""
    MARKET = "MARKET"


@unique
class OrderStatus(str, Enum):
    """Order lifecycle status."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


@unique
class PositionStatus(str, Enum):
    """Open position lifecycle status."""
    OPEN = "OPEN"
    PARTIALLY_CLOSED = "PARTIALLY_CLOSED"
    CLOSED = "CLOSED"


@unique
class RecoveryStatus(str, Enum):
    """Position reconciliation status during startup/reconnect."""
    FOUND = "FOUND"
    NOT_FOUND = "NOT_FOUND"
    BROKER_QUERY_FAILED = "BROKER_QUERY_FAILED"
    UNKNOWN = "UNKNOWN"


@unique
class BullishState(str, Enum):
    """
    11 Sequential States for Bullish Structure Lifecycle.
    Exactly 10 Transitions between State 1 and State 11.
    """
    IDLE = "IDLE"
    BULLISH_SPIKE_DETECTED = "BULLISH_SPIKE_DETECTED"
    WAIT_INITIAL_CORRECTION = "WAIT_INITIAL_CORRECTION"
    FIRST_BOTTOM_FORMED = "FIRST_BOTTOM_FORMED"
    WAIT_RETURN_TO_FIRST_BOTTOM = "WAIT_RETURN_TO_FIRST_BOTTOM"
    M1_MONITORING = "M1_MONITORING"
    BEARISH_PINBAR_DETECTED = "BEARISH_PINBAR_DETECTED"
    BUY_TRIGGER_ARMED = "BUY_TRIGGER_ARMED"
    BUY_EXECUTED = "BUY_EXECUTED"
    MANAGE_TRADE = "MANAGE_TRADE"
    CLOSED = "CLOSED"


@unique
class BearishState(str, Enum):
    """
    11 Sequential States for Bearish Structure Lifecycle.
    Exactly 10 Transitions between State 1 and State 11.
    """
    IDLE = "IDLE"
    BEARISH_SPIKE_DETECTED = "BEARISH_SPIKE_DETECTED"
    WAIT_INITIAL_CORRECTION = "WAIT_INITIAL_CORRECTION"
    FIRST_TOP_FORMED = "FIRST_TOP_FORMED"
    WAIT_RETURN_TO_FIRST_TOP = "WAIT_RETURN_TO_FIRST_TOP"
    M1_MONITORING = "M1_MONITORING"
    BULLISH_PINBAR_DETECTED = "BULLISH_PINBAR_DETECTED"
    SELL_TRIGGER_ARMED = "SELL_TRIGGER_ARMED"
    SELL_EXECUTED = "SELL_EXECUTED"
    MANAGE_TRADE = "MANAGE_TRADE"
    CLOSED = "CLOSED"


# =====================================================================
# STRATEGY NUMERIC SPECIFICATIONS (LOCKED)
# =====================================================================

# Specification Version
SPECIFICATION_VERSION: str = "v1.0"

# Risk Management
TARGET_RISK_FRACTION: float = 0.01  # Target risk is exactly 1% of account equity

# Structure / Spike Definitions
MIN_SPIKE_CANDLES: int = 3          # Minimum 3 consecutive directional candles
MIN_GAP_COUNT: int = 1              # Minimum 1 valid gap within the spike

# M1 Pin Bar Confirmation Geometry
PINBAR_MAX_BODY_RATIO: float = 0.20 # Real body must be <= 20% of total candle range
PINBAR_MIDPOINT_RATIO: float = 0.50 # Body boundary strictly within upper/lower half

# Execution Spread Offsets
BUY_ENTRY_SPREAD_MULTIPLIER: float = 1.0   # BUY_ENTRY = PinBar_High + Spread
BUY_INITIAL_SL_SPREAD_MULTIPLIER: float = 2.0  # BUY_INITIAL_SL = PinBar_Low - 2 * Spread

SELL_ENTRY_SPREAD_MULTIPLIER: float = 1.0  # SELL_ENTRY = PinBar_Low - Spread
SELL_INITIAL_SL_SPREAD_MULTIPLIER: float = 2.0 # SELL_INITIAL_SL = PinBar_High + 2 * Spread

# Trailing Stop Multipliers
TRAILING_STEP_SPREAD_MULTIPLIER: float = 2.0  # SL = TP(n-2) +/- 2 * Spread

# Partial Close Allocation (Locked: 25% / 25% / 25% / 25% of ORIGINAL Volume)
PARTIAL_CLOSE_TP1_RATIO: float = 0.25
PARTIAL_CLOSE_TP2_RATIO: float = 0.25
PARTIAL_CLOSE_TP3_RATIO: float = 0.25
PARTIAL_CLOSE_STRUCTURAL_TARGET_RATIO: float = 0.25

# Timezones
INTERNAL_TIMEZONE: str = "UTC"
TRADING_DAY_TIMEZONE: str = "America/New_York"

# Concurrency Rules
MAX_ACTIVE_POSITIONS_PER_SYMBOL: int = 1
MAX_GLOBAL_ACTIVE_POSITIONS: int = 2

# TradeRecord Invariant Field Count
TRADE_RECORD_EXACT_FIELD_COUNT: int = 35

# Backtest Defaults (Configurable)
DEFAULT_BACKTEST_COMMISSION_PER_LOT: float = 0.0
DEFAULT_BACKTEST_SLIPPAGE_POINTS: float = 2.0
