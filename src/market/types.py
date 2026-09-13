"""
Market Layer Domain Types.
Extends Core domain models with Market Layer specific data representations such as Tick and SpreadProvenance.
Decoupled strictly from broker SDKs (zero MT5 imports).
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from src.core.constants import CanonicalSymbol
from src.core.exceptions import DataValidationError


def _validate_utc_datetime(dt: datetime, field_name: str) -> None:
    """Validate that a datetime is timezone-aware and set to UTC."""
    if not isinstance(dt, datetime):
        raise DataValidationError(f"{field_name} must be a datetime instance, got {type(dt).__name__}")
    if dt.tzinfo is None or dt.tzinfo != timezone.utc:
        raise DataValidationError(
            f"{field_name} must be timezone-aware UTC, got tzinfo={dt.tzinfo}"
        )


class SpreadProvenance(Enum):
    """
    Explicit provenance tracking for market spread data.
    Enforces architectural separation between live execution spread and historical bar spread.
    """
    LIVE_QUOTE_BID_ASK = "LIVE_QUOTE_BID_ASK"        # Direct Quote execution spread: Ask - Bid
    HISTORICAL_BAR_SPREAD = "HISTORICAL_BAR_SPREAD"  # Informational bar spread from data source


class MarketFeedType(Enum):
    """
    Source feed classification for market data streams.
    """
    HISTORICAL = "HISTORICAL"
    LIVE = "LIVE"


@dataclass(frozen=True)
class Tick:
    """
    Immutable representation of an individual market tick for intrabar resolution.
    All timestamps must be strictly timezone-aware UTC.
    """
    symbol: CanonicalSymbol
    timestamp: datetime
    bid: float
    ask: float
    last: float = 0.0
    volume: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, CanonicalSymbol):
            raise DataValidationError(f"symbol must be a CanonicalSymbol, got {self.symbol}")
        _validate_utc_datetime(self.timestamp, "timestamp")
        if self.bid <= 0 or self.ask <= 0:
            raise DataValidationError(f"Bid ({self.bid}) and ask ({self.ask}) must be strictly positive")
        if self.ask < self.bid:
            raise DataValidationError(f"Ask ({self.ask}) cannot be strictly less than bid ({self.bid})")
        if self.volume < 0:
            raise DataValidationError(f"Volume ({self.volume}) cannot be negative")
        if self.last < 0:
            raise DataValidationError(f"Last price ({self.last}) cannot be negative")

    @property
    def spread(self) -> float:
        """Calculate the top-of-book tick spread (Ask - Bid)."""
        return round(self.ask - self.bid, 8)
