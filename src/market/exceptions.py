"""
Market Layer Exceptions.
Subclasses core MarketDataError for granular failure diagnostics.
"""

from typing import Any, Dict, Optional
from src.core.exceptions import MarketDataError


class InvalidSequenceError(MarketDataError):
    """Raised when market candle or tick sequence violates chronological ordering or interval rules."""
    pass


class TimeframeMismatchError(MarketDataError):
    """Raised when returned candle timeframe does not match the requested timeframe."""
    pass


class SymbolMismatchError(MarketDataError):
    """Raised when market data symbol does not match the expected canonical symbol."""
    pass


class SpreadProvenanceError(MarketDataError):
    """Raised when historical bar spread is conflated with live Quote execution spread."""
    pass
