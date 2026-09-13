"""
Market Layer Package.
Provides broker-agnostic market data transport, validation, and historical/live interfaces.
Decoupled strictly from MT5 SDK and trading strategy logic.
"""

from src.market.exceptions import (
    InvalidSequenceError,
    SpreadProvenanceError,
    SymbolMismatchError,
    TimeframeMismatchError,
)
from src.market.in_memory_provider import InMemoryMarketDataProvider
from src.market.interfaces import (
    ICandleProvider,
    IMarketDataProvider,
    IQuoteProvider,
    ISymbolSpecificationProvider,
    ITickProvider,
)
from src.market.provenance import (
    assert_execution_spread_not_from_candle,
    build_spread_audit_record,
    extract_execution_spread,
)
from src.market.session import (
    get_session_date,
    to_session_time,
    validate_time_range,
)
from src.market.types import (
    MarketFeedType,
    SpreadProvenance,
    Tick,
)
from src.market.validator import MarketDataValidator

__all__ = [
    # Types
    "Tick",
    "SpreadProvenance",
    "MarketFeedType",
    # Exceptions
    "InvalidSequenceError",
    "SpreadProvenanceError",
    "SymbolMismatchError",
    "TimeframeMismatchError",
    # Interfaces
    "ICandleProvider",
    "ITickProvider",
    "IQuoteProvider",
    "ISymbolSpecificationProvider",
    "IMarketDataProvider",
    # Validator
    "MarketDataValidator",
    # Provenance
    "extract_execution_spread",
    "assert_execution_spread_not_from_candle",
    "build_spread_audit_record",
    # Session
    "validate_time_range",
    "to_session_time",
    "get_session_date",
    # Implementation
    "InMemoryMarketDataProvider",
]
