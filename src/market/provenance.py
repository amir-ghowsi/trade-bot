"""
Spread Provenance and Execution Spread Boundary Enforcement.
Ensures strict architectural separation between historical bar spread (Candle.spread)
and real-time execution spread (Quote.ask - Quote.bid).
"""

from typing import Any, Dict

from src.core.types import Candle, Quote
from src.market.exceptions import SpreadProvenanceError
from src.market.types import SpreadProvenance


def extract_execution_spread(quote: Quote) -> float:
    """
    Extract execution spread strictly from a live Quote instance.
    Execution spread is mathematically defined as (Ask - Bid).
    Never derived from OHLC or Candle spread.
    """
    if not isinstance(quote, Quote):
        raise SpreadProvenanceError(
            f"Execution spread must be extracted from a Quote instance, got {type(quote).__name__}"
        )
    spread = round(quote.ask - quote.bid, 8)
    if spread < 0:
        raise SpreadProvenanceError(f"Negative execution spread detected in quote: {spread}")
    return spread


def assert_execution_spread_not_from_candle(source: Any) -> None:
    """
    Guard against using Candle or historical bar spread as a live execution spread.
    Raises SpreadProvenanceError if a Candle or non-Quote object is provided.
    """
    if isinstance(source, Candle):
        raise SpreadProvenanceError(
            "Candle.spread represents historical bar data and MUST NOT be used "
            "as live execution spread. Live execution spread requires a Quote."
        )


def build_spread_audit_record(
    quote: Quote,
    provenance: SpreadProvenance = SpreadProvenance.LIVE_QUOTE_BID_ASK,
) -> Dict[str, Any]:
    """
    Generate an immutable audit dictionary for trade execution records.
    """
    if provenance != SpreadProvenance.LIVE_QUOTE_BID_ASK:
        raise SpreadProvenanceError(
            f"Invalid provenance for execution spread: {provenance.value}. "
            f"Must be {SpreadProvenance.LIVE_QUOTE_BID_ASK.value}"
        )
    return {
        "symbol": quote.symbol.value,
        "timestamp": quote.timestamp.isoformat(),
        "bid": quote.bid,
        "ask": quote.ask,
        "execution_spread": extract_execution_spread(quote),
        "provenance": provenance.value,
    }
