"""
In-Memory Market Data Provider.
Deterministic, broker-agnostic implementation of IMarketDataProvider for testing,
simulation, and replay. Completely decoupled from MT5 SDK.
"""

from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from src.core.constants import CanonicalSymbol, Timeframe
from src.core.exceptions import MissingDataError
from src.core.types import Candle, Quote, SymbolSpecification
from src.market.exceptions import InvalidSequenceError
from src.market.interfaces import IMarketDataProvider
from src.market.session import validate_time_range
from src.market.types import Tick
from src.market.validator import MarketDataValidator


class InMemoryMarketDataProvider(IMarketDataProvider):
    """
    In-memory market data provider supporting deterministic ingestion, validation,
    and chronological retrieval of candles, ticks, quotes, and specifications.
    """

    def __init__(self) -> None:
        self._candles: Dict[Tuple[CanonicalSymbol, Timeframe], List[Candle]] = {}
        self._ticks: Dict[CanonicalSymbol, List[Tick]] = {}
        self._quotes: Dict[CanonicalSymbol, Quote] = {}
        self._specifications: Dict[CanonicalSymbol, SymbolSpecification] = {}

    # ------------------------------------------------------------------
    # Ingestion API
    # ------------------------------------------------------------------

    def add_candles(self, candles: Sequence[Candle]) -> None:
        """
        Ingest a batch of candles, validating chronological ordering, timeframe,
        and interval sanity before storing.
        """
        if not candles:
            return

        MarketDataValidator.validate_candle_sequence(candles)

        first = candles[0]
        key = (first.symbol, first.timeframe)
        existing = self._candles.get(key, [])

        if existing:
            combined = existing + list(candles)
            MarketDataValidator.validate_candle_sequence(combined)
            self._candles[key] = combined
        else:
            self._candles[key] = list(candles)

    def add_ticks(self, ticks: Sequence[Tick]) -> None:
        """
        Ingest a batch of ticks, validating non-decreasing timestamps.
        """
        if not ticks:
            return

        MarketDataValidator.validate_tick_sequence(ticks)

        symbol = ticks[0].symbol
        existing = self._ticks.get(symbol, [])

        if existing:
            combined = existing + list(ticks)
            MarketDataValidator.validate_tick_sequence(combined)
            self._ticks[symbol] = combined
        else:
            self._ticks[symbol] = list(ticks)

    def set_quote(self, quote: Quote) -> None:
        """
        Update the current live quote for a canonical symbol.
        """
        MarketDataValidator.validate_quote(quote)
        self._quotes[quote.symbol] = quote

    def set_specification(self, spec: SymbolSpecification) -> None:
        """
        Register instrument specification for a canonical symbol.
        """
        MarketDataValidator.validate_symbol_specification(spec)
        self._specifications[spec.canonical_symbol] = spec

    # ------------------------------------------------------------------
    # IMarketDataProvider Implementation
    # ------------------------------------------------------------------

    def get_historical_candles(
        self,
        symbol: CanonicalSymbol,
        timeframe: Timeframe,
        start_time: datetime,
        end_time: datetime,
    ) -> List[Candle]:
        """
        Retrieve historical candles in range [start_time, end_time].
        """
        validate_time_range(start_time, end_time)
        candles = self._candles.get((symbol, timeframe), [])
        return [c for c in candles if start_time <= c.open_time <= end_time]

    def get_latest_candle(
        self,
        symbol: CanonicalSymbol,
        timeframe: Timeframe,
    ) -> Optional[Candle]:
        """
        Retrieve the latest candle for the requested symbol and timeframe.
        """
        candles = self._candles.get((symbol, timeframe), [])
        if not candles:
            return None
        return candles[-1]

    def get_historical_ticks(
        self,
        symbol: CanonicalSymbol,
        start_time: datetime,
        end_time: datetime,
    ) -> List[Tick]:
        """
        Retrieve historical ticks in range [start_time, end_time].
        """
        validate_time_range(start_time, end_time)
        ticks = self._ticks.get(symbol, [])
        return [t for t in ticks if start_time <= t.timestamp <= end_time]

    def get_latest_tick(
        self,
        symbol: CanonicalSymbol,
    ) -> Optional[Tick]:
        """
        Retrieve the latest tick for the requested symbol.
        """
        ticks = self._ticks.get(symbol, [])
        if not ticks:
            return None
        return ticks[-1]

    def get_live_quote(
        self,
        symbol: CanonicalSymbol,
    ) -> Quote:
        """
        Query current best bid and ask quote.
        Raises MissingDataError if no quote is registered.
        """
        if symbol not in self._quotes:
            raise MissingDataError(f"Live quote unavailable for canonical symbol: {symbol.value}")
        return self._quotes[symbol]

    def get_specification(
        self,
        symbol: CanonicalSymbol,
    ) -> SymbolSpecification:
        """
        Retrieve instrument specification.
        Raises MissingDataError if no specification is registered.
        """
        if symbol not in self._specifications:
            raise MissingDataError(f"Specification unavailable for canonical symbol: {symbol.value}")
        return self._specifications[symbol]
