"""
Market Data Provider Interfaces and Contracts.
Defines clean, broker-agnostic interfaces for historical and live market data.
Decoupled strictly from MT5 SDK and trading strategy logic.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from src.core.constants import CanonicalSymbol, Timeframe
from src.core.types import Candle, Quote, SymbolSpecification
from src.market.types import Tick


class ICandleProvider(ABC):
    """
    Interface for retrieving historical and latest OHLCV candle bars.
    Guarantees canonical symbol usage, explicit timeframe, and UTC timezone-aware datetimes.
    """

    @abstractmethod
    def get_historical_candles(
        self,
        symbol: CanonicalSymbol,
        timeframe: Timeframe,
        start_time: datetime,
        end_time: datetime,
    ) -> List[Candle]:
        """
        Retrieve historical candles for a canonical symbol and timeframe between start_time and end_time (inclusive).
        Returns candles in strictly ascending chronological order.
        """
        pass

    @abstractmethod
    def get_latest_candle(
        self,
        symbol: CanonicalSymbol,
        timeframe: Timeframe,
    ) -> Optional[Candle]:
        """
        Retrieve the most recently completed candle for a canonical symbol and timeframe.
        Returns None if no candle data is available.
        """
        pass


class ITickProvider(ABC):
    """
    Interface for retrieving historical intrabar tick data and latest tick snapshots.
    """

    @abstractmethod
    def get_historical_ticks(
        self,
        symbol: CanonicalSymbol,
        start_time: datetime,
        end_time: datetime,
    ) -> List[Tick]:
        """
        Retrieve historical ticks for a canonical symbol between start_time and end_time (inclusive).
        Returns ticks in non-decreasing chronological order.
        """
        pass

    @abstractmethod
    def get_latest_tick(
        self,
        symbol: CanonicalSymbol,
    ) -> Optional[Tick]:
        """
        Retrieve the latest tick snapshot for a canonical symbol.
        Returns None if no tick data is available.
        """
        pass


class IQuoteProvider(ABC):
    """
    Interface for querying real-time top-of-book market quotes.
    """

    @abstractmethod
    def get_live_quote(
        self,
        symbol: CanonicalSymbol,
    ) -> Quote:
        """
        Query current best bid and ask quotes for a canonical symbol.
        Execution spread is strictly derived as (ask - bid).
        """
        pass


class ISymbolSpecificationProvider(ABC):
    """
    Interface for accessing instrument contract specifications.
    """

    @abstractmethod
    def get_specification(
        self,
        symbol: CanonicalSymbol,
    ) -> SymbolSpecification:
        """
        Retrieve the instrument specification for a canonical symbol.
        """
        pass


class IMarketDataProvider(
    ICandleProvider,
    ITickProvider,
    IQuoteProvider,
    ISymbolSpecificationProvider,
    ABC,
):
    """
    Unified Market Data Provider interface aggregating candle, tick, quote, and specification providers.
    """
    pass
