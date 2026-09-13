"""
Market Data Validator.
Enforces data integrity, chronological causal ordering, timeframe separation,
and geometric invariants across candles, quotes, ticks, and symbol specifications.
"""

from datetime import datetime, timezone
from typing import Optional, Sequence

from src.core.constants import CanonicalSymbol, Timeframe
from src.core.exceptions import DataValidationError, MissingDataError, TimezoneError
from src.core.types import Candle, Quote, SymbolSpecification
from src.market.exceptions import (
    InvalidSequenceError,
    SymbolMismatchError,
    TimeframeMismatchError,
)
from src.market.types import Tick


class MarketDataValidator:
    """
    Stateless validator enforcing strict contracts on market data feeds and domain entities.
    """

    @staticmethod
    def validate_utc_datetime(dt: datetime, field_name: str) -> None:
        """Validate that a datetime is strictly timezone-aware UTC."""
        if not isinstance(dt, datetime):
            raise TimezoneError(f"{field_name} must be a datetime instance, got {type(dt).__name__}")
        if dt.tzinfo is None or dt.tzinfo != timezone.utc:
            raise TimezoneError(
                f"{field_name} must be timezone-aware UTC, got tzinfo={dt.tzinfo}"
            )

    @classmethod
    def validate_candle(
        cls,
        candle: Candle,
        expected_symbol: Optional[CanonicalSymbol] = None,
        expected_timeframe: Optional[Timeframe] = None,
    ) -> None:
        """
        Validate single Candle OHLCV invariants, symbol, timeframe, and timestamps.
        """
        if not isinstance(candle, Candle):
            raise DataValidationError(f"Expected Candle instance, got {type(candle).__name__}")

        cls.validate_utc_datetime(candle.open_time, "open_time")
        cls.validate_utc_datetime(candle.close_time, "close_time")

        if candle.close_time <= candle.open_time:
            raise DataValidationError(
                f"Candle close_time ({candle.close_time.isoformat()}) must be strictly "
                f"after open_time ({candle.open_time.isoformat()})"
            )

        if expected_symbol is not None and candle.symbol != expected_symbol:
            raise SymbolMismatchError(
                f"Candle symbol mismatch: expected {expected_symbol.value}, got {candle.symbol.value}"
            )

        if expected_timeframe is not None and candle.timeframe != expected_timeframe:
            raise TimeframeMismatchError(
                f"Candle timeframe mismatch: expected {expected_timeframe.value}, got {candle.timeframe.value}"
            )

        if candle.high < candle.low:
            raise DataValidationError(f"Candle high ({candle.high}) cannot be less than low ({candle.low})")

        if candle.high < candle.open or candle.high < candle.close:
            raise DataValidationError(
                f"Candle high ({candle.high}) must be >= open ({candle.open}) and close ({candle.close})"
            )

        if candle.low > candle.open or candle.low > candle.close:
            raise DataValidationError(
                f"Candle low ({candle.low}) must be <= open ({candle.open}) and close ({candle.close})"
            )

        if candle.volume < 0:
            raise DataValidationError(f"Candle volume ({candle.volume}) cannot be negative")

        if candle.spread < 0:
            raise DataValidationError(f"Candle spread ({candle.spread}) cannot be negative")

    @classmethod
    def validate_candle_sequence(
        cls,
        candles: Sequence[Candle],
        expected_symbol: Optional[CanonicalSymbol] = None,
        expected_timeframe: Optional[Timeframe] = None,
        require_non_empty: bool = False,
    ) -> None:
        """
        Validate that a list of candles forms a strictly increasing, non-overlapping chronological sequence.
        Rejects backward timestamps, duplicate timestamps, overlapping bars, and symbol/timeframe mismatches.
        """
        if not candles:
            if require_non_empty:
                raise MissingDataError("Candle sequence cannot be empty when require_non_empty is True")
            return

        first = candles[0]
        effective_symbol = expected_symbol or first.symbol
        effective_tf = expected_timeframe or first.timeframe

        for i, c in enumerate(candles):
            cls.validate_candle(c, expected_symbol=effective_symbol, expected_timeframe=effective_tf)

            if i > 0:
                prev = candles[i - 1]

                # Strict monotonic ordering on open_time: no duplicates, no backward time
                if c.open_time == prev.open_time:
                    raise InvalidSequenceError(
                        f"Duplicate candle open_time detected at index {i}: {c.open_time.isoformat()}"
                    )
                if c.open_time < prev.open_time:
                    raise InvalidSequenceError(
                        f"Backward candle open_time detected at index {i} (causality violation): "
                        f"prev={prev.open_time.isoformat()} > current={c.open_time.isoformat()}"
                    )

                # Non-overlapping bar validation
                if c.open_time < prev.close_time:
                    raise InvalidSequenceError(
                        f"Overlapping candle interval detected at index {i}: "
                        f"prev close_time={prev.close_time.isoformat()} > current open_time={c.open_time.isoformat()}"
                    )

    @classmethod
    def validate_quote(
        cls,
        quote: Quote,
        expected_symbol: Optional[CanonicalSymbol] = None,
    ) -> None:
        """
        Validate live Quote contract, timestamp, positive bid/ask, and execution spread consistency.
        """
        if not isinstance(quote, Quote):
            raise DataValidationError(f"Expected Quote instance, got {type(quote).__name__}")

        cls.validate_utc_datetime(quote.timestamp, "timestamp")

        if expected_symbol is not None and quote.symbol != expected_symbol:
            raise SymbolMismatchError(
                f"Quote symbol mismatch: expected {expected_symbol.value}, got {quote.symbol.value}"
            )

        if quote.bid <= 0 or quote.ask <= 0:
            raise DataValidationError(f"Bid ({quote.bid}) and ask ({quote.ask}) must be strictly positive")

        if quote.ask < quote.bid:
            raise DataValidationError(
                f"Ask ({quote.ask}) cannot be strictly less than bid ({quote.bid})"
            )

        # Execution spread consistency verification
        expected_spread = round(quote.ask - quote.bid, 8)
        if abs(quote.spread - expected_spread) > 1e-5:
            raise DataValidationError(
                f"Quote spread inconsistency: reported {quote.spread} != calculated {expected_spread} (ask - bid)"
            )

    @classmethod
    def validate_tick(
        cls,
        tick: Tick,
        expected_symbol: Optional[CanonicalSymbol] = None,
    ) -> None:
        """
        Validate single Tick invariants, timestamp, and price contracts.
        """
        if not isinstance(tick, Tick):
            raise DataValidationError(f"Expected Tick instance, got {type(tick).__name__}")

        cls.validate_utc_datetime(tick.timestamp, "timestamp")

        if expected_symbol is not None and tick.symbol != expected_symbol:
            raise SymbolMismatchError(
                f"Tick symbol mismatch: expected {expected_symbol.value}, got {tick.symbol.value}"
            )

        if tick.bid <= 0 or tick.ask <= 0:
            raise DataValidationError(f"Bid ({tick.bid}) and ask ({tick.ask}) must be strictly positive")

        if tick.ask < tick.bid:
            raise DataValidationError(
                f"Ask ({tick.ask}) cannot be strictly less than bid ({tick.bid})"
            )

        if tick.volume < 0:
            raise DataValidationError(f"Tick volume ({tick.volume}) cannot be negative")

        if tick.last < 0:
            raise DataValidationError(f"Tick last price ({tick.last}) cannot be negative")

    @classmethod
    def validate_tick_sequence(
        cls,
        ticks: Sequence[Tick],
        expected_symbol: Optional[CanonicalSymbol] = None,
        require_non_empty: bool = False,
    ) -> None:
        """
        Validate tick chronological sequence.
        Ticks must have non-decreasing timestamps (multiple ticks can occur within the same millisecond/second).
        Backward timestamps strictly raise InvalidSequenceError.
        """
        if not ticks:
            if require_non_empty:
                raise MissingDataError("Tick sequence cannot be empty when require_non_empty is True")
            return

        effective_symbol = expected_symbol or ticks[0].symbol

        for i, t in enumerate(ticks):
            cls.validate_tick(t, expected_symbol=effective_symbol)

            if i > 0:
                prev = ticks[i - 1]
                if t.timestamp < prev.timestamp:
                    raise InvalidSequenceError(
                        f"Backward tick timestamp detected at index {i}: "
                        f"prev={prev.timestamp.isoformat()} > current={t.timestamp.isoformat()}"
                    )

    @classmethod
    def validate_symbol_specification(
        cls,
        spec: SymbolSpecification,
        expected_symbol: Optional[CanonicalSymbol] = None,
    ) -> None:
        """
        Validate instrument contract parameters.
        """
        if not isinstance(spec, SymbolSpecification):
            raise DataValidationError(f"Expected SymbolSpecification instance, got {type(spec).__name__}")

        if expected_symbol is not None and spec.canonical_symbol != expected_symbol:
            raise SymbolMismatchError(
                f"Specification symbol mismatch: expected {expected_symbol.value}, "
                f"got {spec.canonical_symbol.value}"
            )

        if spec.tick_size <= 0:
            raise DataValidationError(f"tick_size must be strictly positive, got {spec.tick_size}")
        if spec.tick_value <= 0:
            raise DataValidationError(f"tick_value must be strictly positive, got {spec.tick_value}")
        if spec.contract_size <= 0:
            raise DataValidationError(f"contract_size must be strictly positive, got {spec.contract_size}")
        if spec.volume_min <= 0:
            raise DataValidationError(f"volume_min must be strictly positive, got {spec.volume_min}")
        if spec.volume_max < spec.volume_min:
            raise DataValidationError(
                f"volume_max ({spec.volume_max}) cannot be less than volume_min ({spec.volume_min})"
            )
        if spec.volume_step <= 0:
            raise DataValidationError(f"volume_step must be strictly positive, got {spec.volume_step}")
        if spec.point <= 0:
            raise DataValidationError(f"point must be strictly positive, got {spec.point}")
