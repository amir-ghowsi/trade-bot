"""
Average M5 Range Calculator.
Strictly implements the locked mathematical formula:
Candle_Range = High - Low
Average_M5_Range = Sum(Candle_Range) / Number_of_Valid_M5_Candles

Rules:
- Tokyo/Oceania candles are excluded from the calculation.
- ATR is STRICTLY FORBIDDEN.
- Moving averages, Wilder smoothing, or True Range (including previous close) are strictly forbidden.
- Session distinction: Tokyo/Oceania exclusion for Average M5 Range is strictly decoupled
  from London/New York trading session eligibility.
- Session hours are NOT invented here. Data filtering dependency is injected via a callable predicate
  (e.g., is_tokyo_oceania_candle) to allow exact configuration when defined.
"""

from typing import Callable, Optional, Sequence

from src.core.constants import Timeframe
from src.core.types import Candle
from src.strategy.exceptions import DetectorError


def calculate_candle_range(candle: Candle) -> float:
    """
    Calculate pure price action range for a single candle: High - Low.
    Strictly forbids True Range or ATR formulas.
    """
    if not isinstance(candle, Candle):
        raise DetectorError(f"Expected Candle instance, got {type(candle).__name__}")
    return candle.high - candle.low


def calculate_average_m5_range(
    candles: Sequence[Candle],
    is_tokyo_oceania_filter: Optional[Callable[[Candle], bool]] = None,
) -> float:
    """
    Calculate the dynamic Average M5 Range according to the locked specification:
    Average_M5_Range = Sum(High - Low) / Number_of_Valid_M5_Candles

    Args:
        candles: Sequence of M5 candles to evaluate.
        is_tokyo_oceania_filter: Optional predicate returning True if a candle falls in the
            Tokyo/Oceania excluded window. Injected to prevent hardcoding invented session hours.

    Returns:
        float: Exact arithmetic mean of candle ranges.

    Raises:
        DetectorError: If candles sequence is empty, contains non-M5 candles, or has 0 valid candles.
    """
    if not candles:
        raise DetectorError("Cannot calculate Average M5 Range on an empty candle sequence")

    valid_ranges = []
    for idx, c in enumerate(candles):
        if not isinstance(c, Candle):
            raise DetectorError(f"Element at index {idx} is not a Candle")
        if c.timeframe != Timeframe.M5:
            raise DetectorError(
                f"Candle at index {idx} has invalid timeframe {c.timeframe.value}; must be M5"
            )

        # Exclude Tokyo/Oceania candles if filter provided
        if is_tokyo_oceania_filter is not None and is_tokyo_oceania_filter(c):
            continue

        candle_range = calculate_candle_range(c)
        valid_ranges.append(candle_range)

    if not valid_ranges:
        raise DetectorError("Zero valid M5 candles available after applying session exclusions")

    return sum(valid_ranges) / len(valid_ranges)
