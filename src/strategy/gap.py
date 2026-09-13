"""
Fair Value Gap Detector.
Strictly implements locked mathematical rules:
Bullish Gap: High(Candle1) < Low(Candle3)
Bearish Gap: Low(Candle1) > High(Candle3)
Strict boundary: High(C1) == Low(C3) or Low(C1) == High(C3) is NOT a gap.
Zero ATR, zero percentage thresholds, zero indicators.
"""

from typing import List, Optional, Sequence

from src.core.constants import CanonicalSymbol, Direction
from src.core.exceptions import DataValidationError
from src.core.types import Candle, GapEvent
from src.strategy.exceptions import DetectorError


def detect_gap_in_triplet(c1: Candle, c2: Candle, c3: Candle) -> Optional[GapEvent]:
    """
    Evaluate three consecutive chronological candles for a Fair Value Gap.
    Candles must belong to the same canonical symbol and timeframe, in strictly ascending time order.
    """
    if not (isinstance(c1, Candle) and isinstance(c2, Candle) and isinstance(c3, Candle)):
        raise DetectorError("All three elements must be Candle instances")

    if c1.symbol != c2.symbol or c2.symbol != c3.symbol:
        raise DetectorError(
            f"Symbol mismatch across triplet: {c1.symbol.value}, {c2.symbol.value}, {c3.symbol.value}"
        )

    if c1.timeframe != c2.timeframe or c2.timeframe != c3.timeframe:
        raise DetectorError(
            f"Timeframe mismatch across triplet: {c1.timeframe.value}, {c2.timeframe.value}, {c3.timeframe.value}"
        )

    if not (c1.open_time < c2.open_time < c3.open_time):
        raise DetectorError("Candles in triplet must be in strictly ascending chronological order")

    # Bullish Gap: High(Candle1) < Low(Candle3)
    if c1.high < c3.low:
        return GapEvent(
            direction=Direction.BULLISH,
            c1_high=c1.high,
            c1_low=c1.low,
            c3_high=c3.high,
            c3_low=c3.low,
            c1_time=c1.open_time,
            c3_time=c3.open_time,
        )

    # Bearish Gap: Low(Candle1) > High(Candle3)
    if c1.low > c3.high:
        return GapEvent(
            direction=Direction.BEARISH,
            c1_high=c1.high,
            c1_low=c1.low,
            c3_high=c3.high,
            c3_low=c3.low,
            c1_time=c1.open_time,
            c3_time=c3.open_time,
        )

    # Equality or overlap is strictly not a gap
    return None


def detect_gaps_in_sequence(
    candles: Sequence[Candle],
    direction_filter: Optional[Direction] = None,
) -> List[GapEvent]:
    """
    Scan a sequence of chronologically ordered candles for all 3-candle Fair Value Gaps.
    Optionally filter by Direction.BULLISH or Direction.BEARISH.
    """
    if len(candles) < 3:
        return []

    gaps: List[GapEvent] = []
    for i in range(len(candles) - 2):
        gap = detect_gap_in_triplet(candles[i], candles[i + 1], candles[i + 2])
        if gap is not None:
            if direction_filter is None or gap.direction == direction_filter:
                gaps.append(gap)

    return gaps
