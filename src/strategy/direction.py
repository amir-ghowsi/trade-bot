"""
Candle Direction Detector.
Strictly implements locked mathematical rules:
Bullish: Close > Open
Bearish: Open > Close
Neutral: Open == Close
Neutral candles MUST NOT count toward same-direction spike candle counts.
No indicators, no ATR, no body percentage, no volume.
"""

from src.core.constants import Direction
from src.core.types import Candle
from src.strategy.exceptions import DetectorError


def detect_candle_direction(candle: Candle) -> Direction:
    """
    Determine candle direction strictly from Open and Close prices.
    Bullish: Close > Open
    Bearish: Open > Close
    Neutral: Open == Close
    """
    if not isinstance(candle, Candle):
        raise DetectorError(f"Expected Candle instance, got {type(candle).__name__}")

    if candle.close > candle.open:
        return Direction.BULLISH
    elif candle.close < candle.open:
        return Direction.BEARISH
    return Direction.NEUTRAL


def is_bullish(candle: Candle) -> bool:
    """Check if candle is strictly bullish (Close > Open)."""
    return detect_candle_direction(candle) == Direction.BULLISH


def is_bearish(candle: Candle) -> bool:
    """Check if candle is strictly bearish (Open > Close)."""
    return detect_candle_direction(candle) == Direction.BEARISH


def is_neutral(candle: Candle) -> bool:
    """Check if candle is neutral (Open == Close). Neutral candles never count toward directional runs."""
    return detect_candle_direction(candle) == Direction.NEUTRAL
