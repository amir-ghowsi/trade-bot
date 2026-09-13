"""
Setup Invalidation Rules (BLK-14).
Strictly implements locked mathematical rules:

BUY Setup Invalidation:
- M5 Close < BOTTOM_LEVEL
- Wick alone does NOT invalidate (Low < BOTTOM_LEVEL with Close >= BOTTOM_LEVEL is NOT invalidation)

SELL Setup Invalidation:
- M5 Close > TOP_LEVEL
- Wick alone does NOT invalidate (High > TOP_LEVEL with Close <= TOP_LEVEL is NOT invalidation)

Additional Invalidation Conditions:
1. Opposite valid spike occurs:
   - In a BUY setup: A valid Bearish Spike occurs.
   - In a SELL setup: A valid Bullish Spike occurs.
2. Trading session ends before execution.
3. Active position conflict makes the setup non-executable.

MAX_SETUP_LIFETIME is DISABLED (zero timeout invented).
"""

from typing import Optional

from src.core.constants import Direction, TradeDirection
from src.core.exceptions import DataValidationError
from src.core.types import Candle, SpikeEvent


def is_buy_invalidated_by_m5(candle: Candle, bottom_level: float) -> bool:
    """
    Check if a BUY setup is invalidated by an M5 candle.
    Locked Rule: Invalidation requires M5 Close < BOTTOM_LEVEL.
    Wick alone does NOT invalidate (candle.low < bottom_level with candle.close >= bottom_level is valid).
    """
    if bottom_level <= 0:
        raise DataValidationError(f"bottom_level must be strictly positive, got {bottom_level}")

    return candle.close < bottom_level


def is_sell_invalidated_by_m5(candle: Candle, top_level: float) -> bool:
    """
    Check if a SELL setup is invalidated by an M5 candle.
    Locked Rule: Invalidation requires M5 Close > TOP_LEVEL.
    Wick alone does NOT invalidate (candle.high > top_level with candle.close <= top_level is valid).
    """
    if top_level <= 0:
        raise DataValidationError(f"top_level must be strictly positive, got {top_level}")

    return candle.close > top_level


def is_invalidated_by_opposite_spike(
    current_direction: TradeDirection,
    spike: SpikeEvent,
) -> bool:
    """
    Check if a valid spike opposes the active setup direction:
    - BUY setup is invalidated by a BEARISH spike.
    - SELL setup is invalidated by a BULLISH spike.
    """
    if current_direction == TradeDirection.BUY:
        return spike.direction == Direction.BEARISH
    elif current_direction == TradeDirection.SELL:
        return spike.direction == Direction.BULLISH
    else:
        raise DataValidationError(f"Invalid TradeDirection: {current_direction}")
