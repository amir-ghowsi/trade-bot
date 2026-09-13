"""
M1 Pin Bar Detection and Geometric Validation.
Strictly implements locked mathematical rules:
BUY Setup:
- Requires Bearish M1 Pin Bar
- Open > Close
- Range = High - Low
- Body = abs(Open - Close)
- BodyRatio = Body / Range <= 0.20
- MidPoint = Low + 0.5 * Range
- BodyHigh = max(Open, Close) <= MidPoint

SELL Setup:
- Requires Bullish M1 Pin Bar
- Close > Open
- Range = High - Low
- Body = abs(Open - Close)
- BodyRatio = Body / Range <= 0.20
- MidPoint = Low + 0.5 * Range
- BodyLow = min(Open, Close) >= MidPoint

Zero indicators, zero ATR, zero external dependencies.
"""

from typing import Union

from src.core.constants import (
    Direction,
    PINBAR_MAX_BODY_RATIO,
    PINBAR_MIDPOINT_RATIO,
    Timeframe,
    TradeDirection,
)
from src.core.exceptions import DataValidationError
from src.core.types import Candle, PinBarEvent


def validate_buy_pinbar(candle: Candle) -> PinBarEvent:
    """
    Validate an M1 candle as a Bearish Pin Bar required for a BUY setup.
    All conditions are mandatory:
    1. Timeframe == M1
    2. Open > Close (Bearish)
    3. Range = High - Low > 0
    4. BodyRatio = Body / Range <= 0.20
    5. BodyHigh = max(Open, Close) <= Midpoint (Low + 0.5 * Range)
    """
    if candle.timeframe != Timeframe.M1:
        raise DataValidationError(f"PinBar validation requires M1 timeframe, got {candle.timeframe}")

    candle_range = candle.high - candle.low
    if candle_range <= 0:
        return PinBarEvent(
            symbol=candle.symbol,
            timeframe=Timeframe.M1,
            close_time=candle.close_time,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            range=0.0,
            body=0.0,
            body_ratio=0.0,
            midpoint=candle.low,
            is_valid=False,
            direction=Direction.BEARISH,
        )

    body = abs(candle.open - candle.close)
    body_ratio = body / candle_range
    midpoint = candle.low + PINBAR_MIDPOINT_RATIO * candle_range
    body_high = max(candle.open, candle.close)

    # All conditions are mandatory
    is_valid = (
        (candle.open > candle.close)
        and (body_ratio <= PINBAR_MAX_BODY_RATIO)
        and (body_high <= midpoint)
    )

    return PinBarEvent(
        symbol=candle.symbol,
        timeframe=Timeframe.M1,
        close_time=candle.close_time,
        open=candle.open,
        high=candle.high,
        low=candle.low,
        close=candle.close,
        range=candle_range,
        body=body,
        body_ratio=body_ratio,
        midpoint=midpoint,
        is_valid=is_valid,
        direction=Direction.BEARISH,
    )


def validate_sell_pinbar(candle: Candle) -> PinBarEvent:
    """
    Validate an M1 candle as a Bullish Pin Bar required for a SELL setup.
    All conditions are mandatory:
    1. Timeframe == M1
    2. Close > Open (Bullish)
    3. Range = High - Low > 0
    4. BodyRatio = Body / Range <= 0.20
    5. BodyLow = min(Open, Close) >= Midpoint (Low + 0.5 * Range)
    """
    if candle.timeframe != Timeframe.M1:
        raise DataValidationError(f"PinBar validation requires M1 timeframe, got {candle.timeframe}")

    candle_range = candle.high - candle.low
    if candle_range <= 0:
        return PinBarEvent(
            symbol=candle.symbol,
            timeframe=Timeframe.M1,
            close_time=candle.close_time,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            range=0.0,
            body=0.0,
            body_ratio=0.0,
            midpoint=candle.low,
            is_valid=False,
            direction=Direction.BULLISH,
        )

    body = abs(candle.open - candle.close)
    body_ratio = body / candle_range
    midpoint = candle.low + PINBAR_MIDPOINT_RATIO * candle_range
    body_low = min(candle.open, candle.close)

    # All conditions are mandatory
    is_valid = (
        (candle.close > candle.open)
        and (body_ratio <= PINBAR_MAX_BODY_RATIO)
        and (body_low >= midpoint)
    )

    return PinBarEvent(
        symbol=candle.symbol,
        timeframe=Timeframe.M1,
        close_time=candle.close_time,
        open=candle.open,
        high=candle.high,
        low=candle.low,
        close=candle.close,
        range=candle_range,
        body=body,
        body_ratio=body_ratio,
        midpoint=midpoint,
        is_valid=is_valid,
        direction=Direction.BULLISH,
    )


def evaluate_pinbar(candle: Candle, setup_direction: Union[TradeDirection, str]) -> PinBarEvent:
    """
    Evaluate an M1 candle according to the active setup direction.
    BUY requires a Bearish Pin Bar; SELL requires a Bullish Pin Bar.
    """
    if isinstance(setup_direction, str):
        direction_val = setup_direction.upper().strip()
    elif isinstance(setup_direction, TradeDirection):
        direction_val = setup_direction.value
    else:
        raise DataValidationError(f"setup_direction must be TradeDirection or str, got {setup_direction}")

    if direction_val == TradeDirection.BUY.value:
        return validate_buy_pinbar(candle)
    elif direction_val == TradeDirection.SELL.value:
        return validate_sell_pinbar(candle)
    else:
        raise DataValidationError(f"Invalid setup direction: {setup_direction}. Must be BUY or SELL.")
