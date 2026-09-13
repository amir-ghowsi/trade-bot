"""
Proximity Zone and Re-entry Detection (BLK-13).
Strictly implements locked mathematical rules:

BUY Setup:
- Zone Distance: Average_M5_Range
- Exit Condition (must close outside proximity zone):
    M5 Close > BOTTOM_LEVEL + Average_M5_Range
- Re-entry Return Condition (uses LIVE ASK exclusively):
    abs(Live_Ask - BOTTOM_LEVEL) <= Average_M5_Range

SELL Setup:
- Zone Distance: Average_M5_Range
- Exit Condition (must close outside proximity zone):
    M5 Close < TOP_LEVEL - Average_M5_Range
- Re-entry Return Condition (uses LIVE BID exclusively):
    abs(Live_Bid - TOP_LEVEL) <= Average_M5_Range

At least one complete M5 candle must close outside the proximity zone before re-entry is valid.
BUY must use LIVE ASK. SELL must use LIVE BID.
Zero indicators, zero ATR.
"""

from src.core.exceptions import DataValidationError


def is_buy_proximity_exit(m5_close: float, bottom_level: float, average_m5_range: float) -> bool:
    """
    Check if an M5 candle has closed outside the proximity zone for a BUY setup.
    M5 Close > BOTTOM_LEVEL + Average_M5_Range
    """
    if average_m5_range <= 0:
        raise DataValidationError(f"average_m5_range must be strictly positive, got {average_m5_range}")
    if bottom_level <= 0:
        raise DataValidationError(f"bottom_level must be strictly positive, got {bottom_level}")

    return m5_close > (bottom_level + average_m5_range)


def is_buy_proximity_active(live_ask: float, bottom_level: float, average_m5_range: float) -> bool:
    """
    Check if Live Ask is within the proximity zone of BOTTOM_LEVEL for a BUY setup.
    abs(Live_Ask - BOTTOM_LEVEL) <= Average_M5_Range
    Exclusively uses LIVE ASK (never Bid, Last, Close, or Mid).
    """
    if average_m5_range <= 0:
        raise DataValidationError(f"average_m5_range must be strictly positive, got {average_m5_range}")
    if bottom_level <= 0:
        raise DataValidationError(f"bottom_level must be strictly positive, got {bottom_level}")
    if live_ask <= 0:
        raise DataValidationError(f"live_ask must be strictly positive, got {live_ask}")

    return abs(live_ask - bottom_level) <= average_m5_range


def is_sell_proximity_exit(m5_close: float, top_level: float, average_m5_range: float) -> bool:
    """
    Check if an M5 candle has closed outside the proximity zone for a SELL setup.
    M5 Close < TOP_LEVEL - Average_M5_Range
    """
    if average_m5_range <= 0:
        raise DataValidationError(f"average_m5_range must be strictly positive, got {average_m5_range}")
    if top_level <= 0:
        raise DataValidationError(f"top_level must be strictly positive, got {top_level}")

    return m5_close < (top_level - average_m5_range)


def is_sell_proximity_active(live_bid: float, top_level: float, average_m5_range: float) -> bool:
    """
    Check if Live Bid is within the proximity zone of TOP_LEVEL for a SELL setup.
    abs(Live_Bid - TOP_LEVEL) <= Average_M5_Range
    Exclusively uses LIVE BID (never Ask, Last, Close, or Mid).
    """
    if average_m5_range <= 0:
        raise DataValidationError(f"average_m5_range must be strictly positive, got {average_m5_range}")
    if top_level <= 0:
        raise DataValidationError(f"top_level must be strictly positive, got {top_level}")
    if live_bid <= 0:
        raise DataValidationError(f"live_bid must be strictly positive, got {live_bid}")

    return abs(live_bid - top_level) <= average_m5_range
