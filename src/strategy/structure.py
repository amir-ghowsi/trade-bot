"""
Market Structure & Level Detector.
Strictly implements locked mathematical rules:
- Initial Correction:
  - Bullish structure: Bullish Spike -> Initial Bearish Correction -> First Bottom.
  - Bearish structure: Bearish Spike -> Initial Bullish Correction -> First Top.
  - Neutral candles do not count as correction-direction candles.
  - No arbitrary minimum correction candle count.
- First Bottom (LOCKED RULE BLK-02):
  - Confirmed by: BREAK OF THE CORRECTION CANDLE HIGH (subsequent candle high > correction candle high).
  - BOTTOM_LEVEL = First Bottom Low (lowest low of the correction).
  - Once formed: BOTTOM_LEVEL is FIXED and IMMUTABLE (is_locked=True).
- First Top (LOCKED RULE BLK-03):
  - Confirmed by: BREAK OF THE CORRECTION CANDLE LOW (subsequent candle low < correction candle low).
  - TOP_LEVEL = First Top High (highest high of the correction).
  - Once formed: TOP_LEVEL is FIXED and IMMUTABLE (is_locked=True).
Zero ATR, zero indicators, zero look-ahead.
"""

from typing import List, Optional, Sequence

from src.core.constants import CanonicalSymbol, Direction
from src.core.types import Candle, SpikeEvent, StructuralLevel
from src.strategy.direction import detect_candle_direction
from src.strategy.exceptions import DetectorError, InvalidStructureError


def detect_first_bottom(
    spike: SpikeEvent,
    post_spike_candles: Sequence[Candle],
) -> Optional[StructuralLevel]:
    """
    Detect confirmation of First Bottom following a Bullish Spike (BLK-02).

    Rules:
    1. Spike must be BULLISH.
    2. Post-spike candles must be in chronological order, strictly after spike.end_time.
    3. Initial correction must contain at least one bearish candle (Close < Open).
       Neutral candles do not count as correction candles.
    4. First Bottom is confirmed when a subsequent candle breaks the high of the
       correction candle (confirming_candle.high > correction_candle.high).
    5. Structural Level price is the lowest low of the correction (First Bottom Low).
    6. Returns a fixed, immutable StructuralLevel with is_locked=True.
    """
    if not isinstance(spike, SpikeEvent):
        raise DetectorError(f"Expected SpikeEvent, got {type(spike).__name__}")
    if spike.direction != Direction.BULLISH:
        raise InvalidStructureError(f"First Bottom requires BULLISH spike, got {spike.direction}")

    if not post_spike_candles:
        return None

    # Validate post-spike candles
    correction_candles: List[Candle] = []

    for idx, c in enumerate(post_spike_candles):
        if c.symbol != spike.symbol:
            raise DetectorError(f"Symbol mismatch: expected {spike.symbol}, got {c.symbol}")
        if idx == 0 and c.open_time < spike.end_time:
            raise DetectorError("Post-spike candles cannot start before spike end_time")
        if idx > 0 and c.open_time <= post_spike_candles[idx - 1].open_time:
            raise DetectorError(f"Candles not in strictly ascending chronological order at index {idx}")

        c_dir = detect_candle_direction(c)

        if not correction_candles:
            # We are waiting for the initial correction to begin with a bearish candle
            if c_dir == Direction.BEARISH:
                correction_candles.append(c)
            # Neutral candles or continued bullish candles before correction starts do not form correction
            continue

        # Correction has already started.
        # Check if the current candle confirms First Bottom by breaking the correction candle high (BLK-02).
        last_corr = correction_candles[-1]
        if c.high > last_corr.high:
            # First Bottom Confirmed!
            # The First Bottom candle is the correction candle whose high was broken.
            # BOTTOM_LEVEL is strictly the Low of this First Bottom candle.
            first_bottom_low = last_corr.low
            return StructuralLevel(
                spike_id=spike.spike_id,
                symbol=spike.symbol,
                direction=Direction.BULLISH,
                level_type="BOTTOM",
                price=first_bottom_low,
                formation_time=c.close_time,
                is_locked=True,
            )

        # If not breaking high, does it continue the bearish correction?
        if c_dir == Direction.BEARISH:
            correction_candles.append(c)
        # Neutral candles do not count as correction candles, but do not necessarily break it unless confirmed

    return None


def detect_first_top(
    spike: SpikeEvent,
    post_spike_candles: Sequence[Candle],
) -> Optional[StructuralLevel]:
    """
    Detect confirmation of First Top following a Bearish Spike (BLK-03).

    Rules:
    1. Spike must be BEARISH.
    2. Post-spike candles must be in chronological order, strictly after spike.end_time.
    3. Initial correction must contain at least one bullish candle (Close > Open).
       Neutral candles do not count as correction candles.
    4. First Top is confirmed when a subsequent candle breaks the low of the
       correction candle (confirming_candle.low < correction_candle.low).
    5. Structural Level price is the highest high of the correction (First Top High).
    6. Returns a fixed, immutable StructuralLevel with is_locked=True.
    """
    if not isinstance(spike, SpikeEvent):
        raise DetectorError(f"Expected SpikeEvent, got {type(spike).__name__}")
    if spike.direction != Direction.BEARISH:
        raise InvalidStructureError(f"First Top requires BEARISH spike, got {spike.direction}")

    if not post_spike_candles:
        return None

    correction_candles: List[Candle] = []

    for idx, c in enumerate(post_spike_candles):
        if c.symbol != spike.symbol:
            raise DetectorError(f"Symbol mismatch: expected {spike.symbol}, got {c.symbol}")
        if idx == 0 and c.open_time < spike.end_time:
            raise DetectorError("Post-spike candles cannot start before spike end_time")
        if idx > 0 and c.open_time <= post_spike_candles[idx - 1].open_time:
            raise DetectorError(f"Candles not in strictly ascending chronological order at index {idx}")

        c_dir = detect_candle_direction(c)

        if not correction_candles:
            # We are waiting for the initial correction to begin with a bullish candle
            if c_dir == Direction.BULLISH:
                correction_candles.append(c)
            continue

        # Correction has started. Check BLK-03: break of the correction candle low
        last_corr = correction_candles[-1]
        if c.low < last_corr.low:
            # First Top Confirmed!
            # The First Top candle is the correction candle whose low was broken.
            # TOP_LEVEL is strictly the High of this First Top candle.
            first_top_high = last_corr.high
            return StructuralLevel(
                spike_id=spike.spike_id,
                symbol=spike.symbol,
                direction=Direction.BEARISH,
                level_type="TOP",
                price=first_top_high,
                formation_time=c.close_time,
                is_locked=True,
            )

        if c_dir == Direction.BULLISH:
            correction_candles.append(c)

    return None
