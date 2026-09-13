"""
Bullish Spike and Bearish Spike Detector.
Strictly implements locked mathematical rules:
Bullish Spike:
- At least 3 consecutive bullish candles (Close > Open)
- At least 1 bullish gap (High(C1) < Low(C3))
Bearish Spike:
- At least 3 consecutive bearish candles (Open > Close)
- At least 1 bearish gap (Low(C1) > High(C3))
Neutral candles (Open == Close) strictly break consecutive runs.
Opposite direction candles strictly break consecutive runs.
All calculations are strictly deterministic and causal (no look-ahead).
"""

import hashlib
from typing import List, Optional, Sequence

from src.core.constants import (
    CanonicalSymbol,
    Direction,
    MIN_GAP_COUNT,
    MIN_SPIKE_CANDLES,
)
from src.core.types import Candle, GapEvent, SpikeEvent
from src.strategy.direction import detect_candle_direction
from src.strategy.exceptions import DetectorError
from src.strategy.gap import detect_gaps_in_sequence


def generate_spike_id(
    symbol: CanonicalSymbol,
    direction: Direction,
    start_time: str,
    end_time: str,
    high: float,
    low: float,
) -> str:
    """
    Generate a deterministic spike ID using SHA-256 digest over spike attributes.
    Zero randomness, zero UUID4, purely reproducible.
    """
    symbol_short = "DJ" if symbol == CanonicalSymbol.DOW_JONES else "NAS"
    dir_short = "BULL" if direction == Direction.BULLISH else "BEAR"
    canonical_str = f"{symbol.value}|{direction.value}|{start_time}|{end_time}|{high:.5f}|{low:.5f}"
    digest = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()[:8].upper()
    return f"SPK-{symbol_short}-{dir_short}-{digest}"


class SpikeDetector:
    """
    Deterministic detector for Bullish Spike and Bearish Spike events.
    Operates strictly on CanonicalSymbol and UTC timezone-aware Candle sequences.
    """

    @classmethod
    def evaluate_candidate_run(
        cls,
        candles: Sequence[Candle],
        expected_direction: Optional[Direction] = None,
    ) -> Optional[SpikeEvent]:
        """
        Evaluate an exact contiguous run of candles to check if it qualifies as a SpikeEvent.
        Requirements:
        - len(candles) >= 3
        - All candles must have the exact same non-neutral direction
        - At least 1 gap in that direction
        - Same canonical symbol and timeframe across all candles
        - Monotonic ascending timestamps
        """
        if len(candles) < MIN_SPIKE_CANDLES:
            return None

        first = candles[0]
        if not isinstance(first.symbol, CanonicalSymbol):
            raise DetectorError(f"Candles must use CanonicalSymbol, got {first.symbol}")

        first_dir = detect_candle_direction(first)
        if first_dir == Direction.NEUTRAL:
            return None

        if expected_direction is not None and first_dir != expected_direction:
            return None

        # Verify every candle in the run is strictly directional and matches first_dir
        for i, c in enumerate(candles):
            if c.symbol != first.symbol:
                raise DetectorError(
                    f"Symbol mismatch in spike sequence at index {i}: expected {first.symbol.value}, got {c.symbol.value}"
                )
            if c.timeframe != first.timeframe:
                raise DetectorError(
                    f"Timeframe mismatch in spike sequence at index {i}: expected {first.timeframe.value}, got {c.timeframe.value}"
                )
            if i > 0 and c.open_time <= candles[i - 1].open_time:
                raise DetectorError(
                    f"Causality violation: candles not in strictly ascending order at index {i}"
                )

            c_dir = detect_candle_direction(c)
            if c_dir != first_dir:
                # Neutral or opposite candle breaks the run
                return None

        # Find all gaps within the run matching the direction
        gaps = detect_gaps_in_sequence(candles, direction_filter=first_dir)
        if len(gaps) < MIN_GAP_COUNT:
            return None

        highest_high = max(c.high for c in candles)
        lowest_low = min(c.low for c in candles)
        start_time = candles[0].open_time
        end_time = candles[-1].close_time

        spike_id = generate_spike_id(
            symbol=first.symbol,
            direction=first_dir,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            high=highest_high,
            low=lowest_low,
        )

        return SpikeEvent(
            spike_id=spike_id,
            symbol=first.symbol,
            direction=first_dir,
            start_time=start_time,
            end_time=end_time,
            high=highest_high,
            low=lowest_low,
            candle_count=len(candles),
            gap_count=len(gaps),
        )

    @classmethod
    def find_spikes(
        cls,
        candles: Sequence[Candle],
        expected_direction: Optional[Direction] = None,
    ) -> List[SpikeEvent]:
        """
        Scan a full candle series and identify all maximal contiguous spike runs.
        A spike run continues as long as consecutive candles maintain the same direction.
        Once broken, if length >= 3 and gaps >= 1, a SpikeEvent is recorded.
        """
        if len(candles) < MIN_SPIKE_CANDLES:
            return []

        spikes: List[SpikeEvent] = []
        current_run: List[Candle] = []

        for c in candles:
            c_dir = detect_candle_direction(c)

            if c_dir == Direction.NEUTRAL:
                # Neutral candle ends any current run
                if len(current_run) >= MIN_SPIKE_CANDLES:
                    spike = cls.evaluate_candidate_run(current_run, expected_direction=expected_direction)
                    if spike is not None:
                        spikes.append(spike)
                current_run = []
                continue

            if not current_run:
                if expected_direction is None or c_dir == expected_direction:
                    current_run.append(c)
            else:
                run_dir = detect_candle_direction(current_run[0])
                if c_dir == run_dir:
                    current_run.append(c)
                else:
                    # Direction changed: check if previous run qualified
                    if len(current_run) >= MIN_SPIKE_CANDLES:
                        spike = cls.evaluate_candidate_run(current_run, expected_direction=expected_direction)
                        if spike is not None:
                            spikes.append(spike)
                    # Start new run with current candle
                    if expected_direction is None or c_dir == expected_direction:
                        current_run = [c]
                    else:
                        current_run = []

        # Check trailing run at the end of the series
        if len(current_run) >= MIN_SPIKE_CANDLES:
            spike = cls.evaluate_candidate_run(current_run, expected_direction=expected_direction)
            if spike is not None:
                spikes.append(spike)

        return spikes
