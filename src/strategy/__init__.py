"""
Strategy Layer - Strategy Detectors Package.
Pure price-action pattern detection adhering strictly to locked strategy specifications.
Zero MT5 imports, zero third-party indicators (ATR, RSI, etc.), strictly deterministic.
"""

from src.strategy.average_range import (
    calculate_average_m5_range,
    calculate_candle_range,
)
from src.strategy.context import (
    SetupContext,
    StateTransitionRecord,
)
from src.strategy.direction import (
    detect_candle_direction,
    is_bearish,
    is_bullish,
    is_neutral,
)
from src.strategy.exceptions import (
    CausalityViolationError,
    DetectorError,
    InvalidationError,
    InvalidStructureError,
    StateTransitionError,
    StrategyError,
)
from src.strategy.gap import (
    detect_gap_in_triplet,
    detect_gaps_in_sequence,
)
from src.strategy.invalidation import (
    is_buy_invalidated_by_m5,
    is_invalidated_by_opposite_spike,
    is_sell_invalidated_by_m5,
)
from src.strategy.pinbar import (
    evaluate_pinbar,
    validate_buy_pinbar,
    validate_sell_pinbar,
)
from src.strategy.proximity import (
    is_buy_proximity_active,
    is_buy_proximity_exit,
    is_sell_proximity_active,
    is_sell_proximity_exit,
)
from src.strategy.spike import (
    generate_spike_id,
    SpikeDetector,
)
from src.strategy.state_machine import (
    BuyStateMachine,
    SellStateMachine,
)
from src.strategy.structure import (
    detect_first_bottom,
    detect_first_top,
)

__all__ = [
    # Direction
    "detect_candle_direction",
    "is_bullish",
    "is_bearish",
    "is_neutral",
    # Gaps
    "detect_gap_in_triplet",
    "detect_gaps_in_sequence",
    # Spikes
    "generate_spike_id",
    "SpikeDetector",
    # Structures and Levels
    "detect_first_bottom",
    "detect_first_top",
    # Average Range
    "calculate_candle_range",
    "calculate_average_m5_range",
    # Pin Bar
    "validate_buy_pinbar",
    "validate_sell_pinbar",
    "evaluate_pinbar",
    # Proximity
    "is_buy_proximity_exit",
    "is_buy_proximity_active",
    "is_sell_proximity_exit",
    "is_sell_proximity_active",
    # Invalidation
    "is_buy_invalidated_by_m5",
    "is_sell_invalidated_by_m5",
    "is_invalidated_by_opposite_spike",
    # State Machine and Context
    "BuyStateMachine",
    "SellStateMachine",
    "SetupContext",
    "StateTransitionRecord",
    # Exceptions
    "StrategyError",
    "DetectorError",
    "CausalityViolationError",
    "InvalidStructureError",
    "StateTransitionError",
    "InvalidationError",
]
