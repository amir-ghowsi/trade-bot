"""
State Machine Context and Transition Models.
Stores all stateful metadata for a single BUY or SELL setup lifecycle.
Completely decoupled from broker SDKs and third-party libraries.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from src.core.clock import ensure_utc
from src.core.constants import (
    BearishState,
    BullishState,
    CanonicalSymbol,
    TradeDirection,
)
from src.core.exceptions import DataValidationError
from src.core.types import (
    Candle,
    PinBarEvent,
    SignalEvent,
    SpikeEvent,
    StructuralLevel,
)


@dataclass(frozen=True)
class StateTransitionRecord:
    """
    Immutable audit record for a state transition.
    """
    from_state: str
    to_state: str
    timestamp: datetime
    event_type: str
    reason: str

    def __post_init__(self) -> None:
        _ = ensure_utc(self.timestamp)
        if not self.from_state or not self.to_state:
            raise DataValidationError("Transition states cannot be empty")


@dataclass
class SetupContext:
    """
    Mutable container for active setup state tracking.
    State transitions mutate this context deterministically.
    """
    symbol: CanonicalSymbol
    direction: TradeDirection
    tick_size: float
    average_m5_range: float
    current_state: Union[BullishState, BearishState]
    spike: Optional[SpikeEvent] = None
    structural_level: Optional[StructuralLevel] = None
    correction_candles: List[Candle] = field(default_factory=list)
    has_left_proximity_zone: bool = False
    proximity_activation_time: Optional[datetime] = None
    pinbar_event: Optional[PinBarEvent] = None
    setup_id: Optional[str] = None
    signal_event: Optional[SignalEvent] = None
    invalidation_reason: Optional[str] = None
    transitions: List[StateTransitionRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, CanonicalSymbol):
            raise DataValidationError(f"symbol must be CanonicalSymbol, got {self.symbol}")
        if not isinstance(self.direction, TradeDirection):
            raise DataValidationError(f"direction must be TradeDirection, got {self.direction}")
        if self.tick_size <= 0:
            raise DataValidationError("tick_size must be strictly positive")
        if self.average_m5_range <= 0:
            raise DataValidationError("average_m5_range must be strictly positive")

    @property
    def is_closed(self) -> bool:
        """Return True if setup reached terminal CLOSED state."""
        return (
            self.current_state == BullishState.CLOSED
            or self.current_state == BearishState.CLOSED
        )

    def record_transition(
        self,
        to_state: Union[BullishState, BearishState],
        timestamp: datetime,
        event_type: str,
        reason: str = "",
    ) -> None:
        """Record an immutable state transition."""
        utc_ts = ensure_utc(timestamp)
        record = StateTransitionRecord(
            from_state=self.current_state.value,
            to_state=to_state.value,
            timestamp=utc_ts,
            event_type=event_type,
            reason=reason,
        )
        self.transitions.append(record)
        self.current_state = to_state
