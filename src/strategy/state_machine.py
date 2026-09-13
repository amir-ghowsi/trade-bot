"""
Strategy State Machines and Signal Generation.
Strictly implements the locked strategy lifecycle for BUY and SELL setups:

BUY Setup Lifecycle:
IDLE
-> BULLISH_SPIKE_DETECTED
-> WAIT_INITIAL_CORRECTION
-> FIRST_BOTTOM_FORMED
-> WAIT_RETURN_TO_FIRST_BOTTOM
-> M1_MONITORING
-> BEARISH_PINBAR_DETECTED
-> BUY_TRIGGER_ARMED (Emits SignalEvent)
-> CLOSED

SELL Setup Lifecycle:
IDLE
-> BEARISH_SPIKE_DETECTED
-> WAIT_INITIAL_CORRECTION
-> FIRST_TOP_FORMED
-> WAIT_RETURN_TO_FIRST_TOP
-> M1_MONITORING
-> BULLISH_PINBAR_DETECTED
-> SELL_TRIGGER_ARMED (Emits SignalEvent)
-> CLOSED

Zero MT5 dependencies, zero indicators, completely broker-agnostic.
"""

from datetime import datetime
from typing import Optional, Union

from src.core.clock import ensure_utc
from src.core.constants import (
    BearishState,
    BullishState,
    CanonicalSymbol,
    Direction,
    TradeDirection,
)
from src.core.exceptions import DataValidationError
from src.core.types import (
    Candle,
    Quote,
    SignalEvent,
    SpikeEvent,
    StructuralLevel,
    generate_setup_id,
)
from src.market.types import Tick
from src.strategy.context import SetupContext
from src.strategy.direction import detect_candle_direction
from src.strategy.exceptions import (
    CausalityViolationError,
    DetectorError,
    StateTransitionError,
)
from src.strategy.invalidation import (
    is_buy_invalidated_by_m5,
    is_invalidated_by_opposite_spike,
    is_sell_invalidated_by_m5,
)
from src.strategy.pinbar import validate_buy_pinbar, validate_sell_pinbar
from src.strategy.proximity import (
    is_buy_proximity_active,
    is_buy_proximity_exit,
    is_sell_proximity_active,
    is_sell_proximity_exit,
)

PRE_TRADE_BUY_STATES = frozenset({
    BullishState.IDLE,
    BullishState.BULLISH_SPIKE_DETECTED,
    BullishState.WAIT_INITIAL_CORRECTION,
    BullishState.FIRST_BOTTOM_FORMED,
    BullishState.WAIT_RETURN_TO_FIRST_BOTTOM,
    BullishState.M1_MONITORING,
    BullishState.BEARISH_PINBAR_DETECTED,
    BullishState.BUY_TRIGGER_ARMED,
})

PRE_TRADE_SELL_STATES = frozenset({
    BearishState.IDLE,
    BearishState.BEARISH_SPIKE_DETECTED,
    BearishState.WAIT_INITIAL_CORRECTION,
    BearishState.FIRST_TOP_FORMED,
    BearishState.WAIT_RETURN_TO_FIRST_TOP,
    BearishState.M1_MONITORING,
    BearishState.BULLISH_PINBAR_DETECTED,
    BearishState.SELL_TRIGGER_ARMED,
})


class BuyStateMachine:
    """
    Deterministic State Machine for BUY Setup Lifecycle.
    Uses BullishState enumeration from Phase 1.
    """

    def __init__(
        self,
        symbol: CanonicalSymbol,
        tick_size: float,
        average_m5_range: float,
        spike: Optional[SpikeEvent] = None,
    ) -> None:
        if not isinstance(symbol, CanonicalSymbol):
            raise DataValidationError(f"symbol must be CanonicalSymbol, got {symbol}")
        if tick_size <= 0:
            raise DataValidationError("tick_size must be strictly positive")
        if average_m5_range <= 0:
            raise DataValidationError("average_m5_range must be strictly positive")

        self.context = SetupContext(
            symbol=symbol,
            direction=TradeDirection.BUY,
            tick_size=tick_size,
            average_m5_range=average_m5_range,
            current_state=BullishState.IDLE,
        )

        if spike is not None:
            self.on_spike(spike)

    @property
    def current_state(self) -> BullishState:
        return self.context.current_state  # type: ignore

    @property
    def is_closed(self) -> bool:
        return self.context.current_state == BullishState.CLOSED

    @property
    def signal_event(self) -> Optional[SignalEvent]:
        return self.context.signal_event

    @property
    def structural_level(self) -> Optional[StructuralLevel]:
        return self.context.structural_level

    def on_spike(self, spike: SpikeEvent) -> None:
        """Process an initial Bullish Spike."""
        if self.is_closed:
            return

        if not isinstance(spike, SpikeEvent):
            raise DataValidationError(f"Expected SpikeEvent, got {type(spike).__name__}")
        if spike.symbol != self.context.symbol:
            raise DataValidationError(
                f"Spike symbol {spike.symbol} does not match state machine symbol {self.context.symbol}"
            )
        if spike.direction != Direction.BULLISH:
            raise DataValidationError(
                f"BUY setup requires BULLISH spike, got {spike.direction}"
            )

        if self.context.current_state != BullishState.IDLE:
            raise StateTransitionError(
                f"Cannot accept spike in state {self.context.current_state}"
            )

        self.context.spike = spike
        self.context.record_transition(
            to_state=BullishState.BULLISH_SPIKE_DETECTED,
            timestamp=spike.end_time,
            event_type="BULLISH_SPIKE",
            reason=f"Spike {spike.spike_id} detected with {spike.candle_count} candles",
        )

    def on_m5_candle(self, candle: Candle) -> None:
        """Process an incoming M5 candle."""
        if self.context.current_state not in PRE_TRADE_BUY_STATES:
            return

        if candle.symbol != self.context.symbol:
            raise DataValidationError(f"Candle symbol {candle.symbol} does not match state machine symbol {self.context.symbol}")

        # Invalidation check (BLK-14): Only active after First Bottom has formed
        if self.context.structural_level is not None:
            if is_buy_invalidated_by_m5(candle, self.context.structural_level.price):
                self.invalidate(
                    reason="M5_CLOSE_BELOW_BOTTOM_LEVEL",
                    timestamp=candle.close_time,
                )
                return

        # State: BULLISH_SPIKE_DETECTED -> transition to WAIT_INITIAL_CORRECTION
        if self.context.current_state == BullishState.BULLISH_SPIKE_DETECTED:
            self.context.record_transition(
                to_state=BullishState.WAIT_INITIAL_CORRECTION,
                timestamp=candle.open_time,
                event_type="CORRECTION_STARTED",
                reason="Processing initial correction candles",
            )
            self._handle_m5_in_correction(candle)
            return

        # State: WAIT_INITIAL_CORRECTION
        if self.context.current_state == BullishState.WAIT_INITIAL_CORRECTION:
            self._handle_m5_in_correction(candle)
            return

        # State: WAIT_RETURN_TO_FIRST_BOTTOM
        if self.context.current_state == BullishState.WAIT_RETURN_TO_FIRST_BOTTOM:
            # Check if price has left the proximity zone
            bottom_level = self.context.structural_level.price  # type: ignore
            if is_buy_proximity_exit(candle.close, bottom_level, self.context.average_m5_range):
                self.context.has_left_proximity_zone = True
            return

        # State: M1_MONITORING
        # In this state, M5 candle still performs invalidation checks above.

    def _handle_m5_in_correction(self, candle: Candle) -> None:
        """Handle M5 candle evaluation during initial correction."""
        direction = detect_candle_direction(candle)

        if direction == Direction.BEARISH:
            # Bearish correction candle: append to collection
            self.context.correction_candles.append(candle)
            return

        if direction == Direction.BULLISH:
            # Potential confirmation if we have at least one bearish correction candle
            if len(self.context.correction_candles) > 0:
                last_corr = self.context.correction_candles[-1]
                if candle.high > last_corr.high:
                    # First Bottom Confirmed!
                    # BLK-02: BOTTOM_LEVEL is strictly the Low of the First Bottom correction candle
                    bottom_level = last_corr.low
                    spike = self.context.spike
                    assert spike is not None

                    struct_level = StructuralLevel(
                        spike_id=spike.spike_id,
                        symbol=self.context.symbol,
                        direction=Direction.BULLISH,
                        level_type="FIRST_BOTTOM",
                        price=bottom_level,
                        formation_time=candle.close_time,
                        is_locked=True,
                    )
                    self.context.structural_level = struct_level

                    # Deterministic Setup ID generation
                    setup_id = generate_setup_id(
                        symbol=self.context.symbol,
                        direction=TradeDirection.BUY,
                        spike_start_time=spike.start_time,
                        spike_end_time=spike.end_time,
                        level_formed_time=candle.close_time,
                        structural_price=bottom_level,
                        tick_size=self.context.tick_size,
                    )
                    self.context.setup_id = setup_id

                    self.context.record_transition(
                        to_state=BullishState.FIRST_BOTTOM_FORMED,
                        timestamp=candle.close_time,
                        event_type="FIRST_BOTTOM_CONFIRMED",
                        reason=f"High {candle.high} broke correction high {last_corr.high}. Bottom Level = {bottom_level}",
                    )

                    # Transition to return monitoring
                    self.context.record_transition(
                        to_state=BullishState.WAIT_RETURN_TO_FIRST_BOTTOM,
                        timestamp=candle.close_time,
                        event_type="AWAITING_PROXIMITY_RETURN",
                        reason="Monitoring for exit and subsequent re-entry",
                    )

                    # Check if the confirming candle itself already closed outside the proximity zone
                    if is_buy_proximity_exit(candle.close, bottom_level, self.context.average_m5_range):
                        self.context.has_left_proximity_zone = True

    def on_quote(self, quote: Union[Quote, Tick]) -> None:
        """
        Process incoming live quote/tick.
        Evaluates proximity re-entry using LIVE ASK exclusively.
        """
        if self.is_closed:
            return

        if quote.symbol != self.context.symbol:
            raise DataValidationError(f"Quote symbol {quote.symbol} does not match state machine symbol {self.context.symbol}")

        if self.context.current_state != BullishState.WAIT_RETURN_TO_FIRST_BOTTOM:
            return

        # Proximity Re-entry Rule (BLK-13):
        # Must have closed at least one complete M5 candle outside the proximity zone first.
        if not self.context.has_left_proximity_zone:
            return

        bottom_level = self.context.structural_level.price  # type: ignore
        live_ask = quote.ask

        # BUY proximity uses LIVE ASK exclusively
        if is_buy_proximity_active(live_ask, bottom_level, self.context.average_m5_range):
            self.context.proximity_activation_time = ensure_utc(quote.timestamp)
            self.context.record_transition(
                to_state=BullishState.M1_MONITORING,
                timestamp=quote.timestamp,
                event_type="PROXIMITY_ACTIVE",
                reason=f"Live Ask {live_ask} entered proximity zone [<= {self.context.average_m5_range}] of level {bottom_level}",
            )

    def on_tick(self, tick: Union[Quote, Tick]) -> None:
        """Alias for on_quote."""
        self.on_quote(tick)

    def on_m1_candle(self, candle: Candle) -> Optional[SignalEvent]:
        """
        Process incoming M1 candle during M1_MONITORING state.
        Only candles with close_time > proximity_activation_time are evaluated.
        """
        if self.is_closed:
            return None

        if candle.symbol != self.context.symbol:
            raise DataValidationError(f"Candle symbol {candle.symbol} does not match state machine symbol {self.context.symbol}")

        if self.context.current_state != BullishState.M1_MONITORING:
            return None

        act_time = self.context.proximity_activation_time
        assert act_time is not None

        # Causality / Look-Ahead Rule (Section 8):
        # Only M1.close_time > proximity_activation_time may be evaluated.
        # Any M1.close_time <= proximity_activation_time MUST be ignored.
        if candle.close_time <= act_time:
            return None

        # Validate Bearish Pin Bar (Section 9)
        pinbar = validate_buy_pinbar(candle)
        if not pinbar.is_valid:
            return None

        self.context.pinbar_event = pinbar
        self.context.record_transition(
            to_state=BullishState.BEARISH_PINBAR_DETECTED,
            timestamp=candle.close_time,
            event_type="BEARISH_PINBAR",
            reason=f"Valid bearish pin bar: body_ratio={pinbar.body_ratio:.4f}, body_high={pinbar.open} <= midpoint={pinbar.midpoint}",
        )

        # Generate SignalEvent immediately (BLK-04)
        spike = self.context.spike
        assert spike is not None
        assert self.context.setup_id is not None

        signal = SignalEvent(
            setup_id=self.context.setup_id,
            symbol=self.context.symbol,
            direction=TradeDirection.BUY,
            signal_time=candle.close_time,
            pinbar_high=candle.high,
            pinbar_low=candle.low,
            final_structural_target=spike.high,
        )
        self.context.signal_event = signal

        self.context.record_transition(
            to_state=BullishState.BUY_TRIGGER_ARMED,
            timestamp=candle.close_time,
            event_type="SIGNAL_GENERATED",
            reason=f"BUY Signal generated with target {spike.high}",
        )
        return signal

    def on_opposite_spike(self, spike: SpikeEvent) -> None:
        """Invalidate setup if an opposite (BEARISH) spike occurs before execution."""
        if self.context.current_state not in PRE_TRADE_BUY_STATES:
            return

        if is_invalidated_by_opposite_spike(TradeDirection.BUY, spike):
            self.invalidate(
                reason="OPPOSITE_SPIKE_DETECTED",
                timestamp=spike.end_time,
            )

    def on_session_end(self, timestamp: datetime) -> None:
        """Invalidate setup if trading session ends before execution."""
        if self.context.current_state not in PRE_TRADE_BUY_STATES:
            return

        self.invalidate(
            reason="TRADING_SESSION_ENDED",
            timestamp=timestamp,
        )

    def on_position_conflict(self, reason: str, timestamp: datetime) -> None:
        """Invalidate setup if an active position conflict prevents execution."""
        if self.context.current_state not in PRE_TRADE_BUY_STATES:
            return

        self.invalidate(
            reason=f"POSITION_CONFLICT: {reason}",
            timestamp=timestamp,
        )

    def on_order_executed(self, executed_price: float, timestamp: datetime) -> None:
        """Process order execution confirmation from execution layer."""
        if self.is_closed:
            return
        if self.context.current_state != BullishState.BUY_TRIGGER_ARMED:
            raise StateTransitionError(
                f"Cannot execute order from state {self.context.current_state}"
            )
        utc_ts = ensure_utc(timestamp)
        self.context.record_transition(
            to_state=BullishState.BUY_EXECUTED,
            timestamp=utc_ts,
            event_type="ORDER_FILLED",
            reason=f"BUY order executed at {executed_price}",
        )
        self.context.record_transition(
            to_state=BullishState.MANAGE_TRADE,
            timestamp=utc_ts,
            event_type="TRADE_ACTIVE",
            reason="Position active, managing stop loss and profit targets",
        )

    def on_trade_closed(self, reason: str, timestamp: datetime) -> None:
        """Process trade closure from position management layer."""
        if self.is_closed:
            return
        utc_ts = ensure_utc(timestamp)
        self.context.record_transition(
            to_state=BullishState.CLOSED,
            timestamp=utc_ts,
            event_type="TRADE_COMPLETED",
            reason=reason,
        )

    def invalidate(self, reason: str, timestamp: datetime) -> None:
        """Transition setup to CLOSED terminal state (pre-trade only)."""
        if self.context.current_state not in PRE_TRADE_BUY_STATES:
            return

        utc_ts = ensure_utc(timestamp)
        self.context.invalidation_reason = reason
        self.context.record_transition(
            to_state=BullishState.CLOSED,
            timestamp=utc_ts,
            event_type="SETUP_INVALIDATED",
            reason=reason,
        )


class SellStateMachine:
    """
    Deterministic State Machine for SELL Setup Lifecycle.
    Uses BearishState enumeration from Phase 1.
    """

    def __init__(
        self,
        symbol: CanonicalSymbol,
        tick_size: float,
        average_m5_range: float,
        spike: Optional[SpikeEvent] = None,
    ) -> None:
        if not isinstance(symbol, CanonicalSymbol):
            raise DataValidationError(f"symbol must be CanonicalSymbol, got {symbol}")
        if tick_size <= 0:
            raise DataValidationError("tick_size must be strictly positive")
        if average_m5_range <= 0:
            raise DataValidationError("average_m5_range must be strictly positive")

        self.context = SetupContext(
            symbol=symbol,
            direction=TradeDirection.SELL,
            tick_size=tick_size,
            average_m5_range=average_m5_range,
            current_state=BearishState.IDLE,
        )

        if spike is not None:
            self.on_spike(spike)

    @property
    def current_state(self) -> BearishState:
        return self.context.current_state  # type: ignore

    @property
    def is_closed(self) -> bool:
        return self.context.current_state == BearishState.CLOSED

    @property
    def signal_event(self) -> Optional[SignalEvent]:
        return self.context.signal_event

    @property
    def structural_level(self) -> Optional[StructuralLevel]:
        return self.context.structural_level

    def on_spike(self, spike: SpikeEvent) -> None:
        """Process an initial Bearish Spike."""
        if self.is_closed:
            return

        if not isinstance(spike, SpikeEvent):
            raise DataValidationError(f"Expected SpikeEvent, got {type(spike).__name__}")
        if spike.symbol != self.context.symbol:
            raise DataValidationError(
                f"Spike symbol {spike.symbol} does not match state machine symbol {self.context.symbol}"
            )
        if spike.direction != Direction.BEARISH:
            raise DataValidationError(
                f"SELL setup requires BEARISH spike, got {spike.direction}"
            )

        if self.context.current_state != BearishState.IDLE:
            raise StateTransitionError(
                f"Cannot accept spike in state {self.context.current_state}"
            )

        self.context.spike = spike
        self.context.record_transition(
            to_state=BearishState.BEARISH_SPIKE_DETECTED,
            timestamp=spike.end_time,
            event_type="BEARISH_SPIKE",
            reason=f"Spike {spike.spike_id} detected with {spike.candle_count} candles",
        )

    def on_m5_candle(self, candle: Candle) -> None:
        """Process an incoming M5 candle."""
        if self.context.current_state not in PRE_TRADE_SELL_STATES:
            return

        if candle.symbol != self.context.symbol:
            raise DataValidationError(f"Candle symbol {candle.symbol} does not match state machine symbol {self.context.symbol}")

        # Invalidation check (BLK-14): Only active after First Top has formed
        if self.context.structural_level is not None:
            if is_sell_invalidated_by_m5(candle, self.context.structural_level.price):
                self.invalidate(
                    reason="M5_CLOSE_ABOVE_TOP_LEVEL",
                    timestamp=candle.close_time,
                )
                return

        # State: BEARISH_SPIKE_DETECTED -> transition to WAIT_INITIAL_CORRECTION
        if self.context.current_state == BearishState.BEARISH_SPIKE_DETECTED:
            self.context.record_transition(
                to_state=BearishState.WAIT_INITIAL_CORRECTION,
                timestamp=candle.open_time,
                event_type="CORRECTION_STARTED",
                reason="Processing initial correction candles",
            )
            self._handle_m5_in_correction(candle)
            return

        # State: WAIT_INITIAL_CORRECTION
        if self.context.current_state == BearishState.WAIT_INITIAL_CORRECTION:
            self._handle_m5_in_correction(candle)
            return

        # State: WAIT_RETURN_TO_FIRST_TOP
        if self.context.current_state == BearishState.WAIT_RETURN_TO_FIRST_TOP:
            # Check if price has left the proximity zone
            top_level = self.context.structural_level.price  # type: ignore
            if is_sell_proximity_exit(candle.close, top_level, self.context.average_m5_range):
                self.context.has_left_proximity_zone = True
            return

        # State: M1_MONITORING
        # In this state, M5 candle still performs invalidation checks above.

    def _handle_m5_in_correction(self, candle: Candle) -> None:
        """Handle M5 candle evaluation during initial correction."""
        direction = detect_candle_direction(candle)

        if direction == Direction.BULLISH:
            # Bullish correction candle: append to collection
            self.context.correction_candles.append(candle)
            return

        if direction == Direction.BEARISH:
            # Potential confirmation if we have at least one bullish correction candle
            if len(self.context.correction_candles) > 0:
                last_corr = self.context.correction_candles[-1]
                if candle.low < last_corr.low:
                    # First Top Confirmed!
                    # BLK-03: TOP_LEVEL is strictly the High of the First Top correction candle
                    top_level = last_corr.high
                    spike = self.context.spike
                    assert spike is not None

                    struct_level = StructuralLevel(
                        spike_id=spike.spike_id,
                        symbol=self.context.symbol,
                        direction=Direction.BEARISH,
                        level_type="FIRST_TOP",
                        price=top_level,
                        formation_time=candle.close_time,
                        is_locked=True,
                    )
                    self.context.structural_level = struct_level

                    # Deterministic Setup ID generation
                    setup_id = generate_setup_id(
                        symbol=self.context.symbol,
                        direction=TradeDirection.SELL,
                        spike_start_time=spike.start_time,
                        spike_end_time=spike.end_time,
                        level_formed_time=candle.close_time,
                        structural_price=top_level,
                        tick_size=self.context.tick_size,
                    )
                    self.context.setup_id = setup_id

                    self.context.record_transition(
                        to_state=BearishState.FIRST_TOP_FORMED,
                        timestamp=candle.close_time,
                        event_type="FIRST_TOP_CONFIRMED",
                        reason=f"Low {candle.low} broke correction low {last_corr.low}. Top Level = {top_level}",
                    )

                    # Transition to return monitoring
                    self.context.record_transition(
                        to_state=BearishState.WAIT_RETURN_TO_FIRST_TOP,
                        timestamp=candle.close_time,
                        event_type="AWAITING_PROXIMITY_RETURN",
                        reason="Monitoring for exit and subsequent re-entry",
                    )

                    # Check if the confirming candle itself already closed outside the proximity zone
                    if is_sell_proximity_exit(candle.close, top_level, self.context.average_m5_range):
                        self.context.has_left_proximity_zone = True

    def on_quote(self, quote: Union[Quote, Tick]) -> None:
        """
        Process incoming live quote/tick.
        Evaluates proximity re-entry using LIVE BID exclusively.
        """
        if self.is_closed:
            return

        if quote.symbol != self.context.symbol:
            raise DataValidationError(f"Quote symbol {quote.symbol} does not match state machine symbol {self.context.symbol}")

        if self.context.current_state != BearishState.WAIT_RETURN_TO_FIRST_TOP:
            return

        # Proximity Re-entry Rule (BLK-13):
        # Must have closed at least one complete M5 candle outside the proximity zone first.
        if not self.context.has_left_proximity_zone:
            return

        top_level = self.context.structural_level.price  # type: ignore
        live_bid = quote.bid

        # SELL proximity uses LIVE BID exclusively
        if is_sell_proximity_active(live_bid, top_level, self.context.average_m5_range):
            self.context.proximity_activation_time = ensure_utc(quote.timestamp)
            self.context.record_transition(
                to_state=BearishState.M1_MONITORING,
                timestamp=quote.timestamp,
                event_type="PROXIMITY_ACTIVE",
                reason=f"Live Bid {live_bid} entered proximity zone [<= {self.context.average_m5_range}] of level {top_level}",
            )

    def on_tick(self, tick: Union[Quote, Tick]) -> None:
        """Alias for on_quote."""
        self.on_quote(tick)

    def on_m1_candle(self, candle: Candle) -> Optional[SignalEvent]:
        """
        Process incoming M1 candle during M1_MONITORING state.
        Only candles with close_time > proximity_activation_time are evaluated.
        """
        if self.is_closed:
            return None

        if candle.symbol != self.context.symbol:
            raise DataValidationError(f"Candle symbol {candle.symbol} does not match state machine symbol {self.context.symbol}")

        if self.context.current_state != BearishState.M1_MONITORING:
            return None

        act_time = self.context.proximity_activation_time
        assert act_time is not None

        # Causality / Look-Ahead Rule (Section 8):
        # Only M1.close_time > proximity_activation_time may be evaluated.
        # Any M1.close_time <= proximity_activation_time MUST be ignored.
        if candle.close_time <= act_time:
            return None

        # Validate Bullish Pin Bar (Section 10)
        pinbar = validate_sell_pinbar(candle)
        if not pinbar.is_valid:
            return None

        self.context.pinbar_event = pinbar
        self.context.record_transition(
            to_state=BearishState.BULLISH_PINBAR_DETECTED,
            timestamp=candle.close_time,
            event_type="BULLISH_PINBAR",
            reason=f"Valid bullish pin bar: body_ratio={pinbar.body_ratio:.4f}, body_low={pinbar.open} >= midpoint={pinbar.midpoint}",
        )

        # Generate SignalEvent immediately (BLK-04)
        spike = self.context.spike
        assert spike is not None
        assert self.context.setup_id is not None

        signal = SignalEvent(
            setup_id=self.context.setup_id,
            symbol=self.context.symbol,
            direction=TradeDirection.SELL,
            signal_time=candle.close_time,
            pinbar_high=candle.high,
            pinbar_low=candle.low,
            final_structural_target=spike.low,
        )
        self.context.signal_event = signal

        self.context.record_transition(
            to_state=BearishState.SELL_TRIGGER_ARMED,
            timestamp=candle.close_time,
            event_type="SIGNAL_GENERATED",
            reason=f"SELL Signal generated with target {spike.low}",
        )
        return signal

    def on_opposite_spike(self, spike: SpikeEvent) -> None:
        """Invalidate setup if an opposite (BULLISH) spike occurs before execution."""
        if self.context.current_state not in PRE_TRADE_SELL_STATES:
            return

        if is_invalidated_by_opposite_spike(TradeDirection.SELL, spike):
            self.invalidate(
                reason="OPPOSITE_SPIKE_DETECTED",
                timestamp=spike.end_time,
            )

    def on_session_end(self, timestamp: datetime) -> None:
        """Invalidate setup if trading session ends before execution."""
        if self.context.current_state not in PRE_TRADE_SELL_STATES:
            return

        self.invalidate(
            reason="TRADING_SESSION_ENDED",
            timestamp=timestamp,
        )

    def on_position_conflict(self, reason: str, timestamp: datetime) -> None:
        """Invalidate setup if an active position conflict prevents execution."""
        if self.context.current_state not in PRE_TRADE_SELL_STATES:
            return

        self.invalidate(
            reason=f"POSITION_CONFLICT: {reason}",
            timestamp=timestamp,
        )

    def on_order_executed(self, executed_price: float, timestamp: datetime) -> None:
        """Process order execution confirmation from execution layer."""
        if self.is_closed:
            return
        if self.context.current_state != BearishState.SELL_TRIGGER_ARMED:
            raise StateTransitionError(
                f"Cannot execute order from state {self.context.current_state}"
            )
        utc_ts = ensure_utc(timestamp)
        self.context.record_transition(
            to_state=BearishState.SELL_EXECUTED,
            timestamp=utc_ts,
            event_type="ORDER_FILLED",
            reason=f"SELL order executed at {executed_price}",
        )
        self.context.record_transition(
            to_state=BearishState.MANAGE_TRADE,
            timestamp=utc_ts,
            event_type="TRADE_ACTIVE",
            reason="Position active, managing stop loss and profit targets",
        )

    def on_trade_closed(self, reason: str, timestamp: datetime) -> None:
        """Process trade closure from position management layer."""
        if self.is_closed:
            return
        utc_ts = ensure_utc(timestamp)
        self.context.record_transition(
            to_state=BearishState.CLOSED,
            timestamp=utc_ts,
            event_type="TRADE_COMPLETED",
            reason=reason,
        )

    def invalidate(self, reason: str, timestamp: datetime) -> None:
        """Transition setup to CLOSED terminal state (pre-trade only)."""
        if self.context.current_state not in PRE_TRADE_SELL_STATES:
            return

        utc_ts = ensure_utc(timestamp)
        self.context.invalidation_reason = reason
        self.context.record_transition(
            to_state=BearishState.CLOSED,
            timestamp=utc_ts,
            event_type="SETUP_INVALIDATED",
            reason=reason,
        )
