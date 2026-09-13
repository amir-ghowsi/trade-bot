"""
Risk Engine & Deterministic Position Sizing.
Strict implementation of locked 1% target equity risk, broker specification validation,
step normalization, and volume bounds enforcement.
Zero indicators, zero MT5 imports, strictly deterministic.
"""

import math
from decimal import Decimal, getcontext

from src.core.constants import TARGET_RISK_FRACTION
from src.core.types import (
    SymbolSpecification,
    TradeDirection,
)
from src.risk.exceptions import (
    DataValidationError,
    InsufficientEquityError,
    InvalidRiskDistanceError,
    InvalidSymbolSpecificationError,
    MinimumVolumeViolationError,
    RiskError,
)
from src.risk.types import PositionSizingResult

# Set decimal precision high enough for any financial calculation
getcontext().prec = 28


def calculate_risk_distance(
    direction: TradeDirection,
    entry_price: float,
    initial_sl: float,
) -> float:
    """
    Calculate the directional risk distance between entry price and initial stop loss.
    BUY: Entry - InitialSL
    SELL: InitialSL - Entry

    Raises:
        InvalidRiskDistanceError: If risk distance is <= 0 or not finite.
        DataValidationError: If direction or price inputs are invalid, non-finite (NaN/Inf), boolean, or <= 0.
    """
    if not isinstance(direction, TradeDirection):
        raise DataValidationError(f"direction must be a TradeDirection, got {direction}")
    if (
        isinstance(entry_price, bool)
        or isinstance(initial_sl, bool)
        or not isinstance(entry_price, (int, float))
        or not isinstance(initial_sl, (int, float))
    ):
        raise DataValidationError("entry_price and initial_sl must be numeric values")
    if not math.isfinite(entry_price) or not math.isfinite(initial_sl):
        raise DataValidationError("entry_price and initial_sl must be finite numbers (no NaN or Infinity)")
    if entry_price <= 0 or initial_sl <= 0:
        raise DataValidationError("entry_price and initial_sl must be strictly positive (> 0)")

    if direction == TradeDirection.BUY:
        risk_distance = entry_price - initial_sl
    elif direction == TradeDirection.SELL:
        risk_distance = initial_sl - entry_price
    else:
        raise DataValidationError(f"Unsupported direction: {direction}")

    if not math.isfinite(risk_distance) or risk_distance <= 0:
        raise InvalidRiskDistanceError(
            f"Risk distance for {direction.value} must be strictly positive and finite (> 0). "
            f"Entry: {entry_price}, SL: {initial_sl}, Calculated Distance: {risk_distance}"
        )

    return risk_distance


def validate_symbol_specification(spec: SymbolSpecification) -> None:
    """
    Validate that broker SymbolSpecification contains all required positive and finite parameters.

    Raises:
        InvalidSymbolSpecificationError: If any specification rule fails.
    """
    if not isinstance(spec, SymbolSpecification):
        raise InvalidSymbolSpecificationError(f"spec must be a SymbolSpecification instance, got {type(spec)}")

    numeric_fields = {
        "tick_size": spec.tick_size,
        "tick_value": spec.tick_value,
        "volume_step": spec.volume_step,
        "volume_min": spec.volume_min,
        "volume_max": spec.volume_max,
    }

    for name, val in numeric_fields.items():
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise InvalidSymbolSpecificationError(f"{name} must be a finite number (no NaN or Infinity), got {val}")
        if val <= 0:
            raise InvalidSymbolSpecificationError(f"{name} must be strictly positive (> 0), got {val}")

    if spec.volume_max < spec.volume_min:
        raise InvalidSymbolSpecificationError(
            f"volume_max ({spec.volume_max}) cannot be less than volume_min ({spec.volume_min})"
        )


def calculate_position_size(
    equity: float,
    entry_price: float,
    initial_sl: float,
    direction: TradeDirection,
    spec: SymbolSpecification,
) -> PositionSizingResult:
    """
    Calculate normalized position volume targeting exactly locked 1% equity risk (TARGET_RISK_FRACTION = 0.01).

    Formulas:
        TargetRiskMoney = Equity * TARGET_RISK_FRACTION
        RiskDistance = Entry - InitialSL (BUY) or InitialSL - Entry (SELL)
        RiskTicks = RiskDistance / TickSize
        RiskPerLot = RiskTicks * TickValue
        RawVolume = TargetRiskMoney / RiskPerLot
        NormalizedVolume = floor(RawVolume / VolumeStep) * VolumeStep

    Volume Bounds Rules:
        - If NormalizedVolume < VolumeMin: REJECT (raise MinimumVolumeViolationError).
        - If NormalizedVolume > VolumeMax: NormalizedVolume = VolumeMax (cap).

    Args:
        equity: Account equity in account currency (> 0, finite).
        entry_price: Order execution price (> 0, finite).
        initial_sl: Initial stop loss price (> 0, finite).
        direction: TradeDirection.BUY or TradeDirection.SELL.
        spec: SymbolSpecification containing broker contract settings.

    Returns:
        PositionSizingResult: Immutable contract containing all sizing metrics.

    Raises:
        InsufficientEquityError: If equity <= 0, boolean, or not finite.
        InvalidRiskDistanceError: If risk distance <= 0.
        InvalidSymbolSpecificationError: If broker spec is invalid.
        MinimumVolumeViolationError: If normalized volume is below VolumeMin.
    """
    # 1. Preconditions validation
    if isinstance(equity, bool) or not isinstance(equity, (int, float)) or not math.isfinite(equity):
        raise InsufficientEquityError(f"Account equity must be a finite number (no NaN or Infinity), got {equity}")
    if equity <= 0:
        raise InsufficientEquityError(f"Account equity must be strictly positive (> 0), got {equity}")

    validate_symbol_specification(spec)

    # 2. Risk distance calculation
    risk_distance = calculate_risk_distance(direction, entry_price, initial_sl)

    # 3. Deterministic calculation using Decimal
    equity_d = Decimal(str(equity))
    risk_frac_d = Decimal(str(TARGET_RISK_FRACTION))
    target_risk_money_d = equity_d * risk_frac_d

    risk_dist_d = Decimal(str(risk_distance))
    tick_size_d = Decimal(str(spec.tick_size))
    tick_val_d = Decimal(str(spec.tick_value))

    risk_ticks_d = risk_dist_d / tick_size_d
    risk_per_lot_d = risk_ticks_d * tick_val_d

    if risk_per_lot_d <= 0:
        raise RiskError(f"Calculated risk per lot must be strictly positive, got {risk_per_lot_d}")

    raw_volume_d = target_risk_money_d / risk_per_lot_d
    volume_step_d = Decimal(str(spec.volume_step))

    # 4. Floor normalization by volume step
    normalized_volume_d = (raw_volume_d // volume_step_d) * volume_step_d

    volume_min_d = Decimal(str(spec.volume_min))
    volume_max_d = Decimal(str(spec.volume_max))

    # 5. Minimum volume enforcement - REJECT without forcing VolumeMin
    if normalized_volume_d < volume_min_d:
        raise MinimumVolumeViolationError(
            f"Normalized volume ({float(normalized_volume_d):.4f}) is below minimum allowed volume ({spec.volume_min}). "
            f"Target risk (${float(target_risk_money_d):.2f}) cannot be safely allocated without exceeding risk limits."
        )

    # 6. Maximum volume enforcement - CAP at VolumeMax
    is_capped = False
    if normalized_volume_d > volume_max_d:
        normalized_volume_d = volume_max_d
        is_capped = True

    # 7. Actual risk calculation
    actual_risk_money_d = normalized_volume_d * risk_per_lot_d

    return PositionSizingResult(
        target_risk_money=float(target_risk_money_d),
        risk_distance=float(risk_dist_d),
        risk_ticks=float(risk_ticks_d),
        risk_per_lot=float(risk_per_lot_d),
        raw_volume=float(raw_volume_d),
        normalized_volume=float(normalized_volume_d),
        actual_risk_money=float(actual_risk_money_d),
        is_capped=is_capped,
    )


class RiskEngine:
    """
    Deterministic Risk Management Engine.
    Strictly locked to 1% target equity risk model without mutable state.
    """

    def __init__(self) -> None:
        pass

    @property
    def target_risk_fraction(self) -> float:
        """Immutable target risk fraction locked to project specification (1%)."""
        return TARGET_RISK_FRACTION

    def calculate_position_size(
        self,
        equity: float,
        entry_price: float,
        initial_sl: float,
        direction: TradeDirection,
        spec: SymbolSpecification,
    ) -> PositionSizingResult:
        """Calculate position sizing using the locked 1% target risk model."""
        return calculate_position_size(
            equity=equity,
            entry_price=entry_price,
            initial_sl=initial_sl,
            direction=direction,
            spec=spec,
        )

    def validate_risk(
        self,
        equity: float,
        volume: float,
        entry_price: float,
        initial_sl: float,
        direction: TradeDirection,
        spec: SymbolSpecification,
    ) -> bool:
        """
        Validate whether a proposed order volume satisfies the locked 1% risk limit.
        Returns True if actual risk <= target risk money + tolerance, False on invalid risk or sizing violations.
        """
        if isinstance(equity, bool) or not isinstance(equity, (int, float)) or not math.isfinite(equity) or equity <= 0:
            return False
        if isinstance(volume, bool) or not isinstance(volume, (int, float)) or not math.isfinite(volume) or volume <= 0:
            return False

        try:
            validate_symbol_specification(spec)
            risk_dist = calculate_risk_distance(direction, entry_price, initial_sl)
            risk_ticks = risk_dist / spec.tick_size
            risk_per_lot = risk_ticks * spec.tick_value
            actual_risk = volume * risk_per_lot
            target_risk = equity * TARGET_RISK_FRACTION
            return actual_risk <= (target_risk + 1e-9)
        except (DataValidationError, RiskError):
            return False



__all__ = [
    "calculate_risk_distance",
    "validate_symbol_specification",
    "calculate_position_size",
    "RiskEngine",
]
