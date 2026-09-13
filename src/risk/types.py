"""
Risk Engine and Position Sizing Domain Types.
Provides immutable result structures for position sizing and risk calculations.
"""

import math
from dataclasses import dataclass
from src.core.exceptions import DataValidationError


@dataclass(frozen=True)
class PositionSizingResult:
    """
    Immutable result contract for position sizing calculations.
    Contains exact risk and volume metrics evaluated for a trade setup.
    Guarantees all numeric fields are strictly positive and finite (no NaN, +inf, -inf).
    """
    target_risk_money: float
    risk_distance: float
    risk_ticks: float
    risk_per_lot: float
    raw_volume: float
    normalized_volume: float
    actual_risk_money: float
    is_capped: bool = False

    def __post_init__(self) -> None:
        numeric_fields = {
            "target_risk_money": self.target_risk_money,
            "risk_distance": self.risk_distance,
            "risk_ticks": self.risk_ticks,
            "risk_per_lot": self.risk_per_lot,
            "raw_volume": self.raw_volume,
            "normalized_volume": self.normalized_volume,
            "actual_risk_money": self.actual_risk_money,
        }

        for field_name, value in numeric_fields.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise DataValidationError(f"{field_name} must be a numeric value, got {type(value)}")
            if not math.isfinite(value):
                raise DataValidationError(f"{field_name} must be a finite number (no NaN or Infinity), got {value}")
            if value <= 0:
                raise DataValidationError(f"{field_name} must be strictly positive (> 0), got {value}")



__all__ = [
    "PositionSizingResult",
]
