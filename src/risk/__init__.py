"""
Risk Management and Position Sizing Package.
Strict 1% equity risk model and deterministic volume sizing based on broker symbol specifications.
Zero indicators, zero MT5 imports, strictly deterministic.
"""

from src.risk.exceptions import (
    DataValidationError,
    InsufficientEquityError,
    InvalidRiskDistanceError,
    InvalidRiskParameterError,
    InvalidSymbolSpecificationError,
    MinimumVolumeViolationError,
    RiskError,
    RiskLimitExceededError,
)
from src.risk.position_sizing import (
    calculate_position_size,
    calculate_risk_distance,
    RiskEngine,
    validate_symbol_specification,
)
from src.risk.types import PositionSizingResult

__all__ = [
    # Types and Results
    "PositionSizingResult",
    # Calculation Functions
    "calculate_risk_distance",
    "validate_symbol_specification",
    "calculate_position_size",
    # Engine
    "RiskEngine",
    # Exceptions
    "RiskError",
    "MinimumVolumeViolationError",
    "RiskLimitExceededError",
    "InvalidSymbolSpecificationError",
    "DataValidationError",
    "InvalidRiskDistanceError",
    "InsufficientEquityError",
    "InvalidRiskParameterError",
]
