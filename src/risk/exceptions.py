"""
Risk Management and Position Sizing Exceptions.
Typed, structured error classes for risk engine operations.
"""

from src.core.exceptions import (
    DataValidationError,
    InvalidSymbolSpecificationError,
    MinimumVolumeViolationError,
    RiskError,
    RiskLimitExceededError,
)


class InvalidRiskDistanceError(RiskError):
    """Raised when risk distance (Entry - SL) is non-positive or inverted."""
    pass


class InsufficientEquityError(RiskError):
    """Raised when account equity is non-positive or insufficient for risk sizing."""
    pass


class InvalidRiskParameterError(RiskError):
    """Raised when risk fraction or sizing parameters violate limits."""
    pass


__all__ = [
    "RiskError",
    "MinimumVolumeViolationError",
    "RiskLimitExceededError",
    "InvalidSymbolSpecificationError",
    "DataValidationError",
    "InvalidRiskDistanceError",
    "InsufficientEquityError",
    "InvalidRiskParameterError",
]
