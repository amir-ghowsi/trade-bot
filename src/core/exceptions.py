"""
Core Domain Exceptions Hierarchy.
Provides typed, structured error classes for the entire trading system.
Never uses generic exceptions.
"""

from typing import Any, Dict, Optional


class TradingSystemException(Exception):
    """Base exception for all errors produced by the trading system."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# =====================================================================
# Configuration Errors
# =====================================================================

class ConfigurationError(TradingSystemException):
    """Raised when configuration is missing, invalid, or violates invariants."""
    pass


# =====================================================================
# Market Data Errors
# =====================================================================

class MarketDataError(TradingSystemException):
    """Base exception for market data and feed anomalies."""
    pass


class DataValidationError(MarketDataError):
    """Raised when market candle or quote data violates basic geometric rules (e.g. High < Low)."""
    pass


class MissingDataError(MarketDataError):
    """Raised when required candle or historical tick data is unavailable."""
    pass


class TimezoneError(MarketDataError):
    """Raised when naive datetimes or unrecognized timezones are encountered."""
    pass


# =====================================================================
# Strategy & State Machine Errors
# =====================================================================

class StrategyError(TradingSystemException):
    """Base exception for strategy calculation and state anomalies."""
    pass


class InvalidStateTransitionError(StrategyError):
    """Raised when an illegal or non-sequential state machine transition is attempted."""
    pass


class LevelImmutabilityViolationError(StrategyError):
    """Raised when an attempt is made to update or mutate a locked structural level."""
    pass


class CausalityViolationError(StrategyError):
    """Raised when an operation attempts to look ahead or violate M5/M1 causal sequencing."""
    pass


# =====================================================================
# Risk & Sizing Errors
# =====================================================================

class RiskError(TradingSystemException):
    """Base exception for risk management and position sizing violations."""
    pass


class MinimumVolumeViolationError(RiskError):
    """Raised when calculated trade volume is below broker minimum (trade must be rejected)."""
    pass


class RiskLimitExceededError(RiskError):
    """Raised when an order request exceeds the maximum allowed 1% equity risk."""
    pass


class InvalidSymbolSpecificationError(RiskError):
    """Raised when broker symbol specifications are missing or non-positive (tick size, step, etc.)."""
    pass


# =====================================================================
# Broker & Execution Errors
# =====================================================================

class BrokerError(TradingSystemException):
    """Base exception for broker communication and execution failures."""
    pass


class BrokerConnectionError(BrokerError):
    """Raised when connection to broker or terminal fails or is lost."""
    pass


class OrderValidationError(BrokerError):
    """Raised when pre-flight order validation (e.g. order_check) fails."""
    pass


class OrderExecutionError(BrokerError):
    """Raised when order submission (order_send) is rejected by the broker."""
    pass


class PositionNotFoundError(BrokerError):
    """Raised when querying or modifying a position ticket that does not exist."""
    pass


class RecoveryError(BrokerError):
    """Raised when state reconciliation after restart cannot uniquely resolve position status."""
    pass
