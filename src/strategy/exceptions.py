"""
Strategy Layer Domain Exceptions.
Inherits from Core exceptions hierarchy.
Decoupled strictly from MT5 SDK and broker-specific concerns.
"""

from typing import Any, Dict, Optional

from src.core.exceptions import StrategyError


class DetectorError(StrategyError):
    """Raised when a strategy detector encounters an invariant violation or invalid state."""
    pass


class CausalityViolationError(StrategyError):
    """Raised when look-ahead or time-travel logic is detected."""
    pass


class InvalidStructureError(StrategyError):
    """Raised when a market structure sequence violates locked rules."""
    pass


class StateTransitionError(StrategyError):
    """Raised when an illegal or non-causal state transition is attempted."""
    pass


class InvalidationError(StrategyError):
    """Raised when a setup is explicitly invalidated."""
    pass
