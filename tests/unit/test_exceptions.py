"""
Unit Tests for src/core/exceptions.py
Validates the complete domain exception hierarchy and error string representation.
"""

import unittest

from src.core.exceptions import (
    BrokerConnectionError,
    BrokerError,
    CausalityViolationError,
    ConfigurationError,
    DataValidationError,
    InvalidStateTransitionError,
    InvalidSymbolSpecificationError,
    LevelImmutabilityViolationError,
    MarketDataError,
    MinimumVolumeViolationError,
    MissingDataError,
    OrderExecutionError,
    OrderValidationError,
    PositionNotFoundError,
    RecoveryError,
    RiskError,
    RiskLimitExceededError,
    StrategyError,
    TimezoneError,
    TradingSystemException,
)


class TestExceptions(unittest.TestCase):
    """Test suite verifying inheritance, typing, and formatting of all exceptions."""

    def test_base_exception_formatting_without_details(self) -> None:
        exc = TradingSystemException("General fault")
        self.assertEqual(str(exc), "General fault")
        self.assertEqual(exc.details, {})

    def test_base_exception_formatting_with_details(self) -> None:
        exc = TradingSystemException("Detailed fault", details={"code": 404, "symbol": "DJI"})
        self.assertIn("Detailed fault", str(exc))
        self.assertIn("'code': 404", str(exc))
        self.assertIn("'symbol': 'DJI'", str(exc))

    def test_configuration_error_inheritance(self) -> None:
        err = ConfigurationError("Config missing")
        self.assertIsInstance(err, TradingSystemException)

    def test_market_data_error_hierarchy(self) -> None:
        for cls in (DataValidationError, MissingDataError, TimezoneError):
            inst = cls("Market data issue")
            self.assertIsInstance(inst, MarketDataError)
            self.assertIsInstance(inst, TradingSystemException)

    def test_strategy_error_hierarchy(self) -> None:
        for cls in (InvalidStateTransitionError, LevelImmutabilityViolationError, CausalityViolationError):
            inst = cls("Strategy issue")
            self.assertIsInstance(inst, StrategyError)
            self.assertIsInstance(inst, TradingSystemException)

    def test_risk_error_hierarchy(self) -> None:
        for cls in (MinimumVolumeViolationError, RiskLimitExceededError, InvalidSymbolSpecificationError):
            inst = cls("Risk issue")
            self.assertIsInstance(inst, RiskError)
            self.assertIsInstance(inst, TradingSystemException)

    def test_broker_error_hierarchy(self) -> None:
        for cls in (
            BrokerConnectionError,
            OrderValidationError,
            OrderExecutionError,
            PositionNotFoundError,
            RecoveryError,
        ):
            inst = cls("Broker issue")
            self.assertIsInstance(inst, BrokerError)
            self.assertIsInstance(inst, TradingSystemException)


if __name__ == "__main__":
    unittest.main()
