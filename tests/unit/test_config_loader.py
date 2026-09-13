"""
Unit Tests for src/core/config_loader.py
Validates JSON parsing, dictionary parsing, and fail-fast invariant checks.
"""

import json
import tempfile
import unittest
from pathlib import Path

from src.core.config_loader import (
    load_config_from_dict,
    load_config_from_json,
)
from src.core.exceptions import ConfigurationError


class TestConfigLoader(unittest.TestCase):
    """Test suite for system configuration loading and validation."""

    def setUp(self) -> None:
        self.valid_raw_dict = {
            "magic_number": 123456,
            "max_deviation_points": 10,
            "internal_timezone": "UTC",
            "trading_day_timezone": "America/New_York",
            "symbols": {
                "dow_jones": {
                    "canonical_symbol": "DOW_JONES",
                    "broker_symbol": "US100_i",
                },
                "nasdaq": {
                    "canonical_symbol": "NASDAQ",
                    "broker_symbol": "NAS100_i",
                },
            },
            "risk": {
                "target_risk_fraction": 0.01,
                "max_active_positions_per_symbol": 1,
            },
            "strategy": {
                "market_timeframe": "M5",
                "entry_timeframe": "M1",
                "min_spike_candles": 3,
                "min_gap_count": 1,
                "pinbar_max_body_ratio": 0.20,
            },
            "backtest": {
                "commission_per_lot": 0.0,
                "slippage_points": 2.0,
            },
        }

    def test_load_valid_config_from_dict(self) -> None:
        cfg = load_config_from_dict(self.valid_raw_dict)
        self.assertEqual(cfg.magic_number, 123456)
        self.assertEqual(cfg.max_deviation_points, 10)
        self.assertEqual(cfg.internal_timezone, "UTC")
        self.assertEqual(cfg.trading_day_timezone, "America/New_York")
        self.assertEqual(cfg.symbols.dow_jones.canonical_symbol.value, "DOW_JONES")
        self.assertEqual(cfg.symbols.dow_jones.broker_symbol, "US100_i")
        self.assertEqual(cfg.symbols.nasdaq.canonical_symbol.value, "NASDAQ")
        self.assertEqual(cfg.symbols.nasdaq.broker_symbol, "NAS100_i")
        self.assertEqual(cfg.symbols.dow_jones_symbol, "US100_i")
        self.assertEqual(cfg.symbols.nasdaq_symbol, "NAS100_i")
        self.assertEqual(cfg.risk.target_risk_fraction, 0.01)
        self.assertEqual(cfg.risk.max_active_positions_per_symbol, 1)
        self.assertEqual(cfg.strategy.market_timeframe, "M5")
        self.assertEqual(cfg.strategy.entry_timeframe, "M1")
        self.assertEqual(cfg.strategy.min_spike_candles, 3)
        self.assertEqual(cfg.strategy.min_gap_count, 1)
        self.assertEqual(cfg.strategy.pinbar_max_body_ratio, 0.20)
        self.assertEqual(cfg.strategy.partial_close_tp1, 0.25)
        self.assertEqual(cfg.strategy.partial_close_tp2, 0.25)
        self.assertEqual(cfg.strategy.partial_close_tp3, 0.25)
        self.assertEqual(cfg.strategy.partial_close_structural_target, 0.25)
        self.assertEqual(cfg.backtest.commission_per_lot, 0.0)
        self.assertEqual(cfg.backtest.slippage_points, 2.0)

    def test_symbol_bidirectional_mapping(self) -> None:
        cfg = load_config_from_dict(self.valid_raw_dict)
        # Canonical to Broker mapping
        self.assertEqual(cfg.symbols.to_broker_symbol("DOW_JONES"), "US100_i")
        self.assertEqual(cfg.symbols.to_broker_symbol("NASDAQ"), "NAS100_i")
        # Broker to Canonical mapping
        self.assertEqual(cfg.symbols.to_canonical_symbol("US100_i").value, "DOW_JONES")
        self.assertEqual(cfg.symbols.to_canonical_symbol("NAS100_i").value, "NASDAQ")
        # Unknown mapping raises ConfigurationError
        with self.assertRaises(ConfigurationError):
            cfg.symbols.to_broker_symbol("UNKNOWN")
        with self.assertRaises(ConfigurationError):
            cfg.symbols.to_canonical_symbol("UNKNOWN")

    def test_flat_string_symbol_format_supported(self) -> None:
        data = dict(self.valid_raw_dict)
        data["symbols"] = {
            "dow_jones": "US100_i",
            "nasdaq": "NAS100_i",
        }
        cfg = load_config_from_dict(data)
        self.assertEqual(cfg.symbols.dow_jones.broker_symbol, "US100_i")
        self.assertEqual(cfg.symbols.nasdaq.broker_symbol, "NAS100_i")

    def test_rejects_missing_magic_number(self) -> None:
        data = dict(self.valid_raw_dict)
        data.pop("magic_number")
        with self.assertRaises(ConfigurationError):
            load_config_from_dict(data)

    def test_rejects_non_positive_magic_number(self) -> None:
        data = dict(self.valid_raw_dict)
        data["magic_number"] = -5
        with self.assertRaises(ConfigurationError):
            load_config_from_dict(data)

    def test_rejects_invalid_internal_timezone(self) -> None:
        data = dict(self.valid_raw_dict)
        data["internal_timezone"] = "America/New_York"
        with self.assertRaises(ConfigurationError):
            load_config_from_dict(data)

    def test_rejects_unknown_trading_day_timezone(self) -> None:
        data = dict(self.valid_raw_dict)
        data["trading_day_timezone"] = "Invalid/Timezone"
        with self.assertRaises(ConfigurationError):
            load_config_from_dict(data)

    def test_rejects_empty_symbol_mappings(self) -> None:
        data = dict(self.valid_raw_dict)
        data["symbols"] = {"dow_jones": "", "nasdaq": "NAS100"}
        with self.assertRaises(ConfigurationError):
            load_config_from_dict(data)

    def test_risk_fraction_strictly_locked_to_one_percent(self) -> None:
        """
        Verify target_risk_fraction is strictly locked to 0.01 (1%).
        - Omitted -> 0.01
        - Explicit 0.01 -> accepted
        - Explicit 0.02, 0.05, 0.10 -> rejected
        - NaN, Infinity, bool -> rejected
        """
        # 1. Omitted risk section
        data_omitted = dict(self.valid_raw_dict)
        data_omitted.pop("risk")
        cfg_omitted = load_config_from_dict(data_omitted)
        self.assertEqual(cfg_omitted.risk.target_risk_fraction, 0.01)

        # 2. Omitted target_risk_fraction inside risk section
        data_omitted_field = dict(self.valid_raw_dict)
        data_omitted_field["risk"] = {"max_active_positions_per_symbol": 1}
        cfg_omitted_field = load_config_from_dict(data_omitted_field)
        self.assertEqual(cfg_omitted_field.risk.target_risk_fraction, 0.01)

        # 3. Explicit 0.01 accepted
        data_exact = dict(self.valid_raw_dict)
        data_exact["risk"] = {"target_risk_fraction": 0.01, "max_active_positions_per_symbol": 1}
        cfg_exact = load_config_from_dict(data_exact)
        self.assertEqual(cfg_exact.risk.target_risk_fraction, 0.01)

        # 4. Explicit non-0.01 values strictly rejected
        for bad_val in [0.02, 0.05, 0.10, 0.005, -0.01, 0.0, 1.0]:
            data_bad = dict(self.valid_raw_dict)
            data_bad["risk"] = {"target_risk_fraction": bad_val, "max_active_positions_per_symbol": 1}
            with self.assertRaises(ConfigurationError):
                load_config_from_dict(data_bad)

        # 5. Non-finite values and booleans strictly rejected
        for bad_type in [float("nan"), float("inf"), float("-inf"), True, False, "0.01", None]:
            data_bad = dict(self.valid_raw_dict)
            data_bad["risk"] = {"target_risk_fraction": bad_type, "max_active_positions_per_symbol": 1}
            with self.assertRaises(ConfigurationError):
                load_config_from_dict(data_bad)

    def test_rejects_multiple_active_positions_per_symbol(self) -> None:
        data = dict(self.valid_raw_dict)
        data["risk"] = {"target_risk_fraction": 0.01, "max_active_positions_per_symbol": 2}
        with self.assertRaises(ConfigurationError):
            load_config_from_dict(data)

    def test_load_from_json_string(self) -> None:
        json_str = json.dumps(self.valid_raw_dict)
        cfg = load_config_from_json(json_str)
        self.assertEqual(cfg.magic_number, 123456)

    def test_load_from_json_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            json.dump(self.valid_raw_dict, tf)
            temp_path = tf.name

        try:
            cfg = load_config_from_json(temp_path)
            self.assertEqual(cfg.symbols.dow_jones_symbol, "US100_i")
            self.assertEqual(cfg.symbols.nasdaq_symbol, "NAS100_i")
        finally:
            Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
