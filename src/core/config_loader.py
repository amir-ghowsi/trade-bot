"""
System Configuration Loader and Validator.
Provides type-safe, validated dataclasses for all system configurations.
Fails fast on any missing or out-of-bounds parameter.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Union
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.core.constants import (
    CanonicalSymbol,
    INTERNAL_TIMEZONE,
    MIN_GAP_COUNT,
    MIN_SPIKE_CANDLES,
    PINBAR_MAX_BODY_RATIO,
    TARGET_RISK_FRACTION,
    TRADING_DAY_TIMEZONE,
)
from src.core.exceptions import ConfigurationError


@dataclass(frozen=True)
class SingleSymbolMapping:
    """Individual symbol mapping associating canonical symbol with broker symbol."""
    canonical_symbol: CanonicalSymbol
    broker_symbol: str


@dataclass(frozen=True)
class SymbolMappingConfig:
    """
    Mapping between canonical identifiers and broker-specific symbols.
    Core/Strategy exclusively operates with canonical symbols.
    MT5 Adapter exclusively communicates with broker symbols.
    """
    dow_jones: SingleSymbolMapping
    nasdaq: SingleSymbolMapping

    def to_broker_symbol(self, canonical: Union[CanonicalSymbol, str]) -> str:
        """Map canonical symbol to broker symbol."""
        c_val = canonical.value if isinstance(canonical, CanonicalSymbol) else canonical
        if c_val == CanonicalSymbol.DOW_JONES.value:
            return self.dow_jones.broker_symbol
        elif c_val == CanonicalSymbol.NASDAQ.value:
            return self.nasdaq.broker_symbol
        raise ConfigurationError(f"Unknown canonical symbol: '{canonical}'")

    def to_canonical_symbol(self, broker_symbol: str) -> CanonicalSymbol:
        """Map broker symbol to canonical symbol."""
        if broker_symbol == self.dow_jones.broker_symbol:
            return CanonicalSymbol.DOW_JONES
        elif broker_symbol == self.nasdaq.broker_symbol:
            return CanonicalSymbol.NASDAQ
        raise ConfigurationError(f"Unknown broker symbol: '{broker_symbol}'")

    @property
    def dow_jones_symbol(self) -> str:
        """Convenience property for broker symbol of Dow Jones."""
        return self.dow_jones.broker_symbol

    @property
    def nasdaq_symbol(self) -> str:
        """Convenience property for broker symbol of Nasdaq."""
        return self.nasdaq.broker_symbol


@dataclass(frozen=True)
class RiskConfig:
    """Risk management parameters."""
    target_risk_fraction: float
    max_active_positions_per_symbol: int


@dataclass(frozen=True)
class StrategyConfig:
    """Strategy parameters strictly derived from the locked specification."""
    market_timeframe: str
    entry_timeframe: str
    min_spike_candles: int
    min_gap_count: int
    pinbar_max_body_ratio: float
    partial_close_tp1: float
    partial_close_tp2: float
    partial_close_tp3: float
    partial_close_structural_target: float


@dataclass(frozen=True)
class BacktestConfig:
    """Backtesting execution simulation parameters."""
    commission_per_lot: float
    slippage_points: float


@dataclass(frozen=True)
class SystemConfig:
    """Complete root configuration object."""
    magic_number: int
    max_deviation_points: int
    internal_timezone: str
    trading_day_timezone: str
    symbols: SymbolMappingConfig
    risk: RiskConfig
    strategy: StrategyConfig
    backtest: BacktestConfig


def _validate_timezone(tz_name: str, field_name: str) -> str:
    """Validate that a timezone name is recognized by zoneinfo."""
    if not isinstance(tz_name, str) or not tz_name.strip():
        raise ConfigurationError(f"Field '{field_name}' must be a non-empty string.")
    try:
        ZoneInfo(tz_name)
    except ZoneInfoNotFoundError as err:
        raise ConfigurationError(f"Invalid timezone in '{field_name}': '{tz_name}'") from err
    return tz_name


def load_config_from_dict(data: Dict[str, Any]) -> SystemConfig:
    """
    Validate and construct a SystemConfig instance from a dictionary.
    Fails fast on any invalid type or missing required key.
    """
    if not isinstance(data, dict):
        raise ConfigurationError("Configuration root must be a dictionary.")

    # 1. Root scalar fields
    magic_number = data.get("magic_number")
    if not isinstance(magic_number, int) or magic_number <= 0:
        raise ConfigurationError("Field 'magic_number' must be a positive integer.")

    max_deviation = data.get("max_deviation_points")
    if not isinstance(max_deviation, int) or max_deviation < 0:
        raise ConfigurationError("Field 'max_deviation_points' must be a non-negative integer.")

    internal_tz = _validate_timezone(
        data.get("internal_timezone", INTERNAL_TIMEZONE), "internal_timezone"
    )
    if internal_tz != "UTC":
        raise ConfigurationError(f"internal_timezone must be 'UTC', got '{internal_tz}'.")

    trading_day_tz = _validate_timezone(
        data.get("trading_day_timezone", TRADING_DAY_TIMEZONE), "trading_day_timezone"
    )

    # 2. Symbols section
    symbols_data = data.get("symbols")
    if not isinstance(symbols_data, dict):
        raise ConfigurationError("Missing or invalid 'symbols' section in configuration.")

    dow_entry = symbols_data.get("dow_jones")
    if isinstance(dow_entry, dict):
        dow_broker = dow_entry.get("broker_symbol")
        c_dow = dow_entry.get("canonical_symbol", CanonicalSymbol.DOW_JONES.value)
        if c_dow != CanonicalSymbol.DOW_JONES.value:
            raise ConfigurationError(f"dow_jones canonical_symbol must be 'DOW_JONES', got '{c_dow}'.")
    elif isinstance(dow_entry, str):
        dow_broker = dow_entry
    else:
        raise ConfigurationError("symbols.dow_jones must be a string or mapping dictionary.")

    if not isinstance(dow_broker, str) or not dow_broker.strip():
        raise ConfigurationError("symbols.dow_jones broker_symbol must be a non-empty string.")

    nasdaq_entry = symbols_data.get("nasdaq")
    if isinstance(nasdaq_entry, dict):
        nasdaq_broker = nasdaq_entry.get("broker_symbol")
        c_nasdaq = nasdaq_entry.get("canonical_symbol", CanonicalSymbol.NASDAQ.value)
        if c_nasdaq != CanonicalSymbol.NASDAQ.value:
            raise ConfigurationError(f"nasdaq canonical_symbol must be 'NASDAQ', got '{c_nasdaq}'.")
    elif isinstance(nasdaq_entry, str):
        nasdaq_broker = nasdaq_entry
    else:
        raise ConfigurationError("symbols.nasdaq must be a string or mapping dictionary.")

    if not isinstance(nasdaq_broker, str) or not nasdaq_broker.strip():
        raise ConfigurationError("symbols.nasdaq broker_symbol must be a non-empty string.")

    symbol_config = SymbolMappingConfig(
        dow_jones=SingleSymbolMapping(
            canonical_symbol=CanonicalSymbol.DOW_JONES,
            broker_symbol=dow_broker.strip(),
        ),
        nasdaq=SingleSymbolMapping(
            canonical_symbol=CanonicalSymbol.NASDAQ,
            broker_symbol=nasdaq_broker.strip(),
        ),
    )

    # 3. Risk section
    risk_data = data.get("risk", {})
    if not isinstance(risk_data, dict):
        raise ConfigurationError("'risk' section must be a dictionary.")

    if "target_risk_fraction" in risk_data:
        raw_risk = risk_data["target_risk_fraction"]
        if isinstance(raw_risk, bool) or not isinstance(raw_risk, (int, float)) or not math.isfinite(raw_risk):
            raise ConfigurationError("risk.target_risk_fraction must be a finite float equal to 0.01.")
        if abs(float(raw_risk) - TARGET_RISK_FRACTION) > 1e-9:
            raise ConfigurationError(
                f"risk.target_risk_fraction is locked to {TARGET_RISK_FRACTION} (1%). "
                f"Configured value {raw_risk} is rejected."
            )

    max_active_per_sym = risk_data.get("max_active_positions_per_symbol", 1)
    if not isinstance(max_active_per_sym, int) or max_active_per_sym != 1:
        raise ConfigurationError("risk.max_active_positions_per_symbol must be strictly 1.")

    risk_config = RiskConfig(
        target_risk_fraction=TARGET_RISK_FRACTION,
        max_active_positions_per_symbol=max_active_per_sym,
    )

    # 4. Strategy section
    strat_data = data.get("strategy", {})
    if not isinstance(strat_data, dict):
        raise ConfigurationError("'strategy' section must be a dictionary.")

    market_tf = strat_data.get("market_timeframe", "M5")
    if market_tf != "M5":
        raise ConfigurationError(f"strategy.market_timeframe must be 'M5', got '{market_tf}'.")

    entry_tf = strat_data.get("entry_timeframe", "M1")
    if entry_tf != "M1":
        raise ConfigurationError(f"strategy.entry_timeframe must be 'M1', got '{entry_tf}'.")

    strategy_config = StrategyConfig(
        market_timeframe=market_tf,
        entry_timeframe=entry_tf,
        min_spike_candles=int(strat_data.get("min_spike_candles", MIN_SPIKE_CANDLES)),
        min_gap_count=int(strat_data.get("min_gap_count", MIN_GAP_COUNT)),
        pinbar_max_body_ratio=float(strat_data.get("pinbar_max_body_ratio", PINBAR_MAX_BODY_RATIO)),
        partial_close_tp1=0.25,
        partial_close_tp2=0.25,
        partial_close_tp3=0.25,
        partial_close_structural_target=0.25,
    )

    # 5. Backtest section
    bt_data = data.get("backtest", {})
    if not isinstance(bt_data, dict):
        raise ConfigurationError("'backtest' section must be a dictionary.")

    commission = bt_data.get("commission_per_lot", 0.0)
    if not isinstance(commission, (int, float)) or commission < 0.0:
        raise ConfigurationError("backtest.commission_per_lot must be a non-negative float.")

    slippage = bt_data.get("slippage_points", 2.0)
    if not isinstance(slippage, (int, float)) or slippage < 0.0:
        raise ConfigurationError("backtest.slippage_points must be a non-negative float.")

    backtest_config = BacktestConfig(
        commission_per_lot=float(commission),
        slippage_points=float(slippage),
    )

    return SystemConfig(
        magic_number=magic_number,
        max_deviation_points=max_deviation,
        internal_timezone=internal_tz,
        trading_day_timezone=trading_day_tz,
        symbols=symbol_config,
        risk=risk_config,
        strategy=strategy_config,
        backtest=backtest_config,
    )


def load_config_from_json(source: Union[str, Path]) -> SystemConfig:
    """Load configuration from a JSON file path or JSON string."""
    try:
        path = Path(source)
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(str(source))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as err:
        raise ConfigurationError(f"Failed to parse JSON configuration: {err}") from err

    return load_config_from_dict(data)
