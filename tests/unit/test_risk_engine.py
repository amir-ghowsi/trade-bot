"""
Comprehensive Unit Tests for Risk Engine and Deterministic Position Sizing.
Validates locked 1% equity risk model, directional distance, volume normalization,
bounds enforcement, finite numeric validations (NaN/Inf rejection), broker specification
handling, and deterministic invariants.
"""

import math
import unittest
from decimal import Decimal

from src.core.constants import (
    CanonicalSymbol,
    TARGET_RISK_FRACTION,
)
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
from src.risk.position_sizing import (
    calculate_position_size,
    calculate_risk_distance,
    RiskEngine,
    validate_symbol_specification,
)
from src.risk.types import PositionSizingResult


class TestRiskEngine(unittest.TestCase):
    """Test suite for Phase 6 Risk Engine and Position Sizing."""

    def setUp(self) -> None:
        # Standard Symbol Specification for Dow Jones
        self.spec_dj = SymbolSpecification(
            canonical_symbol=CanonicalSymbol.DOW_JONES,
            broker_symbol="US30.cash",
            tick_size=1.0,
            tick_value=1.0,
            contract_size=1.0,
            volume_min=0.1,
            volume_max=100.0,
            volume_step=0.01,
            point=1.0,
        )

        # Standard Symbol Specification for Nasdaq
        self.spec_nq = SymbolSpecification(
            canonical_symbol=CanonicalSymbol.NASDAQ,
            broker_symbol="US100.cash",
            tick_size=0.25,
            tick_value=0.50,
            contract_size=1.0,
            volume_min=0.1,
            volume_max=50.0,
            volume_step=0.01,
            point=0.01,
        )

        self.engine = RiskEngine()

    # -----------------------------------------------------------------
    # 1. Basic BUY Sizing
    # -----------------------------------------------------------------
    def test_basic_buy_position_sizing(self) -> None:
        """Verify correct risk distance, volume, and metrics for a standard BUY setup."""
        equity = 10000.0
        entry = 34100.0
        initial_sl = 34050.0  # Risk distance = 50.0
        # TargetRiskMoney = 10000 * 0.01 = 100.0
        # RiskTicks = 50.0 / 1.0 = 50.0
        # RiskPerLot = 50.0 * 1.0 = 50.0
        # RawVolume = 100.0 / 50.0 = 2.0
        # NormalizedVolume = floor(2.0 / 0.01) * 0.01 = 2.0

        res = calculate_position_size(
            equity=equity,
            entry_price=entry,
            initial_sl=initial_sl,
            direction=TradeDirection.BUY,
            spec=self.spec_dj,
        )

        self.assertIsInstance(res, PositionSizingResult)
        self.assertEqual(res.target_risk_money, 100.0)
        self.assertEqual(res.risk_distance, 50.0)
        self.assertEqual(res.risk_ticks, 50.0)
        self.assertEqual(res.risk_per_lot, 50.0)
        self.assertEqual(res.raw_volume, 2.0)
        self.assertEqual(res.normalized_volume, 2.0)
        self.assertEqual(res.actual_risk_money, 100.0)
        self.assertFalse(res.is_capped)

    # -----------------------------------------------------------------
    # 2. Basic SELL Sizing
    # -----------------------------------------------------------------
    def test_basic_sell_position_sizing(self) -> None:
        """Verify correct risk distance, volume, and metrics for a standard SELL setup."""
        equity = 20000.0
        entry = 18000.0
        initial_sl = 18025.0  # Risk distance = 25.0
        # TargetRiskMoney = 20000 * 0.01 = 200.0
        # Nasdaq: tick_size=0.25, tick_value=0.50
        # RiskTicks = 25.0 / 0.25 = 100.0
        # RiskPerLot = 100.0 * 0.50 = 50.0
        # RawVolume = 200.0 / 50.0 = 4.0
        # NormalizedVolume = 4.0

        res = calculate_position_size(
            equity=equity,
            entry_price=entry,
            initial_sl=initial_sl,
            direction=TradeDirection.SELL,
            spec=self.spec_nq,
        )

        self.assertEqual(res.target_risk_money, 200.0)
        self.assertEqual(res.risk_distance, 25.0)
        self.assertEqual(res.risk_ticks, 100.0)
        self.assertEqual(res.risk_per_lot, 50.0)
        self.assertEqual(res.raw_volume, 4.0)
        self.assertEqual(res.normalized_volume, 4.0)
        self.assertEqual(res.actual_risk_money, 200.0)
        self.assertFalse(res.is_capped)

    # -----------------------------------------------------------------
    # 3. Locked 1% Target Risk Model Invariant
    # -----------------------------------------------------------------
    def test_exact_one_percent_locked_target_risk(self) -> None:
        """TargetRiskMoney must strictly equal Equity * 0.01 for any equity value."""
        for equity in [1000.0, 5000.0, 25430.75, 100000.0]:
            res = calculate_position_size(
                equity=equity,
                entry_price=34100.0,
                initial_sl=34050.0,
                direction=TradeDirection.BUY,
                spec=self.spec_dj,
            )
            expected_target = equity * 0.01
            self.assertAlmostEqual(res.target_risk_money, expected_target, places=8)
            self.assertEqual(TARGET_RISK_FRACTION, 0.01)

    def test_locked_risk_model_cannot_be_overridden(self) -> None:
        """Verify that caller cannot pass a custom target risk fraction to override the 1% model."""
        self.assertEqual(self.engine.target_risk_fraction, 0.01)

        # calculate_position_size must only accept locked parameters (no target_risk_fraction)
        with self.assertRaises(TypeError):
            calculate_position_size(  # type: ignore
                equity=10000.0,
                entry_price=34100.0,
                initial_sl=34050.0,
                direction=TradeDirection.BUY,
                spec=self.spec_dj,
                target_risk_fraction=0.05,
            )

        # RiskEngine constructor must not accept custom fractions
        with self.assertRaises(TypeError):
            RiskEngine(target_risk_fraction=0.05)  # type: ignore

    # -----------------------------------------------------------------
    # 4. VolumeStep Normalization Floor Behavior
    # -----------------------------------------------------------------
    def test_volume_step_normalization_flooring(self) -> None:
        """Normalization must always floor (round down) by volume_step."""
        # Equity = 10000 -> TargetRisk = 100.0
        # Risk distance = 33.0 -> RiskTicks = 33.0 -> RiskPerLot = 33.0
        # RawVolume = 100 / 33 = 3.03030303...
        # Volume step = 0.01 -> floor is 3.03
        res = calculate_position_size(
            equity=10000.0,
            entry_price=34100.0,
            initial_sl=34067.0,
            direction=TradeDirection.BUY,
            spec=self.spec_dj,
        )

        self.assertAlmostEqual(res.raw_volume, 100.0 / 33.0, places=6)
        self.assertEqual(res.normalized_volume, 3.03)
        self.assertLessEqual(res.actual_risk_money, res.target_risk_money)

    def test_volume_step_coarse_granularity(self) -> None:
        """Test coarse volume_step (e.g. 1.0)."""
        spec_coarse = SymbolSpecification(
            canonical_symbol=CanonicalSymbol.DOW_JONES,
            broker_symbol="US30.coarse",
            tick_size=1.0,
            tick_value=1.0,
            contract_size=1.0,
            volume_min=1.0,
            volume_max=100.0,
            volume_step=1.0,
            point=1.0,
        )

        res = calculate_position_size(
            equity=10000.0,
            entry_price=34100.0,
            initial_sl=34067.0,
            direction=TradeDirection.BUY,
            spec=spec_coarse,
        )
        self.assertEqual(res.normalized_volume, 3.0)

    # -----------------------------------------------------------------
    # 5. Raw Volume Below Minimum Rejection
    # -----------------------------------------------------------------
    def test_raw_volume_below_minimum_rejects(self) -> None:
        """If normalized volume < volume_min, the trade must be rejected with MinimumVolumeViolationError."""
        # Equity = 100.0 -> TargetRisk = 1.0
        # Risk distance = 50.0 -> RiskPerLot = 50.0
        # RawVolume = 1.0 / 50.0 = 0.02 < volume_min (0.1)
        with self.assertRaises(MinimumVolumeViolationError):
            calculate_position_size(
                equity=100.0,
                entry_price=34100.0,
                initial_sl=34050.0,
                direction=TradeDirection.BUY,
                spec=self.spec_dj,
            )

    # -----------------------------------------------------------------
    # 6. Raw Volume Exactly Minimum Accepted
    # -----------------------------------------------------------------
    def test_raw_volume_exactly_minimum_accepted(self) -> None:
        """If normalized volume == volume_min, trade is accepted."""
        # Equity = 500.0 -> TargetRisk = 5.0
        # Risk distance = 50.0 -> RiskPerLot = 50.0
        # RawVolume = 5.0 / 50.0 = 0.10 == volume_min (0.1)
        res = calculate_position_size(
            equity=500.0,
            entry_price=34100.0,
            initial_sl=34050.0,
            direction=TradeDirection.BUY,
            spec=self.spec_dj,
        )
        self.assertEqual(res.normalized_volume, 0.1)
        self.assertEqual(res.actual_risk_money, 5.0)

    # -----------------------------------------------------------------
    # 7. Raw Volume Above Maximum Capped at VolumeMax
    # -----------------------------------------------------------------
    def test_raw_volume_above_maximum_capped(self) -> None:
        """If normalized volume > volume_max, capped at volume_max and is_capped set to True."""
        res = calculate_position_size(
            equity=10000000.0,
            entry_price=34100.0,
            initial_sl=34050.0,
            direction=TradeDirection.BUY,
            spec=self.spec_dj,
        )
        self.assertEqual(res.normalized_volume, 100.0)
        self.assertTrue(res.is_capped)
        self.assertEqual(res.actual_risk_money, 100.0 * 50.0)  # 5000.0 <= 100,000.0
        self.assertLess(res.actual_risk_money, res.target_risk_money)

    # -----------------------------------------------------------------
    # 8. FINITE NUMERIC VALIDATION TESTS (NaN & Infinity Rejection)
    # -----------------------------------------------------------------
    def test_equity_nan_rejected(self) -> None:
        """NaN equity must be rejected deterministically."""
        with self.assertRaises(InsufficientEquityError):
            calculate_position_size(
                equity=float("nan"),
                entry_price=34100.0,
                initial_sl=34050.0,
                direction=TradeDirection.BUY,
                spec=self.spec_dj,
            )

    def test_equity_infinity_rejected(self) -> None:
        """+Infinity and -Infinity equity must be rejected."""
        with self.assertRaises(InsufficientEquityError):
            calculate_position_size(
                equity=float("inf"),
                entry_price=34100.0,
                initial_sl=34050.0,
                direction=TradeDirection.BUY,
                spec=self.spec_dj,
            )
        with self.assertRaises(InsufficientEquityError):
            calculate_position_size(
                equity=float("-inf"),
                entry_price=34100.0,
                initial_sl=34050.0,
                direction=TradeDirection.BUY,
                spec=self.spec_dj,
            )

    def test_entry_price_nan_rejected(self) -> None:
        """NaN entry price must be rejected."""
        with self.assertRaises(DataValidationError):
            calculate_position_size(
                equity=10000.0,
                entry_price=float("nan"),
                initial_sl=34050.0,
                direction=TradeDirection.BUY,
                spec=self.spec_dj,
            )

    def test_entry_price_infinity_rejected(self) -> None:
        """Infinity entry price must be rejected."""
        with self.assertRaises(DataValidationError):
            calculate_position_size(
                equity=10000.0,
                entry_price=float("inf"),
                initial_sl=34050.0,
                direction=TradeDirection.BUY,
                spec=self.spec_dj,
            )

    def test_initial_sl_nan_rejected(self) -> None:
        """NaN initial stop loss must be rejected."""
        with self.assertRaises(DataValidationError):
            calculate_position_size(
                equity=10000.0,
                entry_price=34100.0,
                initial_sl=float("nan"),
                direction=TradeDirection.BUY,
                spec=self.spec_dj,
            )

    def test_initial_sl_infinity_rejected(self) -> None:
        """Infinity initial stop loss must be rejected."""
        with self.assertRaises(DataValidationError):
            calculate_position_size(
                equity=10000.0,
                entry_price=34100.0,
                initial_sl=float("inf"),
                direction=TradeDirection.BUY,
                spec=self.spec_dj,
            )

    def test_spec_tick_size_nan_and_infinity_rejected(self) -> None:
        """NaN and Infinity tick_size must be rejected."""
        spec_nan = SymbolSpecification(
            canonical_symbol=CanonicalSymbol.DOW_JONES,
            broker_symbol="US30",
            tick_size=float("nan"),
            tick_value=1.0,
            contract_size=1.0,
            volume_min=0.1,
            volume_max=100.0,
            volume_step=0.01,
            point=1.0,
        )
        with self.assertRaises(InvalidSymbolSpecificationError):
            validate_symbol_specification(spec_nan)

        spec_inf = SymbolSpecification(
            canonical_symbol=CanonicalSymbol.DOW_JONES,
            broker_symbol="US30",
            tick_size=float("inf"),
            tick_value=1.0,
            contract_size=1.0,
            volume_min=0.1,
            volume_max=100.0,
            volume_step=0.01,
            point=1.0,
        )
        with self.assertRaises(InvalidSymbolSpecificationError):
            validate_symbol_specification(spec_inf)

    def test_spec_tick_value_nan_and_infinity_rejected(self) -> None:
        """NaN and Infinity tick_value must be rejected."""
        spec_nan = SymbolSpecification(
            canonical_symbol=CanonicalSymbol.DOW_JONES,
            broker_symbol="US30",
            tick_size=1.0,
            tick_value=float("nan"),
            contract_size=1.0,
            volume_min=0.1,
            volume_max=100.0,
            volume_step=0.01,
            point=1.0,
        )
        with self.assertRaises(InvalidSymbolSpecificationError):
            validate_symbol_specification(spec_nan)

        spec_inf = SymbolSpecification(
            canonical_symbol=CanonicalSymbol.DOW_JONES,
            broker_symbol="US30",
            tick_size=1.0,
            tick_value=float("inf"),
            contract_size=1.0,
            volume_min=0.1,
            volume_max=100.0,
            volume_step=0.01,
            point=1.0,
        )
        with self.assertRaises(InvalidSymbolSpecificationError):
            validate_symbol_specification(spec_inf)

    def test_spec_volume_step_nan_and_infinity_rejected(self) -> None:
        """NaN and Infinity volume_step must be rejected."""
        spec_nan = SymbolSpecification(
            canonical_symbol=CanonicalSymbol.DOW_JONES,
            broker_symbol="US30",
            tick_size=1.0,
            tick_value=1.0,
            contract_size=1.0,
            volume_min=0.1,
            volume_max=100.0,
            volume_step=float("nan"),
            point=1.0,
        )
        with self.assertRaises(InvalidSymbolSpecificationError):
            validate_symbol_specification(spec_nan)

        spec_inf = SymbolSpecification(
            canonical_symbol=CanonicalSymbol.DOW_JONES,
            broker_symbol="US30",
            tick_size=1.0,
            tick_value=1.0,
            contract_size=1.0,
            volume_min=0.1,
            volume_max=100.0,
            volume_step=float("inf"),
            point=1.0,
        )
        with self.assertRaises(InvalidSymbolSpecificationError):
            validate_symbol_specification(spec_inf)

    def test_spec_volume_min_nan_and_infinity_rejected(self) -> None:
        """NaN and Infinity volume_min must be rejected."""
        spec_nan = SymbolSpecification(
            canonical_symbol=CanonicalSymbol.DOW_JONES,
            broker_symbol="US30",
            tick_size=1.0,
            tick_value=1.0,
            contract_size=1.0,
            volume_min=float("nan"),
            volume_max=100.0,
            volume_step=0.01,
            point=1.0,
        )
        with self.assertRaises(InvalidSymbolSpecificationError):
            validate_symbol_specification(spec_nan)

        spec_inf = SymbolSpecification(
            canonical_symbol=CanonicalSymbol.DOW_JONES,
            broker_symbol="US30",
            tick_size=1.0,
            tick_value=1.0,
            contract_size=1.0,
            volume_min=float("inf"),
            volume_max=float("inf"),
            volume_step=0.01,
            point=1.0,
        )
        with self.assertRaises(InvalidSymbolSpecificationError):
            validate_symbol_specification(spec_inf)

    def test_spec_volume_max_nan_and_infinity_rejected(self) -> None:
        """NaN and Infinity volume_max must be rejected."""
        spec_nan = SymbolSpecification(
            canonical_symbol=CanonicalSymbol.DOW_JONES,
            broker_symbol="US30",
            tick_size=1.0,
            tick_value=1.0,
            contract_size=1.0,
            volume_min=0.1,
            volume_max=float("nan"),
            volume_step=0.01,
            point=1.0,
        )
        with self.assertRaises(InvalidSymbolSpecificationError):
            validate_symbol_specification(spec_nan)

        spec_inf = SymbolSpecification(
            canonical_symbol=CanonicalSymbol.DOW_JONES,
            broker_symbol="US30",
            tick_size=1.0,
            tick_value=1.0,
            contract_size=1.0,
            volume_min=0.1,
            volume_max=float("inf"),
            volume_step=0.01,
            point=1.0,
        )
        with self.assertRaises(InvalidSymbolSpecificationError):
            validate_symbol_specification(spec_inf)

    # -----------------------------------------------------------------
    # 9. PositionSizingResult Finite Enforcement
    # -----------------------------------------------------------------
    def test_position_sizing_result_rejects_nan(self) -> None:
        """PositionSizingResult must reject NaN on any field."""
        fields = [
            "target_risk_money", "risk_distance", "risk_ticks",
            "risk_per_lot", "raw_volume", "normalized_volume", "actual_risk_money"
        ]
        base_args = {
            "target_risk_money": 100.0,
            "risk_distance": 50.0,
            "risk_ticks": 50.0,
            "risk_per_lot": 50.0,
            "raw_volume": 2.0,
            "normalized_volume": 2.0,
            "actual_risk_money": 100.0,
            "is_capped": False,
        }

        for f in fields:
            bad_args = dict(base_args)
            bad_args[f] = float("nan")
            with self.assertRaises(DataValidationError, msg=f"Field {f} accepted NaN"):
                PositionSizingResult(**bad_args)

    def test_position_sizing_result_rejects_infinity(self) -> None:
        """PositionSizingResult must reject Infinity on any field."""
        fields = [
            "target_risk_money", "risk_distance", "risk_ticks",
            "risk_per_lot", "raw_volume", "normalized_volume", "actual_risk_money"
        ]
        base_args = {
            "target_risk_money": 100.0,
            "risk_distance": 50.0,
            "risk_ticks": 50.0,
            "risk_per_lot": 50.0,
            "raw_volume": 2.0,
            "normalized_volume": 2.0,
            "actual_risk_money": 100.0,
            "is_capped": False,
        }

        for f in fields:
            bad_args = dict(base_args)
            bad_args[f] = float("inf")
            with self.assertRaises(DataValidationError, msg=f"Field {f} accepted Infinity"):
                PositionSizingResult(**bad_args)

    # -----------------------------------------------------------------
    # 10. Zero and Negative Risk Distance Rejection
    # -----------------------------------------------------------------
    def test_zero_risk_distance_rejects(self) -> None:
        """Zero risk distance (Entry == SL) must be rejected."""
        with self.assertRaises(InvalidRiskDistanceError):
            calculate_position_size(
                equity=10000.0,
                entry_price=34100.0,
                initial_sl=34100.0,
                direction=TradeDirection.BUY,
                spec=self.spec_dj,
            )

    def test_negative_risk_distance_inverted_levels_rejects(self) -> None:
        """Inverted levels (BUY with Entry < SL or SELL with Entry > SL) must be rejected."""
        with self.assertRaises(InvalidRiskDistanceError):
            calculate_position_size(
                equity=10000.0,
                entry_price=34000.0,
                initial_sl=34100.0,
                direction=TradeDirection.BUY,
                spec=self.spec_dj,
            )

        with self.assertRaises(InvalidRiskDistanceError):
            calculate_position_size(
                equity=10000.0,
                entry_price=18100.0,
                initial_sl=18000.0,
                direction=TradeDirection.SELL,
                spec=self.spec_nq,
            )

    # -----------------------------------------------------------------
    # 11. Directional Formulas Validation
    # -----------------------------------------------------------------
    def test_directional_risk_distance_formulas(self) -> None:
        """BUY uses Entry - SL; SELL uses SL - Entry."""
        buy_dist = calculate_risk_distance(TradeDirection.BUY, entry_price=100.0, initial_sl=90.0)
        self.assertEqual(buy_dist, 10.0)

        sell_dist = calculate_risk_distance(TradeDirection.SELL, entry_price=90.0, initial_sl=100.0)
        self.assertEqual(sell_dist, 10.0)

    # -----------------------------------------------------------------
    # 12. Sizing Driven by SymbolSpecification
    # -----------------------------------------------------------------
    def test_sizing_driven_by_different_symbol_specs(self) -> None:
        """Verify position sizing accurately adapts to diverse broker contract specifications."""
        equity = 10000.0  # TargetRisk = 100.0

        # Spec A: tick_size=0.1, tick_value=1.0
        spec_a = SymbolSpecification(
            canonical_symbol=CanonicalSymbol.DOW_JONES,
            broker_symbol="SYM_A",
            tick_size=0.1,
            tick_value=1.0,
            contract_size=1.0,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            point=0.1,
        )
        res_a = calculate_position_size(
            equity=equity,
            entry_price=1000.0,
            initial_sl=990.0,
            direction=TradeDirection.BUY,
            spec=spec_a,
        )
        self.assertEqual(res_a.normalized_volume, 1.0)

        # Spec B: tick_size=0.01, tick_value=0.01
        spec_b = SymbolSpecification(
            canonical_symbol=CanonicalSymbol.NASDAQ,
            broker_symbol="SYM_B",
            tick_size=0.01,
            tick_value=0.01,
            contract_size=1.0,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            point=0.01,
        )
        res_b = calculate_position_size(
            equity=equity,
            entry_price=1000.0,
            initial_sl=990.0,
            direction=TradeDirection.BUY,
            spec=spec_b,
        )
        self.assertEqual(res_b.normalized_volume, 10.0)

    # -----------------------------------------------------------------
    # 13. Strict Determinism
    # -----------------------------------------------------------------
    def test_strict_determinism_identical_runs(self) -> None:
        """Two calls with identical parameters must return identical results."""
        engine1 = RiskEngine()
        engine2 = RiskEngine()

        for _ in range(10):
            res1 = engine1.calculate_position_size(
                equity=25678.90,
                entry_price=34567.8,
                initial_sl=34512.3,
                direction=TradeDirection.BUY,
                spec=self.spec_dj,
            )
            res2 = engine2.calculate_position_size(
                equity=25678.90,
                entry_price=34567.8,
                initial_sl=34512.3,
                direction=TradeDirection.BUY,
                spec=self.spec_dj,
            )
            self.assertEqual(res1, res2)

    # -----------------------------------------------------------------
    # 14. No Minimum-Volume Forcing
    # -----------------------------------------------------------------
    def test_no_minimum_volume_forcing(self) -> None:
        """
        Verify that an undersized normalized position is rejected rather than forced to VolumeMin.
        Forcing VolumeMin would cause actual risk to exceed the 1% target.
        """
        spec_high_min = SymbolSpecification(
            canonical_symbol=CanonicalSymbol.DOW_JONES,
            broker_symbol="US30.institutional",
            tick_size=1.0,
            tick_value=1.0,
            contract_size=1.0,
            volume_min=5.0,  # High minimum
            volume_max=100.0,
            volume_step=0.1,
            point=1.0,
        )

        with self.assertRaises(MinimumVolumeViolationError):
            calculate_position_size(
                equity=10000.0,
                entry_price=34100.0,
                initial_sl=34050.0,
                direction=TradeDirection.BUY,
                spec=spec_high_min,
            )

    # -----------------------------------------------------------------
    # 15. Maximum Volume Cap Safety
    # -----------------------------------------------------------------
    def test_maximum_volume_cap_actual_risk_safety(self) -> None:
        """When capped at volume_max, actual risk must be <= target risk."""
        spec_low_max = SymbolSpecification(
            canonical_symbol=CanonicalSymbol.DOW_JONES,
            broker_symbol="US30.restricted",
            tick_size=1.0,
            tick_value=1.0,
            contract_size=1.0,
            volume_min=0.1,
            volume_max=1.0,  # Low maximum
            volume_step=0.01,
            point=1.0,
        )

        res = calculate_position_size(
            equity=100000.0,
            entry_price=34100.0,
            initial_sl=34050.0,
            direction=TradeDirection.BUY,
            spec=spec_low_max,
        )
        self.assertEqual(res.normalized_volume, 1.0)
        self.assertTrue(res.is_capped)
        self.assertEqual(res.actual_risk_money, 50.0)
        self.assertLessEqual(res.actual_risk_money, res.target_risk_money)

    # -----------------------------------------------------------------
    # 16. Floating Step Edge Cases
    # -----------------------------------------------------------------
    def test_floating_step_boundary_precision(self) -> None:
        """Precision tests around volume step boundaries."""
        spec = SymbolSpecification(
            canonical_symbol=CanonicalSymbol.DOW_JONES,
            broker_symbol="US30",
            tick_size=1.0,
            tick_value=1.0,
            contract_size=1.0,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            point=1.0,
        )

        # Case 1: TargetRisk = 204.99, RiskPerLot = 100 -> RawVolume = 2.0499 -> Normalized = 2.04
        equity1 = 20499.0
        res1 = calculate_position_size(
            equity=equity1,
            entry_price=34100.0,
            initial_sl=34000.0,
            direction=TradeDirection.BUY,
            spec=spec,
        )
        self.assertEqual(res1.normalized_volume, 2.04)

        # Case 2: TargetRisk = 205.01, RiskPerLot = 100 -> RawVolume = 2.0501 -> Normalized = 2.05
        equity2 = 20501.0
        res2 = calculate_position_size(
            equity=equity2,
            entry_price=34100.0,
            initial_sl=34000.0,
            direction=TradeDirection.BUY,
            spec=spec,
        )
        self.assertEqual(res2.normalized_volume, 2.05)

    # -----------------------------------------------------------------
    # 17. Deterministic Invariant Enforcement Across Scenarios
    # -----------------------------------------------------------------
    def test_deterministic_invariants_for_valid_cases(self) -> None:
        """Enforce invariants for valid trade setups without blanket exception swallowing."""
        valid_scenarios = [
            (5000.0, 34000.0, 33950.0, self.spec_dj),      # Target $50, dist 50, vol 1.0
            (10000.0, 34000.0, 33900.0, self.spec_dj),     # Target $100, dist 100, vol 1.0
            (50000.0, 18000.0, 17950.0, self.spec_nq),     # Target $500, dist 50, vol 5.0
            (125000.0, 35000.0, 34975.0, self.spec_dj),    # Target $1250, dist 25, vol 50.0
        ]

        for equity, entry, sl, spec in valid_scenarios:
            res = calculate_position_size(
                equity=equity,
                entry_price=entry,
                initial_sl=sl,
                direction=TradeDirection.BUY,
                spec=spec,
            )

            # Invariant 1: Exact step multiple
            step_d = Decimal(str(spec.volume_step))
            norm_d = Decimal(str(res.normalized_volume))
            self.assertEqual(norm_d % step_d, Decimal("0"))

            # Invariant 2: Within volume bounds
            self.assertGreaterEqual(res.normalized_volume, spec.volume_min)
            self.assertLessEqual(res.normalized_volume, spec.volume_max)

            # Invariant 3: Actual risk <= target risk
            self.assertLessEqual(res.actual_risk_money, res.target_risk_money + 1e-9)

    def test_explicit_rejection_for_undersized_scenarios(self) -> None:
        """Assert explicit MinimumVolumeViolationError for known undersized scenarios."""
        undersized_scenarios = [
            (100.0, 34000.0, 33950.0, self.spec_dj),   # Target $1, dist 50 -> Raw vol 0.02 < 0.1
            (250.0, 34000.0, 33900.0, self.spec_dj),   # Target $2.5, dist 100 -> Raw vol 0.025 < 0.1
            (500.0, 18000.0, 17900.0, self.spec_nq),   # Target $5, dist 100 -> Raw vol 0.025 < 0.1
        ]

        for equity, entry, sl, spec in undersized_scenarios:
            with self.assertRaises(MinimumVolumeViolationError):
                calculate_position_size(
                    equity=equity,
                    entry_price=entry,
                    initial_sl=sl,
                    direction=TradeDirection.BUY,
                    spec=spec,
                )

    # -----------------------------------------------------------------
    # 18. validate_risk Method Verification
    # -----------------------------------------------------------------
    def test_validate_risk_method(self) -> None:
        """Verify RiskEngine.validate_risk deterministic validation without swallowing bugs."""
        engine = RiskEngine()

        # Valid order volume (volume 2.0 -> risk $100 == target $100)
        self.assertTrue(
            engine.validate_risk(
                equity=10000.0,
                volume=2.0,
                entry_price=34100.0,
                initial_sl=34050.0,
                direction=TradeDirection.BUY,
                spec=self.spec_dj,
            )
        )

        # Exceeds risk (volume 2.5 -> risk $125 > target $100) -> returns False
        self.assertFalse(
            engine.validate_risk(
                equity=10000.0,
                volume=2.5,
                entry_price=34100.0,
                initial_sl=34050.0,
                direction=TradeDirection.BUY,
                spec=self.spec_dj,
            )
        )

        # Invalid inputs return False cleanly
        self.assertFalse(
            engine.validate_risk(
                equity=float("nan"),
                volume=2.0,
                entry_price=34100.0,
                initial_sl=34050.0,
                direction=TradeDirection.BUY,
                spec=self.spec_dj,
            )
        )
        self.assertFalse(
            engine.validate_risk(
                equity=10000.0,
                volume=-1.0,
                entry_price=34100.0,
                initial_sl=34050.0,
                direction=TradeDirection.BUY,
                spec=self.spec_dj,
            )
        )
        self.assertFalse(
            engine.validate_risk(
                equity=True,  # type: ignore
                volume=2.0,
                entry_price=34100.0,
                initial_sl=34050.0,
                direction=TradeDirection.BUY,
                spec=self.spec_dj,
            )
        )
        self.assertFalse(
            engine.validate_risk(
                equity=10000.0,
                volume=True,  # type: ignore
                entry_price=34100.0,
                initial_sl=34050.0,
                direction=TradeDirection.BUY,
                spec=self.spec_dj,
            )
        )

    # -----------------------------------------------------------------
    # 19. BOOLEAN INPUT REJECTION TESTS (True/False != Numbers)
    # -----------------------------------------------------------------
    def test_equity_boolean_rejected(self) -> None:
        """True and False for equity must be strictly rejected."""
        for val in [True, False]:
            with self.assertRaises(InsufficientEquityError):
                calculate_position_size(
                    equity=val,  # type: ignore
                    entry_price=34100.0,
                    initial_sl=34050.0,
                    direction=TradeDirection.BUY,
                    spec=self.spec_dj,
                )

    def test_entry_price_boolean_rejected(self) -> None:
        """True and False for entry_price must be strictly rejected."""
        for val in [True, False]:
            with self.assertRaises(DataValidationError):
                calculate_position_size(
                    equity=10000.0,
                    entry_price=val,  # type: ignore
                    initial_sl=34050.0,
                    direction=TradeDirection.BUY,
                    spec=self.spec_dj,
                )

    def test_initial_sl_boolean_rejected(self) -> None:
        """True and False for initial_sl must be strictly rejected."""
        for val in [True, False]:
            with self.assertRaises(DataValidationError):
                calculate_position_size(
                    equity=10000.0,
                    entry_price=34100.0,
                    initial_sl=val,  # type: ignore
                    direction=TradeDirection.BUY,
                    spec=self.spec_dj,
                )

    def test_spec_fields_boolean_rejected(self) -> None:
        """True and False for all numeric fields of SymbolSpecification must be rejected."""
        spec_fields = ["tick_size", "tick_value", "volume_step", "volume_min", "volume_max"]
        for field in spec_fields:
            for bool_val in [True, False]:
                spec_dict = {
                    "canonical_symbol": CanonicalSymbol.DOW_JONES,
                    "broker_symbol": "US30",
                    "tick_size": 1.0,
                    "tick_value": 1.0,
                    "contract_size": 1.0,
                    "volume_min": 0.1,
                    "volume_max": 100.0,
                    "volume_step": 0.01,
                    "point": 1.0,
                }
                spec_dict[field] = bool_val
                # When creating or validating
                try:
                    bad_spec = SymbolSpecification(**spec_dict)
                    with self.assertRaises(
                        (InvalidSymbolSpecificationError, DataValidationError),
                        msg=f"Field {field} allowed boolean {bool_val}",
                    ):
                        validate_symbol_specification(bad_spec)
                except DataValidationError:
                    pass  # SymbolSpecification itself rejected it

    def test_position_sizing_result_rejects_booleans(self) -> None:
        """PositionSizingResult must reject boolean values for all numeric fields."""
        fields = [
            "target_risk_money", "risk_distance", "risk_ticks",
            "risk_per_lot", "raw_volume", "normalized_volume", "actual_risk_money"
        ]
        base_args = {
            "target_risk_money": 100.0,
            "risk_distance": 50.0,
            "risk_ticks": 50.0,
            "risk_per_lot": 50.0,
            "raw_volume": 2.0,
            "normalized_volume": 2.0,
            "actual_risk_money": 100.0,
            "is_capped": False,
        }

        for f in fields:
            for bool_val in [True, False]:
                bad_args = dict(base_args)
                bad_args[f] = bool_val
                with self.assertRaises(DataValidationError, msg=f"Field {f} accepted boolean {bool_val}"):
                    PositionSizingResult(**bad_args)

    # -----------------------------------------------------------------
    # 20. RiskEngine Immutable Structure Verification
    # -----------------------------------------------------------------
    def test_risk_engine_has_no_mutable_target_risk_state(self) -> None:
        """Verify RiskEngine has no mutable internal state for target risk fraction."""
        engine = RiskEngine()
        self.assertFalse(hasattr(engine, "_target_risk_fraction"))
        self.assertEqual(engine.target_risk_fraction, 0.01)

        # Mutating any attribute on engine cannot alter calculate_position_size
        engine.some_custom_attr = 0.05  # type: ignore
        res = engine.calculate_position_size(
            equity=10000.0,
            entry_price=34100.0,
            initial_sl=34050.0,
            direction=TradeDirection.BUY,
            spec=self.spec_dj,
        )
        self.assertEqual(res.target_risk_money, 100.0)  # Always 1% of 10000 = 100.0


if __name__ == "__main__":
    unittest.main()
