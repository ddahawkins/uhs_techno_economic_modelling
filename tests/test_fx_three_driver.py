"""Synthetic three-by-three FX tests; no geological model dependency required."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from dataclasses import dataclass
from enum import StrEnum
from itertools import product
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "cost_model"
LINEAR_FILE = ROOT / "src/uhs_costs/analysis/linear_cost_fit.py"
COST_FILE = ROOT / "scripts/experiments/cost_rates.py"


def load_fx_module():
    """Supply lightweight data classes so the fit can be tested without CoolProp."""
    for name in (
        "uhs_costs", "uhs_costs.analysis", "uhs_costs.cost_model",
        "uhs_costs.cost_model.helpers", "scripts", "scripts.experiments",
    ):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module

    class CostDriver(StrEnum):
        STORAGE = "storage_capacity_kWh"
        INJECTION = "injection_capacity_kw"
        WITHDRAWAL = "withdrawal_capacity_kw"
        INJECTION_THROUGHPUT_KWH = "injection_throughput_kwh"
        WITHDRAWAL_THROUGHPUT_KWH = "withdrawal_throughput_kwh"

    class CostType(StrEnum):
        CAPEX = "capex"
        ABEX = "abex"
        FIXED_OPEX = "fixed_opex"
        VARIABLE_OPEX = "variable_opex"

    class CostUnit(StrEnum):
        EUR_PER_KW = "EUR/kW"
        EUR_PER_KWH = "EUR/kWh"
        EUR_PER_KW_YEAR = "EUR/kW/year"
        EUR_PER_KWH_YEAR = "EUR/kWh/year"

    class HyStoriesGroup(StrEnum):
        SURFACE = "surface"
        SUBSURFACE = "subsurface"

    @dataclass(frozen=True)
    class CostComponent:
        name: str
        value_eur: float
        cost_type: CostType
        hystories_group: HyStoriesGroup
        cost_driver: CostDriver
        driver_value: float
        cost_unit: CostUnit
        notes: str | None = None

    @dataclass(frozen=True)
    class CostBreakdown:
        components: tuple[CostComponent, ...]

    components_module = types.ModuleType("uhs_costs.cost_model.helpers.cost_components")
    for name, value in locals().copy().items():
        if name in {"CostDriver", "CostType", "CostUnit", "HyStoriesGroup", "CostComponent", "CostBreakdown"}:
            setattr(components_module, name, value)
    sys.modules[components_module.__name__] = components_module

    linear_spec = importlib.util.spec_from_file_location("uhs_costs.analysis.linear_cost_fit", LINEAR_FILE)
    linear = importlib.util.module_from_spec(linear_spec)
    sys.modules[linear_spec.name] = linear
    linear_spec.loader.exec_module(linear)

    @dataclass
    class TechnologySpec:
        technology: str = "synthetic"
        default_storage_capacity_gwh_lhv: float = 1.0
        default_epr_days: float = 10.0

    sensitivity = types.ModuleType("scripts.experiments.linearisation_sensitivity")
    sensitivity.TECHNOLOGIES = ()
    sensitivity.TechnologySpec = TechnologySpec
    sensitivity.construct_project = lambda **kwargs: None
    sys.modules[sensitivity.__name__] = sensitivity

    cost_spec = importlib.util.spec_from_file_location("scripts.experiments.cost_rates", COST_FILE)
    cost = importlib.util.module_from_spec(cost_spec)
    sys.modules[cost_spec.name] = cost
    cost_spec.loader.exec_module(cost)
    return cost, linear, CostComponent, CostBreakdown, CostDriver, CostType, CostUnit, HyStoriesGroup


class FxThreeDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (cls.mod, cls.linear, cls.Component, cls.Breakdown, cls.Driver,
         cls.Type, cls.Unit, cls.Group) = load_fx_module()

    def test_three_drivers_are_independently_identified(self):
        own = np.full(16, 20000.0)
        secondary = np.tile(np.array([2., 4., 6., 8.]), 4) * 100.0
        tertiary = np.repeat(np.array([3., 5., 7., 9.]), 4) * 100.0
        costs = .13 * own + 2.4 * secondary + .7 * tertiary
        a, b, c = self.linear.fit_three_driver_cost(own, secondary, tertiary, costs)
        self.assertAlmostEqual(a, .13, places=10)
        self.assertAlmostEqual(b, 2.4, places=10)
        self.assertAlmostEqual(c, .7, places=10)

    def make_cases(self, negative: bool = False):
        D, T, U, G, Component, Breakdown = (
            self.Driver, self.Type, self.Unit, self.Group,
            self.Component, self.Breakdown
        )
        cases = []
        for factor, wtir in product(np.linspace(.5, 2, 5), np.linspace(1, 10, 5)):
            e = 100000.0
            pout = e / (24.0 * 10.0 * factor)
            pin = pout / wtir
            capacities = {D.STORAGE: e, D.INJECTION: pin, D.WITHDRAWAL: pout}
            components = []
            for typ in (T.CAPEX, T.ABEX, T.FIXED_OPEX):
                for source in (D.STORAGE, D.INJECTION, D.WITHDRAWAL):
                    coeff = {D.STORAGE: .02, D.INJECTION: .15, D.WITHDRAWAL: .25}
                    if negative and typ == T.CAPEX and source == D.STORAGE:
                        coeff[D.INJECTION] = -.1
                    value = sum(coeff[driver] * capacity for driver, capacity in capacities.items())
                    components.append(Component(
                        name=f"{typ.value}/{source.value}", value_eur=value,
                        cost_type=typ, hystories_group=G.SURFACE,
                        cost_driver=source, driver_value=capacities[source],
                        cost_unit=U.EUR_PER_KW,
                    ))
            cases.append((capacities, Breakdown(tuple(components))))
        return cases

    def test_source_fits_aggregate_to_three_cem_rates(self):
        mod, cases = self.mod, self.make_cases()
        original = mod.local_training_cases
        mod.local_training_cases = lambda **kwargs: cases
        try:
            centre_capacities, original_breakdown = cases[0]
            result, diagnostics = mod.cross_driver_cost_mapping(
                spec=types.SimpleNamespace(technology="synthetic"),
                detailed=original_breakdown,
                capacities=centre_capacities,
                epr_factor_interval=mod.Interval(.5, 2.0),
                wtir_interval=mod.Interval(1.0, 10.0),
            )
        finally:
            mod.local_training_cases = original

        self.assertEqual(len(diagnostics), 9)  # 3 source groups x 3 cost types
        self.assertEqual(len(result.components), 9)  # 3 destination rates x 3 cost types
        self.assertTrue(all(d.third_driver not in (d.own_driver, d.additional_driver) for d in diagnostics))
        for typ in (self.Type.CAPEX, self.Type.ABEX, self.Type.FIXED_OPEX):
            expected = sum(c.value_eur for c in original_breakdown.components if c.cost_type == typ)
            actual = sum(c.value_eur for c in result.components if c.cost_type == typ)
            self.assertAlmostEqual(actual, expected, places=6)

    def test_any_negative_source_coefficient_fails(self):
        mod, cases = self.mod, self.make_cases(negative=True)
        original = mod.local_training_cases
        mod.local_training_cases = lambda **kwargs: cases
        try:
            with self.assertRaisesRegex(ValueError, "Negative FX three-driver coefficient"):
                mod.cross_driver_cost_mapping(
                    spec=types.SimpleNamespace(technology="synthetic"),
                    detailed=cases[0][1], capacities=cases[0][0],
                    epr_factor_interval=mod.Interval(.5, 2.0),
                    wtir_interval=mod.Interval(1.0, 10.0),
                )
        finally:
            mod.local_training_cases = original

    def test_default_domains_are_updated(self):
        self.assertEqual(self.mod.STUDY_EPR_FACTOR_RANGE, (.5, 2.))
        self.assertEqual(self.mod.STUDY_WTIR_RANGE, (1., 10.))


if __name__ == "__main__":
    unittest.main()
