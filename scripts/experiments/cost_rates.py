
"""Generate CEM-ready UHS cost samples for four storage formulations.

The generator samples a uniform grid in:
    - EPR factor relative to the technology's default EPR;
    - WTIR = withdrawal capacity / injection capacity.

Working-gas capacity remains at the reference size when calibrating costs.

Each sample contains:
    - Detailed CostBreakdown
    - CEM-ready CostBreakdown
    - CAPEX + ABEX investment rates (TCR)
    - Fixed and variable OPEX rates
    - Applicable EPR and WTIR intervals

For the cross-driver method only, coefficients are fitted independently in
each sample's EPR/WTIR cell, with a small storage-capacity sensitivity to
identify both slopes. A sample's fitted total need not reproduce the detailed
cost at its centre; this approximation error is reported rather than removed
by anchoring.

Grid points are bin MIDPOINTS. The reference design is not forced into the
sample set.

Applicability intervals are metadata. The CEM must decide how to enforce
them, including whether ratios are fixed or bounded.

Run from the repository root:

    pixi run python -m scripts.experiments.cost_rates `
        --technology aquifer `
        --method fully_decoupled_cross_driver `
        --n-epr 3 `
        --n-wtir 1
"""

from __future__ import annotations

import argparse
import csv

from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import product
from math import isclose, isfinite
from pathlib import Path

import numpy as np

from uhs_costs.cost_model.helpers.cost_components import (
    CostBreakdown,
    CostComponent,
    CostDriver,
    CostType,
    CostUnit,
    HyStoriesGroup,
)
from uhs_costs.analysis.linear_cost_fit import fit_two_driver_cost

from scripts.experiments.linearisation_sensitivity import (
    TECHNOLOGIES,
    TechnologySpec,
    construct_project,
)


# =============================================================================
# Configuration and output types
# =============================================================================


class CostMethod(StrEnum):
    BUNDLED = "bundled"
    PARTIALLY_DECOUPLED = "partially_decoupled"
    FULLY_DECOUPLED = "fully_decoupled"
    FULLY_DECOUPLED_CROSS_DRIVER = "fully_decoupled_cross_driver"


@dataclass(frozen=True)
class SamplingConfig:
    technology: str
    method: CostMethod | str

    n_epr: int = 1
    n_wtir: int = 1

    # EPR is a factor of the technology-specific reference EPR.
    epr_factor_range: tuple[float, float] = (0.5, 1.5)

    # WTIR is an absolute ratio.
    wtir_range: tuple[float, float] = (1.0, 5.0)

    # Only the cross-driver formulation uses this extra capacity sensitivity
    # to identify coefficients. It does NOT add storage-capacity samples.
    cross_driver_storage_perturbation: float = 0.10

@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float

    lower_inclusive: bool = True
    upper_inclusive: bool = True

    def contains(self, value: float) -> bool:
        lower_ok = (
            value >= self.lower
            if self.lower_inclusive
            else value > self.lower
        )
        upper_ok = (
            value <= self.upper
            if self.upper_inclusive
            else value < self.upper
        )
        return lower_ok and upper_ok


@dataclass(frozen=True)
class CapacityRates:
    """Storage EUR/kWh; injection and withdrawal EUR/kW."""

    storage: float
    injection: float
    withdrawal: float


@dataclass(frozen=True)
class ThroughputRates:
    """Variable operating costs in EUR/kWh of hydrogen throughput."""

    injection: float
    withdrawal: float


@dataclass(frozen=True)
class FitDiagnostic:
    group: str
    cost_type: str
    mean_absolute_percentage_error: float
    max_absolute_percentage_error: float


@dataclass(frozen=True)
class SampledCostSet:
    technology: str
    method: CostMethod
    sample_id: str

    storage_reference_gwh_lhv: float

    # Sample calibration point.
    epr_factor: float
    epr_days: float
    wtir: float

    # Validity region of this sample.
    epr_factor_interval: Interval
    epr_interval_days: Interval
    wtir_interval: Interval

    # Capacities of the reference project at this design point.
    storage_capacity_kwh_lhv: float
    injection_capacity_kw: float
    withdrawal_capacity_kw: float

    # Original detailed cost breakdown and transformed CEM representation.
    detailed_breakdown: CostBreakdown
    cem_breakdown: CostBreakdown

    # Ready for translation into CEM cost parameters.
    tcr_rates: CapacityRates
    fom_rates: CapacityRates
    vom_rates: ThroughputRates

    cross_driver_diagnostics: tuple[FitDiagnostic, ...] = ()

    # Not necessarily zero: unanchored local fits need not match the detailed
    # investment cost at their sample centre.
    centre_tcr_error_pct: float = 0.0


CAPITAL_TYPES = (CostType.CAPEX, CostType.ABEX)

CAPACITY_DRIVERS = (
    CostDriver.STORAGE,
    CostDriver.INJECTION,
    CostDriver.WITHDRAWAL,
)

THROUGHPUT_DRIVERS = (
    CostDriver.INJECTION_THROUGHPUT_KWH,
    CostDriver.WITHDRAWAL_THROUGHPUT_KWH,
)


# =============================================================================
# Reference projects and sampling grid
# =============================================================================


def validate_config(config: SamplingConfig) -> CostMethod:
    method = CostMethod(config.method)

    if type(config.n_epr) is not int or type(config.n_wtir) is not int:
        raise TypeError("n_epr and n_wtir must be integers.")

    if config.n_epr < 1 or config.n_wtir < 1:
        raise ValueError("n_epr and n_wtir must be >= 1.")

    for name, bounds in (
        ("epr_factor_range", config.epr_factor_range),
        ("wtir_range", config.wtir_range),
    ):
        if (
            len(bounds) != 2
            or not all(isfinite(v) for v in bounds)
            or not 0 < bounds[0] < bounds[1]
        ):
            raise ValueError(
                f"{name} must satisfy 0 < lower < upper, "
                "with finite values."
            )


    if not 0 < config.cross_driver_storage_perturbation < 1:
        raise ValueError(
            "cross_driver_storage_perturbation must be strictly between 0 and 1."
        )

    return method


def get_technology(name: str) -> TechnologySpec:
    for spec in TECHNOLOGIES:
        if spec.technology == name:
            return spec

    available = ", ".join(spec.technology for spec in TECHNOLOGIES)
    raise ValueError(f"Unknown technology {name!r}. Available: {available}")


def make_grid(
    bounds: tuple[float, float],
    n: int,
) -> tuple[tuple[float, Interval], ...]:
    """Return n evenly spaced cells and their midpoint sample values."""

    edges = np.linspace(bounds[0], bounds[1], n + 1)

    return tuple(
        (
            float((edges[i] + edges[i + 1]) / 2),
            Interval(
                lower=float(edges[i]),
                upper=float(edges[i + 1]),
                lower_inclusive=(i == 0),
            ),
        )
        for i in range(n)
    )


def design_capacities(
    spec: TechnologySpec,
    epr_days: float,
    wtir: float,
    storage_gwh: float | None = None,
) -> dict[CostDriver, float]:
    """Return reference design capacities in kWh and kW."""

    e_gwh = (
        spec.default_storage_capacity_gwh_lhv
        if storage_gwh is None
        else storage_gwh
    )

    E = e_gwh * 1e6
    P_out = E / (24.0 * epr_days)
    P_in = P_out / wtir

    return {
        CostDriver.STORAGE: E,
        CostDriver.INJECTION: P_in,
        CostDriver.WITHDRAWAL: P_out,
    }


def evaluate_design(
    spec: TechnologySpec,
    storage_gwh: float,
    epr_days: float,
    wtir: float,
) -> CostBreakdown:
    """Run the detailed project constructor and cost model."""

    project = construct_project(
        spec=spec,
        storage_capacity_gwh_lhv=storage_gwh,
        epr_days=epr_days,
        wtir=wtir,
    )

    result = spec.cost_function(project)

    if not isinstance(result, CostBreakdown):
        raise TypeError(
            f"Expected CostBreakdown, received {type(result).__name__}."
        )

    for component in result.components:
        if not (
            isfinite(component.value_eur)
            and isfinite(component.driver_value)
            and component.driver_value > 0
        ):
            raise ValueError(
                f"Invalid cost or driver value: {component.name}"
            )

        if component.cost_type == CostType.VARIABLE_OPEX:
            if component.cost_driver not in THROUGHPUT_DRIVERS:
                raise ValueError(
                    f"Unexpected VOM driver in {component.name}."
                )
        elif component.cost_driver not in CAPACITY_DRIVERS:
            raise ValueError(
                f"Unexpected investment driver in {component.name}."
            )

    return result


# =============================================================================
# Direct cost mappings
# =============================================================================


def direct_cost_mapping(
    detailed: CostBreakdown,
    method: CostMethod,
    capacities: dict[CostDriver, float],
) -> CostBreakdown:
    """Map detailed costs to bundled, partial or fully decoupled CEM drivers.

    Capital and fixed OPEX components are reassigned to the relevant capacity
    variable. Their absolute value at the sample centre stays unchanged.

    Variable OPEX remains attached to hydrogen throughput.
    """

    mapped = []

    for component in detailed.components:

        if component.cost_type == CostType.VARIABLE_OPEX:
            mapped.append(component)
            continue

        driver = component.cost_driver

        if method == CostMethod.BUNDLED:
            driver = CostDriver.STORAGE

        elif (
            method == CostMethod.PARTIALLY_DECOUPLED
            and driver == CostDriver.INJECTION
        ):
            # Fixed WTIR convention:
            # injection investment cost is mapped to withdrawal capacity.
            driver = CostDriver.WITHDRAWAL

        if component.cost_type == CostType.FIXED_OPEX:
            unit = (
                CostUnit.EUR_PER_KWH_YEAR
                if driver == CostDriver.STORAGE
                else CostUnit.EUR_PER_KW_YEAR
            )
        else:
            unit = (
                CostUnit.EUR_PER_KWH
                if driver == CostDriver.STORAGE
                else CostUnit.EUR_PER_KW
            )

        mapped.append(
            replace(
                component,
                cost_driver=driver,
                driver_value=capacities[driver],
                cost_unit=unit,
                notes=(
                    (component.notes or "")
                    + f" | CEM mapping: {method.value}"
                ),
            )
        )

    return CostBreakdown(components=tuple(mapped))


def extract_cem_rates(
    breakdown: CostBreakdown,
    capacities: dict[CostDriver, float],
) -> tuple[CapacityRates, CapacityRates, ThroughputRates]:
    """Extract investment, fixed OPEX and throughput rates."""

    investment = {d: 0.0 for d in CAPACITY_DRIVERS}
    fixed_opex = {d: 0.0 for d in CAPACITY_DRIVERS}
    variable_opex = {d: 0.0 for d in THROUGHPUT_DRIVERS}

    for component in breakdown.components:

        if component.cost_type in CAPITAL_TYPES:
            investment[component.cost_driver] += component.value_eur

        elif component.cost_type == CostType.FIXED_OPEX:
            fixed_opex[component.cost_driver] += component.value_eur

        elif component.cost_type == CostType.VARIABLE_OPEX:
            variable_opex[component.cost_driver] += (
                component.value_eur / component.driver_value
            )

    def capacity_rates(values) -> CapacityRates:
        return CapacityRates(
            storage=(
                values[CostDriver.STORAGE]
                / capacities[CostDriver.STORAGE]
            ),
            injection=(
                values[CostDriver.INJECTION]
                / capacities[CostDriver.INJECTION]
            ),
            withdrawal=(
                values[CostDriver.WITHDRAWAL]
                / capacities[CostDriver.WITHDRAWAL]
            ),
        )

    return (
        capacity_rates(investment),
        capacity_rates(fixed_opex),
        ThroughputRates(
            injection=variable_opex[
                CostDriver.INJECTION_THROUGHPUT_KWH
            ],
            withdrawal=variable_opex[
                CostDriver.WITHDRAWAL_THROUGHPUT_KWH
            ],
        ),
    )


# =============================================================================
# Unanchored, interval-specific cross-driver representation
# =============================================================================


def capital_group_cost(
    breakdown: CostBreakdown,
    cost_type: CostType,
    driver: CostDriver,
) -> float:
    return sum(
        c.value_eur
        for c in breakdown.components
        if c.cost_type == cost_type and c.cost_driver == driver
    )


def local_training_cases(
    spec: TechnologySpec,
    epr_factor_interval: Interval,
    wtir_interval: Interval,
    storage_perturbation: float,
) -> list[tuple[dict[CostDriver, float], CostBreakdown]]:
    """Evaluate a small regular grid *inside* one sample's validity cell.

    All three independent design quantities vary. The three EPR and WTIR
    points lie 20%, 50%, and 80% of the way across each interval so we do not
    evaluate adjacent cells' shared boundaries. The three storage values
    provide independent scale variation without sampling E in the CEM.

    No global EPR >= 0.9 cutoff applies: low-EPR samples must learn from
    their own region, including any nonlinear or discontinuous behaviour.
    """

    def interior_grid(interval: Interval) -> tuple[float, float, float]:
        width = interval.upper - interval.lower
        return tuple(
            interval.lower + fraction * width
            for fraction in (0.2, 0.5, 0.8)
        )

    storage_reference = spec.default_storage_capacity_gwh_lhv
    storage_values = (
        storage_reference * (1 - storage_perturbation),
        storage_reference,
        storage_reference * (1 + storage_perturbation),
    )

    cases = []
    for storage_gwh, epr_factor, wtir in product(
        storage_values,
        interior_grid(epr_factor_interval),
        interior_grid(wtir_interval),
    ):
        epr_days = epr_factor * spec.default_epr_days
        capacities = design_capacities(
            spec, epr_days=epr_days, wtir=wtir, storage_gwh=storage_gwh
        )
        breakdown = evaluate_design(
            spec, storage_gwh=storage_gwh, epr_days=epr_days, wtir=wtir
        )
        cases.append((capacities, breakdown))

    return cases


def fit_local_two_driver(
    cases: list[tuple[dict[CostDriver, float], CostBreakdown]],
    cost_type: CostType,
    source_driver: CostDriver,
    own_driver: CostDriver,
    second_driver: CostDriver,
) -> tuple[dict[CostDriver, float], FitDiagnostic]:
    """Fit two unconstrained slopes and report error on local training cases."""

    own = np.asarray([
        capacities[own_driver]
        for capacities, _ in cases
    ])

    secondary = np.asarray([
        capacities[second_driver]
        for capacities, _ in cases
    ])

    y = np.asarray([
        capital_group_cost(
            breakdown,
            cost_type,
            source_driver,
        )
        for _, breakdown in cases
    ])

    # An absent cost group is represented by a zero coefficient, not a
    # numerically unstable regression against an all-zero target.
    if np.allclose(y, 0.0, atol=1e-8, rtol=0):
        a = b = 0.0
    else:
        a, b = fit_two_driver_cost(
            own_capacity=own,
            secondary_capacity=secondary,
            detailed_cost=y,
        )

    prediction = a * own + b * secondary
    nonzero = np.abs(y) > 1e-8
    percentage_errors = (
        100.0 * np.abs(prediction[nonzero] - y[nonzero]) / np.abs(y[nonzero])
    )
    diagnostic = FitDiagnostic(
        group=source_driver.value,
        cost_type=cost_type.value,
        mean_absolute_percentage_error=(
            float(np.mean(percentage_errors)) if percentage_errors.size else 0.0
        ),
        max_absolute_percentage_error=(
            float(np.max(percentage_errors)) if percentage_errors.size else 0.0
        ),
    )

    return {own_driver: a, second_driver: b}, diagnostic

def cross_driver_cost_mapping(
    spec: TechnologySpec,
    detailed: CostBreakdown,
    capacities: dict[CostDriver, float],
    epr_factor_interval: Interval,
    wtir_interval: Interval,
    storage_perturbation: float = 0.10,
) -> tuple[CostBreakdown, tuple[FitDiagnostic, ...]]:
    """Build an LP-compatible two-slope model for one design-space cell."""

    cases = local_training_cases(
        spec=spec,
        epr_factor_interval=epr_factor_interval,
        wtir_interval=wtir_interval,
        storage_perturbation=storage_perturbation,
    )

    # Detailed source cost -> (primary driver, additional driver).
    #
    # Storage costs can depend on P_out through EPR.
    # Injection costs can depend on P_out as well as P_in through WTIR.
    # Withdrawal costs can depend on E as well as P_out through EPR.
    models = (
        (
            CostDriver.STORAGE,
            CostDriver.STORAGE,
            CostDriver.WITHDRAWAL,
        ),
        (
            CostDriver.INJECTION,
            CostDriver.INJECTION,
            CostDriver.WITHDRAWAL,
        ),
        (
            CostDriver.WITHDRAWAL,
            CostDriver.WITHDRAWAL,
            CostDriver.STORAGE,
        ),
    )

    # Aggregate fitted coefficients by CEM decision variable.
    # CAPEX and ABEX are fitted separately to preserve cost classification.
    coefficients = {
        typ: {driver: 0.0 for driver in CAPACITY_DRIVERS}
        for typ in CAPITAL_TYPES
    }

    diagnostics = []

    for typ in CAPITAL_TYPES:
        for source, own, additional in models:
            rates, diagnostic = fit_local_two_driver(
                cases=cases,
                cost_type=typ,
                source_driver=source,
                own_driver=own,
                second_driver=additional,
            )

            for driver, rate in rates.items():
                coefficients[typ][driver] += rate

            diagnostics.append(diagnostic)

    # Create synthetic capital components with the fitted coefficients.
    # These represent linear objective terms, not physical EPC components.
    mapped_components = []

    for typ in CAPITAL_TYPES:
        for driver in CAPACITY_DRIVERS:
            rate = coefficients[typ][driver]
            capacity = capacities[driver]

            mapped_components.append(
                CostComponent(
                    name=f"cross_driver_{typ.value}_{driver.value}",
                    value_eur=rate * capacity,
                    cost_type=typ,
                    hystories_group=(
                        HyStoriesGroup.SUBSURFACE
                        if driver == CostDriver.STORAGE
                        else HyStoriesGroup.SURFACE
                    ),
                    cost_driver=driver,
                    driver_value=capacity,
                    cost_unit=(
                        CostUnit.EUR_PER_KWH
                        if driver == CostDriver.STORAGE
                        else CostUnit.EUR_PER_KW
                    ),
                    notes=(
                        "Synthetic unanchored local cross-driver investment "
                        "coefficient; not a physical component."
                    ),
                )
            )

    # Retain FOM and throughput VOM without cross-driver fitting.
    mapped_components.extend(
        component
        for component in detailed.components
        if component.cost_type not in CAPITAL_TYPES
    )

    mapped = CostBreakdown(
        components=tuple(mapped_components)
    )

    return mapped, tuple(diagnostics)


# =============================================================================
# Main generator
# =============================================================================


def generate_cost_samples(
    config: SamplingConfig,
) -> list[SampledCostSet]:
    """Generate n_epr * n_wtir cost sets with bounded validity intervals."""

    method = validate_config(config)
    spec = get_technology(config.technology)


    epr_grid = make_grid(
        config.epr_factor_range,
        config.n_epr,
    )

    wtir_grid = make_grid(
        config.wtir_range,
        config.n_wtir,
    )

    samples = []

    for (
        (i, (epr_factor, epr_factor_interval)),
        (j, (wtir, wtir_interval)),
    ) in product(
        enumerate(epr_grid, start=1),
        enumerate(wtir_grid, start=1),
    ):

        epr_days = epr_factor * spec.default_epr_days

        epr_interval_days = Interval(
            lower=(
                epr_factor_interval.lower
                * spec.default_epr_days
            ),
            upper=(
                epr_factor_interval.upper
                * spec.default_epr_days
            ),
            lower_inclusive=epr_factor_interval.lower_inclusive,
            upper_inclusive=epr_factor_interval.upper_inclusive,
        )

        capacities = design_capacities(
            spec,
            epr_days=epr_days,
            wtir=wtir,
        )

        detailed = evaluate_design(
            spec=spec,
            storage_gwh=spec.default_storage_capacity_gwh_lhv,
            epr_days=epr_days,
            wtir=wtir,
        )

        diagnostics = ()

        if method == CostMethod.FULLY_DECOUPLED_CROSS_DRIVER:
            cem_breakdown, diagnostics = cross_driver_cost_mapping(
                spec=spec,
                detailed=detailed,
                capacities=capacities,
                epr_factor_interval=epr_factor_interval,
                wtir_interval=wtir_interval,
                storage_perturbation=config.cross_driver_storage_perturbation,
            )

        else:
            cem_breakdown = direct_cost_mapping(
                detailed=detailed,
                method=method,
                capacities=capacities,
            )

        tcr_rates, fom_rates, vom_rates = extract_cem_rates(
            cem_breakdown,
            capacities,
        )

        # Direct reallocation preserves the detailed investment total at
        # the centre. An unanchored cross-driver fit is allowed to differ.
        original_tcr = sum(
            c.value_eur for c in detailed.components
            if c.cost_type in CAPITAL_TYPES
        )

        mapped_tcr = sum(
            c.value_eur for c in cem_breakdown.components
            if c.cost_type in CAPITAL_TYPES
        )

        centre_tcr_error_pct = (
            100.0 * (mapped_tcr - original_tcr) / original_tcr
            if original_tcr else 0.0
        )

        if (
            method != CostMethod.FULLY_DECOUPLED_CROSS_DRIVER
            and not isclose(
                original_tcr,
                mapped_tcr,
                rel_tol=1e-9,
                abs_tol=1e-3,
            )
        ):
            raise AssertionError(
                "Direct CEM reallocation failed to preserve sample-centre TCR."
            )

        # Negative total capacity coefficients can be problematic in a
        # continuously expandable LP. Flag them rather than exporting
        # a potentially unbounded cost representation.
        if min(
            tcr_rates.storage,
            tcr_rates.injection,
            tcr_rates.withdrawal,
        ) < -1e-9:
            raise ValueError(
                f"Negative fitted TCR coefficient for "
                f"{spec.technology}, sample ({i}, {j}). "
                "Inspect the local fit or narrow the sample interval."
            )

        sample = SampledCostSet(
            technology=spec.technology,
            method=method,
            sample_id=(
                f"{spec.technology}__{method.value}"
                f"__epr{i:02d}_wtir{j:02d}"
            ),
            storage_reference_gwh_lhv=(
                spec.default_storage_capacity_gwh_lhv
            ),
            epr_factor=epr_factor,
            epr_days=epr_days,
            wtir=wtir,
            epr_factor_interval=epr_factor_interval,
            epr_interval_days=epr_interval_days,
            wtir_interval=wtir_interval,
            storage_capacity_kwh_lhv=capacities[CostDriver.STORAGE],
            injection_capacity_kw=capacities[CostDriver.INJECTION],
            withdrawal_capacity_kw=capacities[CostDriver.WITHDRAWAL],
            detailed_breakdown=detailed,
            cem_breakdown=cem_breakdown,
            tcr_rates=tcr_rates,
            fom_rates=fom_rates,
            vom_rates=vom_rates,
            cross_driver_diagnostics=diagnostics,
            centre_tcr_error_pct=centre_tcr_error_pct,
        )

        samples.append(sample)

    return samples


# =============================================================================
# Optional CSV export
# =============================================================================


def write_sample_summary(
    samples: list[SampledCostSet],
    filename: str | Path,
) -> None:
    """Write flattened CEM coefficients and applicability bounds to CSV."""

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for s in samples:
        rows.append({
            "sample_id": s.sample_id,
            "technology": s.technology,
            "method": s.method.value,
            "E_reference_GWh": s.storage_reference_gwh_lhv,
            "EPR_factor": s.epr_factor,
            "EPR_days": s.epr_days,
            "WTIR": s.wtir,
            "EPR_factor_lower": s.epr_factor_interval.lower,
            "EPR_factor_upper": s.epr_factor_interval.upper,
            "EPR_factor_lower_inclusive": s.epr_factor_interval.lower_inclusive,
            "EPR_factor_upper_inclusive": s.epr_factor_interval.upper_inclusive,
            "EPR_days_lower": s.epr_interval_days.lower,
            "EPR_days_upper": s.epr_interval_days.upper,
            "WTIR_lower": s.wtir_interval.lower,
            "WTIR_upper": s.wtir_interval.upper,
            "WTIR_lower_inclusive": s.wtir_interval.lower_inclusive,
            "WTIR_upper_inclusive": s.wtir_interval.upper_inclusive,
            "TCR_storage_EUR_per_kWh": s.tcr_rates.storage,
            "TCR_injection_EUR_per_kW": s.tcr_rates.injection,
            "TCR_withdrawal_EUR_per_kW": s.tcr_rates.withdrawal,
            "FOM_storage_EUR_per_kWh_year": s.fom_rates.storage,
            "FOM_injection_EUR_per_kW_year": s.fom_rates.injection,
            "FOM_withdrawal_EUR_per_kW_year": s.fom_rates.withdrawal,
            "VOM_injection_EUR_per_kWh": s.vom_rates.injection,
            "VOM_withdrawal_EUR_per_kWh": s.vom_rates.withdrawal,
            "centre_TCR_error_pct": s.centre_tcr_error_pct,
            "local_fit_max_APE_pct": (
                max(d.max_absolute_percentage_error for d in s.cross_driver_diagnostics)
                if s.cross_driver_diagnostics else 0.0
            ),
        })

    if not rows:
        return

    with filename.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Command-line entry point
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CEM-ready UHS cost samples."
    )

    parser.add_argument(
        "--technology",
        required=True,
        choices=[s.technology for s in TECHNOLOGIES],
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=[m.value for m in CostMethod],
    )
    parser.add_argument("--n-epr", type=int, default=1)
    parser.add_argument("--n-wtir", type=int, default=1)

    parser.add_argument("--epr-min-factor", type=float, default=0.5)
    parser.add_argument("--epr-max-factor", type=float, default=1.5)
    parser.add_argument("--wtir-min", type=float, default=1.0)
    parser.add_argument("--wtir-max", type=float, default=5.0)
    parser.add_argument(
        "--cross-driver-storage-perturbation",
        type=float,
        default=0.10,
        help="Fractional +/- E perturbation for local cross-driver regression only.",
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV summary destination.",
    )

    args = parser.parse_args()

    config = SamplingConfig(
        technology=args.technology,
        method=args.method,
        n_epr=args.n_epr,
        n_wtir=args.n_wtir,
        epr_factor_range=(
            args.epr_min_factor,
            args.epr_max_factor,
        ),
        wtir_range=(
            args.wtir_min,
            args.wtir_max,
        ),
        cross_driver_storage_perturbation=(
            args.cross_driver_storage_perturbation
        ),
    )

    samples = generate_cost_samples(config)

    for s in samples:
        print(
            f"{s.sample_id}: "
            f"EPR={s.epr_days:.3g} days, "
            f"WTIR={s.wtir:.3g}, "
            f"TCR=[{s.tcr_rates.storage:.4f} EUR/kWh, "
            f"{s.tcr_rates.injection:.2f} EUR/kW, "
            f"{s.tcr_rates.withdrawal:.2f} EUR/kW]"
            + (
                f", centre TCR error={s.centre_tcr_error_pct:+.2f}%"
                if s.method == CostMethod.FULLY_DECOUPLED_CROSS_DRIVER
                else ""
            )
        )

    if args.csv is not None:
        write_sample_summary(samples, args.csv)
        print(f"Saved {len(samples)} samples to {args.csv}")


if __name__ == "__main__":
    main()
