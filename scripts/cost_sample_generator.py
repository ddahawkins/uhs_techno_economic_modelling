
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

Grid points are bin MIDPOINTS. The reference design is not forced into the
sample set.

Applicability intervals are metadata. The CEM must decide how to enforce
them, including whether ratios are fixed or bounded.

Run from the repository root:

    pixi run python -m scripts.cost_sample_generator `
        --technology aquifer `
        --method fully_decoupled `
        --n-epr 1 `
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

    # Local relative perturbation for anchored cross-driver fitting.
    # This does NOT create additional CEM investment samples.
    cross_driver_fit_step: float = 0.10


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

    if not 0 < config.cross_driver_fit_step < 0.5:
        raise ValueError(
            "cross_driver_fit_step must lie between 0 and 0.5."
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
# Reference-anchored cross-driver representation
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
    epr_days: float,
    wtir: float,
    epr_factor_interval: Interval,
    wtir_interval: Interval,
    step: float,
) -> list[tuple[dict[CostDriver, float], CostBreakdown]]:
    """Construct nearby designs used to fit local cross-driver coefficients.

    The first case is always the sample centre. This anchors the regression
    to its own calibration point, NOT necessarily the global reference design.
    """

    E0 = spec.default_storage_capacity_gwh_lhv

    designs = [(E0, epr_days, wtir)]

    # Vary E to learn how approximately fixed costs scale with storage size.
    for direction in (-1, 1):
        designs.append((
            E0 * (1 + direction * step),
            epr_days,
            wtir,
        ))

    # EPR and WTIR perturbations remain inside the sample's validity cell.
    epr_cell_width_days = (
        epr_factor_interval.upper - epr_factor_interval.lower
    ) * spec.default_epr_days

    epr_delta = min(
        epr_days * step,
        0.4 * epr_cell_width_days,
    )

    wtir_delta = min(
        wtir * step,
        0.4 * (wtir_interval.upper - wtir_interval.lower),
    )

    for direction in (-1, 1):
        designs.append((
            E0,
            epr_days + direction * epr_delta,
            wtir,
        ))
        designs.append((
            E0,
            epr_days,
            wtir + direction * wtir_delta,
        ))

    cases = []

    for E, epr, ratio in designs:
        try:
            costs = evaluate_design(spec, E, epr, ratio)
        except (ValueError, ArithmeticError) as exc:
            raise ValueError(
                f"Local cross-driver fit failed for {spec.technology}: "
                f"E={E:.4g} GWh, EPR={epr:.4g} days, "
                f"WTIR={ratio:.4g}. {exc}"
            ) from exc

        caps = design_capacities(spec, epr, ratio, E)
        cases.append((caps, costs))

    return cases


def anchored_two_driver_fit(
    cases: list[tuple[dict[CostDriver, float], CostBreakdown]],
    cost_type: CostType,
    source_driver: CostDriver,
    own_driver: CostDriver,
    second_driver: CostDriver,
) -> tuple[dict[CostDriver, float], FitDiagnostic]:
    """Fit C = a * own + b * second, anchored at the sample centre.

    Using normalized capacities:
        x = own / own_0
        z = second / second_0

    The exact reference constraint is:
        a_scaled + b_scaled = C0

    Therefore:
        C = C0*x + b_scaled*(z-x)

    This yields a one-dimensional, intercept-free least-squares problem.
    """

    caps0, breakdown0 = cases[0]

    C0 = capital_group_cost(
        breakdown0, cost_type, source_driver
    )

    own0 = caps0[own_driver]
    second0 = caps0[second_driver]

    x = np.asarray(
        [caps[own_driver] / own0 for caps, _ in cases],
        dtype=float,
    )
    z = np.asarray(
        [caps[second_driver] / second0 for caps, _ in cases],
        dtype=float,
    )
    y = np.asarray(
        [
            capital_group_cost(costs, cost_type, source_driver)
            for _, costs in cases
        ],
        dtype=float,
    )

    delta = z - x
    denominator = float(np.dot(delta, delta))

    if denominator < 1e-12:
        raise ValueError(
            f"Degenerate cross-driver fit for {source_driver.value}."
        )

    b_scaled = float(
        np.dot(delta, y - C0 * x) / denominator
    )
    a_scaled = C0 - b_scaled

    rates = {
        own_driver: a_scaled / own0,
        second_driver: b_scaled / second0,
    }

    predictions = a_scaled * x + b_scaled * z

    mask = np.abs(y) > 1e-9
    if mask.any():
        errors = 100 * np.abs(
            (predictions[mask] - y[mask]) / y[mask]
        )
        mean_error = float(np.mean(errors))
        max_error = float(np.max(errors))
    else:
        mean_error = 0.0
        max_error = 0.0

    assert isclose(
        sum(rates[d] * caps0[d] for d in rates),
        C0,
        rel_tol=1e-9,
        abs_tol=1e-4,
    )

    return rates, FitDiagnostic(
        group=source_driver.value,
        cost_type=cost_type.value,
        mean_absolute_percentage_error=mean_error,
        max_absolute_percentage_error=max_error,
    )


def cross_driver_cost_mapping(
    spec: TechnologySpec,
    detailed: CostBreakdown,
    capacities: dict[CostDriver, float],
    epr_days: float,
    wtir: float,
    epr_factor_interval: Interval,
    wtir_interval: Interval,
    fit_step: float,
) -> tuple[CostBreakdown, tuple[FitDiagnostic, ...]]:
    """Build the anchored LP-compatible cross-driver cost representation."""

    cases = local_training_cases(
        spec=spec,
        epr_days=epr_days,
        wtir=wtir,
        epr_factor_interval=epr_factor_interval,
        wtir_interval=wtir_interval,
        step=fit_step,
    )

    # Detailed source cost -> (primary driver, additional driver).
    #
    # Storage costs can depend on P_out through EPR.
    # Injection costs can include a capacity-linked component.
    # Withdrawal costs can include a capacity-linked component.
    models = (
        (
            CostDriver.STORAGE,
            CostDriver.STORAGE,
            CostDriver.WITHDRAWAL,
        ),
        (
            CostDriver.INJECTION,
            CostDriver.INJECTION,
            CostDriver.STORAGE,
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
            rates, diagnostic = anchored_two_driver_fit(
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
                        "Synthetic reference-anchored cross-driver "
                        "investment coefficient; not a physical component."
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
                epr_days=epr_days,
                wtir=wtir,
                epr_factor_interval=epr_factor_interval,
                wtir_interval=wtir_interval,
                fit_step=config.cross_driver_fit_step,
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

        # Check that transformed investment costs reproduce the detailed
        # model at the sample centre.
        original_tcr = sum(
            c.value_eur for c in detailed.components
            if c.cost_type in CAPITAL_TYPES
        )

        mapped_tcr = sum(
            c.value_eur for c in cem_breakdown.components
            if c.cost_type in CAPITAL_TYPES
        )

        if not isclose(
            original_tcr,
            mapped_tcr,
            rel_tol=1e-9,
            abs_tol=1e-3,
        ):
            raise AssertionError(
                "CEM representation failed to reproduce "
                "the sample-centre investment cost."
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
                "Inspect the anchored fit or narrow the domain."
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
            "EPR_days_lower": s.epr_interval_days.lower,
            "EPR_days_upper": s.epr_interval_days.upper,
            "WTIR_lower": s.wtir_interval.lower,
            "WTIR_upper": s.wtir_interval.upper,
            "TCR_storage_EUR_per_kWh": s.tcr_rates.storage,
            "TCR_injection_EUR_per_kW": s.tcr_rates.injection,
            "TCR_withdrawal_EUR_per_kW": s.tcr_rates.withdrawal,
            "FOM_storage_EUR_per_kWh_year": s.fom_rates.storage,
            "FOM_injection_EUR_per_kW_year": s.fom_rates.injection,
            "FOM_withdrawal_EUR_per_kW_year": s.fom_rates.withdrawal,
            "VOM_injection_EUR_per_kWh": s.vom_rates.injection,
            "VOM_withdrawal_EUR_per_kWh": s.vom_rates.withdrawal,
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
        )

    if args.csv is not None:
        write_sample_summary(samples, args.csv)
        print(f"Saved {len(samples)} samples to {args.csv}")


if __name__ == "__main__":
    main()