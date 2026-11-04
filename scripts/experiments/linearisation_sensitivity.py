"""Analyse linearisation sensitivity of UHS CEM cost rates.

This experiment asks how strongly the CEM-ready unit cost coefficients depend on
which facility design point is used to derive them.

The detailed UHS model is evaluated while varying one of three CEM-facing design
quantities at a time:

* working-gas storage capacity, E;
* energy-to-power ratio (EPR), expressed as full-withdrawal duration;
* withdrawal-to-injection ratio (WTIR = P_out / P_in).

For E and EPR, all technologies are evaluated at the same set of multipliers
relative to their technology-specific default design point. WTIR is shown in its
natural units because all technologies share the same reference value and the
HyStories-style 1--5 range is directly interpretable.

For every detailed project, total-capital-requirement (TCR = CAPEX + ABEX)
rates are derived for the three CEM investment dimensions:

* Injection:  EUR/kW of CEM-facing injection capacity
* Storage:    EUR/kWh of recoverable working-gas capacity
* Withdrawal: EUR/kW of CEM-facing usable withdrawal capacity

The default design point defines the fixed linearised CEM parameterisation. If
the detailed-to-CEM conversion were perfectly linear, re-costing projects away
from this design point would return the same unit cost rates. The main component
figure therefore plots each re-costed TCR rate relative to its default value:

    relative TCR rate = re-costed TCR rate / default TCR rate

A value of 1 means that the unit cost is unchanged from the coefficient supplied
to the CEM; 1.5 means 50% higher; 0.75 means 25% lower.

A second figure reports the implied total-project error if the default CEM rates
are retained away from their calibration point:

    error = 100 * (default-linearised TCR - detailed TCR) / detailed TCR

Positive values therefore mean that the fixed CEM parameterisation overestimates
the detailed project cost.

Outputs
-------
outputs/linearisation_sensitivity/
    linearisation_sensitivity_results.csv
    figures/
        component_tcr_rate_factors.pdf
        total_linearisation_error.pdf

Run from the repository root, for example:

    pixi run python scripts/experiments/linearisation_sensitivity.py

Notes
-----
* The multiplier range is an experiment range, not an uncertainty interval.
* The multiplier 1.0 MUST be present: it is the calibration design point.
* The default WTIR MUST be present in the WTIR sweep.
* Absolute design values and absolute TCR rates are retained in the CSV even
  though the manuscript-style figures use normalised axes/rates.
* All non-varied design assumptions retain the technology constructor defaults.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import csv
import math
from typing import Any, Callable
import numpy as np

import matplotlib.pyplot as plt

from uhs_costs.design.aquifer import (
    construct_project as construct_aquifer_project,
    ASSUMPTIONS as AQUIFER_DESIGN_ASSUMPTIONS,
    REFERENCE_CAPACITY_GWH_LHV as AQUIFER_REFERENCE_CAPACITY,
    REFERENCE_EPR_DAYS as AQUIFER_REFERENCE_EPR,
    REFERENCE_WTIR as AQUIFER_REFERENCE_WTIR
    )

from uhs_costs.design.depleted_gas_field import (
    construct_project as construct_dgf_project,
    ASSUMPTIONS as DGF_DESIGN_ASSUMPTIONS,
    REFERENCE_CAPACITY_GWH_LHV as DGF_REFERENCE_CAPACITY,
    REFERENCE_EPR_DAYS as DGF_REFERENCE_EPR,
    REFERENCE_WTIR as DGF_REFERENCE_WTIR
)
from uhs_costs.design.lined_rock_cavern import (
    construct_project as construct_lrc_project,
    ASSUMPTIONS as LRC_DESIGN_ASSUMPTIONS,
    REFERENCE_CAPACITY_GWH_LHV as LRC_REFERENCE_CAPACITY,
    REFERENCE_EPR_DAYS as LRC_REFERENCE_EPR,
    REFERENCE_WTIR as LRC_REFERENCE_WTIR
)
from uhs_costs.design.salt_cavern import (
    construct_project as construct_salt_cavern_project,
    ASSUMPTIONS as SC_DESIGN_ASSUMPTIONS,
    REFERENCE_CAPACITY_GWH_LHV as SC_REFERENCE_CAPACITY,
    REFERENCE_EPR_DAYS as SC_REFERENCE_EPR,
    REFERENCE_WTIR as SC_REFERENCE_WTIR
)

from uhs_costs.cost_model.aquifer import (
    calculate_cost_components as calculate_aquifer_cost_components,
)
from uhs_costs.cost_model.depleted_gas_field import (
    calculate_depleted_gas_field_cost_components,
)
from uhs_costs.cost_model.lined_rock_cavern import (
    calculate_lined_rock_cavern_cost_components,
)
from uhs_costs.cost_model.salt_cavern import (
    calculate_salt_cavern_cost_components,
)

from uhs_costs.analysis.linear_cost_fit import fit_two_driver_cost


# =============================================================================
# User configuration
# =============================================================================

OUTPUT_DIR = Path("outputs/linearisation_sensitivity")
OUTPUT_CSV = OUTPUT_DIR / "linearisation_sensitivity_results.csv"
FIGURE_DIR = OUTPUT_DIR / "figures"

# Set these to a subset when developing/debugging the experiment.
RUN_TECHNOLOGIES = (
    "salt_cavern",
    "depleted_gas_field",
    "aquifer",
    "lined_rock_cavern",
)
RUN_SENSITIVITIES = (
    "storage_capacity_gwh_lhv",
    "epr_days",
    "wtir",
)
PLOT_SENSITIVITIES = (
    "epr_days",
    "wtir",
)

# Common normalised design-point sweep for both E and EPR.
#
# The same multipliers are used for every technology, which makes the x-axis
# directly comparable across technologies. The corresponding absolute E and EPR
# values are still written to the CSV.
DESIGN_POINT_MULTIPLIERS = np.linspace(0.5,1.5,11) 
# Minimum EPR factor for the *global withdrawal figure* regression only.
# Local cross-driver CEM samples are fitted independently over their own
# EPR/WTIR cells, including cells below 0.9. This plot is a diagnostic of the
# smoother high-EPR region, not a source of sampler coefficients.
CROSS_DRIVER_EPR_MIN_FACTOR = 0.9

# WTIR is deliberately kept in its natural units rather than normalised.
WTIR_SWEEP = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)

# Figure switches make it easy to suppress one output during development.
PLOT_COMPONENT_RATE_FACTORS = True
PLOT_TOTAL_ERROR = False


@dataclass(frozen=True)
class TechnologySpec:
    """Reference design used to derive one technology's default CEM rates."""

    technology: str
    label: str
    case_name: str
    project_constructor: Callable[..., Any]
    cost_function: Callable[..., Any]

    design_assumptions: Any

    # Reference facility used to derive the fixed linear CEM coefficients.
    default_storage_capacity_gwh_lhv: float
    default_epr_days: float
    default_wtir: float


# Reference designs are imported from the technology-specific constructors.
# These are the authoritative reference assumptions for this experiment.
TECHNOLOGIES: tuple[TechnologySpec, ...] = (
    TechnologySpec(
        technology="salt_cavern",
        label="Salt cavern",
        case_name="salt_cavern",
        project_constructor=construct_salt_cavern_project,
        cost_function=calculate_salt_cavern_cost_components,
        design_assumptions=SC_DESIGN_ASSUMPTIONS,
        default_storage_capacity_gwh_lhv=SC_REFERENCE_CAPACITY,
        default_epr_days=SC_REFERENCE_EPR,
        default_wtir=SC_REFERENCE_WTIR,
    ),
    TechnologySpec(
        technology="depleted_gas_field",
        label="Depleted gas field",
        case_name="depleted_gas_field",
        project_constructor=construct_dgf_project,
        cost_function=calculate_depleted_gas_field_cost_components,
        design_assumptions=DGF_DESIGN_ASSUMPTIONS,
        default_storage_capacity_gwh_lhv=DGF_REFERENCE_CAPACITY,
        default_epr_days=DGF_REFERENCE_EPR,
        default_wtir=DGF_REFERENCE_WTIR,
    ),
    TechnologySpec(
        technology="aquifer",
        label="Aquifer",
        case_name="aquifer",
        project_constructor=construct_aquifer_project,
        cost_function=calculate_aquifer_cost_components,
        design_assumptions=AQUIFER_DESIGN_ASSUMPTIONS,
        default_storage_capacity_gwh_lhv=AQUIFER_REFERENCE_CAPACITY,
        default_epr_days=AQUIFER_REFERENCE_EPR,
        default_wtir=AQUIFER_REFERENCE_WTIR,
    ),
    TechnologySpec(
        technology="lined_rock_cavern",
        label="Lined rock cavern",
        case_name="lined_rock_cavern",
        project_constructor=construct_lrc_project,
        cost_function=calculate_lined_rock_cavern_cost_components,
        design_assumptions=LRC_DESIGN_ASSUMPTIONS,
        default_storage_capacity_gwh_lhv=LRC_REFERENCE_CAPACITY,
        default_epr_days=LRC_REFERENCE_EPR,
        default_wtir=LRC_REFERENCE_WTIR,
    ),
)


COMPONENT_GROUPS = ("Injection", "Storage", "Withdrawal")

SENSITIVITY_LABELS = {
    "storage_capacity_gwh_lhv": "Working-gas capacity",
    "epr_days": "Energy-to-power ratio",
    "wtir": "Withdrawal-to-injection ratio",
}

# Plot labels: E and EPR are shown relative to their technology-specific default;
# WTIR retains its natural numerical value.
SENSITIVITY_AXIS_LABELS = {
    "storage_capacity_gwh_lhv": "Working-gas capacity,\n $E/E_0$ [-]",
    "epr_days": "Energy-to-power ratio,\n $EPR/EPR_0$ [-]",
    "wtir": "WTIR [-]",
}

COMPONENT_FACTOR_AXIS_LABELS = {
    "Injection": "Injection TCR / default [-]",
    "Storage": "Storage TCR / default [-]",
    "Withdrawal": "Withdrawal TCR / default [-]",
}


# =============================================================================
# Generic helpers
# =============================================================================


def normalise_name(value: Any) -> str:
    """Convert enum/string-like values to a comparable lower-case string."""

    if value is None:
        return ""
    if hasattr(value, "value"):
        value = value.value
    return str(value).lower().strip()



def safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    """Return numerator / denominator, or None for missing/zero denominator."""

    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator



def get_nested_attr(obj: Any, attr_path: str, default: Any = None) -> Any:
    """Safely retrieve an attribute using a dotted path."""

    current = obj
    for attr in attr_path.split("."):
        if current is None or not hasattr(current, attr):
            return default
        current = getattr(current, attr)
    return current



def is_close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)



def get_selected_specs() -> tuple[TechnologySpec, ...]:
    selected = tuple(
        spec for spec in TECHNOLOGIES if spec.technology in RUN_TECHNOLOGIES
    )
    missing = set(RUN_TECHNOLOGIES) - {spec.technology for spec in selected}
    if missing:
        raise KeyError(f"Unknown RUN_TECHNOLOGIES entries: {sorted(missing)}")
    return selected



def validate_configuration() -> None:
    """Fail early if a configured experiment range is invalid."""

    valid_sensitivities = set(SENSITIVITY_LABELS)
    unknown = set(RUN_SENSITIVITIES) - valid_sensitivities
    if unknown:
        raise KeyError(f"Unknown RUN_SENSITIVITIES entries: {sorted(unknown)}")

    if any(value <= 0 for value in DESIGN_POINT_MULTIPLIERS):
        raise ValueError("DESIGN_POINT_MULTIPLIERS must all be positive.")
    if len(DESIGN_POINT_MULTIPLIERS) != len(set(DESIGN_POINT_MULTIPLIERS)):
        raise ValueError("DESIGN_POINT_MULTIPLIERS contains duplicates.")
    if not any(is_close(value, 1.0) for value in DESIGN_POINT_MULTIPLIERS):
        raise ValueError("DESIGN_POINT_MULTIPLIERS must contain 1.0.")

    if any(value <= 0 for value in WTIR_SWEEP):
        raise ValueError("WTIR_SWEEP must contain only positive values.")
    if len(WTIR_SWEEP) != len(set(WTIR_SWEEP)):
        raise ValueError("WTIR_SWEEP contains duplicates.")

    for spec in get_selected_specs():
        if not any(is_close(value, spec.default_wtir) for value in WTIR_SWEEP):
            raise ValueError(
                f"WTIR_SWEEP must contain the default WTIR "
                f"({spec.default_wtir}) for {spec.technology}."
            )


# =============================================================================
# Facility design-point construction
# =============================================================================


def derive_power_from_storage_and_design(
    *,
    storage_capacity_gwh_lhv: float,
    epr_days: float,
    wtir: float,
) -> tuple[float, float]:
    """Return net CEM-facing withdrawal and injection capacities in kW H2 LHV."""

    if storage_capacity_gwh_lhv <= 0:
        raise ValueError("storage_capacity_gwh_lhv must be positive.")
    if epr_days <= 0:
        raise ValueError("epr_days must be positive.")
    if wtir <= 0:
        raise ValueError("wtir must be positive.")

    storage_capacity_kwh_lhv = storage_capacity_gwh_lhv * 1_000_000.0
    withdrawal_flow_kw_h2_lhv = storage_capacity_kwh_lhv / (epr_days * 24.0)
    injection_flow_kw_h2_lhv = withdrawal_flow_kw_h2_lhv / wtir

    return withdrawal_flow_kw_h2_lhv, injection_flow_kw_h2_lhv



def construct_project(
    *,
    spec: TechnologySpec,
    storage_capacity_gwh_lhv: float,
    epr_days: float,
    wtir: float,
) -> Any:
    """Build one detailed UHS project at a specified CEM design point."""

    withdrawal_kw, injection_kw = derive_power_from_storage_and_design(
        storage_capacity_gwh_lhv=storage_capacity_gwh_lhv,
        epr_days=epr_days,
        wtir=wtir,
    )

    return spec.project_constructor(
        working_gas_capacity_kwh_lhv=storage_capacity_gwh_lhv * 1_000_000.0,
        withdrawal_flow_kw_h2_lhv=withdrawal_kw,
        injection_flow_kw_h2_lhv=injection_kw,
        assumptions=spec.design_assumptions,
        case_name=spec.case_name,
    )


# =============================================================================
# Cost extraction
# =============================================================================


def get_cost_components(cost_breakdown: Any) -> tuple[Any, ...]:
    if hasattr(cost_breakdown, "components"):
        return tuple(cost_breakdown.components)
    if isinstance(cost_breakdown, (tuple, list)):
        return tuple(cost_breakdown)
    raise TypeError(
        f"Unsupported cost breakdown type: {type(cost_breakdown).__name__}"
    )



def get_component_group(cost_component: Any) -> str | None:
    cost_driver = normalise_name(cost_component.cost_driver)
    name = normalise_name(cost_component.name)

    if "injection" in cost_driver or "injection" in name:
        return "Injection"
    if "withdrawal" in cost_driver or "withdrawal" in name:
        return "Withdrawal"
    if "storage" in cost_driver or "subsurface" in name:
        return "Storage"
    return None



def get_cost_type_label(cost_component: Any) -> str:
    cost_type = normalise_name(cost_component.cost_type)
    labels = {
        "capex": "CAPEX",
        "abex": "ABEX",
        "fixed_opex": "Fixed OPEX",
        "variable_opex": "Variable OPEX",
    }
    if cost_type not in labels:
        raise ValueError(f"Unknown cost type: {cost_component.cost_type}")
    return labels[cost_type]



def get_cost_unit_label(cost_component: Any) -> str:
    if hasattr(cost_component.cost_unit, "value"):
        return str(cost_component.cost_unit.value)
    return str(cost_component.cost_unit)



def get_cost_rate(cost_component: Any) -> float:
    if cost_component.driver_value == 0:
        raise ValueError(
            f"Cannot calculate rate for {cost_component.name}: driver value is zero."
        )
    return float(cost_component.value_eur) / float(cost_component.driver_value)



def summarise_tcr(cost_breakdown: Any) -> dict[str, dict[str, float | str | None]]:
    """Summarise CAPEX + ABEX by CEM investment component."""

    rates: dict[str, float] = defaultdict(float)
    costs: dict[str, float] = defaultdict(float)
    units: dict[str, str] = {}

    for component in get_cost_components(cost_breakdown):
        group = get_component_group(component)
        if group is None:
            continue

        cost_type = get_cost_type_label(component)
        if cost_type not in {"CAPEX", "ABEX"}:
            continue

        rates[group] += get_cost_rate(component)
        costs[group] += float(component.value_eur)
        units[group] = get_cost_unit_label(component)

    summary: dict[str, dict[str, float | str | None]] = {}
    for group in COMPONENT_GROUPS:
        if group not in rates:
            summary[group] = {
                "rate": None,
                "absolute_cost_eur": None,
                "unit": None,
            }
        else:
            summary[group] = {
                "rate": rates[group],
                "absolute_cost_eur": costs[group],
                "unit": units.get(group),
            }

    return summary


# =============================================================================
# Experiment construction
# =============================================================================


def get_sweep(spec: TechnologySpec, sensitivity: str) -> tuple[float, ...]:
    """Return absolute physical values to evaluate for one sensitivity."""

    if sensitivity == "storage_capacity_gwh_lhv":
        return tuple(
            spec.default_storage_capacity_gwh_lhv * multiplier
            for multiplier in DESIGN_POINT_MULTIPLIERS
        )
    if sensitivity == "epr_days":
        return tuple(
            spec.default_epr_days * multiplier
            for multiplier in DESIGN_POINT_MULTIPLIERS
        )
    if sensitivity == "wtir":
        return WTIR_SWEEP
    raise KeyError(sensitivity)



def get_default(spec: TechnologySpec, sensitivity: str) -> float:
    if sensitivity == "storage_capacity_gwh_lhv":
        return spec.default_storage_capacity_gwh_lhv
    if sensitivity == "epr_days":
        return spec.default_epr_days
    if sensitivity == "wtir":
        return spec.default_wtir
    raise KeyError(sensitivity)



def get_parameter_factor(
    spec: TechnologySpec,
    sensitivity: str,
    parameter_value: float,
) -> float:
    """Return parameter value relative to its technology-specific default."""

    return parameter_value / get_default(spec, sensitivity)



def get_plot_x(
    spec: TechnologySpec,
    sensitivity: str,
    parameter_value: float,
) -> float:
    """Return x coordinate used in the compact manuscript figures."""

    if sensitivity in {"storage_capacity_gwh_lhv", "epr_days"}:
        return get_parameter_factor(spec, sensitivity, parameter_value)
    if sensitivity == "wtir":
        return parameter_value
    raise KeyError(sensitivity)



def get_default_plot_x(spec: TechnologySpec, sensitivity: str) -> float:
    """Return the x coordinate of the calibration design point."""

    if sensitivity in {"storage_capacity_gwh_lhv", "epr_days"}:
        return 1.0
    if sensitivity == "wtir":
        return spec.default_wtir
    raise KeyError(sensitivity)



def build_case(
    *,
    spec: TechnologySpec,
    sensitivity: str,
    parameter_value: float,
) -> dict[str, float]:
    """Create an OAT design-point case around one technology's reference."""

    case = {
        "storage_capacity_gwh_lhv": spec.default_storage_capacity_gwh_lhv,
        "epr_days": spec.default_epr_days,
        "wtir": spec.default_wtir,
    }
    case[sensitivity] = float(parameter_value)
    return case



def extract_project_diagnostics(project: Any) -> dict[str, Any]:
    """Extract diagnostics that help explain steps/non-linearities in the curves."""

    required_storage_volume_m3 = get_nested_attr(
        project, "inventory.required_storage_volume_m3"
    )
    design_electric_power_kw = get_nested_attr(
        project, "compression.total_design_electric_power_kw"
    )

    return {
        "number_caverns": get_nested_attr(project, "wells.number_caverns"),
        "number_well_heads": get_nested_attr(project, "wells.number_well_heads"),
        "number_production_wells": get_nested_attr(
            project, "wells.number_production_wells"
        ),
        "number_observation_wells": get_nested_attr(
            project, "wells.number_observation_wells"
        ),
        "required_storage_volume_million_m3": (
            None
            if required_storage_volume_m3 is None
            else float(required_storage_volume_m3) / 1_000_000.0
        ),
        "compression_electricity_kwh_per_kwh_h2": get_nested_attr(
            project, "compression.electricity_kwh_per_kwh_h2_lhv"
        ),
        "compression_design_electric_power_mw": (
            None
            if design_electric_power_kw is None
            else float(design_electric_power_kw) / 1000.0
        ),
        "purification_hydrogen_recovery": get_nested_attr(
            project, "purification.hydrogen_recovery"
        ),
        "purification_factor": get_nested_attr(
            project, "purification.purification_factor"
        ),
    }


# =============================================================================
# Reference linearisation
# =============================================================================


def build_reference_profiles() -> dict[str, dict[str, Any]]:
    """Build the fixed CEM cost coefficients at each technology's default point."""

    references: dict[str, dict[str, Any]] = {}

    for spec in get_selected_specs():
        project = construct_project(
            spec=spec,
            storage_capacity_gwh_lhv=spec.default_storage_capacity_gwh_lhv,
            epr_days=spec.default_epr_days,
            wtir=spec.default_wtir,
        )
        cost_breakdown = spec.cost_function(project)
        tcr = summarise_tcr(cost_breakdown)

        withdrawal_kw, injection_kw = derive_power_from_storage_and_design(
            storage_capacity_gwh_lhv=spec.default_storage_capacity_gwh_lhv,
            epr_days=spec.default_epr_days,
            wtir=spec.default_wtir,
        )

        references[spec.technology] = {
            "project": project,
            "tcr": tcr,
            "storage_capacity_kwh": spec.default_storage_capacity_gwh_lhv
            * 1_000_000.0,
            "injection_flow_kw": injection_kw,
            "withdrawal_flow_kw": withdrawal_kw,
        }

    return references


# =============================================================================
# Experiment execution
# =============================================================================


def run_experiments() -> list[dict[str, Any]]:
    validate_configuration()
    references = build_reference_profiles()
    rows: list[dict[str, Any]] = []

    for sensitivity in RUN_SENSITIVITIES:
        print(f"\n[{SENSITIVITY_LABELS[sensitivity]}]")

        for spec in get_selected_specs():
            reference = references[spec.technology]
            reference_tcr = reference["tcr"]
            parameter_default = get_default(spec, sensitivity)

            print(f"  {spec.label}")

            for parameter_value in get_sweep(spec, sensitivity):
                case = build_case(
                    spec=spec,
                    sensitivity=sensitivity,
                    parameter_value=parameter_value,
                )

                parameter_factor = get_parameter_factor(
                    spec, sensitivity, parameter_value
                )
                plot_x_value = get_plot_x(spec, sensitivity, parameter_value)

                withdrawal_kw, injection_kw = derive_power_from_storage_and_design(
                    storage_capacity_gwh_lhv=case["storage_capacity_gwh_lhv"],
                    epr_days=case["epr_days"],
                    wtir=case["wtir"],
                )
                storage_kwh = case["storage_capacity_gwh_lhv"] * 1_000_000.0

                project = construct_project(
                    spec=spec,
                    storage_capacity_gwh_lhv=case["storage_capacity_gwh_lhv"],
                    epr_days=case["epr_days"],
                    wtir=case["wtir"],
                )
                cost_breakdown = spec.cost_function(project)
                detailed_tcr = summarise_tcr(cost_breakdown)
                diagnostics = extract_project_diagnostics(project)

                # Fixed CEM linearisation: apply the unit rates derived at the
                # default design point to the current CEM capacities.
                cem_drivers = {
                    "Injection": injection_kw,
                    "Storage": storage_kwh,
                    "Withdrawal": withdrawal_kw,
                }

                detailed_total_eur = 0.0
                default_linearised_total_eur = 0.0
                component_values: dict[str, dict[str, float | None]] = {}

                for group in COMPONENT_GROUPS:
                    detailed_rate = detailed_tcr[group]["rate"]
                    detailed_cost = detailed_tcr[group]["absolute_cost_eur"]
                    default_rate = reference_tcr[group]["rate"]

                    if (
                        detailed_rate is None
                        or detailed_cost is None
                        or default_rate is None
                    ):
                        component_values[group] = {
                            "default_rate": None,
                            "detailed_rate": None,
                            "tcr_rate_factor": None,
                            "detailed_cost_eur": None,
                            "default_linearised_cost_eur": None,
                            "rate_difference_pct": None,
                            "component_linearisation_error_pct": None,
                        }
                        continue

                    default_linearised_cost = float(default_rate) * cem_drivers[group]
                    tcr_rate_factor = float(detailed_rate) / float(default_rate)
                    rate_difference_pct = 100.0 * (tcr_rate_factor - 1.0)
                    component_error_pct = (
                        100.0
                        * (default_linearised_cost - float(detailed_cost))
                        / float(detailed_cost)
                    )

                    detailed_total_eur += float(detailed_cost)
                    default_linearised_total_eur += default_linearised_cost

                    component_values[group] = {
                        "default_rate": float(default_rate),
                        "detailed_rate": float(detailed_rate),
                        "tcr_rate_factor": tcr_rate_factor,
                        "detailed_cost_eur": float(detailed_cost),
                        "default_linearised_cost_eur": default_linearised_cost,
                        "rate_difference_pct": rate_difference_pct,
                        "component_linearisation_error_pct": component_error_pct,
                    }

                total_linearisation_error_pct = (
                    100.0
                    * (default_linearised_total_eur - detailed_total_eur)
                    / detailed_total_eur
                )

                is_default = is_close(parameter_value, parameter_default)

                # At the calibration point, the fixed linearisation must recover
                # the detailed total TCR apart from floating-point noise.
                if is_default and not math.isclose(
                    detailed_total_eur,
                    default_linearised_total_eur,
                    rel_tol=1e-8,
                    abs_tol=1e-4,
                ):
                    raise RuntimeError(
                        f"Reference linearisation does not close for "
                        f"{spec.technology}/{sensitivity}: detailed="
                        f"{detailed_total_eur}, linearised="
                        f"{default_linearised_total_eur}."
                    )

                for group in COMPONENT_GROUPS:
                    values = component_values[group]
                    rows.append(
                        {
                            "sensitivity": sensitivity,
                            "sensitivity_label": SENSITIVITY_LABELS[sensitivity],
                            "technology": spec.technology,
                            "technology_label": spec.label,
                            "component_group": group,
                            # Parameter metadata: retain both physical value and
                            # normalised factor irrespective of plotting choice.
                            "parameter_value": parameter_value,
                            "parameter_default": parameter_default,
                            "parameter_factor": parameter_factor,
                            "plot_x_value": plot_x_value,
                            "is_default_case": is_default,
                            # Complete facility design point.
                            "storage_capacity_gwh_lhv": case[
                                "storage_capacity_gwh_lhv"
                            ],
                            "storage_capacity_factor": (
                                case["storage_capacity_gwh_lhv"]
                                / spec.default_storage_capacity_gwh_lhv
                            ),
                            "epr_days": case["epr_days"],
                            "epr_hours": case["epr_days"] * 24.0,
                            "epr_factor": case["epr_days"] / spec.default_epr_days,
                            "wtir": case["wtir"],
                            "wtir_factor": case["wtir"] / spec.default_wtir,
                            "withdrawal_flow_gw_h2_lhv": withdrawal_kw / 1_000_000.0,
                            "injection_flow_gw_h2_lhv": injection_kw / 1_000_000.0,
                            # Component TCR rates: absolute and relative.
                            "tcr_rate": values["detailed_rate"],
                            "default_linearised_tcr_rate": values["default_rate"],
                            "tcr_rate_factor": values["tcr_rate_factor"],
                            "cost_unit": detailed_tcr[group]["unit"],
                            "tcr_rate_difference_pct": values[
                                "rate_difference_pct"
                            ],
                            # Absolute investment outcomes.
                            "detailed_component_tcr_meur": (
                                None
                                if values["detailed_cost_eur"] is None
                                else values["detailed_cost_eur"] / 1_000_000.0
                            ),
                            "default_linearised_component_tcr_meur": (
                                None
                                if values["default_linearised_cost_eur"] is None
                                else values["default_linearised_cost_eur"]
                                / 1_000_000.0
                            ),
                            "component_linearisation_error_pct": values[
                                "component_linearisation_error_pct"
                            ],
                            "detailed_total_tcr_meur": detailed_total_eur / 1_000_000.0,
                            "default_linearised_total_tcr_meur": (
                                default_linearised_total_eur / 1_000_000.0
                            ),
                            "total_linearisation_error_pct": (
                                total_linearisation_error_pct
                            ),
                            **diagnostics,
                        }
                    )

                print(
                    f"    {parameter_value:g} "
                    f"({parameter_factor:.2f} x default): "
                    f"total linearisation error "
                    f"{total_linearisation_error_pct:+.2f}%"
                )

    return rows


# =============================================================================
# CSV output
# =============================================================================


def write_rows_to_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No experiment rows were generated.")

    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Plotting
# =============================================================================


def save_figure(fig: Any, output_stem: Path) -> None:
    """Save one publication-quality PDF figure."""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)



def get_rows(
    rows: list[dict[str, Any]],
    *,
    sensitivity: str,
    technology: str,
    component_group: str | None = None,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row["sensitivity"] == sensitivity
        and row["technology"] == technology
        and (
            component_group is None
            or row["component_group"] == component_group
        )
    ]
    return sorted(selected, key=lambda row: float(row["plot_x_value"]))



def plot_component_rate_factors(
    rows: list[dict[str, Any]],
    figure_dir: Path,
) -> None:
    """Plot one compact 3 x 3 figure of relative component TCR rates.

    Rows are Injection / Storage / Withdrawal.
    Columns are E / EPR / WTIR.
    Each panel contains one line per storage technology.

    E and EPR use the common normalised design-point axis. WTIR is shown using
    its actual numerical value. The default design point is indicated only by a
    marker; no horizontal default-rate line is needed because y=1 is already
    inherent in the normalisation.
    """

    specs = get_selected_specs()
    sensitivities = tuple(PLOT_SENSITIVITIES)

    fig, axes = plt.subplots(
        nrows=len(COMPONENT_GROUPS),
        ncols=len(sensitivities),
        figsize=(9, 8),
        squeeze=False,
        sharex="col",
        sharey=True
    )

    for col, sensitivity in enumerate(sensitivities):
        for row_idx, group in enumerate(COMPONENT_GROUPS):
            ax = axes[row_idx][col]

            for spec in specs:
                selected = get_rows(
                    rows,
                    sensitivity=sensitivity,
                    technology=spec.technology,
                    component_group=group,
                )

                x = [float(item["plot_x_value"]) for item in selected]
                y = [float(item["tcr_rate_factor"]) for item in selected]

                ax.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=1.5,
                    markersize=3.8,
                    label=spec.label,
                )

            # All technologies intersect at the same calibration point in this
            # normalised view, so a single marker is clearer than four stacked
            # technology markers.
            default_x_values = {
                get_default_plot_x(spec, sensitivity) for spec in specs
            }
            if len(default_x_values) != 1:
                raise RuntimeError(
                    f"Expected a shared default x coordinate for {sensitivity}."
                )
            default_x = default_x_values.pop()
            ax.scatter(
                [default_x],
                [1.0],
                marker="x",
                s=48,
                linewidths=1.5,
                zorder=6,
                label="Default design point",
            )

            ax.grid(True, linewidth=0.4, alpha=0.30)

            if row_idx == 0:
                ax.set_title(SENSITIVITY_LABELS[sensitivity])
            if col == 0:
                ax.set_ylabel(COMPONENT_FACTOR_AXIS_LABELS[group])
            if row_idx == len(COMPONENT_GROUPS) - 1:
                ax.set_xlabel(SENSITIVITY_AXIS_LABELS[sensitivity])

    # All panels use the same technology encoding and default marker.
    handles, labels = axes[0][0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
    )

    # fig.suptitle("Sensitivity of CEM TCR rates to linearisation design point", y=1.01)
    fig.tight_layout(rect=(0, 0.0, 1, 0.96))
    save_figure(fig, figure_dir / "component_tcr_rate_factors")



def plot_total_linearisation_error(
    rows: list[dict[str, Any]],
    figure_dir: Path,
) -> None:
    """Plot one compact 1 x 3 figure of total-project linearisation error."""

    specs = get_selected_specs()
    sensitivities = tuple(PLOT_SENSITIVITIES)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(sensitivities),
        figsize=(12, 9),
        squeeze=False,
        sharey=True,
    )

    for col, sensitivity in enumerate(sensitivities):
        ax = axes[0][col]

        for spec in specs:
            # Total values are repeated for the three component rows, so use
            # Injection as one representative row per experiment case.
            selected = get_rows(
                rows,
                sensitivity=sensitivity,
                technology=spec.technology,
                component_group="Injection",
            )

            x = [float(item["plot_x_value"]) for item in selected]
            y = [float(item["total_linearisation_error_pct"]) for item in selected]

            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.5,
                markersize=3.8,
                label=spec.label,
            )

        # Zero is a meaningful sign boundary here rather than a default-rate
        # reference, so retain a light horizontal baseline.
        ax.axhline(0.0, linewidth=0.8, linestyle="--")

        default_x_values = {
            get_default_plot_x(spec, sensitivity) for spec in specs
        }
        if len(default_x_values) != 1:
            raise RuntimeError(
                f"Expected a shared default x coordinate for {sensitivity}."
            )
        default_x = default_x_values.pop()
        ax.scatter(
            [default_x],
            [0.0],
            marker="x",
            s=48,
            linewidths=1.5,
            zorder=6,
            label="Default design point",
        )

        ax.set_title(SENSITIVITY_LABELS[sensitivity])
        ax.set_xlabel(SENSITIVITY_AXIS_LABELS[sensitivity])
        ax.grid(True, linewidth=0.4, alpha=0.30)

    axes[0][0].set_ylabel(
        "Default-linearised TCR error [%]\n"
        "(+ = CEM overestimates detailed cost)"
    )

    handles, labels = axes[0][0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )

    fig.suptitle("Total-project linearisation error", y=1.03)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    # save_figure(fig, figure_dir / "total_linearisation_error")


# =============================================================================
# Console summary
# =============================================================================


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("\nLINEARISATION SENSITIVITY SUMMARY")
    print("---------------------------------")

    for sensitivity in RUN_SENSITIVITIES:
        print(f"\n{SENSITIVITY_LABELS[sensitivity]}")
        for spec in get_selected_specs():
            selected = get_rows(
                rows,
                sensitivity=sensitivity,
                technology=spec.technology,
                component_group="Injection",
            )
            errors = [float(row["total_linearisation_error_pct"]) for row in selected]
            print(
                f"  {spec.label:<22} "
                f"min={min(errors):+7.2f}%  max={max(errors):+7.2f}%"
            )


# =============================================================================
# Linear versus affine cost representation
# =============================================================================

RUN_COST_FIT_ANALYSIS = True

COST_FIT_RESULTS_CSV = OUTPUT_DIR / "linear_cost_fit_results.csv"

# The restricted domain tests accuracy around the calibration design.
COST_FIT_LOCAL_MIN = 0.5
COST_FIT_LOCAL_MAX = 1.5
COST_FIT_LOCAL_WTIR_MIN = 1.0
COST_FIT_LOCAL_WTIR_MAX = 3.0


def get_unique_cost_cases(
    rows: list[dict[str, Any]],
    technology: str,
    component_group: str,
    domain: str,
) -> list[dict[str, Any]]:
    """Retrieve distinct facility designs for cost fitting.

    The original experiment repeats the reference configuration in
    the E, EPR and WTIR sweeps. These duplicates are removed to avoid
    giving the reference configuration artificial statistical weight.
    """

    representative_group = (
        "Injection" if component_group == "Total" else component_group
    )

    selected = []
    seen = set()

    key_fields = (
        "storage_capacity_gwh_lhv",
        "injection_flow_gw_h2_lhv",
        "withdrawal_flow_gw_h2_lhv",
    )

    for row in rows:
        if row["technology"] != technology:
            continue

        if row["component_group"] != representative_group:
            continue

        if domain == "near_reference":
            e_factor = float(row["storage_capacity_factor"])
            epr_factor = float(row["epr_factor"])
            wtir = float(row["wtir"])

            if not (
                COST_FIT_LOCAL_MIN <= e_factor <= COST_FIT_LOCAL_MAX
                and COST_FIT_LOCAL_MIN <= epr_factor <= COST_FIT_LOCAL_MAX
                and COST_FIT_LOCAL_WTIR_MIN
                <= wtir
                <= COST_FIT_LOCAL_WTIR_MAX
            ):
                continue

        key = tuple(
            round(float(row[field]), 8)
            for field in key_fields
        )

        if key in seen:
            continue

        seen.add(key)
        selected.append(row)

    return selected


def get_fit_target(
    rows: list[dict[str, Any]],
    component_group: str,
) -> np.ndarray:
    """Return detailed absolute TCR in MEUR."""

    field = (
        "detailed_total_tcr_meur"
        if component_group == "Total"
        else "detailed_component_tcr_meur"
    )

    return np.asarray(
        [float(row[field]) for row in rows],
        dtype=float,
    )


def get_reference_lp_prediction(
    rows: list[dict[str, Any]],
    component_group: str,
) -> np.ndarray:
    """Return costs predicted using existing reference CEM coefficients."""

    field = (
        "default_linearised_total_tcr_meur"
        if component_group == "Total"
        else "default_linearised_component_tcr_meur"
    )

    return np.asarray(
        [float(row[field]) for row in rows],
        dtype=float,
    )


def get_fit_model_definitions(
    component_group: str,
) -> list[tuple[str, tuple[str, ...], bool]]:
    """Define alternative cost representations.

    Entries contain:
        model name,
        capacity drivers,
        whether a constant intercept is included.

    E is in GWh, power in GW, and costs in MEUR.

    Consequently:
        MEUR/GWh = EUR/kWh
        MEUR/GW  = EUR/kW
    """

    E = "storage_capacity_gwh_lhv"
    P_IN = "injection_flow_gw_h2_lhv"
    P_OUT = "withdrawal_flow_gw_h2_lhv"

    if component_group == "Injection":
        own_driver = (P_IN,)
        coupled_drivers = (E, P_IN)

    elif component_group == "Storage":
        own_driver = (E,)
        coupled_drivers = (E, P_OUT)

    elif component_group == "Withdrawal":
        own_driver = (P_OUT,)
        coupled_drivers = (E, P_OUT)

    elif component_group == "Total":
        own_driver = ()
        coupled_drivers = (E, P_IN, P_OUT)

    else:
        raise KeyError(component_group)

    models = [
        ("reference_lp", (), False),
    ]

    if component_group != "Total":
        models.append(
            ("single_driver_lp", own_driver, False)
        )

    models.extend(
        [
            ("coupled_lp", coupled_drivers, False),
            ("affine", coupled_drivers, True),
        ]
    )

    return models


def build_fit_matrix(
    rows: list[dict[str, Any]],
    drivers: tuple[str, ...],
    include_intercept: bool,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build the regression design matrix."""

    columns = []

    if include_intercept:
        columns.append(
            np.ones(len(rows), dtype=float)
        )

    for driver in drivers:
        columns.append(
            np.asarray(
                [float(row[driver]) for row in rows],
                dtype=float,
            )
        )

    X = np.column_stack(columns)

    names = (
        ("intercept",) + drivers
        if include_intercept
        else drivers
    )

    return X, names


def fit_linear_cost_model(
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Ordinary least squares with column scaling.

    Scaling improves numerical conditioning because storage capacities
    in GWh can be much larger numerically than power capacities in GW.
    """

    column_scales = np.linalg.norm(X, axis=0)

    column_scales = np.where(
        column_scales > 0,
        column_scales,
        1.0,
    )

    X_scaled = X / column_scales

    coefficients_scaled, *_ = np.linalg.lstsq(
        X_scaled,
        y,
        rcond=None,
    )

    return coefficients_scaled / column_scales


def leave_one_out_predictions(
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Calculate leave-one-design-point-out predictions."""

    predictions = np.full(len(y), np.nan)

    for index in range(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[index] = False

        X_train = X[mask]
        y_train = y[mask]

        if np.linalg.matrix_rank(X_train) < X.shape[1]:
            continue

        coefficients = fit_linear_cost_model(
            X_train,
            y_train,
        )

        predictions[index] = (
            X[index] @ coefficients
        )

    return predictions


def calculate_fit_metrics(
    y: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calculate absolute and relative prediction errors."""

    valid = (
        np.isfinite(y)
        & np.isfinite(predicted)
        & (np.abs(y) > 1e-12)
    )

    if not np.any(valid):
        return {
            "mae_pct": float("nan"),
            "median_abs_pct": float("nan"),
            "max_abs_pct": float("nan"),
            "rmse_pct": float("nan"),
        }

    actual = y[valid]
    fitted = predicted[valid]

    relative_error_pct = (
        100.0 * (fitted - actual) / actual
    )

    return {
        "mae_pct": float(
            np.mean(np.abs(relative_error_pct))
        ),
        "median_abs_pct": float(
            np.median(np.abs(relative_error_pct))
        ),
        "max_abs_pct": float(
            np.max(np.abs(relative_error_pct))
        ),
        "rmse_pct": float(
            np.sqrt(np.mean(relative_error_pct**2))
        ),
    }


def run_cost_fit_analysis(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare reference, refitted-linear and affine TCR models.

    The fits are diagnostic. Coefficients should not be inserted into
    the CEM without checking physical signs, baseline calibration
    accuracy and out-of-sample performance.
    """

    results = []

    domains = ("full", "near_reference")

    groups = (
        "Injection",
        "Storage",
        "Withdrawal",
        "Total",
    )

    for spec in get_selected_specs():
        for domain in domains:
            for group in groups:

                cases = get_unique_cost_cases(
                    rows,
                    technology=spec.technology,
                    component_group=group,
                    domain=domain,
                )

                if not cases:
                    continue

                y = get_fit_target(cases, group)

                reference_index = next(
                    (
                        i
                        for i, case in enumerate(cases)
                        if str(
                            case["is_default_case"]
                        ).lower() in {"true", "1"}
                    ),
                    None,
                )

                models = get_fit_model_definitions(group)

                for model_name, drivers, has_intercept in models:

                    coefficients = {}
                    coefficient_names = ()
                    negative_slopes = False

                    if model_name == "reference_lp":
                        predicted = get_reference_lp_prediction(
                            cases,
                            group,
                        )

                        cv_predicted = predicted.copy()

                    else:
                        X, coefficient_names = build_fit_matrix(
                            cases,
                            drivers,
                            has_intercept,
                        )

                        if (
                            len(y) <= X.shape[1]
                            or np.linalg.matrix_rank(X)
                            < X.shape[1]
                        ):
                            continue

                        beta = fit_linear_cost_model(X, y)

                        predicted = X @ beta

                        cv_predicted = leave_one_out_predictions(
                            X,
                            y,
                        )

                        coefficients = dict(
                            zip(
                                coefficient_names,
                                beta,
                                strict=True,
                            )
                        )

                        negative_slopes = any(
                            float(value) < -1e-9
                            for name, value in coefficients.items()
                            if name != "intercept"
                        )

                    metrics = calculate_fit_metrics(
                        y,
                        predicted,
                    )

                    cv_metrics = calculate_fit_metrics(
                        y,
                        cv_predicted,
                    )

                    if reference_index is None:
                        reference_error_pct = float("nan")
                    else:
                        reference_error_pct = (
                            100.0
                            * (
                                predicted[reference_index]
                                - y[reference_index]
                            )
                            / y[reference_index]
                        )

                    result = {
                        "technology": spec.technology,
                        "technology_label": spec.label,
                        "component_group": group,
                        "domain": domain,
                        "model": model_name,
                        "number_cases": len(cases),

                        "intercept_meur": coefficients.get(
                            "intercept", 0.0
                        ),

                        "storage_coefficient_eur_per_kwh": (
                            coefficients.get(
                                "storage_capacity_gwh_lhv",
                                0.0,
                            )
                        ),

                        "injection_coefficient_eur_per_kw": (
                            coefficients.get(
                                "injection_flow_gw_h2_lhv",
                                0.0,
                            )
                        ),

                        "withdrawal_coefficient_eur_per_kw": (
                            coefficients.get(
                                "withdrawal_flow_gw_h2_lhv",
                                0.0,
                            )
                        ),

                        "reference_cost_meur": (
                            float(y[reference_index])
                            if reference_index is not None
                            else None
                        ),

                        "reference_error_pct": reference_error_pct,

                        "mae_pct": metrics["mae_pct"],
                        "median_abs_pct": metrics["median_abs_pct"],
                        "max_abs_pct": metrics["max_abs_pct"],
                        "rmse_pct": metrics["rmse_pct"],

                        "loocv_mae_pct": cv_metrics["mae_pct"],
                        "loocv_median_abs_pct": (
                            cv_metrics["median_abs_pct"]
                        ),
                        "loocv_max_abs_pct": (
                            cv_metrics["max_abs_pct"]
                        ),

                        "negative_capacity_coefficient": (
                            negative_slopes
                        ),
                    }

                    results.append(result)

    return results


def plot_cost_fit_comparison(
    fit_results: list[dict[str, Any]],
    figure_dir: Path,
) -> None:
    """Compare total-project prediction errors across model types."""

    specs = get_selected_specs()

    models = (
        "reference_lp",
        "coupled_lp",
        "affine",
    )

    labels = {
        "reference_lp": "Reference LP",
        "coupled_lp": "Refitted LP",
        "affine": "Affine",
    }

    domains = (
        "near_reference",
        "full",
    )

    lookup = {
        (
            row["technology"],
            row["component_group"],
            row["domain"],
            row["model"],
        ): row
        for row in fit_results
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
        sharey=False,
    )

    x = np.arange(len(specs))
    width = 0.24

    for ax, domain in zip(axes, domains, strict=True):

        for model_index, model in enumerate(models):
            values = []

            for spec in specs:
                result = lookup.get(
                    (
                        spec.technology,
                        "Total",
                        domain,
                        model,
                    )
                )

                values.append(
                    float(result["loocv_median_abs_pct"])
                    if result is not None
                    else np.nan
                )

            ax.bar(
                x + (model_index - 1) * width,
                values,
                width,
                label=labels[model],
            )

        ax.set_xticks(
            x,
            [spec.label for spec in specs],
            rotation=20,
            ha="right",
        )

        ax.set_ylabel(
            "Median absolute prediction error [%]"
        )

        ax.set_title(
            "Near reference"
            if domain == "near_reference"
            else "Full design range"
        )

        ax.grid(
            axis="y",
            alpha=0.3,
            linewidth=0.5,
        )

        ax.set_axisbelow(True)

    handles, legend_labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.93))

    save_figure(
        fig,
        figure_dir / "linear_cost_fit_comparison",
    )

def plot_withdrawal_cost_model_comparison(
    rows: list[dict[str, Any]],
    figure_dir: Path,
) -> None:
    """Compare linear cost representations for withdrawal TCR.

    Fits use both working-gas capacity and EPR sensitivities.
    The figure displays predictions along the EPR sensitivity.

    Models:
      - Constant reference unit rate
      - Cross-driver linear cost: a*P_out + b*E
      - Affine cost: F + a*P_out + b*E
    """

    specs = get_selected_specs()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 7),
        sharex=True,
    )

    axes = axes.flatten()

    for ax, spec in zip(axes, specs):

        # Data to display: EPR variation.
        epr_rows = get_rows(
            rows,
            sensitivity="epr_days",
            technology=spec.technology,
            component_group="Withdrawal",
        )

        # Additional data needed to identify the E coefficient.
        capacity_rows = get_rows(
            rows,
            sensitivity="storage_capacity_gwh_lhv",
            technology=spec.technology,
            component_group="Withdrawal",
        )

        # Fit to both sensitivities, not only the displayed EPR sweep.
        fit_rows = epr_rows + capacity_rows

        # Remove duplicated design configurations.
        unique = {}

        for row in fit_rows:

            E = float(row["storage_capacity_gwh_lhv"]) * 1e6
            P = float(row["withdrawal_flow_gw_h2_lhv"]) * 1e6

            key = (round(E), round(P))
            unique[key] = row

        fit_rows = list(unique.values())

        # Reference capacities.
        E0 = spec.default_storage_capacity_gwh_lhv * 1e6
        P0 = E0 / (spec.default_epr_days * 24.0)

        # Normalised regression drivers.
        X = np.array([
            [
                float(r["withdrawal_flow_gw_h2_lhv"]) * 1e6 / P0,
                float(r["storage_capacity_gwh_lhv"]) * 1e6 / E0,
            ]
            for r in fit_rows
        ])

        # Detailed withdrawal investment cost, MEUR.
        y = np.array([
            float(r["detailed_component_tcr_meur"])
            for r in fit_rows
        ])

        # ---------------------------------------------------------
        # Unanchored cross-driver regression (diagnostic figure).
        # ---------------------------------------------------------

        # Use both the EPR and working-gas sweeps; exclude only the
        # low-EPR observations with strongly discontinuous behaviour.
        training_rows = [
            row
            for row in fit_rows
            if float(row["epr_factor"]) >= CROSS_DRIVER_EPR_MIN_FACTOR
        ]

        if len(training_rows) < 2:
            raise ValueError(
                f"Insufficient EPR training points for {spec.technology}"
            )

        # Normalised capacities.
        p = np.asarray([
            float(row["withdrawal_flow_gw_h2_lhv"]) * 1e6 / P0
            for row in training_rows
        ])

        e = np.asarray([
            float(row["storage_capacity_gwh_lhv"]) * 1e6 / E0
            for row in training_rows
        ])

        # Detailed withdrawal investment cost (MEUR).
        y_training = np.asarray([
            float(row["detailed_component_tcr_meur"])
            for row in training_rows
        ])

        # Two free slopes, zero fixed intercept, no reference anchor.
        beta_P, beta_E = fit_two_driver_cost(
            own_capacity=p,
            secondary_capacity=e,
            detailed_cost=y_training,
        )

        beta_coupled = np.array([
            beta_P,
            beta_E,
        ])

        # Affine model, with intercept.
        X_affine = np.column_stack([
            np.ones(len(X)),
            X,
        ])

        beta_affine = np.linalg.lstsq(
            X_affine, y, rcond=None
        )[0]

        # Predict on the displayed EPR sweep.
        E_plot = np.array([
            float(r["storage_capacity_gwh_lhv"]) * 1e6
            for r in epr_rows
        ])

        P_plot = np.array([
            float(r["withdrawal_flow_gw_h2_lhv"]) * 1e6
            for r in epr_rows
        ])

        x_plot = np.array([
            float(r["epr_factor"])
            for r in epr_rows
        ])

        observed = np.array([
            float(r["tcr_rate"])
            for r in epr_rows
        ])

        X_plot = np.column_stack([
            P_plot / P0,
            E_plot / E0,
        ])

        # Predicted absolute costs (MEUR).
        cost_coupled = X_plot @ beta_coupled

        cost_affine = (
            np.column_stack([
                np.ones(len(X_plot)),
                X_plot,
            ])
            @ beta_affine
        )

        # Convert predicted costs to withdrawal EUR/kW.
        rate_coupled = cost_coupled * 1e6 / P_plot
        rate_affine = cost_affine * 1e6 / P_plot

        # Constant reference rate.
        reference_rate = float(
            epr_rows[0]["default_linearised_tcr_rate"]
        )

        rate_constant = np.full_like(
            x_plot,
            reference_rate,
        )

        # ---------------------------------------------------------
        # Plot
        # ---------------------------------------------------------

        ax.plot(
            x_plot,
            observed,
            "ko",
            markersize=4,
            label="Detailed model",
            zorder=5,
        )

        ax.plot(
            x_plot,
            rate_constant,
            color="0.5",
            linestyle="--",
            linewidth=1.7,
            label="Constant unit rate",
        )

        ax.plot(
            x_plot,
            rate_coupled,
            color="tab:blue",
            linewidth=2,
            label="Cross-driver linear",
        )

        ax.plot(
            x_plot,
            rate_affine,
            color="tab:orange",
            linestyle=":",
            linewidth=2,
            label="Affine",
        )

        ax.axvline(
            1.0,
            color="0.7",
            linestyle=":",
            linewidth=0.8,
        )

        ax.set_title(spec.label)
        ax.set_xlabel("EPR / reference EPR [-]")
        ax.set_ylabel("Withdrawal TCR [EUR/kW]")
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))

    save_figure(
        fig,
        figure_dir / "withdrawal_cost_model_comparison",
    )


def plot_injection_cost_model_comparison(
    rows: list[dict[str, Any]],
    figure_dir: Path,
) -> None:
    """Compare linear cost representations for injection TCR.

    Fits use the storage-capacity, EPR, and WTIR sensitivities.
    The figure displays predictions along the WTIR sensitivity.

    Models:
      - Constant reference unit rate
      - Cross-driver linear cost: a*P_in + b*P_out
      - Affine cost: F + a*P_in + b*P_out
    """

    specs = get_selected_specs()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 7),
        sharex=True,
    )

    axes = axes.flatten()

    for ax, spec in zip(axes, specs):

        # Data to display: WTIR variation.
        wtir_rows = get_rows(
            rows,
            sensitivity="wtir",
            technology=spec.technology,
            component_group="Injection",
        )

        # Fit to all sensitivities, not just the displayed WTIR sweep.
        fit_rows = []
        for sensitivity in (
            "storage_capacity_gwh_lhv",
            "epr_days",
            "wtir",
        ):
            fit_rows.extend(
                get_rows(
                    rows,
                    sensitivity=sensitivity,
                    technology=spec.technology,
                    component_group="Injection",
                )
            )
        # Remove duplicated design configurations.
        unique = {}

        for row in fit_rows:

            E = float(row["storage_capacity_gwh_lhv"]) * 1e6

            P_in = float(
                row["injection_flow_gw_h2_lhv"]
            ) * 1e6

            P_out = float(
                row["withdrawal_flow_gw_h2_lhv"]
            ) * 1e6

            key = (
                round(E),
                round(P_in),
                round(P_out),
            )

            unique[key] = row

        fit_rows = list(unique.values())

        # Reference capacities.
        E0 = spec.default_storage_capacity_gwh_lhv * 1e6

        P_out0 = E0 / (spec.default_epr_days * 24.0)

        P0 = P_out0 / spec.default_wtir

        # Normalised regression drivers.
        X = np.array([
            [
                float(r["injection_flow_gw_h2_lhv"]) * 1e6 / P0,
                float(r["withdrawal_flow_gw_h2_lhv"]) * 1e6 / P_out0,
            ]
            for r in fit_rows
        ])

        # Detailed injection investment cost, MEUR.
        y = np.array([
            float(r["detailed_component_tcr_meur"])
            for r in fit_rows
        ])

        # ---------------------------------------------------------
        # Unanchored cross-driver regression (diagnostic figure).
        # ---------------------------------------------------------

        # X already contains:
        # Column 0: P_in / P_in,0
        # Column 1: P_out / P_out,0
        p = X[:, 0]
        p_out = X[:, 1]

        beta_P, beta_P_out = fit_two_driver_cost(
            own_capacity=p,
            secondary_capacity=p_out,
            detailed_cost=y,
        )

        beta_coupled = np.array([
            beta_P,
            beta_P_out,
        ])

        # Affine model, with intercept.
        X_affine = np.column_stack([
            np.ones(len(X)),
            X,
        ])

        beta_affine = np.linalg.lstsq(
            X_affine, y, rcond=None
        )[0]

        # Predict on the displayed WTIR sweep.
        P_out_plot = np.array([
            float(r["withdrawal_flow_gw_h2_lhv"]) * 1e6
            for r in wtir_rows
        ])

        P_plot = np.array([
            float(r["injection_flow_gw_h2_lhv"]) * 1e6
            for r in wtir_rows
        ])

        x_plot = np.array([
            float(r["wtir"])
            for r in wtir_rows
        ])

        observed = np.array([
            float(r["tcr_rate"])
            for r in wtir_rows
        ])

        X_plot = np.column_stack([
            P_plot / P0,
            P_out_plot / P_out0,
        ])

        # Predicted absolute costs (MEUR).
        cost_coupled = X_plot @ beta_coupled

        cost_affine = (
            np.column_stack([
                np.ones(len(X_plot)),
                X_plot,
            ])
            @ beta_affine
        )

        # Convert predicted costs to injection EUR/kW.
        rate_coupled = cost_coupled * 1e6 / P_plot
        rate_affine = cost_affine * 1e6 / P_plot

        # Constant reference rate.
        reference_rate = float(
            wtir_rows[0]["default_linearised_tcr_rate"]
        )

        rate_constant = np.full_like(
            x_plot,
            reference_rate,
        )

        # ---------------------------------------------------------
        # Plot
        # ---------------------------------------------------------

        ax.plot(
            x_plot,
            observed,
            "ko",
            markersize=4,
            label="Detailed model",
            zorder=5,
        )

        ax.plot(
            x_plot,
            rate_constant,
            color="0.5",
            linestyle="--",
            linewidth=1.7,
            label="Constant unit rate",
        )

        ax.plot(
            x_plot,
            rate_coupled,
            color="tab:blue",
            linewidth=2,
            label="Cross-driver linear",
        )

        ax.plot(
            x_plot,
            rate_affine,
            color="tab:orange",
            linestyle=":",
            linewidth=2,
            label="Affine",
        )

        ax.axvline(
            spec.default_wtir,
            color="0.7",
            linestyle=":",
            linewidth=0.8,
        )

        ax.set_title(spec.label)
        ax.set_xlabel("Withdrawal-to-injection ratio [-]")
        ax.set_ylabel("Injection TCR [EUR/kW]")
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))

    save_figure(
        fig,
        figure_dir / "injection_cost_model_comparison",
    )


def print_cost_fit_summary(
    fit_results: list[dict[str, Any]],
) -> None:
    """Print diagnostic results for the total-project cost models."""

    print("\nLINEAR COST REPRESENTATION COMPARISON")
    print("-------------------------------------")

    for spec in get_selected_specs():
        print(f"\n{spec.label}")

        for domain in ("near_reference", "full"):
            print(f"  {domain}:")

            selected = [
                row
                for row in fit_results
                if row["technology"] == spec.technology
                and row["component_group"] == "Total"
                and row["domain"] == domain
            ]

            for row in selected:
                print(
                    f"    {row['model']:<20} "
                    f"LOOCV median error="
                    f"{row['loocv_median_abs_pct']:.2f}%  "
                    f"max={row['loocv_max_abs_pct']:.2f}%  "
                    f"F={row['intercept_meur']:.2f} MEUR"
                )

                if row["negative_capacity_coefficient"]:
                    print(
                        "      WARNING: One or more fitted capacity "
                        "coefficients are negative."
                    )

# =============================================================================
# Main
# =============================================================================


def main() -> None:
    rows = run_experiments()
    write_rows_to_csv(rows, OUTPUT_CSV)

    if PLOT_COMPONENT_RATE_FACTORS:
        plot_component_rate_factors(rows, FIGURE_DIR)
    if PLOT_TOTAL_ERROR:
        plot_total_linearisation_error(rows, FIGURE_DIR)

    print_summary(rows)

    if RUN_COST_FIT_ANALYSIS:
        fit_results = run_cost_fit_analysis(rows)

        write_rows_to_csv(
            fit_results,
            COST_FIT_RESULTS_CSV,
        )

        plot_cost_fit_comparison(
            fit_results,
            FIGURE_DIR,
        )

        print_cost_fit_summary(fit_results)

        plot_withdrawal_cost_model_comparison(
            rows,
            FIGURE_DIR,
        )

        plot_injection_cost_model_comparison(
            rows,
            FIGURE_DIR,
        )
    
    print(f"\nSaved results: {OUTPUT_CSV}")
    print(f"Saved figures: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
