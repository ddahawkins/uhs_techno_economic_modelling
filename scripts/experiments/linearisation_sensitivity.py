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

import matplotlib.pyplot as plt

from uhs_costs.design.aquifer import construct_project as construct_aquifer_project
from uhs_costs.design.depleted_gas_field import (
    construct_project as construct_dgf_project,
)
from uhs_costs.design.lined_rock_cavern import (
    construct_project as construct_lrc_project,
)
from uhs_costs.design.salt_cavern import (
    construct_project as construct_salt_cavern_project,
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

# Common normalised design-point sweep for both E and EPR.
#
# The same multipliers are used for every technology, which makes the x-axis
# directly comparable across technologies. The corresponding absolute E and EPR
# values are still written to the CSV.
DESIGN_POINT_MULTIPLIERS = (
    0.25,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    2.00,
)

# WTIR is deliberately kept in its natural units rather than normalised.
WTIR_SWEEP = (1.0, 2.0, 3.0, 4.0, 5.0)

# Figure switches make it easy to suppress one output during development.
PLOT_COMPONENT_RATE_FACTORS = True
PLOT_TOTAL_ERROR = True


@dataclass(frozen=True)
class TechnologySpec:
    """Reference design used to derive one technology's default CEM rates."""

    technology: str
    label: str
    case_name: str
    project_constructor: Callable[..., Any]
    cost_function: Callable[..., Any]

    # Reference facility used to derive the fixed linear CEM coefficients.
    default_storage_capacity_gwh_lhv: float
    default_epr_days: float
    default_wtir: float


# Defaults reproduce scripts/cost_rates.py.
TECHNOLOGIES: tuple[TechnologySpec, ...] = (
    TechnologySpec(
        technology="salt_cavern",
        label="Salt cavern",
        case_name="salt_cavern",
        project_constructor=construct_salt_cavern_project,
        cost_function=calculate_salt_cavern_cost_components,
        default_storage_capacity_gwh_lhv=800.0,
        default_epr_days=12.0,
        default_wtir=2.0,
    ),
    TechnologySpec(
        technology="depleted_gas_field",
        label="Depleted gas field",
        case_name="depleted_gas_field",
        project_constructor=construct_dgf_project,
        cost_function=calculate_depleted_gas_field_cost_components,
        default_storage_capacity_gwh_lhv=1700.0,
        default_epr_days=66.0,
        default_wtir=2.0,
    ),
    TechnologySpec(
        technology="aquifer",
        label="Aquifer",
        case_name="aquifer",
        project_constructor=construct_aquifer_project,
        cost_function=calculate_aquifer_cost_components,
        default_storage_capacity_gwh_lhv=1700.0,
        default_epr_days=66.0,
        default_wtir=2.0,
    ),
    TechnologySpec(
        technology="lined_rock_cavern",
        label="Lined rock cavern",
        case_name="lined_rock_cavern",
        project_constructor=construct_lrc_project,
        cost_function=calculate_lined_rock_cavern_cost_components,
        default_storage_capacity_gwh_lhv=20.0,
        default_epr_days=8.0,
        default_wtir=2.0,
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
    sensitivities = tuple(RUN_SENSITIVITIES)

    fig, axes = plt.subplots(
        nrows=len(COMPONENT_GROUPS),
        ncols=len(sensitivities),
        figsize=(12, 9),
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
    sensitivities = tuple(RUN_SENSITIVITIES)

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
    save_figure(fig, figure_dir / "total_linearisation_error")


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

    print(f"\nSaved results: {OUTPUT_CSV}")
    print(f"Saved figures: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
