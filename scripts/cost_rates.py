from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from uhs_costs.design.aquifer import construct_project as construct_aquifer_project
from uhs_costs.design.salt_cavern import construct_project as construct_salt_cavern_project
from uhs_costs.design.depleted_gas_field import construct_project as construct_dgf_project
from uhs_costs.design.lined_rock_cavern import construct_project as construct_lrc_project

from uhs_costs.cost_model.aquifer import (
    calculate_cost_components as calculate_aquifer_cost_components,
)
from uhs_costs.cost_model.salt_cavern import (
    calculate_salt_cavern_cost_components,
)
from uhs_costs.cost_model.depleted_gas_field import (
    calculate_depleted_gas_field_cost_components,
)

from uhs_costs.cost_model.lined_rock_cavern import (
    calculate_lined_rock_cavern_cost_components,
)

from uhs_costs.design.helpers.project import StorageProject


# =============================================================================
# Generic helpers
# =============================================================================


def get_nested_attr(obj: Any, attr_path: str, default: Any = None) -> Any:
    """Safely get a nested attribute using a dotted path."""

    current = obj

    for attr in attr_path.split("."):
        if current is None:
            return default

        if not hasattr(current, attr):
            return default

        current = getattr(current, attr)

    return current


def get_first_attr(
    obj: Any,
    candidate_names: tuple[str, ...],
    default: Any = None,
) -> Any:
    """Return the first available attribute from a list of candidate names."""

    for name in candidate_names:
        if hasattr(obj, name):
            return getattr(obj, name)

    return default


def normalise_name(value: Any) -> str:
    """Convert enums/strings/objects to lower-case comparable strings."""

    if value is None:
        return ""

    if hasattr(value, "value"):
        value = value.value

    return str(value).lower().strip()


def format_value(value: Any, decimals: int = 2) -> str:
    """Format values for table output."""

    if value is None:
        return "n/a"

    if isinstance(value, int):
        return f"{value:,}"

    if isinstance(value, float):
        return f"{value:,.{decimals}f}"

    return str(value)


def format_ratio(value: Any, decimals: int = 1) -> str:
    """Format a ratio as a percentage."""

    if value is None:
        return "n/a"

    return f"{100 * value:,.{decimals}f}%"


def print_table(rows: list[dict[str, str]], columns: list[str]) -> None:
    """Print a simple aligned table."""

    if not rows:
        print("No rows to print.")
        return

    column_widths = {
        column: max(
            len(column),
            *(len(str(row.get(column, ""))) for row in rows),
        )
        for column in columns
    }

    header = " | ".join(
        column.ljust(column_widths[column]) for column in columns
    )
    separator = "-+-".join(
        "-" * column_widths[column] for column in columns
    )

    print(header)
    print(separator)

    for row in rows:
        print(
            " | ".join(
                str(row.get(column, "")).ljust(column_widths[column])
                for column in columns
            )
        )


# =============================================================================
# Design characteristic extraction
# =============================================================================


def extract_design_characteristics(project: StorageProject) -> dict[str, Any]:
    """Extract key design characteristics from a StorageProject."""

    return {
        # ---------------------------------------------------------------------
        # High-level size
        # ---------------------------------------------------------------------
        "Recoverable Working gas capacity [GWh LHV]": (
            project.inventory.recoverable_working_gas_capacity_kwh_lhv / 1_000_000
        ),
        "Required storage volume [million m3]": (
            project.inventory.required_storage_volume_m3 / 1_000_000
        ),
        # "Working gas H2 volume [million Sm3]": (
        #     project.inventory.working_gas_h2_volume_sm3 / 1_000_000
        # ),
        "Working gas H2 mass [kt]": (
            project.inventory.working_gas_h2_mass_kg / 1_000_000
        ),

        # ---------------------------------------------------------------------
        # Cushion gas / total inventory
        # ---------------------------------------------------------------------
        # "Cushion gas H2 volume [million Sm3]": (
        #     project.inventory.cushion_gas_h2_volume_sm3 / 1_000_000
        # ),
        "Total gas at max SOC [million Sm3]": (
            project.inventory.total_gas_at_max_soc_volume_sm3 / 1_000_000
        ),
        "Total cushion gas volume [million Sm3]": (
            (
                project.inventory.cushion_gas_h2_volume_sm3
                + (project.inventory.abandonment_gas_methane_volume_sm3 or 0.0)
            )
            / 1_000_000
        ),
        "Cushion / total gas volume ratio": (
            project.inventory.cushion_gas_to_total_gas_volume_ratio
        ),
        "Methane component of cushion gas [million Sm3]":(
            None
            if project.inventory.abandonment_gas_methane_volume_sm3 is None
            else project.inventory.abandonment_gas_methane_volume_sm3 / 1_000_000
        ),
        "H2 component of cushion gas [million Sm3]":(
            project.inventory.cushion_gas_h2_volume_sm3 / 1_000_000
        ),
        "Mass of H2 cushion gas [kt]": (
            project.inventory.cushion_gas_h2_mass_kg / 1_000_000
        ),

        # ---------------------------------------------------------------------
        # Flow sizing
        # ---------------------------------------------------------------------
        "Withdrawal flow [GW H2 LHV]": (
            project.flows.withdrawal_flow_kw_h2_lhv / 1_000_000
        ),
        "Injection flow [GW H2 LHV]": (
            project.flows.injection_flow_kw_h2_lhv / 1_000_000
        ),
        "Withdrawal flow [million Sm3/day]": (
            project.flows.withdrawal_flow_million_sm3_per_day
        ),
        "Injection flow [million Sm3/day]": (
            project.flows.injection_flow_million_sm3_per_day
        ),
        # "Withdrawal flow [kg/s]": project.flows.withdrawal_flow_kg_per_s,
        # "Injection flow [kg/s]": project.flows.injection_flow_kg_per_s,
        # "Withdrawal / injection ratio": project.flows.withdrawal_to_injection_ratio,

        # ---------------------------------------------------------------------
        # Pressures
        # ---------------------------------------------------------------------
        "Minimum operating pressure [bar]": (
            project.pressures.minimum_operating_pressure_bar
        ),
        "Maximum operating pressure [bar]": (
            project.pressures.maximum_operating_pressure_bar
        ),
        "Abandonment pressure [bar]": (
            project.pressures.abandonment_pressure_pa / 1e5
        ),
        "Pipeline pressure [bar]": project.pressures.pipeline_pressure_bar,
        # "Operating pressure ratio": project.pressures.operating_pressure_ratio,
        # "Maximum compression ratio": project.pressures.maximum_compression_ratio,

        # ---------------------------------------------------------------------
        # Wells / caverns / drilling
        # ---------------------------------------------------------------------
        "Number of well heads": project.wells.number_well_heads,
        "Number of caverns": project.wells.number_caverns,
        "Number of production wells": project.wells.number_production_wells,
        "Number of observation wells": project.wells.number_observation_wells,
        "Reservoir depth [m]": (
            project.drilling.last_cemented_casing_shoe_m
        ),
        "Reservoir temperature [k]": project.wells.well_temperature_k,
        # "Drilling complexity index": project.drilling.drilling_complexity_index,
        # "Field line length per well head [km]": (
        #     project.field_interconnection.field_line_length_per_well_head_km
        # ),
        "Total field line length [km]": (
            project.field_interconnection.additional_field_line_length_km
        ),

        # ---------------------------------------------------------------------
        # Salt cavern-specific
        # ---------------------------------------------------------------------
        # "Fresh water pipeline length [km]": get_nested_attr(
        #     project, "salt_leaching.fresh_water_pipeline_length_km"
        # ),
        # "Brine disposal pipeline length [km]": get_nested_attr(
        #     project, "salt_leaching.brine_disposal_pipeline_length_km"
        # ),
        # "Free gas volume per cavern [thousand m3]": get_nested_attr(
        #     project, "salt_leaching.free_gas_volume_per_cavern_thousand_m3"
        # ),
        # "Leaching duration per cavern [months]": get_nested_attr(
        #     project, "salt_leaching_process.leaching_duration_per_cavern_months"
        # ),
        "Leaching duration [years]": get_nested_attr(
            project, "salt_leaching_process.total_leaching_duration_years"
        ),
        # "Debrining duration per cavern [days]": get_nested_attr(
        #     project, "salt_conversion_process.debrining_duration_per_cavern_days"
        # ),
        "Conversion duration [years]": get_nested_attr(
            project, "salt_conversion_process.total_conversion_duration_years"
        ),

        # ---------------------------------------------------------------------
        # Porous-media-specific
        # ---------------------------------------------------------------------
        "First fill duration [years]": get_nested_attr(
            project, "porous_first_fill_process.first_gas_fill_duration_years"
        ),

        # ---------------------------------------------------------------------
        # Compression / purification
        # ---------------------------------------------------------------------
        "Compression method": project.compression.method.value,
        "Compression electricity [kWh_e/kWh_H2]": (
            project.compression.electricity_kwh_per_kwh_h2_lhv
        ),
        "Purification H2 recovery": project.purification.hydrogen_recovery,
        "Purification cost factor": project.purification.purification_factor,
        # "Compression inlet temperature [K]": project.compression.inlet_temperature_k,
        "Compression inlet pressure [bar]": project.compression.inlet_pressure_pa / 1e5,
        "Compression outlet pressure [bar]": project.compression.outlet_pressure_pa / 1e5,
        # "Compression mass flow [kg/s]": project.compression.mass_flow_kg_s,
        "Number of compression stages": project.compression.number_of_stages,
        "Total brake power [MW]": project.compression.total_brake_power_kw / 1_000,
        "Total electric power [MW]": project.compression.total_electric_power_kw / 1_000,
        "Total design brake power [MW]": (
            project.compression.total_design_brake_power_kw / 1_000
        ),
        "Total design electric power [MW]": (
            project.compression.total_design_electric_power_kw / 1_000
        ),
        # "Number of compressor trains": project.compression.number_of_trains,
        # "Design power factor": project.compression.design_power_factor,
        "Purification factor": project.purification.purification_factor,
    }


def format_design_value(metric: str, value: Any) -> str:
    """Apply metric-specific formatting."""

    if value is None:
        return "n/a"

    metric_lower = metric.lower()

    if " ratio" in metric_lower and "pressure ratio" not in metric_lower:
        return format_ratio(value)

    if "[gwh" in metric_lower:
        return format_value(value, decimals=1)

    if "[gw" in metric_lower:
        return format_value(value, decimals=2)

    if "[million" in metric_lower:
        return format_value(value, decimals=2)

    if "[kt]" in metric_lower:
        return format_value(value, decimals=2)

    if "[bar]" in metric_lower:
        return format_value(value, decimals=1)

    if "[k]" in metric_lower:
        return format_value(value, decimals=1)

    if "[m]" in metric_lower:
        return format_value(value, decimals=0)

    if "[km]" in metric_lower:
        return format_value(value, decimals=2)

    if "[kg/s]" in metric_lower:
        return format_value(value, decimals=2)

    if "[mw]" in metric_lower:
        return format_value(value, decimals=2)

    if "number of" in metric_lower:
        return format_value(value, decimals=0)

    if isinstance(value, float):
        return format_value(value, decimals=2)

    return str(value)


def print_design_characteristics(projects: list[StorageProject]) -> None:
    """Print design characteristics with technologies as columns."""

    characteristics_by_case = {
        project.case_name: extract_design_characteristics(project)
        for project in projects
    }

    metric_names = list(next(iter(characteristics_by_case.values())).keys())

    columns = ["Metric", *characteristics_by_case.keys()]
    rows: list[dict[str, str]] = []

    for metric in metric_names:
        row = {"Metric": metric}

        for case_name, characteristics in characteristics_by_case.items():
            value = characteristics.get(metric)
            row[case_name] = format_design_value(metric, value)

        rows.append(row)

    print()
    print("DESIGN CHARACTERISTICS")
    print_table(rows, columns)


# =============================================================================
# Cost rate extraction
# =============================================================================

# =============================================================================
# Cost rate extraction
# =============================================================================


def get_cost_components(cost_breakdown: Any) -> tuple[Any, ...]:
    """Return cost components from a cost breakdown object.

    Expected structure:
        CostBreakdown(components=(CostComponent(...), ...))
    """

    if hasattr(cost_breakdown, "components"):
        return cost_breakdown.components

    if isinstance(cost_breakdown, tuple | list):
        return tuple(cost_breakdown)

    raise TypeError(
        f"Unsupported cost breakdown type: {type(cost_breakdown).__name__}"
    )


def get_component_group(cost_component: Any) -> str | None:
    """Map a CostComponent to Injection, Storage, or Withdrawal."""

    cost_driver = normalise_name(cost_component.cost_driver)
    name = normalise_name(cost_component.name)

    if "injection" in cost_driver or "injection" in name:
        return "Injection"

    if "withdrawal" in cost_driver or "withdrawal" in name:
        return "Withdrawal"

    if "storage" in cost_driver:
        return "Storage"

    if "subsurface" in name:
        return "Storage"

    return None


def get_cost_type_label(cost_component: Any) -> str:
    """Return a clean cost type label."""

    cost_type = normalise_name(cost_component.cost_type)

    if cost_type == "capex":
        return "CAPEX"

    if cost_type == "abex":
        return "ABEX"

    if cost_type == "fixed_opex":
        return "Fixed OPEX"

    if cost_type == "variable_opex":
        return "Variable OPEX"

    raise ValueError(f"Unknown cost type: {cost_component.cost_type}")


def get_cost_unit_label(cost_component: Any) -> str:
    """Return the cost unit as a string."""

    if hasattr(cost_component.cost_unit, "value"):
        return cost_component.cost_unit.value

    return str(cost_component.cost_unit)


def get_cost_rate(cost_component: Any) -> float:
    """Calculate cost per driver."""

    if cost_component.driver_value == 0:
        raise ValueError(
            f"Cannot calculate cost rate for {cost_component.name}: "
            "driver_value is zero."
        )

    return cost_component.value_eur / cost_component.driver_value


def summarise_cost_rates(cost_breakdown: Any) -> dict[tuple[str, str], dict[str, Any]]:
    """Summarise cost rates by component group and cost type.

    Returns
    -------
    dict
        Keys are tuples such as:
            ("Injection", "CAPEX")
            ("Storage", "Fixed OPEX")

        Values contain:
            rate: summed cost rate
            unit: cost unit
    """

    grouped_rates: dict[tuple[str, str], float] = defaultdict(float)
    grouped_units: dict[tuple[str, str], str] = {}

    for cost_component in get_cost_components(cost_breakdown):
        component_group = get_component_group(cost_component)

        if component_group is None:
            continue

        cost_type = get_cost_type_label(cost_component)
        cost_unit = get_cost_unit_label(cost_component)
        cost_rate = get_cost_rate(cost_component)

        key = (component_group, cost_type)

        grouped_rates[key] += cost_rate
        grouped_units[key] = cost_unit

    return {
        key: {
            "rate": rate,
            "unit": grouped_units.get(key),
        }
        for key, rate in grouped_rates.items()
    }


def get_total_capital_requirement_rate(
    summary: dict[tuple[str, str], dict[str, Any]],
    component_group: str,
) -> dict[str, Any]:
    """Calculate total capital requirement as CAPEX + ABEX."""

    capex = summary.get((component_group, "CAPEX"), {}).get("rate")
    abex = summary.get((component_group, "ABEX"), {}).get("rate")

    if capex is None and abex is None:
        return {"rate": None, "unit": None}

    unit = (
        summary.get((component_group, "CAPEX"), {}).get("unit")
        or summary.get((component_group, "ABEX"), {}).get("unit")
    )

    return {
        "rate": (capex or 0.0) + (abex or 0.0),
        "unit": unit,
    }


def format_cost_rate(value: float | None) -> str:
    """Format cost rates with enough significant digits for small values."""

    if value is None:
        return "n/a"

    abs_value = abs(value)

    if abs_value == 0:
        return "0"

    # Large cost rates: no need for excessive decimals
    if abs_value >= 100:
        return f"{value:,.0f}"

    if abs_value >= 10:
        return f"{value:,.1f}"

    if abs_value >= 1:
        return f"{value:,.2f}"

    # Small values: use significant figures
    return f"{value:.3g}"


def build_cost_rate_rows(
    cost_breakdowns_by_case: dict[str, Any],
) -> list[dict[str, str]]:
    """Build rows for the cost-rate table."""

    summaries_by_case = {
        case_name: summarise_cost_rates(cost_breakdown)
        for case_name, cost_breakdown in cost_breakdowns_by_case.items()
    }

    component_groups = ["Injection", "Storage", "Withdrawal"]

    row_specs = []

    for component_group in component_groups:
        row_specs.extend(
            [
                (component_group, "CAPEX", "CAPEX"),
                (
                    component_group,
                    "Total capital requirement",
                    "Total capital requirement",
                ),
                (component_group, "Fixed OPEX", "Fixed OPEX"),
                (component_group, "Variable OPEX", "Variable OPEX"),
            ]
        )

    rows: list[dict[str, str]] = []

    for component_group, label, lookup_type in row_specs:
        unit = None

        # Use the first available unit across technologies.
        for summary in summaries_by_case.values():
            if lookup_type == "Total capital requirement":
                result = get_total_capital_requirement_rate(
                    summary,
                    component_group,
                )
            else:
                result = summary.get((component_group, lookup_type), {})

            unit = result.get("unit")

            if unit is not None:
                break

        metric_label = f"{component_group} {label}"

        if unit is not None:
            metric_label = f"{metric_label} [{unit}]"

        row = {"Metric": metric_label}

        for case_name, summary in summaries_by_case.items():
            if lookup_type == "Total capital requirement":
                result = get_total_capital_requirement_rate(
                    summary,
                    component_group,
                )
            else:
                result = summary.get((component_group, lookup_type), {})

            row[case_name] = format_cost_rate(result.get("rate"))

        rows.append(row)

    return rows


def print_cost_rate_outcomes(
    cost_breakdowns_by_case: dict[str, Any],
) -> None:
    """Print key cost outcomes with technologies as columns."""

    rows = build_cost_rate_rows(cost_breakdowns_by_case)
    columns = ["Metric", *cost_breakdowns_by_case.keys()]

    print()
    print("KEY COST OUTCOMES")
    print_table(rows, columns)

# =============================================================================
# Optional debug helper for cost breakdowns
# =============================================================================


def debug_print_cost_breakdown_fields(cost_breakdown: Any, max_items: int = 5) -> None:
    """Print available fields on cost breakdown items.

    Use this if the cost table prints many 'n/a' values. It helps identify the
    actual attribute names used by your cost component objects.
    """

    items = flatten_cost_breakdown(cost_breakdown)

    print()
    print("DEBUG: COST BREAKDOWN ITEM FIELDS")

    for index, item in enumerate(items[:max_items], start=1):
        print()
        print(f"Item {index}: {type(item).__name__}")
        print(item)

        if hasattr(item, "__dataclass_fields__"):
            print("Dataclass fields:")
            for field_name in item.__dataclass_fields__:
                print(f"  - {field_name}: {getattr(item, field_name)}")
        elif hasattr(item, "__dict__"):
            print("__dict__ fields:")
            for field_name, value in vars(item).items():
                print(f"  - {field_name}: {value}")
        else:
            print("No dataclass fields or __dict__ available.")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    aquifer_scale = 2_571_000_000 #kWh
    aquifer_project = construct_aquifer_project(
        working_gas_capacity_kwh_lhv=aquifer_scale,
        withdrawal_flow_kw_h2_lhv=aquifer_scale/(24*66),
        injection_flow_kw_h2_lhv=aquifer_scale/(24*66)/2,
        case_name="aquifer",
    )
    aquifer_cost_breakdown = calculate_aquifer_cost_components(aquifer_project)

    dgf_scale = 2_336_660_000 
    dgf_project = construct_dgf_project(
        working_gas_capacity_kwh_lhv=dgf_scale,
        withdrawal_flow_kw_h2_lhv=dgf_scale/(24*66),
        injection_flow_kw_h2_lhv=dgf_scale/(24*66)/2,
        case_name="depleted_gas_field",
    )
    dgf_cost_breakdown = calculate_depleted_gas_field_cost_components(dgf_project)

    salt_cavern_scale = 1_064_730_000 #kWh
    salt_cavern_project = construct_salt_cavern_project(
        working_gas_capacity_kwh_lhv=salt_cavern_scale,
        withdrawal_flow_kw_h2_lhv=salt_cavern_scale/(24*12),
        injection_flow_kw_h2_lhv=salt_cavern_scale/(24*12)/2,
        case_name="salt_cavern",
    )
    salt_cavern_cost_breakdown = calculate_salt_cavern_cost_components(
        salt_cavern_project
    )

    lrc_scale = 21_330_000 #kWh
    lrc_project = construct_lrc_project(
        working_gas_capacity_kwh_lhv=lrc_scale, 
        withdrawal_flow_kw_h2_lhv=lrc_scale/(24*7.5),
        injection_flow_kw_h2_lhv=lrc_scale/(24*7.5)/2,
        case_name="lined_rock_cavern",
    )

    lrc_cavern_cost_breakdown = calculate_lined_rock_cavern_cost_components(
        lrc_project
    )

    projects = [salt_cavern_project, dgf_project, aquifer_project, lrc_project]

    cost_breakdowns_by_case = {
        salt_cavern_project.case_name: salt_cavern_cost_breakdown,
        dgf_project.case_name: dgf_cost_breakdown,
        aquifer_project.case_name: aquifer_cost_breakdown,
        lrc_project.case_name: lrc_cavern_cost_breakdown
    }

    print_design_characteristics(projects)
    print_cost_rate_outcomes(cost_breakdowns_by_case)

    # Uncomment this if the cost table shows many 'n/a' values:
    # debug_print_cost_breakdown_fields(salt_cavern_cost_breakdown)


if __name__ == "__main__":
    main()