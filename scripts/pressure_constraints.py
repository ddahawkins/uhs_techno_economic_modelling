from pathlib import Path

from uhs_costs.design.salt_cavern import (
    construct_project as construct_salt_cavern_project,
    REFERENCE_CAPACITY_GWH_LHV as sc_reference_capacity,
    REFERENCE_EPR_DAYS as sc_reference_epr,
    REFERENCE_WTIR as sc_reference_wtir,
)

from uhs_costs.analysis.pressure_constraint import (
    calculate_pressure_limited_flow_curves_for_projects,
    plot_soc_endpoint_bounds,
)


def main() -> None:

    # Convert reference working-gas capacity from GWh to kWh.
    capacity_kwh = sc_reference_capacity * 1e6

    project = construct_salt_cavern_project(
        working_gas_capacity_kwh_lhv=capacity_kwh,
        withdrawal_flow_kw_h2_lhv=(
            capacity_kwh / (sc_reference_epr * 24)
        ),
        injection_flow_kw_h2_lhv=(
            capacity_kwh / (sc_reference_epr * 24 * sc_reference_wtir)
        ),
        case_name="salt_cavern",
    )

    project_name = "Reference salt cavern"

    results = calculate_pressure_limited_flow_curves_for_projects(
        projects={project_name: project},
        pressure_step_pa=1e5,
        pressure_change_limit_pa_per_day=10e5,
    )

    output_dir = Path("outputs/pressure_constraints")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_soc_endpoint_bounds(
        results={"default": results[project_name]},
        show_linear_fit=False,
        title=None,
        output_path=output_dir / "pressure_constraint_future_soc_linear.pdf",
        show=False,
    )


if __name__ == "__main__":
    main()