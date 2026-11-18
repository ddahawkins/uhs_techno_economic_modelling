from uhs_costs.design.salt_cavern import (
        construct_project as construct_salt_cavern_project, 
        REFERENCE_CAPACITY_GWH_LHV as sc_reference_capacity, 
        REFERENCE_EPR_DAYS as sc_reference_epr, 
        REFERENCE_WTIR as sc_reference_wtir
    )
from uhs_costs.analysis.pressure_constraint import (
    calculate_pressure_limited_flow_curves_for_projects,
    plot_curves,
    plot_soc_change_bounds,
    plot_soc_endpoint_bounds,
    save_results,
    summarise_soc_endpoint_coefficients,
    run_pressure_ramp_limit_sensitivity
)
from uhs_costs.design.helpers.project import StorageProject, StorageTechnology

from uhs_costs.design.default_design_assumptions import HyStoriesDesignAssumptions, construct_hystories_design_assumptions


from pathlib import Path




# =============================================================================
# Main
# =============================================================================


def main() -> None:

    salt_cavern_project = construct_salt_cavern_project(
        working_gas_capacity_kwh_lhv=sc_reference_capacity,
        withdrawal_flow_kw_h2_lhv=sc_reference_capacity/(sc_reference_epr*24),
        injection_flow_kw_h2_lhv=sc_reference_capacity/(sc_reference_epr*24*sc_reference_wtir),
        case_name="salt_cavern",
    )

    low_pressure_range_assumptions =  construct_hystories_design_assumptions(
        storage_technology=StorageTechnology.SALT_CAVERN,
        general_overrides={
            'reference_reservoir_depth_m': 1000,
            'minimum_operating_pressure_pa': 7.0e6,
            'maximum_operating_pressure_pa': 22e6,
            'abandonment_pressure_pa': 0,
        }
    )

    salt_cavern_project_low_pressure = construct_salt_cavern_project(
        working_gas_capacity_kwh_lhv=800_000_000,
        withdrawal_flow_kw_h2_lhv=800_000_000/(24*12),
        injection_flow_kw_h2_lhv=800_000_000/(24*12)/2,
        case_name="salt_cavern",
        assumptions=low_pressure_range_assumptions
    )

    high_pressure_range_assumptions =  construct_hystories_design_assumptions(
        storage_technology=StorageTechnology.SALT_CAVERN,
        general_overrides={
            'reference_reservoir_depth_m': 1000,
            'minimum_operating_pressure_pa': 11e6,
            'maximum_operating_pressure_pa': 26e6,
            'abandonment_pressure_pa': 0,
        }
    )

    salt_cavern_project_high_pressure = construct_salt_cavern_project(
        working_gas_capacity_kwh_lhv=800_000_000,
        withdrawal_flow_kw_h2_lhv=800_000_000/(24*12),
        injection_flow_kw_h2_lhv=800_000_000/(24*12)/2,
        case_name="salt_cavern",
        assumptions=high_pressure_range_assumptions
    )

    narrow_pressure_range_assumptions =  construct_hystories_design_assumptions(
        storage_technology=StorageTechnology.SALT_CAVERN,
        general_overrides={
            'reference_reservoir_depth_m': 1000,
            'minimum_operating_pressure_pa': 11e6,
            'maximum_operating_pressure_pa': 22e6,
            'abandonment_pressure_pa': 0,
        }
    )

    salt_cavern_project_narrow_pressure = construct_salt_cavern_project(
        working_gas_capacity_kwh_lhv=800_000_000,
        withdrawal_flow_kw_h2_lhv=800_000_000/(24*12),
        injection_flow_kw_h2_lhv=800_000_000/(24*12)/2,
        case_name="salt_cavern",
        assumptions=narrow_pressure_range_assumptions
    )

    wide_pressure_range_assumptions =  construct_hystories_design_assumptions(
        storage_technology=StorageTechnology.SALT_CAVERN,
        general_overrides={
            'reference_reservoir_depth_m': 1000,
            'minimum_operating_pressure_pa': 7e6,
            'maximum_operating_pressure_pa': 26e6,
            'abandonment_pressure_pa': 0,
        }
    )

    salt_cavern_project_wide_pressure = construct_salt_cavern_project(
        working_gas_capacity_kwh_lhv=800_000_000,
        withdrawal_flow_kw_h2_lhv=800_000_000/(24*12),
        injection_flow_kw_h2_lhv=800_000_000/(24*12)/2,
        case_name="salt_cavern",
        assumptions=wide_pressure_range_assumptions
    )

    output_dir = Path("outputs/pressure_constraints")
    output_dir.mkdir(parents=True, exist_ok=True)

    ramp_limit_coefficients = run_pressure_ramp_limit_sensitivity(
        default_project=salt_cavern_project,
        output_dir=output_dir,
    )


    projects: dict[str, StorageProject] = {
        "Default:90-240bar": salt_cavern_project,
        "Low:70-220bar": salt_cavern_project_low_pressure,
        "High:110-260bar": salt_cavern_project_high_pressure,
        "Narrow:110-220bar": salt_cavern_project_narrow_pressure,
        "Wide:70-260bar": salt_cavern_project_wide_pressure,
    }

    if not projects:
        raise ValueError(
            "No projects provided. Add StorageProject objects to the `projects` "
            "dictionary in main()."
        )

    # Optional plotting colours. Matplotlib colour names, hex codes, or RGB tuples
    # are all supported.
    project_colours = {
        "Default:90-240bar": (13/255, 8/255, 135/255),
        "Low:70-220bar": (126/255, 3/255, 168/255),
        "High:110-260bar": (204/255, 71/255, 120/255),
        "Narrow:110-220bar": (248/255, 149/255, 64/255),
        "Wide:70-260bar": (240/255, 249/255, 33/255),
    }
    bound_colours = {
        "min": "tab:red",
        "max": "tab:green",
    }


    results_by_project = calculate_pressure_limited_flow_curves_for_projects(
        projects=projects,
        pressure_step_pa=1e5,                 
        pressure_change_limit_pa_per_day=10e5, 
    )



    save_results(
        results_by_project=results_by_project,
        output_dir=output_dir,
    )

    plot_curves(
        results=results_by_project,
        projects=projects,
        show_design_capacity=True,
        x="soc",
        y="withdrawal_kw_lhv",
        y_scale=1 / 1e6,
        xlabel="State of charge [-]",
        ylabel="Average 24-hour  Withdrawal capacity [GW LHV]",
        title="Pressure-constrained withdrawal capacity",
        line_colors=project_colours,
        output_path=output_dir / "withdrawal_capacity_by_soc.png",
    )

    plot_curves(
        results=results_by_project,
        projects=projects,
        x="soc",
        y="withdrawal_gw_per_twh",
        xlabel="State of charge [-]",
        ylabel="Average 24-hour withdrawal capacity [GW/TWh]",
        title="Capacity-normalised pressure-constrained withdrawal capacity",
        line_colors=project_colours,
        output_path=output_dir / "withdrawal_capacity_per_twh_by_soc.png",
    )

    plot_curves(
        results=results_by_project,
        projects=projects,
        show_design_capacity=True,
        x="soc",
        y="injection_kw_lhv",
        y_scale=1 / 1e6,
        xlabel="State of charge [-]",
        ylabel="Average Injection capacity over 24-hours [GW LHV]",
        title="Pressure-constrained injection capacity",
        line_colors=project_colours,
        output_path=output_dir / "injection_capacity_by_soc.png",
    )

    plot_soc_change_bounds(
        results=results_by_project,
        x="soc",
        xlabel="State of charge at time t [-]",
        ylabel="Signed 24-hour change in state of charge [-]",
        title="Pressure-constrained 24-hour SoC change bounds",
        line_colors=project_colours,
        show_linear_fit=False,
        output_path=output_dir / "soc_change_bounds_by_soc.png",
    )

    plot_soc_endpoint_bounds(
        results={"default": results_by_project["Default:90-240bar"]},
        show_linear_fit=False,
        title=None,
        output_path=output_dir / "pressure_constraint_future_soc_default.pdf",
        show=True,
    )
    plot_curves(
        results=results_by_project,
        projects=projects,
        show_design_capacity=True,
        x="pressure_bar",
        y="withdrawal_kw_lhv",
        y_scale=1 / 1e6,
        xlabel="Pressure [bar]",
        ylabel="Average Withdrawal capacity over 24-hours [GW LHV]",
        title="Pressure-constrained withdrawal capacity",
        line_colors=project_colours,
        output_path=output_dir / "withdrawal_capacity_by_pressure.png",
    )

    plot_curves(
        results=results_by_project,
        x="pressure_bar",
        y="h2_inventory_kWh",
        y_scale=1 / 1e6,
        xlabel="Pressure [bar]",
        ylabel="Hydrogen inventory [GWh]",
        title="Hydrogen inventory as a function of pressure",
        line_colors=project_colours,
        output_path=output_dir / "inventory_by_pressure.png",
    )

    pressure_range_coefficients = summarise_soc_endpoint_coefficients(
        results_by_project=results_by_project,
        conservative=True,
        output_csv_path="outputs/pressure_range_endpoint_coefficients.csv",
    )

    print(pressure_range_coefficients)




if __name__ == "__main__":
    main()