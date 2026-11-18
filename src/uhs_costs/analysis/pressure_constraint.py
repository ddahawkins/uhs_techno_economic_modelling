from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from uhs_costs.constants import (
    HYDROGEN_LHV_KWH_PER_KG,
    SECONDS_PER_DAY,
    STANDARD_PRESSURE_PA,
    STANDARD_TEMPERATURE_K,
)
from uhs_costs.design.helpers.project import StorageProject
from uhs_costs.gas_properties import hydrogen, methane


BAR_TO_PA = 1e5


@dataclass
class PressureLimitedFlowPoint:
    pressure_pa: float
    pressure_bar: float
    soc: float
    h2_inventory_kWh: float
    h2_inventory_kg: float
    h2_inventory_after_withdrawal_kWh: float
    h2_inventory_after_injection_kWh: float
    soc_after_withdrawal_pressure_basis: float
    soc_after_injection_pressure_basis: float
    withdrawal_kg_per_day: float
    injection_kg_per_day: float
    withdrawal_kg_s: float
    injection_kg_s: float
    withdrawal_kwh_per_day: float
    injection_kwh_per_day: float
    withdrawal_kw_lhv: float
    injection_kw_lhv: float


def total_pressure_from_inventories(
    m_h2_kg: float,
    m_ch4_kg: float,
    volume_m3: float,
    temperature_k: float,
) -> float:
    """
    Estimate total storage pressure from fixed methane inventory and variable H2 inventory.

    Methane and hydrogen are treated as immiscible pressure-contributing gas components
    sharing the same effective storage volume and temperature. Methane is fixed;
    only hydrogen moves in and out.
    """
    if volume_m3 <= 0:
        raise ValueError("volume_m3 must be greater than zero.")

    if m_h2_kg < 0:
        raise ValueError("m_h2_kg cannot be negative.")

    if m_ch4_kg < 0:
        raise ValueError("m_ch4_kg cannot be negative.")

    if m_h2_kg > 0:
        rho_h2 = m_h2_kg / volume_m3
        p_h2 = hydrogen.pressure_from_density(
            density_kg_m3=rho_h2,
            temperature_k=temperature_k,
        )
    else:
        p_h2 = 0.0

    if m_ch4_kg > 0:
        rho_ch4 = m_ch4_kg / volume_m3
        p_ch4 = methane.pressure_from_density(
            density_kg_m3=rho_ch4,
            temperature_k=temperature_k,
        )
    else:
        p_ch4 = 0.0

    return p_h2 + p_ch4


def hydrogen_inventory_from_total_pressure(
    target_pressure_pa: float,
    m_ch4_fixed_kg: float,
    volume_m3: float,
    temperature_k: float,
    h2_upper_bound_kg: float,
) -> float:
    """
    Find the hydrogen inventory required to reach a target total pressure.

    The methane mass is fixed. The hydrogen inventory is solved numerically.
    """

    def residual(m_h2_kg: float) -> float:
        return (
            total_pressure_from_inventories(
                m_h2_kg=m_h2_kg,
                m_ch4_kg=m_ch4_fixed_kg,
                volume_m3=volume_m3,
                temperature_k=temperature_k,
            )
            - target_pressure_pa
        )

    residual_at_zero = residual(0.0)

    if residual_at_zero > 0:
        methane_only_pressure_bar = residual_at_zero + target_pressure_pa
        methane_only_pressure_bar /= BAR_TO_PA

        raise ValueError(
            "Target pressure is below the pressure caused by fixed methane alone. "
            f"Methane-only pressure is approximately {methane_only_pressure_bar:.2f} bar. "
            "This implies the assumed methane residual/cushion mass is too high "
            "for the specified operating pressure."
        )

    upper_bound = h2_upper_bound_kg
    residual_at_upper = residual(upper_bound)

    while residual_at_upper < 0:
        upper_bound *= 2
        residual_at_upper = residual(upper_bound)

    return brentq(
        residual,
        0.0,
        upper_bound,
    )


def methane_mass_from_standard_volume(
    methane_volume_sm3: float | None,
) -> float:
    """
    Convert methane standard volume [Sm3] to methane mass [kg].

    Sm3 is an accounting volume at standard conditions, not the physical
    reservoir volume.
    """
    if methane_volume_sm3 is None:
        return 0.0

    if methane_volume_sm3 < 0:
        raise ValueError("methane_volume_sm3 cannot be negative.")

    if methane_volume_sm3 == 0:
        return 0.0

    return methane_volume_sm3 * methane.density(
        pressure_pa=STANDARD_PRESSURE_PA,
        temperature_k=STANDARD_TEMPERATURE_K,
    )


def calculate_pressure_limited_flow_curve(
    project: StorageProject,
    pressure_step_pa: float = 1 * BAR_TO_PA,
    pressure_change_limit_pa_per_day: float = 10 * BAR_TO_PA,
) -> pd.DataFrame:
    """
    Calculate pressure-limited injection and withdrawal curves for one StorageProject.

    The pressure-change limit is interpreted as a daily pressure ramp limit.
    Outputs `withdrawal_kw_lhv` and `injection_kw_lhv` are therefore average daily
    LHV power values implied by that pressure limit.
    """
    if pressure_step_pa <= 0:
        raise ValueError("pressure_step_pa must be greater than zero.")

    if pressure_change_limit_pa_per_day <= 0:
        raise ValueError("pressure_change_limit_pa_per_day must be greater than zero.")

    p_min_pa = project.pressures.minimum_operating_pressure_pa
    p_max_pa = project.pressures.maximum_operating_pressure_pa
    temperature_k = project.wells.well_temperature_k
    volume_m3 = project.inventory.required_storage_volume_m3

    if p_max_pa <= p_min_pa:
        raise ValueError("Maximum operating pressure must be greater than minimum pressure.")

    m_ch4_fixed_kg = methane_mass_from_standard_volume(
        project.inventory.abandonment_gas_methane_volume_sm3
    )

    rho_h2_max = hydrogen.density(
        pressure_pa=p_max_pa,
        temperature_k=temperature_k,
    )

    h2_upper_bound_kg = rho_h2_max * volume_m3 * 1.1

    m_h2_min = hydrogen_inventory_from_total_pressure(
        target_pressure_pa=p_min_pa,
        m_ch4_fixed_kg=m_ch4_fixed_kg,
        volume_m3=volume_m3,
        temperature_k=temperature_k,
        h2_upper_bound_kg=h2_upper_bound_kg,
    )

    m_h2_max = hydrogen_inventory_from_total_pressure(
        target_pressure_pa=p_max_pa,
        m_ch4_fixed_kg=m_ch4_fixed_kg,
        volume_m3=volume_m3,
        temperature_k=temperature_k,
        h2_upper_bound_kg=h2_upper_bound_kg,
    )

    if m_h2_max <= m_h2_min:
        raise ValueError("Maximum H2 inventory must be greater than minimum H2 inventory.")

    pressures_pa = np.arange(
        p_min_pa,
        p_max_pa,
        pressure_step_pa,
    )

    pressures_pa = np.append(pressures_pa, p_max_pa)
    pressures_pa = np.unique(pressures_pa)

    points: list[PressureLimitedFlowPoint] = []

    for p_current in pressures_pa:
        m_h2_current = hydrogen_inventory_from_total_pressure(
            target_pressure_pa=p_current,
            m_ch4_fixed_kg=m_ch4_fixed_kg,
            volume_m3=volume_m3,
            temperature_k=temperature_k,
            h2_upper_bound_kg=h2_upper_bound_kg,
        )

        soc = (m_h2_current - m_h2_min) / (m_h2_max - m_h2_min)

        p_after_withdrawal = max(
            # p_min_pa,
            0,
            p_current - pressure_change_limit_pa_per_day,
        )

        p_after_injection = min(
            # p_max_pa,
            1e12,
            p_current + pressure_change_limit_pa_per_day,
        )

        m_h2_after_withdrawal = hydrogen_inventory_from_total_pressure(
            target_pressure_pa=p_after_withdrawal,
            m_ch4_fixed_kg=m_ch4_fixed_kg,
            volume_m3=volume_m3,
            temperature_k=temperature_k,
            h2_upper_bound_kg=h2_upper_bound_kg,
        )

        m_h2_after_injection = hydrogen_inventory_from_total_pressure(
            target_pressure_pa=p_after_injection,
            m_ch4_fixed_kg=m_ch4_fixed_kg,
            volume_m3=volume_m3,
            temperature_k=temperature_k,
            h2_upper_bound_kg=h2_upper_bound_kg,
        )

        h2_inventory_current_kWh = m_h2_current * HYDROGEN_LHV_KWH_PER_KG
        h2_inventory_after_withdrawal_kWh = (
            m_h2_after_withdrawal * HYDROGEN_LHV_KWH_PER_KG
        )
        h2_inventory_after_injection_kWh = (
            m_h2_after_injection * HYDROGEN_LHV_KWH_PER_KG
        )

        soc_after_withdrawal_pressure_basis = (
            (m_h2_after_withdrawal - m_h2_min) / (m_h2_max - m_h2_min)
        )
        soc_after_injection_pressure_basis = (
            (m_h2_after_injection - m_h2_min) / (m_h2_max - m_h2_min)
        )

        withdrawal_kg_per_day = m_h2_current - m_h2_after_withdrawal
        injection_kg_per_day = m_h2_after_injection - m_h2_current

        withdrawal_kg_s = withdrawal_kg_per_day / SECONDS_PER_DAY
        injection_kg_s = injection_kg_per_day / SECONDS_PER_DAY

        withdrawal_kwh_per_day = withdrawal_kg_per_day * HYDROGEN_LHV_KWH_PER_KG
        injection_kwh_per_day = injection_kg_per_day * HYDROGEN_LHV_KWH_PER_KG

        withdrawal_kw_lhv = withdrawal_kwh_per_day / 24
        injection_kw_lhv = injection_kwh_per_day / 24

        points.append(
            PressureLimitedFlowPoint(
                pressure_pa=p_current,
                pressure_bar=p_current / BAR_TO_PA,
                soc=soc,
                h2_inventory_kWh=h2_inventory_current_kWh,
                h2_inventory_kg=m_h2_current,
                h2_inventory_after_withdrawal_kWh=h2_inventory_after_withdrawal_kWh,
                h2_inventory_after_injection_kWh=h2_inventory_after_injection_kWh,
                soc_after_withdrawal_pressure_basis=soc_after_withdrawal_pressure_basis,
                soc_after_injection_pressure_basis=soc_after_injection_pressure_basis,
                withdrawal_kg_per_day=withdrawal_kg_per_day,
                injection_kg_per_day=injection_kg_per_day,
                withdrawal_kg_s=withdrawal_kg_s,
                injection_kg_s=injection_kg_s,
                withdrawal_kwh_per_day=withdrawal_kwh_per_day,
                injection_kwh_per_day=injection_kwh_per_day,
                withdrawal_kw_lhv=withdrawal_kw_lhv,
                injection_kw_lhv=injection_kw_lhv,
            )
        )

    return pd.DataFrame([point.__dict__ for point in points])


def calculate_pressure_limited_flow_curves_for_projects(
    projects: Mapping[str, StorageProject],
    *,
    pressure_step_pa: float = 1 * BAR_TO_PA,
    pressure_change_limit_pa_per_day: float = 10 * BAR_TO_PA,
    print_cem_constraints: bool = True,
    conservative_cem_fit: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Calculate pressure-limited flow curves for several StorageProject objects.

    If print_cem_constraints=True, also fit and print linear CEM-ready
    24-hour pressure-ramp constraints using:

        SoC_t     = stored hydrogen energy [kWh]
        SoC_max   = installed storage capacity [kWh]
    """
    if not projects:
        raise ValueError("No projects provided.")

    results_by_project: dict[str, pd.DataFrame] = {}

    for label, project in projects.items():
        df = calculate_pressure_limited_flow_curve(
            project=project,
            pressure_step_pa=pressure_step_pa,
            pressure_change_limit_pa_per_day=pressure_change_limit_pa_per_day,
        )

        df = add_capacity_normalised_columns(
            df=df,
            working_gas_capacity_kwh_lhv=project.inventory.recoverable_working_gas_capacity_kwh_lhv,
        )

        results_by_project[label] = df

    if print_cem_constraints:
        print_cem_pressure_constraint(
            results_by_project=results_by_project,
            direction="withdrawal",
            y="withdrawal_kw_per_kwh",
            conservative=conservative_cem_fit,
        )

        print_cem_pressure_constraint(
            results_by_project=results_by_project,
            direction="injection",
            y="injection_kw_per_kwh",
            conservative=conservative_cem_fit,
        )

        print_cem_soc_endpoint_constraints(
            results_by_project=results_by_project,
            conservative=conservative_cem_fit,
        )

    return results_by_project
def plot_curves(
    results: pd.DataFrame | Mapping[str, pd.DataFrame],
    x: str = "soc",
    y: str = "withdrawal_kw_lhv",
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    y_scale: float = 1.0,
    x_scale: float = 1.0,
    projects: Mapping[str, StorageProject] | None = None,
    show_design_capacity: bool = False,
    horizontal_lines: Mapping[str, float] | None = None,
    line_colors: Mapping[str, str] | Sequence[str] | None = None,
    horizontal_line_colors: Mapping[str, str] | None = None,
    design_capacity_colors: Mapping[str, str] | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Flexibly plot any dataframe column against any other column.

    If show_design_capacity=True and projects are provided, design capacity
    reference lines are added for withdrawal_kw_lhv or injection_kw_lhv.
    """
    if isinstance(results, pd.DataFrame):
        labelled_results = {"case": results}
    else:
        labelled_results = dict(results)

    if not labelled_results:
        raise ValueError("No results provided for plotting.")

    fig, ax = plt.subplots(figsize=(8, 5))

    if isinstance(line_colors, Mapping):
        colour_lookup = dict(line_colors)
    elif line_colors is not None:
        colour_lookup = {
            label: colour
            for label, colour in zip(labelled_results.keys(), line_colors)
        }
    else:
        colour_lookup = {}

    for label, df in labelled_results.items():
        if x not in df.columns:
            raise ValueError(
                f"Column {x!r} not found in dataframe for {label!r}. "
                f"Available columns: {list(df.columns)}"
            )

        if y not in df.columns:
            raise ValueError(
                f"Column {y!r} not found in dataframe for {label!r}. "
                f"Available columns: {list(df.columns)}"
            )

        ax.plot(
            df[x] * x_scale,
            df[y] * y_scale,
            label=label,
            color=colour_lookup.get(label),
        )

    # User-specified horizontal lines
    if horizontal_lines is not None:
        for label, value in horizontal_lines.items():
            ax.axhline(
                value * y_scale,
                linestyle="--",
                label=label,
                color=(horizontal_line_colors or {}).get(label),
            )

    # Automatically inferred design-capacity lines
    if show_design_capacity:
        if projects is None:
            raise ValueError(
                "projects must be provided when show_design_capacity=True."
            )

        for label, project in projects.items():
            if y == "withdrawal_kw_lhv":
                design_capacity_kw = project.flows.withdrawal_flow_kw_h2_lhv
                design_label = f"{label} design withdrawal"

            elif y == "injection_kw_lhv":
                design_capacity_kw = project.flows.injection_flow_kw_h2_lhv
                design_label = f"{label} design injection"

            else:
                continue

            # ax.axhline(
            #     design_capacity_kw * y_scale,
            #     linestyle="--",
            #     label=design_label,
            #     color=(design_capacity_colors or {}).get(label),
            # )

    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)

    if title is not None:
        ax.set_title(title)

    if (
        len(labelled_results) > 1
        or horizontal_lines is not None
        or show_design_capacity
    ):
        ax.legend()

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_soc_change_bounds(
    results: pd.DataFrame | Mapping[str, pd.DataFrame],
    *,
    x: str = "soc",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    show_linear_fit: bool = True,
    conservative_fit: bool = True,
    line_colors: Mapping[str, str] | Sequence[str] | None = None,
    min_color: str | None = None,
    max_color: str | None = None,
    min_fit_color: str | None = None,
    max_fit_color: str | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plot calculated minimum and maximum signed 24-hour SoC changes.

    The minimum change is negative and represents the withdrawal-side limit. The
    maximum change is positive and represents the injection-side limit. If
    show_linear_fit=True, dashed lines show linearised constraints fitted across
    all supplied projects.
    """
    if isinstance(results, pd.DataFrame):
        labelled_results = {"case": results}
    else:
        labelled_results = dict(results)

    if not labelled_results:
        raise ValueError("No results provided for plotting.")

    required_columns = {x, "soc_change_min_24h", "soc_change_max_24h"}
    for label, df in labelled_results.items():
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing columns for {label!r}: {sorted(missing)}. "
                f"Available columns: {list(df.columns)}"
            )

    if isinstance(line_colors, Mapping):
        colour_lookup = dict(line_colors)
    elif line_colors is not None:
        colour_lookup = {
            label: colour
            for label, colour in zip(labelled_results.keys(), line_colors)
        }
    else:
        colour_lookup = {}

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, df in labelled_results.items():
        project_color = colour_lookup.get(label)
        lower_color = min_color or project_color
        upper_color = max_color or project_color

        ax.plot(
            df[x],
            df["soc_change_min_24h"],
            label=f"{label} minimum 24h SoC change",
            color=lower_color,
        )
        ax.plot(
            df[x],
            df["soc_change_max_24h"],
            label=f"{label} maximum 24h SoC change",
            color=upper_color,
        )

    if show_linear_fit:
        x_fit = np.linspace(0.0, 1.0, 200)
        lower_intercept, lower_slope = fit_linear_soc_endpoint_bound(
            results_by_project=labelled_results,
            y="soc_change_min_24h",
            x=x,
            bound="lower",
            conservative=conservative_fit,
        )
        upper_intercept, upper_slope = fit_linear_soc_endpoint_bound(
            results_by_project=labelled_results,
            y="soc_change_max_24h",
            x=x,
            bound="upper",
            conservative=conservative_fit,
        )

        ax.plot(
            x_fit,
            lower_intercept + lower_slope * x_fit,
            linestyle="--",
            color=min_fit_color or min_color,
            label="Linearised minimum 24h SoC change",
        )
        ax.plot(
            x_fit,
            upper_intercept + upper_slope * x_fit,
            linestyle="--",
            color=max_fit_color or max_color,
            label="Linearised maximum 24h SoC change",
        )

    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or "Signed 24-hour change in state of charge [-]")

    if title is not None:
        ax.set_title(title)

    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

def save_results(
    results_by_project: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
) -> pd.DataFrame:
    """
    Save each project dataframe and a combined dataframe.

    Returns the combined dataframe.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = pd.concat(
        [
            df.assign(project=label)
            for label, df in results_by_project.items()
        ],
        ignore_index=True,
    )

    for label, df in results_by_project.items():
        safe_label = label.lower().replace(" ", "_").replace("/", "_")
        df.to_csv(
            output_dir / f"pressure_limited_flow_{safe_label}.csv",
            index=False,
        )

    combined.to_csv(
        output_dir / "pressure_limited_flow_all_projects.csv",
        index=False,
    )

    return combined

def add_capacity_normalised_columns(
    df: pd.DataFrame,
    working_gas_capacity_kwh_lhv: float,
) -> pd.DataFrame:
    """
    Add capacity-normalised pressure-limit columns.

    Power-per-capacity is reported as GW/TWh, which is equivalent to:
        (kW / 1e6) / (kWh / 1e9)
    """
    df = df.copy()

    working_gas_capacity_gwh = working_gas_capacity_kwh_lhv / 1e6
    working_gas_capacity_twh = working_gas_capacity_kwh_lhv / 1e9

    df["working_gas_capacity_kwh_lhv"] = working_gas_capacity_kwh_lhv
    df["working_gas_capacity_gwh_lhv"] = working_gas_capacity_gwh
    df["working_gas_capacity_twh_lhv"] = working_gas_capacity_twh

    df["withdrawal_gw_lhv"] = df["withdrawal_kw_lhv"] / 1e6
    df["injection_gw_lhv"] = df["injection_kw_lhv"] / 1e6

    df["withdrawal_gw_per_twh"] = (
        df["withdrawal_gw_lhv"] / working_gas_capacity_twh
    )

    df["injection_gw_per_twh"] = (
        df["injection_gw_lhv"] / working_gas_capacity_twh
    )

    df["withdrawal_gwh_per_day"] = df["withdrawal_kwh_per_day"] / 1e6
    df["injection_gwh_per_day"] = df["injection_kwh_per_day"] / 1e6

    df["withdrawal_gwh_per_day_per_twh"] = (
        df["withdrawal_gwh_per_day"] / working_gas_capacity_twh
    )

    df["injection_gwh_per_day_per_twh"] = (
        df["injection_gwh_per_day"] / working_gas_capacity_twh
    )

    df["withdrawal_kw_per_kwh"] = (
        df["withdrawal_kw_lhv"] / working_gas_capacity_kwh_lhv
    )

    df["injection_kw_per_kwh"] = (
        df["injection_kw_lhv"] / working_gas_capacity_kwh_lhv
    )

    # Signed 24-hour change in fractional state of charge.
    # Negative values correspond to withdrawal; positive values correspond to injection.
    df["soc_change_min_24h"] = -(
        df["withdrawal_kwh_per_day"] / working_gas_capacity_kwh_lhv
    )
    df["soc_change_max_24h"] = (
        df["injection_kwh_per_day"] / working_gas_capacity_kwh_lhv
    )

    # Minimum and maximum admissible future state of charge after 24 hours.
    df["soc_t_plus_24_min"] = df["soc"] + df["soc_change_min_24h"]
    df["soc_t_plus_24_max"] = df["soc"] + df["soc_change_max_24h"]

    return df


def fit_linear_soc_endpoint_bound(
    results_by_project: Mapping[str, pd.DataFrame],
    *,
    y: str,
    x: str = "soc",
    bound: str,
    soc_min: float = 0.0,
    soc_max: float = 1.0,
    conservative: bool = True,
) -> tuple[float, float]:
    """
    Fit a linear future-SoC endpoint bound.

    Fits:
        y = intercept + slope * x

    Usually:
        x = current fractional SoC [-]
        y = admissible future fractional SoC 24 hours later [-]

    If conservative=True:
        - for a lower bound, the intercept is shifted upward so that the fitted
          line is never below the calculated lower endpoint;
        - for an upper bound, the intercept is shifted downward so that the fitted
          line is never above the calculated upper endpoint.
    """
    if bound not in {"lower", "upper"}:
        raise ValueError("bound must be either 'lower' or 'upper'.")

    combined = pd.concat(
        [
            df.assign(project=label)
            for label, df in results_by_project.items()
        ],
        ignore_index=True,
    )

    fit_df = combined[
        (combined[x] >= soc_min)
        & (combined[x] <= soc_max)
    ].copy()

    if fit_df.empty:
        raise ValueError(
            "No data available for fitting. Check soc_min, soc_max, and input data."
        )

    if y not in fit_df.columns:
        raise ValueError(
            f"Column {y!r} not found. Available columns: {list(fit_df.columns)}"
        )

    slope, intercept = np.polyfit(
        fit_df[x].to_numpy(),
        fit_df[y].to_numpy(),
        deg=1,
    )

    if conservative:
        y_pred = intercept + slope * fit_df[x].to_numpy()
        y_true = fit_df[y].to_numpy()

        if bound == "lower":
            max_underestimate = np.max(y_true - y_pred)
            if max_underestimate > 0:
                intercept += max_underestimate
        else:
            max_overestimate = np.max(y_pred - y_true)
            if max_overestimate > 0:
                intercept -= max_overestimate

    return intercept, slope


def print_cem_soc_endpoint_constraints(
    results_by_project: Mapping[str, pd.DataFrame],
    *,
    soc_min: float = 0.0,
    soc_max: float = 1.0,
    conservative: bool = True,
) -> None:
    """
    Print CEM-ready 24-hour endpoint SoC constraints.

    These are equivalent to the withdrawal and injection change constraints, but
    are written directly as lower and upper bounds on the state of charge 24
    hours later:

        SoC_t_plus_24 >= a_min * SoC_max + b_min * SoC_t
        SoC_t_plus_24 <= a_max * SoC_max + b_max * SoC_t

    where SoC_t and SoC_max are both expressed in energy units [kWh].
    """
    lower_intercept, lower_slope = fit_linear_soc_endpoint_bound(
        results_by_project=results_by_project,
        y="soc_t_plus_24_min",
        x="soc",
        bound="lower",
        soc_min=soc_min,
        soc_max=soc_max,
        conservative=conservative,
    )

    upper_intercept, upper_slope = fit_linear_soc_endpoint_bound(
        results_by_project=results_by_project,
        y="soc_t_plus_24_max",
        x="soc",
        bound="upper",
        soc_min=soc_min,
        soc_max=soc_max,
        conservative=conservative,
    )

    print()
    print("=" * 88)
    print("CEM pressure-ramp constraint: future SoC endpoint bounds")
    print("=" * 88)
    print()
    print("Fitted future-SoC endpoint curves:")
    print(
        "    soc_t_plus_24_min = "
        f"{lower_intercept:.12f} + ({lower_slope:.12f}) * fractional_soc_t"
    )
    print(
        "    soc_t_plus_24_max = "
        f"{upper_intercept:.12f} + ({upper_slope:.12f}) * fractional_soc_t"
    )
    print()
    print("Using:")
    print("    fractional_soc_t = SoC_t / SoC_max")
    print("    SoC_t             = stored hydrogen energy at time t [kWh]")
    print("    SoC_t_plus_24     = stored hydrogen energy 24 hours later [kWh]")
    print("    SoC_max           = installed storage capacity [kWh]")
    print()
    print("CEM constraints:")
    print(
        "    SoC_t_plus_24 >= "
        f"{lower_intercept:.12f} * SoC_max "
        f"+ ({lower_slope:.12f}) * SoC_t"
    )
    print(
        "    SoC_t_plus_24 <= "
        f"{upper_intercept:.12f} * SoC_max "
        f"+ ({upper_slope:.12f}) * SoC_t"
    )
    print()
    print("Units:")
    print("    SoC_t_plus_24: kWh")
    print("    SoC_t:         kWh")
    print("    SoC_max:       kWh")
    print("    coefficients:  kWh per kWh over 24 hours")
    print()
    print("=" * 88)
    print()

def plot_soc_endpoint_bounds(
    results: pd.DataFrame | Mapping[str, pd.DataFrame],
    *,
    x: str = "soc",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    show_linear_fit: bool = False,
    conservative_fit: bool = True,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plot calculated minimum and maximum admissible future SoC endpoints.

    This version is formatted for the main-text figure:
    - plasma-based colours
    - no linearised fit by default
    - no title by default
    - faint grey y=x reference line
    - LaTeX-friendly axis labels
    """

    if isinstance(results, pd.DataFrame):
        labelled_results = {"case": results}
    else:
        labelled_results = dict(results)

    if not labelled_results:
        raise ValueError("No results provided for plotting.")

    required_columns = {x, "soc_t_plus_24_min", "soc_t_plus_24_max"}
    for label, df in labelled_results.items():
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing columns for {label!r}: {sorted(missing)}. "
                f"Available columns: {list(df.columns)}"
            )

    # Plasma-based colours
    cmap = plt.cm.plasma
    lower_color = cmap(0.20)
    upper_color = cmap(0.80)

    fig, ax = plt.subplots(figsize=(5, 5))

    for label, df in labelled_results.items():
        x_vals = df[x].to_numpy()
        y_min = df["soc_t_plus_24_min"].to_numpy()
        y_max = df["soc_t_plus_24_max"].to_numpy()

        # Faint feasible band between lower and upper bounds
        ax.fill_between(
            x_vals,
            y_min,
            y_max,
            color=cmap(0.55),
            alpha=0.12,
            linewidth=0,
        )

        ax.plot(
            x_vals,
            y_min,
            color=lower_color,
            linewidth=2.0,
            label=r"Lower bound",
        )
        ax.plot(
            x_vals,
            y_max,
            color=upper_color,
            linewidth=2.0,
            label=r"Upper bound",
        )

    # Optional linear fit, off by default
    if show_linear_fit:
        x_fit = np.linspace(0.0, 1.0, 200)
        lower_intercept, lower_slope = fit_linear_soc_endpoint_bound(
            results_by_project=labelled_results,
            y="soc_t_plus_24_min",
            x=x,
            bound="lower",
            conservative=conservative_fit,
        )
        upper_intercept, upper_slope = fit_linear_soc_endpoint_bound(
            results_by_project=labelled_results,
            y="soc_t_plus_24_max",
            x=x,
            bound="upper",
            conservative=conservative_fit,
        )

        ax.plot(
            x_fit,
            lower_intercept + lower_slope * x_fit,
            linestyle="--",
            color=lower_color,
            alpha=0.8,
            label=r"Linearised lower bound",
        )
        ax.plot(
            x_fit,
            upper_intercept + upper_slope * x_fit,
            linestyle="--",
            color=upper_color,
            alpha=0.8,
            label=r"Linearised upper bound",
        )

    # Faint constant-inventory line: y = x
    x_ref = np.linspace(0.0, 1.0, 200)
    ax.plot(
        x_ref,
        x_ref,
        linestyle="--",
        color="0.65",
        linewidth=1.0,
        alpha=0.8,
        label=r"$E_{s,t+24h} = E_{s,t}$",
        zorder=0,
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    ax.set_xlabel(
        xlabel or r"$E_{s,t} / \bar{E}_{s}\;[-]$"
    )
    ax.set_ylabel(
        ylabel or r"$E_{s,t+24h} / \bar{E}_{s}\;[-]$"
    )

    if title is not None:
        ax.set_title(title)

    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

def fit_capacity_normalised_pressure_limit(
    results_by_project: Mapping[str, pd.DataFrame],
    *,
    y: str,
    x: str = "soc",
    soc_min: float = 0.0,
    soc_max: float = 1.0,
    conservative: bool = True,
) -> tuple[float, float]:
    """
    Fit a linear capacity-normalised pressure-limit curve.

    Fits:
        y = intercept + slope * x

    Usually:
        x = fractional SoC [-]
        y = pressure-limited capacity [kW/kWh]

    If conservative=True, the intercept is shifted downward so that the fitted
    line never exceeds the calculated pressure-limit curve over the fitted range.
    """
    combined = pd.concat(
        [
            df.assign(project=label)
            for label, df in results_by_project.items()
        ],
        ignore_index=True,
    )

    fit_df = combined[
        (combined[x] >= soc_min)
        & (combined[x] <= soc_max)
    ].copy()

    if fit_df.empty:
        raise ValueError(
            "No data available for fitting. Check soc_min, soc_max, and input data."
        )

    if y not in fit_df.columns:
        raise ValueError(
            f"Column {y!r} not found. Available columns: {list(fit_df.columns)}"
        )

    slope, intercept = np.polyfit(
        fit_df[x].to_numpy(),
        fit_df[y].to_numpy(),
        deg=1,
    )

    if conservative:
        y_pred = intercept + slope * fit_df[x].to_numpy()
        y_true = fit_df[y].to_numpy()

        max_overestimate = np.max(y_pred - y_true)

        if max_overestimate > 0:
            intercept -= max_overestimate

    return intercept, slope


def print_cem_pressure_constraint(
    results_by_project: Mapping[str, pd.DataFrame],
    *,
    direction: str,
    y: str,
    soc_min: float = 0.0,
    soc_max: float = 1.0,
    conservative: bool = True,
) -> None:
    """
    Print CEM-ready 24-hour pressure-ramp constraint coefficients.

    Here we use the user's preferred notation:

        SoC_t     = stored hydrogen energy at time t [kWh]
        SoC_max   = installed storage energy capacity [kWh]

    The fitted pressure-limit curve is:

        capacity_limit_kw_per_kwh = a + b * fractional_soc

    where:

        fractional_soc = SoC_t / SoC_max

    The 24-hour energy-change limit is:

        delta_SoC_24h_kwh <= 24 * (a + b * SoC_t / SoC_max) * SoC_max

    which simplifies to:

        delta_SoC_24h_kwh <= 24a * SoC_max + 24b * SoC_t

    This is linear.
    """
    if direction not in {"withdrawal", "injection"}:
        raise ValueError("direction must be either 'withdrawal' or 'injection'.")

    intercept, slope = fit_capacity_normalised_pressure_limit(
        results_by_project=results_by_project,
        y=y,
        x="soc",
        soc_min=soc_min,
        soc_max=soc_max,
        conservative=conservative,
    )

    coefficient_soc_max = 24 * intercept
    coefficient_soc_t = 24 * slope

    print()
    print("=" * 88)
    print(f"CEM pressure-ramp constraint: {direction}")
    print("=" * 88)
    print()
    print("Fitted capacity-normalised pressure-limit curve:")
    print(f"    {y} = {intercept:.12f} + ({slope:.12f}) * fractional_soc_t")
    print()
    print("Using:")
    print("    fractional_soc_t = SoC_t / SoC_max")
    print("    SoC_t             = stored hydrogen energy at time t [kWh]")
    print("    SoC_t_plus_24     = stored hydrogen energy 24 hours later [kWh]")
    print("    SoC_max           = installed storage capacity [kWh]")
    print()

    if direction == "withdrawal":
        print("CEM constraint:")
        print(
            "    SoC_t - SoC_t_plus_24 <= "
            f"{coefficient_soc_max:.12f} * SoC_max "
            f"+ ({coefficient_soc_t:.12f}) * SoC_t"
        )
    else:
        print("CEM constraint:")
        print(
            "    SoC_t_plus_24 - SoC_t <= "
            f"{coefficient_soc_max:.12f} * SoC_max "
            f"+ ({coefficient_soc_t:.12f}) * SoC_t"
        )

    print()
    print("Units:")
    print("    left-hand side: kWh over 24 hours")
    print("    SoC_t:          kWh")
    print("    SoC_max:        kWh")
    print("    coefficients:   kWh per kWh over 24 hours")
    print()
    print("=" * 88)
    print()

def summarise_soc_endpoint_coefficients(
    results_by_project: Mapping[str, pd.DataFrame],
    *,
    # case_metadata: Mapping[str, Mapping[str, float | str]] | None = None,
    x: str = "soc",
    conservative: bool = True,
    output_csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Fit CEM-ready 24-hour endpoint SoC constraints for each case separately.

    The fitted constraints are:

        E_{s,t+Delta t24} >= a_min * Ebar_s + b_min * E_{s,t}
        E_{s,t+Delta t24} <= a_max * Ebar_s + b_max * E_{s,t}

    The returned dataframe also reports the implied admissible 24-hour
    withdrawal and injection at e_{s,t}=0.5.
    """

    if not results_by_project:
        raise ValueError("No results provided.")

    rows: list[dict[str, float | str]] = []

    for label, df in results_by_project.items():
        one_case = {label: df}

        a_min, b_min = fit_linear_soc_endpoint_bound(
            results_by_project=one_case,
            y="soc_t_plus_24_min",
            x=x,
            bound="lower",
            conservative=conservative,
        )

        a_max, b_max = fit_linear_soc_endpoint_bound(
            results_by_project=one_case,
            y="soc_t_plus_24_max",
            x=x,
            bound="upper",
            conservative=conservative,
        )

        e_mid = 0.5
        e_min_mid = a_min + b_min * e_mid
        e_max_mid = a_max + b_max * e_mid

        # Positive values for readability:
        # withdrawal = how far below current SoC the lower bound is
        # injection = how far above current SoC the upper bound is
        withdrawal_at_mid = e_mid - e_min_mid
        injection_at_mid = e_max_mid - e_mid



        row = {
            "case": label,
            "a_min": a_min,
            "b_min": b_min,
            "a_max": a_max,
            "b_max": b_max,
            "withdrawal_at_e_0_5": withdrawal_at_mid,
            "injection_at_e_0_5": injection_at_mid,
        }

        rows.append(row)

    summary = pd.DataFrame(rows)

    if output_csv_path is not None:
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_csv_path, index=False)

    return summary

def run_pressure_ramp_limit_sensitivity(
    *,
    default_project: StorageProject,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Run pressure-change-limit sensitivity for the default salt cavern design only.

    The cavern design is fixed at the default pressure range, while the allowable
    daily pressure-change limit is varied.
    """

    ramp_limit_cases = {
        "Low ramp limit: 5 bar/day": 5e5,
        "Default: 10 bar/day": 10e5,
        "High ramp limit: 15 bar/day": 15e5,
    }

    ramp_results: dict[str, pd.DataFrame] = {}
    ramp_projects: dict[str, StorageProject] = {}

    for case_label, pressure_change_limit_pa_per_day in ramp_limit_cases.items():
        case_results = calculate_pressure_limited_flow_curves_for_projects(
            projects={case_label: default_project},
            pressure_step_pa=1e5,
            pressure_change_limit_pa_per_day=pressure_change_limit_pa_per_day,
            print_cem_constraints=False,
        )

        ramp_results[case_label] = case_results[case_label]
        ramp_projects[case_label] = default_project

    ramp_output_dir = output_dir / "ramp_limit_sensitivity"
    ramp_output_dir.mkdir(parents=True, exist_ok=True)

    save_results(
        results_by_project=ramp_results,
        output_dir=ramp_output_dir,
    )

    ramp_colours = {
        "Low ramp limit: 5 bar/day": (13 / 255, 8 / 255, 135 / 255),
        "Default ramp limit: 10 bar/day": (204 / 255, 71 / 255, 120 / 255),
        "High ramp limit: 15 bar/day": (240 / 255, 249 / 255, 33 / 255),
    }

    plot_curves(
        results=ramp_results,
        projects=ramp_projects,
        show_design_capacity=True,
        x="soc",
        y="withdrawal_kw_lhv",
        y_scale=1 / 1e6,
        xlabel="State of charge [-]",
        ylabel="Average 24-hour withdrawal capacity [GW LHV]",
        title="Pressure-change-limit sensitivity",
        line_colors=ramp_colours,
        output_path=ramp_output_dir / "withdrawal_capacity_by_soc_ramp_limit.png",
    )

    ramp_limit_coefficients = summarise_soc_endpoint_coefficients(
        results_by_project=ramp_results,
        conservative=True,
        output_csv_path=ramp_output_dir / "ramp_limit_endpoint_coefficients.csv",
    )

    print("\nPressure-change-limit sensitivity coefficients")
    print(
        ramp_limit_coefficients.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    return ramp_limit_coefficients