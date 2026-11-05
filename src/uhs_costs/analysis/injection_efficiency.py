from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Any
from matplotlib.lines import Line2D


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uhs_costs.constants import (
    HYDROGEN_LHV_KWH_PER_KG,
    HYDROGEN_STANDARD_DENSITY_KG_PER_M3,
)
from uhs_costs.design.helpers.project import StorageProject

from uhs_costs.analysis.pressure_constraint import (
    BAR_TO_PA,
    hydrogen_inventory_from_total_pressure,
    methane_mass_from_standard_volume,
)

from uhs_costs.design.helpers.compression_model import (
    CompressionInput,
    CompressionMethod,
    calculate_compression,
)


@dataclass(frozen=True)
class InjectionEfficiencySummary:
    project: str
    min_efficiency: float
    mean_efficiency: float
    efficiency_at_soc_50: float
    max_compression_intensity_kwh_e_per_kwh_h2: float
    mean_compression_intensity_kwh_e_per_kwh_h2: float
    compression_intensity_at_soc_50_kwh_e_per_kwh_h2: float


def _get_first_attr(obj: Any, names: tuple[str, ...]) -> Any:
    """
    Return the first existing non-None attribute from an object.

    This is useful because the exact location of some project parameters may
    differ slightly across project classes or refactors.
    """
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value

    raise AttributeError(
        f"None of the attributes {names} were found on {type(obj).__name__}."
    )


def h2_lhv_power_kw_to_mass_flow_kg_s(
    h2_lhv_power_kw: float,
    hydrogen_lhv_kwh_per_kg: float = HYDROGEN_LHV_KWH_PER_KG,
) -> float:
    """
    Convert hydrogen LHV power [kW] to mass flow [kg/s].

    kW = kWh/h
    kg/h = kWh/h / kWh/kg
    kg/s = kg/h / 3600
    """
    if h2_lhv_power_kw <= 0:
        raise ValueError("h2_lhv_power_kw must be greater than zero.")

    if hydrogen_lhv_kwh_per_kg <= 0:
        raise ValueError("hydrogen_lhv_kwh_per_kg must be greater than zero.")

    return h2_lhv_power_kw / hydrogen_lhv_kwh_per_kg / 3600


def compression_intensity_to_injection_efficiency(
    compression_intensity_kwh_e_per_kwh_h2: float,
) -> float:
    """
    Convert compression electricity intensity to process-equivalent injection efficiency.

        eta_inj = 1 / (1 + lambda_comp)

    where lambda_comp is kWh_e per kWh_H2,LHV.
    """
    if compression_intensity_kwh_e_per_kwh_h2 < 0:
        raise ValueError("compression_intensity_kwh_e_per_kwh_h2 cannot be negative.")

    return 1 / (1 + compression_intensity_kwh_e_per_kwh_h2)


def pressure_at_soc_profile(
    project: StorageProject,
    *,
    n_points: int = 101,
) -> pd.DataFrame:
    """
    Generate a pressure and hydrogen inventory profile across fractional SoC.

    This uses the same conceptual pressure/inventory relationship as the
    pressure-constraint analysis:

    - fixed methane inventory
    - variable hydrogen inventory
    - fixed effective storage volume
    - fixed storage temperature
    """
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")

    p_min_pa = project.pressures.minimum_operating_pressure_pa
    p_max_pa = project.pressures.maximum_operating_pressure_pa
    temperature_k = project.wells.well_temperature_k
    volume_m3 = project.inventory.required_storage_volume_m3

    if p_max_pa <= p_min_pa:
        raise ValueError("Maximum operating pressure must exceed minimum operating pressure.")

    m_ch4_fixed_kg = methane_mass_from_standard_volume(
        project.inventory.abandonment_gas_methane_volume_sm3
    )

    # Use an intentionally generous upper bound. The pressure solver expands
    # this if required, but this keeps the first guess sensible.
    h2_upper_bound_kg = (
        project.inventory.physical_working_gas_capacity_kwh_lhv
        / HYDROGEN_LHV_KWH_PER_KG
        * 2
    )

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
        raise ValueError("Maximum H2 inventory must exceed minimum H2 inventory.")

    soc_values = np.linspace(0.0, 1.0, n_points)
    pressure_values_pa = np.linspace(p_min_pa, p_max_pa, n_points)

    rows: list[dict[str, float]] = []

    for pressure_pa in pressure_values_pa:
        m_h2_current = hydrogen_inventory_from_total_pressure(
            target_pressure_pa=pressure_pa,
            m_ch4_fixed_kg=m_ch4_fixed_kg,
            volume_m3=volume_m3,
            temperature_k=temperature_k,
            h2_upper_bound_kg=h2_upper_bound_kg,
        )

        soc = (m_h2_current - m_h2_min) / (m_h2_max - m_h2_min)

        rows.append(
            {
                "soc": soc,
                "pressure_pa": pressure_pa,
                "pressure_bar": pressure_pa / BAR_TO_PA,
                "h2_inventory_kg": m_h2_current,
                "h2_inventory_kwh_lhv": m_h2_current * HYDROGEN_LHV_KWH_PER_KG,
            }
        )

    df = pd.DataFrame(rows)

    # The pressure spacing is linear, but the SoC spacing may not be exactly
    # linear because inventory-pressure relationships are not necessarily linear.
    # Sorting makes interpolation and integration safer.
    df = df.sort_values("soc").reset_index(drop=True)

    return df


def calculate_injection_efficiency_curve(
    project: StorageProject,
    *,
    n_points: int = 101,
    compression_method: CompressionMethod | str | None = None,
    inlet_pressure_pa: float | None = None,
    inlet_temperature_k: float | None = None,
    injection_flow_kw_h2_lhv: float | None = None,
    standard_density_kg_per_m3: float = HYDROGEN_STANDARD_DENSITY_KG_PER_M3,
) -> pd.DataFrame:
    """
    Calculate compression electricity intensity and injection efficiency across SoC.

    The curve varies outlet pressure with storage pressure. The compressor inlet
    pressure is normally the pipeline pressure. The mass flow is kept constant
    at the design injection flow to isolate the pressure effect.

    Notes
    -----
    For the HGSM polytropic method, electricity intensity should be independent
    of mass flow because both electric power and H2 throughput scale linearly.

    For the HyStories TICBP method, the equation is also flow-linear, so the
    ratio should likewise be mostly independent of the selected mass flow.
    """

    pressure_df = pressure_at_soc_profile(
        project=project,
        n_points=n_points,
    )

    if inlet_pressure_pa is None:
        inlet_pressure_pa = _get_first_attr(
            project.pressures,
            (
                "pipeline_pressure_pa",
                "compressor_inlet_pressure_pa",
                "inlet_pressure_pa",
            ),
        )

    if inlet_temperature_k is None:
        inlet_temperature_k = project.wells.well_temperature_k

    if injection_flow_kw_h2_lhv is None:
        injection_flow_kw_h2_lhv = project.flows.injection_flow_kw_h2_lhv

    mass_flow_kg_s = h2_lhv_power_kw_to_mass_flow_kg_s(
        injection_flow_kw_h2_lhv
    )

    

    if compression_method is None:
        # If the project already has a compression result/config, use its method.
        # Otherwise, fall back to HGSM.
        if hasattr(project, "compression") and hasattr(project.compression, "method"):
            compression_method = project.compression.method
        else:
            compression_method = CompressionMethod.HGSM_POLYTROPIC

    p_max_pa = project.pressures.maximum_operating_pressure_pa

    max_pressure_result = calculate_compression(
        CompressionInput(
            inlet_pressure_pa=inlet_pressure_pa,
            outlet_pressure_pa=p_max_pa,
            inlet_temperature_k=inlet_temperature_k,
            mass_flow_kg_s=mass_flow_kg_s,
            method=compression_method,
        ),
        standard_density_kg_per_m3=standard_density_kg_per_m3,
    )

    fixed_number_of_stages = max_pressure_result.number_of_stages

    rows: list[dict[str, float | str | None]] = []

    for row in pressure_df.itertuples(index=False):
        outlet_pressure_pa = float(row.pressure_pa)

        if outlet_pressure_pa <= inlet_pressure_pa:
            total_electric_power_kw = 0.0
            h2_lhv_flow_kw = injection_flow_kw_h2_lhv
            compression_intensity = 0.0
            injection_efficiency = 1.0
            overall_pressure_ratio = 1.0
            number_of_stages = 0

        else:
            result = calculate_compression(
                CompressionInput(
                    inlet_pressure_pa=inlet_pressure_pa,
                    outlet_pressure_pa=outlet_pressure_pa,
                    inlet_temperature_k=inlet_temperature_k,
                    mass_flow_kg_s=mass_flow_kg_s,
                    method=compression_method,
                    number_of_stages=fixed_number_of_stages,
                ),
                standard_density_kg_per_m3=standard_density_kg_per_m3,
            )

            total_electric_power_kw = result.total_electric_power_kw
            h2_lhv_flow_kw = result.h2_lhv_flow_kw

            if total_electric_power_kw is None or h2_lhv_flow_kw is None:
                raise ValueError(
                    "Compression result must contain total_electric_power_kw "
                    "and h2_lhv_flow_kw."
                )

            compression_intensity = total_electric_power_kw / h2_lhv_flow_kw
            injection_efficiency = compression_intensity_to_injection_efficiency(
                compression_intensity
            )

            overall_pressure_ratio = result.overall_pressure_ratio
            number_of_stages = result.number_of_stages

        rows.append(
            {
                "soc": float(row.soc),
                "pressure_pa": outlet_pressure_pa,
                "pressure_bar": float(row.pressure_bar),
                "h2_inventory_kg": float(row.h2_inventory_kg),
                "h2_inventory_kwh_lhv": float(row.h2_inventory_kwh_lhv),
                "inlet_pressure_pa": inlet_pressure_pa,
                "inlet_pressure_bar": inlet_pressure_pa / BAR_TO_PA,
                "outlet_pressure_pa": outlet_pressure_pa,
                "outlet_pressure_bar": outlet_pressure_pa / BAR_TO_PA,
                "overall_pressure_ratio": overall_pressure_ratio,
                "number_of_stages": number_of_stages,
                "mass_flow_kg_s": mass_flow_kg_s,
                "h2_lhv_flow_kw": h2_lhv_flow_kw,
                "total_electric_power_kw": total_electric_power_kw,
                "compression_intensity_kwh_e_per_kwh_h2_lhv": compression_intensity,
                "injection_efficiency": injection_efficiency,
            }
        )

    return pd.DataFrame(rows).sort_values("soc").reset_index(drop=True)


def summarise_injection_efficiency_curve(
    df: pd.DataFrame,
    *,
    project_label: str = "project",
) -> InjectionEfficiencySummary:
    """
    Summarise an injection-efficiency curve.

    The mean efficiency is calculated as an area-weighted mean over fractional SoC.
    This is preferable to a simple arithmetic mean if the SoC points are not
    perfectly equally spaced.
    """
    required_columns = {
        "soc",
        "injection_efficiency",
        "compression_intensity_kwh_e_per_kwh_h2_lhv",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    curve = df.sort_values("soc").copy()

    soc = curve["soc"].to_numpy()
    efficiency = curve["injection_efficiency"].to_numpy()
    intensity = curve["compression_intensity_kwh_e_per_kwh_h2_lhv"].to_numpy()

    if soc[0] > 0 or soc[-1] < 1:
        raise ValueError("Efficiency curve should cover the full SoC range 0 to 1.")

    mean_efficiency = np.trapezoid(efficiency, soc) / (soc[-1] - soc[0])
    mean_intensity = np.trapezoid(intensity, soc) / (soc[-1] - soc[0])

    efficiency_at_soc_50 = float(np.interp(0.5, soc, efficiency))
    intensity_at_soc_50 = float(np.interp(0.5, soc, intensity))

    return InjectionEfficiencySummary(
        project=project_label,
        min_efficiency=float(np.min(efficiency)),
        mean_efficiency=float(mean_efficiency),
        efficiency_at_soc_50=efficiency_at_soc_50,
        max_compression_intensity_kwh_e_per_kwh_h2=float(np.max(intensity)),
        mean_compression_intensity_kwh_e_per_kwh_h2=float(mean_intensity),
        compression_intensity_at_soc_50_kwh_e_per_kwh_h2=intensity_at_soc_50,
    )


def calculate_injection_efficiency_curves_for_projects(
    projects: Mapping[str, StorageProject],
    *,
    n_points: int = 101,
    compression_method: CompressionMethod | str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Calculate injection-efficiency curves for several StorageProject objects.
    """
    if not projects:
        raise ValueError("No projects provided.")

    return {
        label: calculate_injection_efficiency_curve(
            project=project,
            n_points=n_points,
            compression_method=compression_method,
        )
        for label, project in projects.items()
    }


def summarise_injection_efficiency_curves(
    results_by_project: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Summarise several injection-efficiency curves into a dataframe.
    """
    summaries = [
        summarise_injection_efficiency_curve(
            df=df,
            project_label=label,
        )
        for label, df in results_by_project.items()
    ]

    return pd.DataFrame([summary.__dict__ for summary in summaries])


def save_injection_efficiency_results(
    results_by_project: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Save individual curve CSVs, a combined curve CSV, and summary CSV.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        combined_results, summary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for label, df in results_by_project.items():
        safe_label = label.lower().replace(" ", "_").replace("/", "_")
        df.to_csv(
            output_dir / f"injection_efficiency_{safe_label}.csv",
            index=False,
        )

    combined = pd.concat(
        [
            df.assign(project=label)
            for label, df in results_by_project.items()
        ],
        ignore_index=True,
    )

    combined.to_csv(
        output_dir / "injection_efficiency_all_projects.csv",
        index=False,
    )

    summary = summarise_injection_efficiency_curves(results_by_project)

    summary.to_csv(
        output_dir / "injection_efficiency_summary.csv",
        index=False,
    )

    return combined, summary

def plot_injection_efficiency_curve(
    results: pd.DataFrame | Mapping[str, pd.DataFrame],
    *,
    title: str = "Injection efficiency across state of charge",
    xlabel: str = "State of charge [-]",
    ylabel: str = "Injection efficiency [-]",
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plot injection efficiency against SoC.

    For each project, the plot highlights:
    - efficiency curve
    - efficiency at SoC = 50%

    Curves are coloured dynamically using the plasma colormap.
    The SoC = 50% markers use the same colour as their corresponding curve,
    but are represented by a single grey proxy marker in the legend.
    """
    if isinstance(results, pd.DataFrame):
        labelled_results = {"case": results}
    else:
        labelled_results = dict(results)

    if not labelled_results:
        raise ValueError("No results provided for plotting.")

    fig, ax = plt.subplots(figsize=(8, 5))

    n_cases = len(labelled_results)
    cmap = plt.get_cmap("plasma")

    if n_cases == 1:
        colour_values = [0.5]
    else:
        colour_values = np.linspace(0.12, 0.88, n_cases)

    line_handles = []

    for (label, df), colour_value in zip(labelled_results.items(), colour_values):
        colour = cmap(colour_value)

        if "soc" not in df.columns:
            raise ValueError(f"Column 'soc' not found for {label!r}.")

        if "injection_efficiency" not in df.columns:
            raise ValueError(f"Column 'injection_efficiency' not found for {label!r}.")

        curve = df.sort_values("soc").copy()

        soc = curve["soc"].to_numpy()
        efficiency = curve["injection_efficiency"].to_numpy()

        summary = summarise_injection_efficiency_curve(
            curve,
            project_label=label,
        )

        eff_50 = summary.efficiency_at_soc_50

        line, = ax.plot(
            soc,
            efficiency,
            color=colour,
            label=label,
        )
        line_handles.append(line)

        ax.scatter(
            [0.5],
            [eff_50],
            marker="x",
            color=colour,
            s=55,
            linewidths=1.8,
            label="_nolegend_",
            zorder=3,
        )

    soc_50_handle = Line2D(
        [0],
        [0],
        marker="x",
        linestyle="None",
        color="0.45",
        markersize=7,
        markeredgewidth=1.8,
        label="Efficiency when SoC = 50%",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    ax.legend(
        handles=[*line_handles, soc_50_handle],
        frameon=True,
    )

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_compression_intensity_curve(
    results: pd.DataFrame | Mapping[str, pd.DataFrame],
    *,
    title: str = "Compression electricity intensity across state of charge",
    xlabel: str = "State of charge [-]",
    ylabel: str = "Compression intensity [kWh_e/kWh_H2,LHV]",
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Optional companion plot: electricity intensity rather than efficiency.
    """
    if isinstance(results, pd.DataFrame):
        labelled_results = {"case": results}
    else:
        labelled_results = dict(results)

    if not labelled_results:
        raise ValueError("No results provided for plotting.")

    fig, ax = plt.subplots(figsize=(8, 5))

    y_col = "compression_intensity_kwh_e_per_kwh_h2_lhv"

    for label, df in labelled_results.items():
        curve = df.sort_values("soc").copy()

        if y_col not in curve.columns:
            raise ValueError(f"Column {y_col!r} not found for {label!r}.")

        ax.plot(
            curve["soc"],
            curve[y_col],
            label=label,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)