"""Generate CEM-ready Parquet/CSV tables of UHS cost and operating parameters.

Declare the technologies and experiments below, then run from the repository root:

    pixi run python -m scripts.generate_costs_parquet --output outputs/cost_rate_samples/WTIR3/uhs_cost_rates.parquet --reference-wtir 3

Each row represents one CEM technology instance. Filter by ``experiment_id``
and ``technology`` to load a scenario. The tables contain TCR, FOM, VOM,
EPR/WTIR applicability bounds and selected operational parameters:

* Compression electricity intensity evaluated at 50% working-gas SoC.
* Equivalent compression-process efficiency, for reference only. If the CEM
  explicitly consumes electricity, do NOT apply it as an H2 conversion loss.
* Fixed physical purification recovery (separate from geological recovery).
* Optional per-technology hourly standing loss, configured explicitly below.
* Salt-cavern-only conservative linear 24-hour pressure-ramp SoC constraints.

An absent process or unspecified assumption is exported as a null/blank.

The study exports one instance per formulation and technology. The reference
design remains at its technology-specific EPR and WTIR. Cross-driver
capital rates are fitted on the expanded EPR/WTIR training grid while holding
the reference working-gas capacity fixed.

Requires pandas, scipy and a Parquet engine (pyarrow): ``pixi add pyarrow`` if needed.
"""

from __future__ import annotations

import argparse
import importlib
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from uhs_costs.analysis.injection_efficiency import (
    compression_intensity_to_injection_efficiency,
    pressure_at_soc_profile,
)
from uhs_costs.analysis.pressure_constraint import (
    BAR_TO_PA,
    add_capacity_normalised_columns,
    calculate_pressure_limited_flow_curve,
    methane_mass_from_standard_volume,
    summarise_soc_endpoint_coefficients,
    total_pressure_from_inventories,
)
from uhs_costs.design.helpers.compression_model import (
    CompressionInput,
    calculate_compression,
)

from scripts.experiments.cost_rates import (
    CostMethod,
    SampledCostSet,
    SamplingConfig,
    STUDY_EPR_FACTOR_RANGE,
    STUDY_WTIR_RANGE,
    STUDY_EPR_FACTOR_TRAINING,
    STUDY_WTIR_TRAINING,
    generate_cost_samples,
    get_technology,
)
from scripts.experiments.linearisation_sensitivity import (
    construct_project,
)
from scripts.experiments.reference_provenance import write_reference_manifest


# -----------------------------------------------------------------------------
# Edit these settings to define the CEM experiment catalogue.
# -----------------------------------------------------------------------------

EXPORT_SIGNIFICANT_DIGITS = 6

TECHNOLOGIES = (
    "aquifer",
    "salt_cavern",
    "depleted_gas_field",
    "lined_rock_cavern",
)

# Annual *standing* hydrogen loss assumptions, not inferred by the UHS model.
# The porous-media values below are the previously adopted 1.5%/year assumption.
# Technologies without a justified assumption are left blank (None). Change
# or remove values here if your CEM assumptions differ.
ANNUAL_STORAGE_LOSS_FRACTION: dict[str, float | None] = {
    "aquifer": 0.015,
    "depleted_gas_field": 0.015,
    "salt_cavern": None,
    "lined_rock_cavern": None,
}
HOURS_PER_YEAR = 8760

# The 24-hour pressure-ramp constraint is ONLY applied to salt caverns.
SALT_CAVERN_PRESSURE_CHANGE_LIMIT_BAR_PER_DAY = 10.0
SALT_CAVERN_PRESSURE_GRID_STEP_BAR = 1.0


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    method: CostMethod | str
    n_epr: int
    n_wtir: int
    epr_factor_range: tuple[float, float] = STUDY_EPR_FACTOR_RANGE
    wtir_range: tuple[float, float] = STUDY_WTIR_RANGE


EXPERIMENTS = (
    Experiment("B1", "bundled", 1, 1),
    Experiment("P1", "partially_decoupled", 1, 1),
    Experiment("F1", "fully_decoupled", 1, 1),
    Experiment("FX1", "fully_decoupled_cross_driver", 1, 1),
)


def _pressure_at_half_soc_pa(project) -> float:
    """Calculate the pressure at exactly 50% H2 working-gas SoC.

    This uses the physical inventory/pressure relationship, rather than
    assuming that 50% SoC corresponds to the midpoint in pressure.
    """

    end_points = pressure_at_soc_profile(project, n_points=2)
    h2_at_half_soc_kg = (
        0.5 * float(end_points["h2_inventory_kg"].iloc[0])
        + 0.5 * float(end_points["h2_inventory_kg"].iloc[-1])
    )
    methane_mass_kg = methane_mass_from_standard_volume(
        project.inventory.abandonment_gas_methane_volume_sm3
    )
    return float(
        total_pressure_from_inventories(
            m_h2_kg=h2_at_half_soc_kg,
            m_ch4_kg=methane_mass_kg,
            volume_m3=project.inventory.required_storage_volume_m3,
            temperature_k=project.wells.well_temperature_k,
        )
    )


def _compression_intensity_at_half_soc(project, pressure_pa: float, *, injection: bool) -> float | None:
    """Return electrical kWh per kWh of H2, with compressor configuration fixed."""

    compressor = project.compression if injection else project.withdrawal_compression
    if compressor is None:
        return None  # No compressor exists on this side of the storage.

    pipeline_pa = project.pressures.pipeline_pressure_pa
    inlet_pa = pipeline_pa if injection else pressure_pa
    outlet_pa = pressure_pa if injection else pipeline_pa

    # A compressor may exist to meet the worst-case pressure condition but
    # require no electricity at the 50%-SoC evaluation point.
    if outlet_pa <= inlet_pa:
        return 0.0

    flow_kg_s = (
        project.flows.injection_flow_kg_per_s
        if injection
        else project.flows.withdrawal_flow_kg_per_s
    )
    result = calculate_compression(
        CompressionInput(
            inlet_pressure_pa=inlet_pa,
            outlet_pressure_pa=outlet_pa,
            inlet_temperature_k=project.wells.well_temperature_k,
            mass_flow_kg_s=flow_kg_s,
            method=compressor.method,
            number_of_stages=compressor.number_of_stages,
        )
    )
    if result.electricity_kwh_per_kwh_h2_lhv is None:
        raise ValueError("Compression model did not report electricity intensity.")
    return float(result.electricity_kwh_per_kwh_h2_lhv)


def _compression_equivalent_efficiency(intensity: float | None) -> float | None:
    """Convert electricity intensity to a combined-energy equivalent efficiency."""
    return (
        compression_intensity_to_injection_efficiency(intensity)
        if intensity is not None
        else None
    )


def _salt_cavern_soc_24h_constraints(project, label: str) -> dict[str, float]:
    """Recompute a conservative 24-hour pressure-ramp fit for this exact design."""

    pressure_curve = calculate_pressure_limited_flow_curve(
        project,
        pressure_step_pa=SALT_CAVERN_PRESSURE_GRID_STEP_BAR * BAR_TO_PA,
        pressure_change_limit_pa_per_day=(
            SALT_CAVERN_PRESSURE_CHANGE_LIMIT_BAR_PER_DAY * BAR_TO_PA
        ),
    )
    normalised_curve = add_capacity_normalised_columns(
        pressure_curve,
        working_gas_capacity_kwh_lhv=(
            project.inventory.recoverable_working_gas_capacity_kwh_lhv
        ),
    )
    fit = summarise_soc_endpoint_coefficients(
        {label: normalised_curve},
        conservative=True,
    ).iloc[0]
    return {
        "soc_24h_a_min": float(fit["a_min"]),
        "soc_24h_b_min": float(fit["b_min"]),
        "soc_24h_a_max": float(fit["a_max"]),
        "soc_24h_b_max": float(fit["b_max"]),
    }


def operational_record(sample: SampledCostSet) -> dict[str, float | None]:
    """Reconstruct the sampled physical design and extract operating parameters."""

    spec = get_technology(sample.technology)
    project = construct_project(
        spec=spec,
        storage_capacity_gwh_lhv=sample.storage_reference_gwh_lhv,
        epr_days=sample.epr_days,
        wtir=sample.wtir,
    )

    pressure_at_half_soc_pa = _pressure_at_half_soc_pa(project)
    injection_electricity = _compression_intensity_at_half_soc(
        project, pressure_at_half_soc_pa, injection=True
    )
    withdrawal_electricity = _compression_intensity_at_half_soc(
        project, pressure_at_half_soc_pa, injection=False
    )

    annual_loss = ANNUAL_STORAGE_LOSS_FRACTION.get(sample.technology)
    if annual_loss is not None and not 0 <= annual_loss < 1:
        raise ValueError(f"Invalid annual standing loss for {sample.technology}.")
    hourly_loss = (
        1.0 - (1.0 - annual_loss) ** (1.0 / HOURS_PER_YEAR)
        if annual_loss is not None else None
    )

    operating = {
        "injection_compression_electricity_kWh_e_per_kWh_H2": injection_electricity,
        "withdrawal_compression_electricity_kWh_e_per_kWh_H2": withdrawal_electricity,
        "injection_compression_equivalent_efficiency": (
            _compression_equivalent_efficiency(injection_electricity)
        ),
        "withdrawal_compression_equivalent_efficiency": (
            _compression_equivalent_efficiency(withdrawal_electricity)
        ),
        "purification_hydrogen_recovery": (
            project.purification.hydrogen_recovery
            if project.purification is not None else None
        ),
        "storage_loss_per_hour": hourly_loss,
        "soc_24h_a_min": None,
        "soc_24h_b_min": None,
        "soc_24h_a_max": None,
        "soc_24h_b_max": None,
    }

    if sample.technology == "salt_cavern":
        operating.update(_salt_cavern_soc_24h_constraints(project, sample.sample_id))

    return operating


def sample_record(experiment_id: str, sample: SampledCostSet) -> dict:
    """Flatten one sampled technology to a Parquet-safe CEM parameter row."""

    return {
        # The underlying sampler's sample_id is not unique across experiments.
        "instance_id": f"{experiment_id}__{sample.sample_id}",
        "experiment_id": experiment_id,
        "sample_id": sample.sample_id,
        "technology": sample.technology,
        "method": sample.method.value,
        "E_reference_GWh": sample.storage_reference_gwh_lhv,
        "EPR_factor": sample.epr_factor,
        "EPR_days": sample.epr_days,
        "WTIR": sample.wtir,
        # Applicability ranges; enforcement is the responsibility of the CEM.
        "EPR_factor_lower": sample.epr_factor_interval.lower,
        "EPR_factor_upper": sample.epr_factor_interval.upper,
        "EPR_factor_lower_inclusive": sample.epr_factor_interval.lower_inclusive,
        "EPR_factor_upper_inclusive": sample.epr_factor_interval.upper_inclusive,
        "EPR_days_lower": sample.epr_interval_days.lower,
        "EPR_days_upper": sample.epr_interval_days.upper,
        "EPR_days_lower_inclusive": sample.epr_interval_days.lower_inclusive,
        "EPR_days_upper_inclusive": sample.epr_interval_days.upper_inclusive,
        "WTIR_lower": sample.wtir_interval.lower,
        "WTIR_upper": sample.wtir_interval.upper,
        "WTIR_lower_inclusive": sample.wtir_interval.lower_inclusive,
        "WTIR_upper_inclusive": sample.wtir_interval.upper_inclusive,
        # TCR includes CAPEX + ABEX (not merely CAPEX).
        "TCR_storage_EUR_per_kWh": sample.tcr_rates.storage,
        "TCR_injection_EUR_per_kW": sample.tcr_rates.injection,
        "TCR_withdrawal_EUR_per_kW": sample.tcr_rates.withdrawal,
        "FOM_storage_EUR_per_kWh_year": sample.fom_rates.storage,
        "FOM_injection_EUR_per_kW_year": sample.fom_rates.injection,
        "FOM_withdrawal_EUR_per_kW_year": sample.fom_rates.withdrawal,
        "VOM_injection_EUR_per_kWh": sample.vom_rates.injection,
        "VOM_withdrawal_EUR_per_kWh": sample.vom_rates.withdrawal,
        "centre_TCR_error_pct": sample.centre_tcr_error_pct,
        "local_fit_max_APE_pct": max(
            (
                d.max_absolute_percentage_error
                for d in sample.cross_driver_diagnostics
                if d.cost_type in ("capex", "abex")
            ),
            default=0.0,
        ),
        "local_FOM_fit_max_APE_pct": max(
            (
                d.max_absolute_percentage_error
                for d in sample.cross_driver_diagnostics
                if d.cost_type == "fixed_opex"
            ),
            default=0.0,
        ),
        **operational_record(sample),
    }


def build_experiment_table(*, reference_wtir: float | None = None,
                           reference_epr_factor: float | None = None,
                           epr_factor_range: tuple[float, float] = STUDY_EPR_FACTOR_RANGE,
                           wtir_range: tuple[float, float] = STUDY_WTIR_RANGE,
                           fx_training_epr_points: int = len(STUDY_EPR_FACTOR_TRAINING),
                           fx_training_wtir_points: int = len(STUDY_WTIR_TRAINING),
                           ) -> tuple[pd.DataFrame, list[SampledCostSet]]:
    """Run all declared experiment/technology combinations once."""

    for name, bounds in (("EPR factor", epr_factor_range), ("WTIR", wtir_range)):
        if not all(math.isfinite(v) for v in bounds) or not 0 < bounds[0] < bounds[1]:
            raise ValueError(f"Invalid {name} domain: {bounds}")
    for name, count in (("EPR", fx_training_epr_points), ("WTIR", fx_training_wtir_points)):
        if count < 2:
            raise ValueError(f"FX training needs at least two {name} points")

    def grid(bounds: tuple[float, float], count: int) -> tuple[float, ...]:
        return tuple(bounds[0] + (bounds[1] - bounds[0]) * i / (count - 1)
                     for i in range(count))

    epr_training = grid(epr_factor_range, fx_training_epr_points)
    wtir_training = grid(wtir_range, fx_training_wtir_points)

    ids = [experiment.experiment_id for experiment in EXPERIMENTS]
    if not ids or any(not name.strip() for name in ids) or len(ids) != len(set(ids)):
        raise ValueError("EXPERIMENTS must have unique, nonempty experiment IDs.")
    if not TECHNOLOGIES or len(TECHNOLOGIES) != len(set(TECHNOLOGIES)):
        raise ValueError("TECHNOLOGIES must list unique technologies.")

    records: list[dict] = []
    all_samples: list[SampledCostSet] = []
    for experiment in EXPERIMENTS:
        for technology in TECHNOLOGIES:
            config = SamplingConfig(
                technology=technology,
                method=experiment.method,
                n_epr=experiment.n_epr,
                n_wtir=experiment.n_wtir,
                epr_factor_range=epr_factor_range,
                wtir_range=wtir_range,
                # No implicit numerical WTIR/EPR. Technology files are the
                # authority unless this scenario explicitly overrides them.
                epr_factor_override=reference_epr_factor,
                wtir_override=reference_wtir,
                # One global FX coefficient set over the declared domain.
                fx_training_epr_factors=(
                    epr_training
                    if experiment.method == CostMethod.FULLY_DECOUPLED_CROSS_DRIVER.value
                    else None
                ),
                fx_training_wtir_values=(
                    wtir_training
                    if experiment.method == CostMethod.FULLY_DECOUPLED_CROSS_DRIVER.value
                    else None
                ),
            )
            samples = generate_cost_samples(config)
            if len(samples) != experiment.n_epr * experiment.n_wtir:
                raise RuntimeError(f"Unexpected sample count for {technology}/{experiment.experiment_id}")
            records.extend(sample_record(experiment.experiment_id, sample) for sample in samples)
            all_samples.extend(samples)
            print(f"{experiment.experiment_id:>4} | {technology:<20} | {len(samples):>3} samples")

    table = pd.DataFrame.from_records(records)
    if table["instance_id"].duplicated().any():
        raise ValueError("Duplicate instance IDs; check the experiment catalogue.")
    return table, all_samples


def verify_reference_expectations(*, expected_salt_cavern_site: str | None,
                                  expected_wtir: float | None,
                                  reference_wtir: float | None) -> None:
    """Fail early if the active design files do not match the named scenario."""
    if expected_salt_cavern_site is not None:
        spec = get_technology("salt_cavern")
        module = importlib.import_module(spec.project_constructor.__module__)
        actual_site = getattr(module, "REFERENCE_SITE", None)
        if actual_site != expected_salt_cavern_site:
            raise ValueError(
                f"Salt cavern reference site is {actual_site!r}, "
                f"expected {expected_salt_cavern_site!r}"
            )
    if expected_wtir is not None:
        if not math.isfinite(expected_wtir) or expected_wtir <= 0:
            raise ValueError("Expected reference WTIR must be positive and finite")
        for technology in TECHNOLOGIES:
            spec = get_technology(technology)
            actual = reference_wtir if reference_wtir is not None else spec.default_wtir
            if not math.isclose(float(actual), expected_wtir, rel_tol=1e-9):
                raise ValueError(
                    f"{technology} reference WTIR {actual} differs from "
                    f"expected {expected_wtir}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export UHS costs with reproducible reference provenance")
    parser.add_argument("--output", type=Path, required=True,
                        help="Explicit scenario-specific output .parquet; existing outputs are not overwritten")
    parser.add_argument("--reference-wtir", type=float, default=None,
                        help="Explicit override; omitted = technology design REFERENCE_WTIR")
    parser.add_argument("--reference-epr-factor", type=float, default=None,
                        help="Explicit override; omitted = technology design REFERENCE_EPR_DAYS")
    parser.add_argument("--epr-factor-min", type=float, default=STUDY_EPR_FACTOR_RANGE[0],
                        help="Minimum fitted EPR factor, relative to the design-file reference")
    parser.add_argument("--epr-factor-max", type=float, default=STUDY_EPR_FACTOR_RANGE[1],
                        help="Maximum fitted EPR factor, relative to the design-file reference")
    parser.add_argument("--wtir-min", type=float, default=STUDY_WTIR_RANGE[0],
                        help="Minimum fitted WTIR ratio")
    parser.add_argument("--wtir-max", type=float, default=STUDY_WTIR_RANGE[1],
                        help="Maximum fitted WTIR ratio")
    parser.add_argument("--fx-training-epr-points", type=int, default=len(STUDY_EPR_FACTOR_TRAINING),
                        help="Number of inclusive EPR-factor training coordinates")
    parser.add_argument("--fx-training-wtir-points", type=int, default=len(STUDY_WTIR_TRAINING),
                        help="Number of inclusive WTIR training coordinates")
    parser.add_argument("--expect-salt-cavern-site", type=str, default=None,
                        help="Guard against accidentally using another geological reference site")
    parser.add_argument("--expect-reference-wtir", type=float, default=None,
                        help="Guard against unexpectedly changed physical WTIR assumptions")
    args = parser.parse_args()
    output_file = args.output
    if output_file.suffix != ".parquet":
        parser.error("--output must end in .parquet")
    csv_file = output_file.with_suffix(".csv")
    manifest_file = output_file.with_suffix(".reference.json")
    for target in (output_file, csv_file):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite historical cost data: {target}")
    verify_reference_expectations(
        expected_salt_cavern_site=args.expect_salt_cavern_site,
        expected_wtir=args.expect_reference_wtir,
        reference_wtir=args.reference_wtir,
    )
    table, all_samples = build_experiment_table(
        reference_wtir=args.reference_wtir,
        reference_epr_factor=args.reference_epr_factor,
        epr_factor_range=(args.epr_factor_min, args.epr_factor_max),
        wtir_range=(args.wtir_min, args.wtir_max),
        fx_training_epr_points=args.fx_training_epr_points,
        fx_training_wtir_points=args.fx_training_wtir_points,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_reference_manifest(Path(__file__).resolve().parents[1], all_samples, manifest_file)

    # Six significant digits remove floating-point tails while retaining
    # numeric values (not strings) in the Parquet. Change this one constant
    # above if the precision requirements of the CEM change.
    export_table = table.copy()
    for column in export_table.select_dtypes(include="floating").columns:
        export_table[column] = export_table[column].map(
            lambda value: (
                float(f"{value:.{EXPORT_SIGNIFICANT_DIGITS}g}")
                if pd.notna(value) else value
            )
        )

    try:
        export_table.to_parquet(output_file, engine="pyarrow", index=False)
    except ImportError as exc:
        raise RuntimeError(
            "Parquet export requires pyarrow in your Pixi environment. "
            "Run `pixi add pyarrow` and retry."
        ) from exc
    export_table.to_csv(
        csv_file,
        index=False,
        float_format=f"%.{EXPORT_SIGNIFICANT_DIGITS}g",
    )
    print(f"Saved {len(export_table)} CEM technology instances to {output_file}")
    print(f"Saved inspection CSV to {csv_file}")
    print(f"Saved reference provenance to {manifest_file}")


if __name__ == "__main__":
    main()
