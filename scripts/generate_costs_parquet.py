"""Generate a single CEM-ready Parquet table of UHS cost-rate experiments.

Declare the technologies and experiments below, then run from the repository root:

    pixi run python -m scripts.experiments.generate_cost_rate_parquet

Each row represents one CEM technology instance. Filter by ``experiment_id``
and ``technology`` to load a scenario. The Parquet contains investment (TCR),
fixed O&M (FOM), variable O&M (VOM), and EPR/WTIR applicability bounds.

The sampler's original fitting logic is unchanged. In particular, cross-driver
rates are fitted locally within each sample's EPR/WTIR interval, holding the
reference working-gas capacity fixed.

Requires pandas and a Parquet engine (pyarrow): ``pixi add pyarrow`` if needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scripts.experiments.cost_rates import (
    CostMethod,
    SampledCostSet,
    SamplingConfig,
    generate_cost_samples,
)


# -----------------------------------------------------------------------------
# Edit these settings to define the CEM experiment catalogue.
# -----------------------------------------------------------------------------

OUTPUT_FILE = Path("outputs/cost_rate_samples/uhs_cost_rates.parquet")

TECHNOLOGIES = (
    "aquifer",
    "salt_cavern",
    "depleted_gas_field",
    "lined_rock_cavern",
)


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    method: CostMethod | str
    n_epr: int
    n_wtir: int
    epr_factor_range: tuple[float, float] = (0.5, 1.5)
    wtir_range: tuple[float, float] = (1.0, 5.0)


EXPERIMENTS = (
    Experiment("B30", "bundled", 6, 5),
    Experiment("PD1", "partially_decoupled", 1, 1),
    Experiment("FD1", "fully_decoupled", 1, 1),
    Experiment("FX1", "fully_decoupled_cross_driver", 1, 1),
    Experiment("FX6", "fully_decoupled_cross_driver", 3, 2),
)


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
            (d.max_absolute_percentage_error for d in sample.cross_driver_diagnostics),
            default=0.0,
        ),
    }


def build_experiment_table() -> pd.DataFrame:
    """Run all declared experiment/technology combinations once."""

    ids = [experiment.experiment_id for experiment in EXPERIMENTS]
    if not ids or any(not name.strip() for name in ids) or len(ids) != len(set(ids)):
        raise ValueError("EXPERIMENTS must have unique, nonempty experiment IDs.")
    if not TECHNOLOGIES or len(TECHNOLOGIES) != len(set(TECHNOLOGIES)):
        raise ValueError("TECHNOLOGIES must list unique technologies.")

    records: list[dict] = []
    for experiment in EXPERIMENTS:
        for technology in TECHNOLOGIES:
            config = SamplingConfig(
                technology=technology,
                method=experiment.method,
                n_epr=experiment.n_epr,
                n_wtir=experiment.n_wtir,
                epr_factor_range=experiment.epr_factor_range,
                wtir_range=experiment.wtir_range,
            )
            samples = generate_cost_samples(config)
            if len(samples) != experiment.n_epr * experiment.n_wtir:
                raise RuntimeError(f"Unexpected sample count for {technology}/{experiment.experiment_id}")
            records.extend(sample_record(experiment.experiment_id, sample) for sample in samples)
            print(f"{experiment.experiment_id:>4} | {technology:<20} | {len(samples):>3} samples")

    table = pd.DataFrame.from_records(records)
    if table["instance_id"].duplicated().any():
        raise ValueError("Duplicate instance IDs; check the experiment catalogue.")
    return table


def main() -> None:
    table = build_experiment_table()
    # Remove floating-point artefacts while retaining 12 significant digits
    export_table = table.copy()

    for column in export_table.select_dtypes(include="floating").columns:
        export_table[column] = export_table[column].map(
            lambda x: float(f"{x:.6g}") if pd.notna(x) else x
        )
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        export_table.to_parquet(OUTPUT_FILE, engine="pyarrow", index=False,)
        export_table.to_csv(OUTPUT_FILE.with_suffix(".csv"),index=False, float_format="%.6g",)
    except ImportError as exc:
        raise RuntimeError(
            "Parquet export requires pyarrow in your Pixi environment. "
            "Run `pixi add pyarrow` and retry."
        ) from exc
    print(f"Saved {len(table)} CEM technology instances to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
