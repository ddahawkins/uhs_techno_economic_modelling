"""Compare one-sample B, P, F and FX capital-cost representations.

For every requested UHS technology, produce two publication-style PDFs:

* ``<technology>_tcr_slices.pdf``: detailed TCR and CEM approximations along
  EPR and WTIR, holding the other ratio at its reference value.
* ``<technology>_tcr_error_maps.pdf``: F and FX percentage errors over the
  full EPR-factor x WTIR evaluation domain, with FX training points marked.

The two maps are diagnostics only. B is restricted to its reference point;
P is valid along the EPR slice at reference WTIR. F and FX can vary both
ratios. All comparisons hold working-gas capacity at its technology-specific
reference value. This is NOT a CEM optimisation or a refitting procedure.

Example (run from the project root with the uhs_costs package installed):

    pixi run python -m scripts.experiments.compare_cost_formulations

To test a single technology or reduce detailed-model evaluations:

    pixi run python -m scripts.experiments.compare_cost_formulations \
        --technologies salt_cavern --grid-points 11 --slice-points 25
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from uhs_costs.cost_model.helpers.cost_components import CostDriver, CostType

from scripts.experiments.cost_rates import (
    SamplingConfig,
    STUDY_EPR_FACTOR_RANGE,
    STUDY_WTIR_RANGE,
    STUDY_EPR_FACTOR_TRAINING,
    STUDY_WTIR_TRAINING,
    design_capacities,
    evaluate_design,
    generate_cost_samples,
    get_technology,
)


OUTPUT_DIR = Path("outputs/cost_rate_samples/comparison")
TECHNOLOGIES = (
    "salt_cavern",
    "depleted_gas_field",
    "aquifer",
    "lined_rock_cavern",
)
FORMULATIONS = {
    "B": "bundled",
    "P": "partially_decoupled",
    "F": "fully_decoupled",
    "FX": "fully_decoupled_cross_driver",
}

# 190 mm wide: Elsevier two-column format. Font sizes remain legible at 100%.
WIDTH_INCH = 190 / 25.4
PURPLE = "#7e03a8"
PLASMA = (
    "#4c02a1",
    "#aa2395",
    "#e66c5c",
    "#fdc527",

)


def _cost_predictions(technology: str):
    """Create exactly one sample per formulation at original reference ratios."""
    samples = {}
    for label, method in FORMULATIONS.items():
        sample = generate_cost_samples(SamplingConfig(
            technology=technology,
            method=method,
            n_epr=1,
            n_wtir=1,
            epr_factor_range=STUDY_EPR_FACTOR_RANGE,
            wtir_range=STUDY_WTIR_RANGE,
            epr_factor_override=1.0,
            wtir_override=2,
            fx_training_epr_factors=(
                STUDY_EPR_FACTOR_TRAINING if label == "FX" else None
            ),
            fx_training_wtir_values=(
                STUDY_WTIR_TRAINING if label == "FX" else None
            ),
        ))
        if len(sample) != 1:
            raise AssertionError("Each study formulation must have one cost set.")
        samples[label] = sample[0]
    return samples


def _sample_tcr(sample, capacities: dict[CostDriver, float]) -> float:
    return sum((
        sample.tcr_rates.storage * capacities[CostDriver.STORAGE],
        sample.tcr_rates.injection * capacities[CostDriver.INJECTION],
        sample.tcr_rates.withdrawal * capacities[CostDriver.WITHDRAWAL],
    ))


def _total_tcr(breakdown) -> float:
    return sum(
        component.value_eur for component in breakdown.components
        if component.cost_type in (CostType.CAPEX, CostType.ABEX)
    )


def _analysis(technology: str, samples, slice_points: int, grid_points: int):
    """Cache detailed evaluations shared by the two plots and metrics."""
    spec = get_technology(technology)
    storage_kwh = spec.default_storage_capacity_gwh_lhv * 1e6
    cache = {}

    def evaluate(epr_factor: float, wtir: float):
        key = (round(float(epr_factor), 12), round(float(wtir), 12))
        if key not in cache:
            epr_days = key[0] * spec.default_epr_days
            capacities = design_capacities(spec, epr_days, key[1])
            breakdown = evaluate_design(
                spec, spec.default_storage_capacity_gwh_lhv, epr_days, key[1]
            )
            detailed_tcr = _total_tcr(breakdown) / storage_kwh
            represented_tcr = {
                label: _sample_tcr(sample, capacities) / storage_kwh
                for label, sample in samples.items()
            }
            cache[key] = (detailed_tcr, represented_tcr)
        return cache[key]

    epr_slice = np.unique(np.r_[np.linspace(*STUDY_EPR_FACTOR_RANGE, slice_points), 1.0])
    wtir_slice = np.unique(np.r_[np.linspace(*STUDY_WTIR_RANGE, slice_points), 2])

    epr_traces = [evaluate(factor, 2) for factor in epr_slice]
    wtir_traces = [evaluate(1.0, value) for value in wtir_slice]

    epr_grid = np.unique(np.r_[np.linspace(*STUDY_EPR_FACTOR_RANGE, grid_points), 1.0])
    wtir_grid = np.unique(np.r_[np.linspace(*STUDY_WTIR_RANGE, grid_points), 2])
    detailed = np.empty((len(wtir_grid), len(epr_grid)))
    fitted = {label: np.empty_like(detailed) for label in ("F", "FX")}

    for row, wtir in enumerate(wtir_grid):
        for col, epr_factor in enumerate(epr_grid):
            source, represented = evaluate(epr_factor, wtir)
            detailed[row, col] = source
            for label in fitted:
                fitted[label][row, col] = represented[label]

    with np.errstate(divide="ignore", invalid="ignore"):
        errors = {
            label: 100.0 * (model - detailed) / detailed
            for label, model in fitted.items()
        }
    if any(not np.isfinite(error).all() for error in errors.values()):
        raise ValueError(f"Invalid detailed TCR or percentage error for {technology}.")

    return epr_slice, epr_traces, wtir_slice, wtir_traces, epr_grid, wtir_grid, errors


def _plot_slices(technology, samples, analysis, output_dir: Path):
    epr, epr_traces, wtir, wtir_traces, *_ = analysis
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_INCH, 3.3), constrained_layout=True)

    for ax, x, traces, variable in (
        (axes[0], epr, epr_traces, "epr"),
        (axes[1], wtir, wtir_traces, "wtir"),
    ):
        ax.plot(x, [point[0] for point in traces], color="black", lw=1.9,ls="-",
                label="Detailed")


        reference_x = 1.0 if variable == "epr" else 2
        reference_tcr = _sample_tcr(
            samples["B"],
            design_capacities(
                get_technology(technology), samples["B"].epr_days, 2
            ),
        ) / (get_technology(technology).default_storage_capacity_gwh_lhv * 1e6)

        ax.plot([reference_x], [reference_tcr], marker="x", ms=6,
                mew=1.4, color=PLASMA[0], ls="none", label="B reference", zorder=10)
        # P can vary EPR at fixed WTIR, but WTIR is fixed by construction.
        if variable == "epr":
            ax.plot(x, [point[1]["P"] for point in traces], color=PLASMA[1],
                    ls="--", lw=1.2, label="P")
        ax.plot(x, [point[1]["F"] for point in traces], color=PLASMA[2],
                        lw=1.5, ls="--", label="F")
        ax.plot(x, [point[1]["FX"] for point in traces], color=PLASMA[3],ls="--",
                        lw=1.6, label="FX")
        
        ax.axvline(reference_x, color="0.6", ls="--", lw=0.7, zorder=0)
        ax.set_xlabel("EPR / reference EPR [-]" if variable == "epr"
                      else "WTIR (withdrawal / injection) [-]")
        # ax.grid(axis="y", color="0.9", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Total capital requirement [€/kWh$_{\\mathrm{H_2}}$]")
    axes[0].set_xlim(*STUDY_EPR_FACTOR_RANGE)
    axes[1].set_xlim(*STUDY_WTIR_RANGE)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5,
               fontsize=8, frameon=False)

    path = output_dir / f"{technology}_tcr_slices.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_error_maps(technology: str, analysis, output_dir: Path):
    *_, epr, wtir, errors = analysis
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_INCH, 3.4),
                             sharex=True, sharey=True, constrained_layout=True)
    magnitude = max(float(np.max(np.abs(error))) for error in errors.values())
    scale = max(magnitude, 0.01)
    norm = TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale)

    for ax, label in zip(axes, ("F", "FX")):
        image = ax.pcolormesh(epr, wtir, errors[label], shading="nearest",
                              cmap="RdBu_r", norm=norm, rasterized=True)
        ax.plot(1.0, 2, marker="x", ms=6, mew=1.3, color="black")
        if label == "FX":
            xx, yy = np.meshgrid(STUDY_EPR_FACTOR_TRAINING, STUDY_WTIR_TRAINING)
            ax.scatter(xx, yy, marker=".", color="black", s=5, alpha=0.5)
        ax.set_title(f"{label}: represented − detailed", fontsize=9)
        ax.set_xlabel("EPR / reference EPR [-]")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("WTIR [-]")
    fig.colorbar(image, ax=axes, location="bottom", shrink=0.75,
                 pad=0.12, label="Total TCR error [%]")
    path = output_dir / f"{technology}_tcr_error_maps.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--technologies", nargs="+", choices=TECHNOLOGIES,
                        default=TECHNOLOGIES)
    parser.add_argument("--slice-points", type=int, default=41)
    parser.add_argument("--grid-points", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.slice_points < 3 or args.grid_points < 3:
        parser.error("--slice-points and --grid-points must be at least 3")

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9,
                         "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "pdf.fonttype": 42})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for technology in args.technologies:
        print(f"Calibrating B/P/F/FX for {technology} ...", flush=True)
        samples = _cost_predictions(technology)
        analysis = _analysis(technology, samples, args.slice_points, args.grid_points)
        *_, errors = analysis
        _plot_slices(technology, samples, analysis, args.output_dir)
        _plot_error_maps(technology, analysis, args.output_dir)
        for label in ("F", "FX"):
            error = errors[label]
            row = {
                "technology": technology,
                "formulation": label,
                "reference_EPR_days": samples[label].epr_days,
                "reference_WTIR": samples[label].wtir,
                "evaluation_points": error.size,
                "mean_signed_error_pct": float(np.mean(error)),
                "mean_absolute_error_pct": float(np.mean(np.abs(error))),
                "p95_absolute_error_pct": float(np.percentile(np.abs(error), 95)),
                "maximum_absolute_error_pct": float(np.max(np.abs(error))),
                "reference_TCR_error_pct": samples[label].centre_tcr_error_pct,
                "TCR_storage_EUR_per_kWh": samples[label].tcr_rates.storage,
                "TCR_injection_EUR_per_kW": samples[label].tcr_rates.injection,
                "TCR_withdrawal_EUR_per_kW": samples[label].tcr_rates.withdrawal,
            }
            summary.append(row)
            print(f"  {label}: mean absolute TCR error {row['mean_absolute_error_pct']:.2f}%, "
                  f"max {row['maximum_absolute_error_pct']:.2f}%", flush=True)

    path = args.output_dir / "comparison_tcr_errors.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved comparisons under {args.output_dir}")


if __name__ == "__main__":
    main()
