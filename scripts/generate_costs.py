import argparse
from pathlib import Path

from scripts.experiments.cost_rates import (
    SamplingConfig,
    STUDY_EPR_FACTOR_RANGE,
    STUDY_WTIR_RANGE,
    STUDY_EPR_FACTOR_TRAINING,
    STUDY_WTIR_TRAINING,
    generate_cost_samples,
    write_sample_summary,
    write_sample_plots,
)
from scripts.experiments.reference_provenance import write_reference_manifest


# The study exports one cost set per technology/formulation. Keep the 1 suffix
# in the output file names for compatibility with existing scenario files.
TECHNOLOGIES = [
    "aquifer",
    "salt_cavern",
    "depleted_gas_field",
    "lined_rock_cavern",
]

EXPERIMENTS = {
    "B1": ("bundled", 1, 1),
    "P1": ("partially_decoupled", 1, 1),
    "F1": ("fully_decoupled", 1, 1),
    "FX1": ("fully_decoupled_cross_driver", 1, 1),
}

# Individual TCR PDFs can be expensive (they re-evaluate the detailed model).
# For side-by-side B/P/F/FX comparisons use compare_cost_formulations.py.
GENERATE_PLOTS = False


def main():
    parser = argparse.ArgumentParser(description="Generate CEM costs from technology references")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Unique, explicit scenario directory; existing cost files are never overwritten")
    parser.add_argument("--reference-wtir", type=float, default=None,
                        help="Explicit scenario override; omitted = technology design REFERENCE_WTIR")
    parser.add_argument("--reference-epr-factor", type=float, default=None,
                        help="Explicit scenario override; omitted = technology design REFERENCE_EPR_DAYS")
    args = parser.parse_args()
    pending = []
    all_samples = []
    for technology in TECHNOLOGIES:
        for experiment, (method, n_epr, n_wtir) in EXPERIMENTS.items():

            config = SamplingConfig(
                technology=technology,
                method=method,
                n_epr=n_epr,
                n_wtir=n_wtir,
                epr_factor_range=STUDY_EPR_FACTOR_RANGE,
                wtir_range=STUDY_WTIR_RANGE,
                epr_factor_override=args.reference_epr_factor,
                wtir_override=args.reference_wtir,
                fx_training_epr_factors=(
                    STUDY_EPR_FACTOR_TRAINING if experiment == "FX1" else None
                ),
                fx_training_wtir_values=(
                    STUDY_WTIR_TRAINING if experiment == "FX1" else None
                ),
            )

            # Generate the samples
            samples = generate_cost_samples(config)

            # Export cost rates and interval constraints
            filename = args.output_dir / f"{technology}_{experiment}.csv"
            if filename.exists():
                raise FileExistsError(f"Refusing to overwrite historical cost table: {filename}")
            pending.append((samples, filename, technology, experiment))
            all_samples.extend(samples)

    write_reference_manifest(Path(__file__).resolve().parents[1], all_samples,
                             args.output_dir / "reference_manifest.json")
    for samples, filename, technology, experiment in pending:
        write_sample_summary(samples, filename)

        # Optional diagnostic figures
        if GENERATE_PLOTS:
            write_sample_plots(
                samples,
                args.output_dir / "plots" / f"{technology}_{experiment}",
            )

        print(f"Saved {len(samples)} samples: {filename}")


if __name__ == "__main__":
    main()
