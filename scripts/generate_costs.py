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


OUTPUT_DIR = Path("outputs/cost_rate_samples")

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
    for technology in TECHNOLOGIES:
        for experiment, (method, n_epr, n_wtir) in EXPERIMENTS.items():

            config = SamplingConfig(
                technology=technology,
                method=method,
                n_epr=n_epr,
                n_wtir=n_wtir,
                epr_factor_range=STUDY_EPR_FACTOR_RANGE,
                wtir_range=STUDY_WTIR_RANGE,
                epr_factor_override=1.0,
                wtir_override=3,
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
            filename = OUTPUT_DIR / f"{technology}_{experiment}.csv"
            write_sample_summary(samples, filename)

            # Optional diagnostic figures
            if GENERATE_PLOTS:
                write_sample_plots(
                    samples,
                    OUTPUT_DIR / "plots" / f"{technology}_{experiment}",
                )

            print(f"Saved {len(samples)} samples: {filename}")


if __name__ == "__main__":
    main()
