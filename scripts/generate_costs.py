from pathlib import Path

from scripts.experiments.cost_rates import (
    SamplingConfig,
    generate_cost_samples,
    write_sample_summary,
    write_sample_plots,
)


OUTPUT_DIR = Path("outputs/cost_rate_samples")

# Technology and experimental configurations
TECHNOLOGIES = [
    "aquifer",
    "salt_cavern",
    "depleted_gas_field",
    "lined_rock_cavern",
]

EXPERIMENTS = {
    "B1": ("bundled", 1, 1),
    "B30": ("bundled", 6, 5),
    "P1": ("partially_decoupled", 1, 1),
    "F1": ("fully_decoupled", 1, 1),
    "FX1": ("fully_decoupled_cross_driver", 1, 1),
    "FX6": ("fully_decoupled_cross_driver", 3, 2),
}

GENERATE_PLOTS = False


def main():
    for technology in TECHNOLOGIES:
        for experiment, (method, n_epr, n_wtir) in EXPERIMENTS.items():

            config = SamplingConfig(
                technology=technology,
                method=method,
                n_epr=n_epr,
                n_wtir=n_wtir,
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