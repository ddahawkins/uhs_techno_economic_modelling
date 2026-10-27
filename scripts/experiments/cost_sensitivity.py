"""Broad factor-based sensitivity screening for UHS techno-economic cost rates.

This experiment is deliberately separate from ``linearisation_sensitivity.py``.
The linearisation experiment varies the CEM-facing design quantities E, EPR and
WTIR and asks how transferable the derived unit-cost coefficients are across the
CEM decision space. This script instead holds those reference capacities fixed
and varies exogenous engineering/economic *factors* around the reference design.

A factor may comprise one model parameter or a physically correlated bundle of
parameters. The ordinary screening registry is restricted to factors for which a
low/default/high interpretation is meaningful. Reservoir depth and three coupled pressure-coordinate screens are retained:
absolute pressure level, operating pressure-ratio headroom, and (DGF only)
abandonment fraction. Every pressure screen preserves pressure ordering.

Documented reservoir configurations are evaluated separately from the OAT screen.
They are loaded from ``RESERVOIR_SAMPLES_CSV`` and applied as coupled site cases
(e.g. depth + minimum pressure + maximum pressure + abandonment pressure). This
preserves observed/site-specific combinations without forcing them into an
artificial low/default/high ordering.

Each enabled ``SensitivitySpec`` defines a low and high factor case around an
explicit reference/default case. For every applicable technology the ordinary
screen therefore evaluates:

    low -> default -> high

The default case is evaluated once per technology and reused across all
sensitivities. The principal screening metric is the change in CEM-ready total
capital requirement (TCR = CAPEX + ABEX) rate for the three independently
investable UHS components:

* injection [EUR/kW]
* storage [EUR/kWh]
* withdrawal [EUR/kW]

The manuscript-style figure is a three-panel horizontal range plot. Each factor
occupies one row. Within each row, technologies are vertically jittered. A
horizontal segment joins the low and high factor outcomes; marker shape denotes
the low/high *factor case*, not the direction of the resulting cost response.
The reference case is x = 0 by construction.

The broad ranges are screening ranges, not probability distributions. The
``range_basis``, ``source`` and ``note`` fields distinguish literature-derived
values from literature-anchored or deliberately broad engineering screens.

Reference E/EPR/WTIR values are stored below for consistency with the paper's
assumption table, but their low/high ranges are *not* executed in this script;
those CEM-facing quantities belong to ``linearisation_sensitivity.py``.

Outputs
-------
outputs/cost_sensitivity/
    sensitivity_results.csv
    sensitivity_specification.csv
    reservoir_sample_results.csv
    figures/
        tcr_sensitivity_ranges.pdf
        reservoir_sample_tcr_rates.pdf

Run from the repository root, e.g.::

    pixi run python scripts/experiments/cost_sensitivity.py

Notes
-----
* The paper reference case is explicit in this script. Reference/default values
  in the sensitivity registry are applied on top of the package's constructor
  defaults before either the baseline or a sensitivity case is evaluated. This
  makes the experiment reproducible even while the package defaults are being
  refactored.
* Changing reservoir depth, either in the OAT screen or a reservoir-sample
  case, recalculates reservoir temperature using the same screening-level
  geothermal relationship used by the design assumptions.
* LRC minimum storage pressure is coupled to the pipeline-pressure factor rather
  than to LRC depth because the current assumption treats it as a network-
  interface requirement rather than a geomechanical depth constraint.
* Compressor efficiency/design-margin sensitivities are intentionally not
  implemented because those parameters remain internal defaults of
  ``CompressionInput`` rather than exposed project assumptions.
* Only PDF figures are generated.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, fields, replace
from pathlib import Path
import csv
import math
from typing import Any, Callable, Literal, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from uhs_costs.design.aquifer import (
    ASSUMPTIONS as AQUIFER_DESIGN_ASSUMPTIONS,
    construct_project as construct_aquifer_project,
)
from uhs_costs.design.depleted_gas_field import (
    ASSUMPTIONS as DGF_DESIGN_ASSUMPTIONS,
    construct_project as construct_dgf_project,
)
from uhs_costs.design.lined_rock_cavern import (
    ASSUMPTIONS as LRC_DESIGN_ASSUMPTIONS,
    construct_project as construct_lrc_project,
)
from uhs_costs.design.salt_cavern import (
    ASSUMPTIONS as SALT_DESIGN_ASSUMPTIONS,
    construct_project as construct_salt_cavern_project,
)
from uhs_costs.design.default_design_assumptions import (
    calculate_temperature_from_depth_k,
)

from uhs_costs.cost_model.aquifer import (
    calculate_cost_components as calculate_aquifer_cost_components,
)
from uhs_costs.cost_model.depleted_gas_field import (
    calculate_depleted_gas_field_cost_components,
)
from uhs_costs.cost_model.lined_rock_cavern import (
    calculate_lined_rock_cavern_cost_components,
)
from uhs_costs.cost_model.salt_cavern import (
    calculate_salt_cavern_cost_components,
)
from uhs_costs.cost_model.default_cost_assumptions import (
    construct_hystories_cost_assumptions,
    construct_lined_rock_cavern_cost_assumptions,
)


# =============================================================================
# User configuration
# =============================================================================

OUTPUT_DIR = Path("outputs/cost_sensitivity")
OUTPUT_RESULTS_CSV = OUTPUT_DIR / "sensitivity_results.csv"
OUTPUT_SPEC_CSV = OUTPUT_DIR / "sensitivity_specification.csv"
OUTPUT_RESERVOIR_SAMPLE_RESULTS_CSV = OUTPUT_DIR / "reservoir_sample_results.csv"
FIGURE_DIR = OUTPUT_DIR / "figures"

# Empirical/site-specific reservoir configurations. The CSV is intentionally
# separate from the low/default/high sensitivity registry. Add future SC/LRC
# rows by using storage_class=SC or storage_class=LRC and populating whichever
# of the recognised reservoir columns are applicable.
RESERVOIR_SAMPLES_CSV = Path("data/resources/porous_media_reservoir_pressure_profiles.csv")
RUN_RESERVOIR_SAMPLE_SWEEP = True
RUN_RESERVOIR_STORAGE_CLASSES: tuple[str, ...] | None = None
RUN_RESERVOIR_SITES: tuple[str, ...] | None = None
RESERVOIR_SAMPLE_SORT: Literal["input", "site", "storage_tcr"] = "input"

RUN_TECHNOLOGIES = (
    "salt_cavern",
    "depleted_gas_field",
    "aquifer",
    "lined_rock_cavern",
)

# None -> run every enabled spec. Otherwise provide sensitivity keys.
RUN_SENSITIVITIES: tuple[str, ...] | None = None

# Broad screening should normally continue if one deliberately extreme case
# turns out to be infeasible. Failed cases are written to the CSV with the
# exception message and omitted from the figure.
FAIL_FAST = True

# Figure layout / behaviour.
SORT_WITHIN_CATEGORY_BY_MAX_IMPACT = True
SHOW_RANGE_IN_YLABEL = False
PLOT_X_LIMIT_PERCENT: float | None = None  # e.g. 100.0 to force +/-100%

# Only show sensitivities that change at least one component TCR rate
# by this amount in any applicable technology.
PLOT_MATERIALITY_THRESHOLD_PERCENT: float | None = 10.0

# Paper-wide colour convention requested by the author: sample plasma from 0
# through 0.9. Technology colours are therefore consistent within this figure.
PLASMA_MIN = 0.0
PLASMA_MAX = 0.9

# Low/high parameter-setting marker convention. Note that the low parameter
# value can still increase cost and the high parameter value can reduce it.
LOW_MARKER = "<"
HIGH_MARKER = ">"


# =============================================================================
# Technology/reference-facility configuration
# =============================================================================


@dataclass(frozen=True)
class ReferenceFacilityRange:
    """Low/reference/high CEM-facing design values retained for traceability."""

    storage_capacity_gwh_lhv: tuple[float, float, float]
    epr_days: tuple[float, float, float]
    wtir: tuple[float, float, float]


REFERENCE_FACILITY_RANGES: dict[str, ReferenceFacilityRange] = {
    # These ranges are documented here for consistency with the paper table.
    # They are not executed by this broad exogenous-cost screen; E/EPR/WTIR are
    # explored in linearisation_sensitivity.py.
    "salt_cavern": ReferenceFacilityRange(
        storage_capacity_gwh_lhv=(500.0, 750.0, 1000.0),
        epr_days=(6.0, 12.0, 18.0),
        wtir=(1.0, 2.0, 3.0),
    ),
    "depleted_gas_field": ReferenceFacilityRange(
        storage_capacity_gwh_lhv=(960.0, 1700.0, 2900.0),
        epr_days=(33.0, 66.0, 100.0),
        wtir=(1.0, 2.0, 3.0),
    ),
    "aquifer": ReferenceFacilityRange(
        storage_capacity_gwh_lhv=(960.0, 1700.0, 2900.0),
        epr_days=(33.0, 66.0, 100.0),
        wtir=(1.0, 2.0, 3.0),
    ),
    "lined_rock_cavern": ReferenceFacilityRange(
        storage_capacity_gwh_lhv=(10.0, 20.0, 20.0),
        epr_days=(5.0, 7.5, 10.0),
        wtir=(1.0, 2.0, 3.0),
    ),
}


@dataclass(frozen=True)
class TechnologySpec:
    technology: str
    label: str
    case_name: str
    project_constructor: Callable[..., Any]
    cost_function: Callable[..., Any]
    design_assumptions: Any

    default_storage_capacity_gwh_lhv: float
    default_epr_days: float
    default_wtir: float = 2.0


TECHNOLOGIES: tuple[TechnologySpec, ...] = (
    TechnologySpec(
        technology="salt_cavern",
        label="Salt cavern",
        case_name="salt_cavern",
        project_constructor=construct_salt_cavern_project,
        cost_function=calculate_salt_cavern_cost_components,
        design_assumptions=SALT_DESIGN_ASSUMPTIONS,
        default_storage_capacity_gwh_lhv=REFERENCE_FACILITY_RANGES[
            "salt_cavern"
        ].storage_capacity_gwh_lhv[1],
        default_epr_days=REFERENCE_FACILITY_RANGES["salt_cavern"].epr_days[1],
        default_wtir=REFERENCE_FACILITY_RANGES["salt_cavern"].wtir[1],
    ),
    TechnologySpec(
        technology="depleted_gas_field",
        label="Depleted gas field",
        case_name="depleted_gas_field",
        project_constructor=construct_dgf_project,
        cost_function=calculate_depleted_gas_field_cost_components,
        design_assumptions=DGF_DESIGN_ASSUMPTIONS,
        default_storage_capacity_gwh_lhv=REFERENCE_FACILITY_RANGES[
            "depleted_gas_field"
        ].storage_capacity_gwh_lhv[1],
        default_epr_days=REFERENCE_FACILITY_RANGES[
            "depleted_gas_field"
        ].epr_days[1],
        default_wtir=REFERENCE_FACILITY_RANGES["depleted_gas_field"].wtir[1],
    ),
    TechnologySpec(
        technology="aquifer",
        label="Aquifer",
        case_name="aquifer",
        project_constructor=construct_aquifer_project,
        cost_function=calculate_aquifer_cost_components,
        design_assumptions=AQUIFER_DESIGN_ASSUMPTIONS,
        default_storage_capacity_gwh_lhv=REFERENCE_FACILITY_RANGES[
            "aquifer"
        ].storage_capacity_gwh_lhv[1],
        default_epr_days=REFERENCE_FACILITY_RANGES["aquifer"].epr_days[1],
        default_wtir=REFERENCE_FACILITY_RANGES["aquifer"].wtir[1],
    ),
    TechnologySpec(
        technology="lined_rock_cavern",
        label="Lined rock cavern",
        case_name="lined_rock_cavern",
        project_constructor=construct_lrc_project,
        cost_function=calculate_lined_rock_cavern_cost_components,
        design_assumptions=LRC_DESIGN_ASSUMPTIONS,
        default_storage_capacity_gwh_lhv=REFERENCE_FACILITY_RANGES[
            "lined_rock_cavern"
        ].storage_capacity_gwh_lhv[1],
        default_epr_days=REFERENCE_FACILITY_RANGES[
            "lined_rock_cavern"
        ].epr_days[1],
        default_wtir=REFERENCE_FACILITY_RANGES["lined_rock_cavern"].wtir[1],
    ),
)

ALL_TECHS = tuple(spec.technology for spec in TECHNOLOGIES)
POROUS_TECHS = ("depleted_gas_field", "aquifer")

# External reservoir-sample class labels -> internal technology keys. This is
# deliberately extensible so the same CSV-driven sweep can later include salt
# caverns and lined-rock caverns without changing the execution logic.
RESERVOIR_STORAGE_CLASS_TO_TECHNOLOGY: dict[str, str] = {
    "DGF": "depleted_gas_field",
    "SA": "aquifer",
    "SC": "salt_cavern",
    "LRC": "lined_rock_cavern",
}

# CSV column -> (GeneralDesignAssumptions field, input scale). Blank cells are
# simply ignored, which lets different storage classes provide different subsets.
RESERVOIR_SAMPLE_PARAMETER_MAP: dict[str, tuple[str, float]] = {
    "depth_m_representative": ("reference_reservoir_depth_m", 1.0),
    "model_abandonment_pressure_bar": ("abandonment_pressure_pa", 1e5),
    "min_storage_pressure_bar": ("minimum_operating_pressure_pa", 1e5),
    "max_storage_pressure_bar": ("maximum_operating_pressure_pa", 1e5),
}


# =============================================================================
# Sensitivity specification
# =============================================================================

OverrideScope = Literal[
    "design_general",
    "design_technology",
    "cost_hystories",
    "cost_lrc",
]
CaseLevel = Literal["low", "default", "high"]
RangeValue = float | int | Mapping[str, float | int]


@dataclass(frozen=True)
class SensitivityOverride:
    """One model parameter changed as part of a sensitivity factor.

    ``low``, ``default`` and ``high`` are given in human-readable ``unit``.
    ``input_scale`` converts them to the model's internal units (e.g. bar -> Pa).

    ``technologies=None`` means the override applies to every technology listed
    in the parent ``SensitivitySpec``. A subset can be supplied where one factor
    has technology-specific constituent parameters.
    """

    override_scope: OverrideScope
    parameter: str
    low: RangeValue
    default: RangeValue
    high: RangeValue
    unit: str
    technologies: tuple[str, ...] | None = None
    input_scale: float = 1.0
    integer: bool = False


@dataclass(frozen=True)
class SensitivitySpec:
    """One low/reference/high uncertain factor.

    A factor can contain several correlated ``SensitivityOverride`` objects.
    All constituent overrides are applied atomically before rebuilding the
    project, so low/high cases can represent coherent site or process archetypes.
    """

    key: str
    label: str
    category: str
    technologies: tuple[str, ...]
    overrides: tuple[SensitivityOverride, ...]

    range_basis: str = "screening"
    source: str = ""
    note: str = ""
    enabled: bool = True


@dataclass(frozen=True)
class ReferenceOnlyOverride:
    """Reference value required by the paper table but not sensitivity-screened."""

    override_scope: OverrideScope
    parameter: str
    value: RangeValue
    technologies: tuple[str, ...]
    unit: str
    input_scale: float = 1.0
    integer: bool = False
    note: str = ""


CATEGORY_ORDER = (
    "Geological / storage design",
    "Wellfield / cavern design",
    "Surface / process",
    "Commodity / material economics",
    "Project-cost uncertainty",
    "LRC cost basis",
)


# Reference-only values from the paper table. They are applied to every baseline
# and sensitivity case, but are not plotted as uncertain factors because no
# low/high range is currently being claimed for them.
REFERENCE_ONLY_OVERRIDES: tuple[ReferenceOnlyOverride, ...] = (
    # Absolute reference pressure coordinates (Pa). The factors below modify
    # these coherently rather than claiming independent low/high pressures.
    ReferenceOnlyOverride("design_general", "minimum_operating_pressure_pa",
                          {"salt_cavern": 72.0, "depleted_gas_field": 77.83,
                           "aquifer": 118.59, "lined_rock_cavern": 55.0},
                          ALL_TECHS, "bar", input_scale=1e5),
    ReferenceOnlyOverride("design_general", "maximum_operating_pressure_pa",
                          {"salt_cavern": 180.0, "depleted_gas_field": 151.62,
                           "aquifer": 143.67, "lined_rock_cavern": 200.0},
                          ALL_TECHS, "bar", input_scale=1e5),
    ReferenceOnlyOverride("design_general", "abandonment_pressure_pa", 46.04,
                          ("depleted_gas_field",), "bar", input_scale=1e5),

    ReferenceOnlyOverride(
        override_scope="design_general",
        parameter="recovery_factor",
        value=0.9983,
        technologies=("salt_cavern",),
        unit="fraction",
        note=(
            "Salt-cavern recoverability is fixed at 0.9983 in the present paper "
            "configuration; no separate broad sensitivity is claimed here."
        ),
    ),
    ReferenceOnlyOverride(
        override_scope="design_general",
        parameter="recovery_factor",
        value=1.0,
        technologies=("lined_rock_cavern",),
        unit="fraction",
        note=(
            "The engineered LRC working-gas inventory is treated as fully "
            "recoverable in the present formulation."
        ),
    ),
)


# The registry below is the executable version of the paper sensitivity table.
# Literature-backed values are distinguished from deliberately broad screening
# ranges. Correlated values are bundled where varying them independently would
# create physically awkward combinations.
SENSITIVITY_SPECS: tuple[SensitivitySpec, ...] = (
    # -------------------------------------------------------------------------
    # Geological / storage design
    # -------------------------------------------------------------------------
    SensitivitySpec(
        key="reservoir_depth",
        label="Reservoir depth",
        category="Geological / storage design",
        technologies=ALL_TECHS,
        overrides=(
            SensitivityOverride(
                override_scope="design_general",
                parameter="reference_reservoir_depth_m",
                low={
                    "salt_cavern": 500.0,
                    "depleted_gas_field": 800.0,
                    "aquifer": 800.0,
                    "lined_rock_cavern": 100.0,
                },
                default={
                    "salt_cavern": 1000.0,
                    "depleted_gas_field": 1957.0,
                    "aquifer": 1162.0,
                    "lined_rock_cavern": 150.0,
                },
                high={
                    "salt_cavern": 1500.0,
                    "depleted_gas_field": 3000.0,
                    "aquifer": 3000.0,
                    "lined_rock_cavern": 200.0,
                },
                unit="m",
            ),
        ),
        range_basis="literature + screening",
        source="HyStories D7.1-1 porous-media/salt conceptual design; LRC literature",
        note=(
            "OAT diagnostic only: reservoir depth is varied while operating "
            "pressures remain at the explicit paper reference values. Changing "
            "depth also recalculates reservoir temperature using the model's "
            "geothermal relationship. Site-specific coupled depth/pressure cases "
            "are evaluated separately by the reservoir-sample sweep."
        ),
    ),
    # Pressure screens are derived in build_case_configuration, not independent
    # absolute pressure overrides. The factor is recorded as a non-design
    # coordinate in the case definition and specification output via a marker
    # override (see apply_pressure_coordinates).
    SensitivitySpec(
        key="reservoir_pressure_level",
        label="Reservoir pressure level (coupled)",
        category="Geological / storage design",
        technologies=ALL_TECHS,
        overrides=(SensitivityOverride(
            override_scope="design_general", parameter="minimum_operating_pressure_pa",
            low=0.5, default=1.0, high=1.5, unit="pressure scale ×",
        ),),
        range_basis="symmetric factor screen",
        source="Study-defined 0.5–1.5 multiplier",
        note=("Scale all physical pressure coordinates by 0.5, 1.0, 1.5. "
              "The operating ratio and DGF abandonment fraction are constant; "
              "the absolute pressure difference changes. For LRC the interface "
              "pipeline pressure is also scaled to preserve the model's equality "
              "between minimum cavern and pipeline pressure. Depth is fixed."),
    ),
    SensitivitySpec(
        key="reservoir_pressure_ratio",
        label="Operating pressure-ratio headroom",
        category="Geological / storage design",
        technologies=ALL_TECHS,
        overrides=(SensitivityOverride(
            override_scope="design_general", parameter="maximum_operating_pressure_pa",
            low=0.5, default=1.0, high=1.5, unit="ratio excess scale ×",
        ),),
        range_basis="symmetric factor screen",
        source="Study-defined 0.5–1.5 multiplier",
        note=("Scale R-1, where R=Pmax/Pmin, by 0.5, 1.0, 1.5; "
              "hold Pmin, Pab and depth fixed. This is NOT 0.5 × R: "
              "scaling R directly can imply Pmax<Pmin for narrow aquifer windows."),
    ),
    SensitivitySpec(
        key="reservoir_abandonment_fraction",
        label="DGF abandonment-pressure fraction",
        category="Geological / storage design",
        technologies=("depleted_gas_field",),
        overrides=(SensitivityOverride(
            override_scope="design_general", parameter="abandonment_pressure_pa",
            low=0.5, default=1.0, high=1.5, unit="Pab/Pmin scale ×",
        ),),
        range_basis="symmetric factor screen",
        source="Study-defined 0.5–1.5 multiplier",
        note=("Scale A=Pab/Pmin by 0.5, 1.0, 1.5 at fixed "
              "Pmin, operating ratio, and depth; enforce 0<A<1."),
    ),
    SensitivitySpec(
        key="recoverability_factor",
        label="Recoverability factor",
        category="Geological / storage design",
        technologies=POROUS_TECHS,
        overrides=(
            SensitivityOverride(
                override_scope="design_general",
                parameter="recovery_factor",
                technologies=("depleted_gas_field",),
                low=0.67,
                default=0.80,
                high=0.85,
                unit="fraction",
            ),
            SensitivityOverride(
                override_scope="design_general",
                parameter="recovery_factor",
                technologies=("aquifer",),
                low=0.75,
                default=0.90,
                high=0.96,
                unit="fraction",
            ),
        ),
        range_basis="literature",
        source="Lardhi et al. (DGF); Zhao et al. (saline aquifer, mature-cycle values)",
        note=(
            "Recoverability is a storage-sizing assumption: it increases the "
            "physical development required to provide one unit of recoverable "
            "working-gas capacity. It is not applied as a recurring cycle loss "
            "in the CEM state-of-charge balance. DGF values represent the range "
            "reported for mature depleted-field operation; aquifer values use "
            "the mature/long-run withdrawal-efficiency range reported by Zhao "
            "et al., rather than first-cycle values."
        ),
    ),

    # -------------------------------------------------------------------------
    # Wellfield / cavern design
    # -------------------------------------------------------------------------
    SensitivitySpec(
        key="drilling_complexity",
        label="Drilling complexity index",
        category="Wellfield / cavern design",
        technologies=ALL_TECHS,
        overrides=(
            SensitivityOverride(
                override_scope="design_general",
                parameter="drilling_complexity_index",
                low=1.0,
                default=1.0,
                high=2.0,
                unit="index",
            ),
        ),
        range_basis="literature",
        source="HyStories D7.2-1 drilling-complexity factor",
        note=(
            "HyStories uses a drilling-complexity index spanning approximately "
            "1-2. The reference design sits at the lower end, so the low case is "
            "identical to the baseline and the sensitivity is intentionally "
            "one-sided toward more difficult drilling conditions."
        ),
    ),
    SensitivitySpec(
        key="salt_cavern_module_design",
        label="Salt cavern module design",
        category="Wellfield / cavern design",
        technologies=("salt_cavern",),
        overrides=(
            SensitivityOverride(
                override_scope="design_technology",
                parameter="working_gas_volume_per_cavern_sm3",
                low=15.625,
                default=31.25,
                high=62.5,
                unit="million Sm3/cavern",
                input_scale=1e6,
            ),
            SensitivityOverride(
                override_scope="design_technology",
                parameter="maximum_withdrawal_flow_per_cavern_million_sm3_per_day",
                low=1.5,
                default=2.8,
                high=5.9,
                unit="million Sm3/day/cavern",
            ),
        ),
        range_basis="literature",
        source="HyStories D7.1-1 conceptual salt-cavern cases",
        note=(
            "Working-gas capacity per cavern and withdrawal capability are varied "
            "together because the HyStories conceptual cases define them as "
            "paired cavern designs. Treating them independently would create "
            "small-cavern/high-deliverability or large-cavern/low-deliverability "
            "combinations that are not the source cases. Values are ordered by "
            "physical module size/deliverability rather than by HyStories' "
            "low/mid/high CAPEX labels."
        ),
    ),
    
    SensitivitySpec(
        key="aquifer_min_saturation_design",
        label="Aquifer minimum gas saturation",
        category="Wellfield / cavern design",
        technologies=("aquifer",),
        overrides=(
            SensitivityOverride(
                override_scope="design_general",
                parameter="minimum_gas_saturation",
                low=0.50,
                default=0.60,
                high=0.70,
                unit="fraction",
            ),
        ),
        range_basis="literature-anchored screening",
        source=(
            "Heinemann et al. (2021), "
            "https://doi.org/10.1016/j.ijhydene.2021.09.174; "
            "Döpp et al. (2025), "
            "https://doi.org/10.1007/s40948-025-00982-y"
        ),
        note=(
            "The minimum gas saturation represents the effective hydrogen-filled "
            "fraction of the developed aquifer pore volume at minimum operating "
            "inventory. The reference value of 0.60 is a screening assumption "
            "informed by reservoir-scale simulations, rather than a directly "
            "measured minimum saturation. Döpp et al. (2025) report near-well "
            "gas saturations of approximately 0.50-0.60 in their Ketzin model, "
            "with a base-case value of 0.64 and values around 0.61-0.70 in "
            "several geological sensitivity cases. Their low-permeability case "
            "reaches only 0.44, illustrating less favourable conditions outside "
            "the intended screening envelope. Heinemann et al. (2021) impose "
            "a gas-saturation shutdown threshold of 0.70 beneath the well "
            "perforations to prevent water breakthrough. This is an operational "
            "criterion, not a measured reservoir-wide minimum. The 0.50-0.70 "
            "range therefore explores alternative minimum-inventory states "
            "for selected, hydraulically suitable aquifers. Applying these "
            "local simulated saturations to the entire effective developed "
            "pore volume is an explicit first-order modelling approximation. "
            "Higher minimum saturation increases the calculated hydrogen "
            "cushion requirement, all else equal."
        ),
    ),
    SensitivitySpec(
        key="aquifer_max_saturation_design",
        label="Aquifer maximum gas saturation",
        category="Wellfield / cavern design",
        technologies=("aquifer",),
        overrides=(
            SensitivityOverride(
                override_scope="design_general",
                parameter="maximum_gas_saturation",
                low=0.75,
                default=0.80,
                high=0.90,
                unit="fraction",
            ),
        ),
        range_basis="experimental literature + screening",
        source=(
            "Hydrogen relative-permeability measurements in Bentheimer sandstone, "
            "https://doi.org/10.1016/j.ijhydene.2023.07.270; "
            "Thaysen et al. (2024), "
            "https://doi.org/10.1016/j.est.2024.112916; "
            "Hydrogen pore-scale imaging study (2025), "
            "https://doi.org/10.1016/j.ijhydene.2025.02.066; "
            "Ren et al. (2026), "
            "https://doi.org/10.1016/j.enconman.2026.121195"
        ),
        note=(
            "The maximum gas saturation represents the effective hydrogen-filled "
            "fraction of developed aquifer pore volume at maximum inventory, "
            "allowing for brine remaining after hydrogen drainage. Experimental "
            "hydrogen-brine measurements in Bentheimer sandstone report an "
            "irreducible water saturation of approximately 0.24, corresponding "
            "to a gas saturation of 0.76. A separate Bentheimer pore-scale "
            "imaging study reports an average hydrogen saturation of 0.82 "
            "after drainage. Thaysen et al. (2024) report irreducible water "
            "saturations of 0.126-0.140 in Clashach sandstone at a specified "
            "capillary pressure, corresponding to gas saturations of "
            "approximately 0.86-0.87. These results motivate the 0.80 "
            "reference for favourable sandstone formations. The 0.75 lower "
            "case is close to the Bentheimer irreducible-water endpoint, "
            "while 0.85 is a screening bound based on"
            "the measured Clashach values, not a universal geological limit. "
            "Ren et al. (2026) additionally report geological input values "
            "for irreducible brine saturation spanning 0.20-0.65, demonstrating "
            "substantial site variability; these are not direct hydrogen "
            "drainage measurements. The assumed endpoint represents an "
            "effective reservoir-wide saturation rather than a resolved "
            "spatial saturation distribution. Lower maximum saturation "
            "increases the required developed pore volume and hydrogen "
            "cushion inventory for a fixed working capacity."
        ),
    ),
    SensitivitySpec(
        key="porous_wellfield_design",
        label="Porous-media wellfield design",
        category="Wellfield / cavern design",
        technologies=POROUS_TECHS,
        overrides=(
            SensitivityOverride(
                override_scope="design_technology",
                parameter="number_production_well_heads",
                low=12,
                default=24,
                high=36,
                unit="wells",
                integer=True,
            ),
            SensitivityOverride(
                override_scope="design_technology",
                parameter="number_observation_well_heads",
                low=1,
                default=6,
                high=12,
                unit="wells",
                integer=True,
            ),
        ),
        range_basis="literature",
        source="HyStories D7.1-1 conceptual porous-media wellfield cases",
        note=(
            "Production and observation wells are varied as a coupled wellfield "
            "design using the low/reference/high conceptual cases reported by "
            "HyStories. This is preferable to independently combining well counts "
            "which were not defined as separate site archetypes in the source."
        ),
    ),
    SensitivitySpec(
        key="salt_fresh_water_pipeline",
        label="Salt fresh-water pipeline length",
        category="Wellfield / cavern design",
        technologies=("salt_cavern",),
        overrides=(
            SensitivityOverride(
                override_scope="design_technology",
                parameter="fresh_water_pipeline_length_km",
                low=5.0,
                default=15.0,
                high=30.0,
                unit="km",
            ),
        ),
        range_basis="screening",
        source="HyStories reference design for the default; bounds are screening",
        note=(
            "The 15 km reference is retained from the HyStories-based salt-cavern "
            "design. The 5-30 km bounds are a site-layout screen intended to test "
            "sensitivity to access to fresh-water infrastructure rather than a "
            "claimed empirical distribution."
        ),
    ),
    SensitivitySpec(
        key="salt_brine_pipeline",
        label="Salt brine-disposal pipeline length",
        category="Wellfield / cavern design",
        technologies=("salt_cavern",),
        overrides=(
            SensitivityOverride(
                override_scope="design_technology",
                parameter="brine_disposal_pipeline_length_km",
                low=10.0,
                default=30.0,
                high=60.0,
                unit="km",
            ),
        ),
        range_basis="screening",
        source="HyStories reference design for the default; bounds are screening",
        note=(
            "The reference brine-disposal distance is inherited from the "
            "HyStories-based design. Low/high cases deliberately span materially "
            "shorter and longer disposal routes to expose site-location effects. "
        ),
    ),
    SensitivitySpec(
        key="salt_debrining_rate",
        label="Salt debrining flow rate",
        category="Wellfield / cavern design",
        technologies=("salt_cavern",),
        overrides=(
            SensitivityOverride(
                override_scope="design_technology",
                parameter="debrining_flowrate_per_cavern_m3_per_hour",
                low=100.0,
                default=200.0,
                high=300.0,
                unit="m3/h/cavern",
            ),
        ),
        range_basis="screening",
        source="HyStories-based reference design; low/high are engineering screening",
        note=(
            "Debrining rate affects conversion duration and associated salt-cavern "
            "development costs. The reference is the current HyStories-derived "
            "assumption; the surrounding bounds are a broad engineering screen. "
        ),
    ),
    SensitivitySpec(
        key="salt_leaching_pumps",
        label="Salt leaching pumps",
        category="Wellfield / cavern design",
        technologies=("salt_cavern",),
        overrides=(
            SensitivityOverride(
                override_scope="design_technology",
                parameter="number_leaching_pumps",
                low=2,
                default=4,
                high=8,
                unit="pumps",
                integer=True,
            ),
        ),
        range_basis="screening",
        source="HyStories-based reference design; low/high are engineering screening",
        note=(
            "The pump count controls the scale/cost of the leaching system. The "
            "four-pump reference follows the current implementation, while 2 and "
            "8 pumps are deliberately broad screening cases rather than a "
            "literature-derived uncertainty interval."
        ),
    ),
    SensitivitySpec(
        key="lrc_h2_per_well",
        label="LRC maximum H2 per well",
        category="Wellfield / cavern design",
        technologies=("lined_rock_cavern",),
        overrides=(
            SensitivityOverride(
                override_scope="design_technology",
                parameter="maximum_h2_mass_per_well_kg",
                low=1.5,
                default=3.0,
                high=6.0,
                unit="kt H2/well",
                input_scale=1e6,
            ),
        ),
        range_basis="screening",
        source="Current LRC engineering assumption; +/- factor-of-two screen",
        note=(
            "This parameter controls when additional LRC access wells/modules are "
            "required. The reference and bounds are retained as an engineering "
            "screen rather than claimed as a transferable literature range. "
        ),
    ),
    SensitivitySpec(
        key="lrc_cavern_radius",
        label="LRC maximum cavern radius",
        category="Wellfield / cavern design",
        technologies=("lined_rock_cavern",),
        overrides=(
            SensitivityOverride(
                override_scope="design_technology",
                parameter="maximum_cavern_radius_m",
                low=10.0,
                default=20.0,
                high=30.0,
                unit="m",
            ),
        ),
        range_basis="screening",
        source="LRC reference geometry informed by Huang et al.; bounds are screening",
        note=(
            "Radius is varied around the current 20 m engineered-cavern reference "
            "to expose geometric scaling of excavation and lining requirements. "
            "The bounds should be interpreted as screening values, not as a "
            "general geotechnical feasibility envelope. "
        ),
    ),
    SensitivitySpec(
        key="lrc_cavern_height",
        label="LRC cavern height",
        category="Wellfield / cavern design",
        technologies=("lined_rock_cavern",),
        overrides=(
            SensitivityOverride(
                override_scope="design_technology",
                parameter="cavern_height_m",
                low=50.0,
                default=100.0,
                high=150.0,
                unit="m",
            ),
        ),
        range_basis="screening",
        source="LRC reference geometry informed by Huang et al.; bounds are screening",
        note=(
            "Cavern height is varied independently as a geometry screen around the "
            "current 100 m reference. It changes excavated volume and lining area "
            "and therefore tests geometric cost scaling."
        ),
    ),
    SensitivitySpec(
        key="lrc_effective_volume_fraction",
        label="LRC effective volume fraction",
        category="Wellfield / cavern design",
        technologies=("lined_rock_cavern",),
        overrides=(
            SensitivityOverride(
                override_scope="design_technology",
                parameter="effective_volume_fraction",
                low=0.85,
                default=0.95,
                high=1.00,
                unit="fraction",
            ),
        ),
        range_basis="screening",
        source="Huang et al. for the 0.95 reference; bounds are screening",
        note=(
            "The effective-volume fraction converts excavated cavern geometry to "
            "usable gas volume after allowing for lining/geometry effects. The "
            "0.95 reference follows the current Huang-derived formulation; 0.85 "
            "and 1.0 test more conservative and idealised utilisation."
        ),
    ),
    SensitivitySpec(
        key="lrc_steel_lining_thickness",
        label="LRC steel lining thickness",
        category="Wellfield / cavern design",
        technologies=("lined_rock_cavern",),
        overrides=(
            SensitivityOverride(
                override_scope="design_technology",
                parameter="steel_lining_thickness_m",
                low=0.010,
                default=0.015,
                high=0.025,
                unit="m",
            ),
        ),
        range_basis="literature + screening",
        source="Huang et al. for the 15 mm reference; bounds are screening",
        note=(
            "The 15 mm reference follows the LRC cost/design basis adapted from "
            "Huang et al. The 10-25 mm cases test sensitivity of steel mass and "
            "lining cost to plausible design variation; the bounds are not "
            "presented as a universal code-compliant range."
        ),
    ),
    SensitivitySpec(
        key="lrc_concrete_lining_thickness",
        label="LRC concrete lining thickness",
        category="Wellfield / cavern design",
        technologies=("lined_rock_cavern",),
        overrides=(
            SensitivityOverride(
                override_scope="design_technology",
                parameter="concrete_lining_thickness_m",
                low=0.30,
                default=0.50,
                high=0.80,
                unit="m",
            ),
        ),
        range_basis="screening",
        source="Current Huang-derived LRC implementation; bounds are screening",
        note=(
            "Concrete thickness changes both material quantity and effective "
            "storage geometry. The 0.50 m reference is retained from the current "
            "LRC implementation and the surrounding range is an engineering "
            "screen."
        ),
    ),

    # -------------------------------------------------------------------------
    # Surface / process
    # -------------------------------------------------------------------------
    SensitivitySpec(
        key="pipeline_pressure",
        label="Pipeline / compressor inlet pressure",
        category="Surface / process",
        technologies=ALL_TECHS,
        overrides=(
            SensitivityOverride(
                override_scope="design_general",
                parameter="pipeline_pressure_pa",
                low=46.2,
                default=55.0,
                high=66.2,
                unit="bar",
                input_scale=1e5,
            ),
            SensitivityOverride(
                override_scope="design_general",
                parameter="minimum_operating_pressure_pa",
                technologies=("lined_rock_cavern",),
                low=46.2,
                default=55.0,
                high=66.2,
                unit="bar",
                input_scale=1e5,
            ),
        ),
        range_basis="literature",
        source="Hynetwork 2026 Transport Agreement, Annex 7 Pressure Specification",
        note=(
            "The sensitivity spans Hynetwork's envisaged long-term operating "
            "range of 46.2-66.2 bar(g), with 55 bar used as a representative "
            "mid-range reference. For LRCs the minimum storage pressure is "
            "changed with the network pressure because the present formulation "
            "treats that minimum as a delivery/interface requirement rather than "
            "a depth-controlled geomechanical limit."
        ),
    ),
    SensitivitySpec(
        key="gas_plant_distance",
        label="Gas plant to nearest wellhead distance",
        category="Surface / process",
        technologies=ALL_TECHS,
        overrides=(
            SensitivityOverride(
                override_scope="design_general",
                parameter="gas_plant_to_nearest_wellhead_distance_km",
                low=1.0,
                default=2.0,
                high=3.0,
                unit="km",
            ),
        ),
        range_basis="literature-anchored screening",
        source="HyStories D7.2-1 / Table 35 reference geometry",
        note=(
            "HyStories uses a 2 km gas-plant-to-nearest-wellhead reference. The "
            "1 and 3 km cases are a symmetric site-layout screen around that "
            "reference. In the local EPC3 implementation any baseline field-line "
            "distance already represented in EPC2 should remain subtracted to "
            "avoid double counting."
        ),
    ),
    SensitivitySpec(
        key="purification_requirement",
        label="Purification requirement",
        category="Surface / process",
        technologies=POROUS_TECHS,
        overrides=(
            SensitivityOverride(
                override_scope="design_general",
                parameter="purification_factor",
                technologies=("depleted_gas_field",),
                low=0.33,
                default=1.50,
                high=3.00,
                unit="factor",
            ),
            SensitivityOverride(
                override_scope="design_general",
                parameter="purification_hydrogen_recovery",
                technologies=("depleted_gas_field",),
                low=0.95,
                default=0.8996,
                high=0.85, 
                unit="fraction",
            ),
            SensitivityOverride(
                override_scope="design_general",
                parameter="purification_factor",
                technologies=("aquifer",),
                low=0.00,
                default=0.33,
                high=1.50,
                unit="factor",
            ),
            SensitivityOverride(
                override_scope="design_general",
                parameter="purification_hydrogen_recovery",
                technologies=("aquifer",),
                low=1.00,
                default=0.95,
                high=0.90,
                unit="fraction",
            ),
        ),
        range_basis="literature + screening",
        source="HyStories D7.2-1 purification-cost factor; Yousefi et al. for DGF PSA recovery",
        note=(
            "Low/high denote purification *burden*, so constituent numerical "
            "parameters deliberately move in opposite directions: more severe "
            "treatment increases K_purif and lowers usable-H2 recovery. For DGF, "
            "the 1.5 reference cost factor follows HyStories and 0.8996 is the "
            "recovery produced by the simplified PSA proxy under the reference "
            "Roden-like gas composition. The 0.33-3.0 K_purif range follows the "
            "HyStories discussion. For saline aquifers HyStories indicates a "
            "smaller cleaning burden but does not provide an aquifer-specific "
            "coefficient; 0/0.33/1.5 therefore represents no cleaning, the "
            "HyStories lower-bound treatment, and treatment comparable to the "
            "generic porous-media reference. Aquifer H2 recoveries 1.00/0.95/0.90 "
            "are explicitly screening assumptions for a cleaner feed stream. "
        ),
    ),

    # -------------------------------------------------------------------------
    # Commodity / material economics
    # -------------------------------------------------------------------------
    SensitivitySpec(
        key="hydrogen_price",
        label="Construction / cushion H2 price",
        category="Commodity / material economics",
        technologies=ALL_TECHS,
        overrides=(
            SensitivityOverride(
                override_scope="cost_hystories",
                parameter="hydrogen_cost_eur_per_kg",
                low=4.12,
                default=6.71,
                high=7.42,
                unit="EUR/kg H2",
            ),
        ),
        range_basis="economic scenario",
        source="European Hydrogen Observatory, 2024 hydrogen production costs",
        note=(
            "The three values are pathway-based cost scenarios rather than a "
            "probability interval: approximately 4.12 EUR/kg for SMR+CCS, 6.71 "
            "EUR/kg for electrolysis directly connected to renewables (used as "
            "the reference), and 7.42 EUR/kg for grid-connected electrolysis. "
            "The value is used to cost hydrogen tied up in cushion/commissioning "
            "inventory; it is not a market-clearing hydrogen price forecast."
        ),
    ),
    SensitivitySpec(
        key="construction_electricity_price",
        label="Construction electricity price",
        category="Commodity / material economics",
        technologies=ALL_TECHS,
        overrides=(
            SensitivityOverride(
                override_scope="cost_hystories",
                parameter="cost_of_electricity_eur_per_mwh",
                low=60.0,
                default=80.0,
                high=100.0,
                unit="EUR/MWh",
            ),
        ),
        range_basis="literature-anchored screening",
        source="IEA quarterly average wholesale electricity prices for Europe, 2025 context",
        note=(
            "The 80 EUR/MWh reference and 60-100 EUR/MWh range are intended to "
            "represent contemporary European wholesale-electricity conditions "
            "for construction/commissioning energy and retained ancillary OPEX "
            "terms. They are not a site-specific industrial power tariff. "
        ),
    ),
    SensitivitySpec(
        key="material_factor_injection",
        label="Injection material cost factor",
        category="Commodity / material economics",
        technologies=ALL_TECHS,
        overrides=(
            SensitivityOverride(
                override_scope="cost_hystories",
                parameter="material_cost_factor_injection",
                low=1.0,
                default=1.5,
                high=3.5,
                unit="factor",
            ),
        ),
        range_basis="literature + reference adjustment",
        source="HyStories D7.2-1 material-cost-factor basis",
        note=(
            "HyStories uses a factor of about 1 for carbon steel and about 3.5 "
            "for 316L stainless steel. The intermediate reference factor of 1.5 "
            "is an explicit study assumption intended to avoid treating the "
            "original carbon-steel cost basis as fully representative after cost "
            "escalation/material requirements; it is not itself a HyStories "
            "material category."
        ),
    ),
    SensitivitySpec(
        key="material_factor_withdrawal",
        label="Withdrawal material cost factor",
        category="Commodity / material economics",
        technologies=ALL_TECHS,
        overrides=(
            SensitivityOverride(
                override_scope="cost_hystories",
                parameter="material_cost_factor_withdrawal",
                low=1.0,
                default=1.5,
                high=3.5,
                unit="factor",
            ),
        ),
        range_basis="literature + reference adjustment",
        source="HyStories D7.2-1 material-cost-factor basis",
        note=(
            "As for injection equipment, the low and high cases correspond to "
            "the approximate carbon-steel and 316L material factors in HyStories. "
            "The 1.5 reference is a study-specific intermediate uplift rather "
            "than a separate material class."
        ),
    ),

    # -------------------------------------------------------------------------
    # Project-cost uncertainty
    # -------------------------------------------------------------------------
    SensitivitySpec(
        key="surface_contingency",
        label="Surface contingency",
        category="Project-cost uncertainty",
        technologies=ALL_TECHS,
        overrides=(
            SensitivityOverride(
                override_scope="cost_hystories",
                parameter="surface_contingency_fraction",
                low=0.10,
                default=0.20,
                high=0.30,
                unit="fraction",
            ),
        ),
        range_basis="literature + screening",
        source="HyStories D7.2-1 reference contingency treatment",
        note=(
            "A 20% surface contingency is retained as the reference project "
            "allowance. The 10-30% range tests how an explicit uncertainty uplift "
            "propagates through the final component TCR rather than representing "
            "independent physical uncertainty in the storage technology."
        ),
    ),
    SensitivitySpec(
        key="subsurface_contingency",
        label="Subsurface contingency",
        category="Project-cost uncertainty",
        technologies=ALL_TECHS,
        overrides=(
            SensitivityOverride(
                override_scope="cost_hystories",
                parameter="subsurface_contingency_fraction",
                low=0.10,
                default=0.20,
                high=0.30,
                unit="fraction",
            ),
        ),
        range_basis="literature + screening",
        source="HyStories D7.2-1 reference contingency treatment",
        note=(
            "The subsurface contingency screen mirrors the surface treatment and "
            "is retained mainly to quantify how project-level allowances are "
            "diluted or amplified in the final storage-capacity TCR coefficient."
        ),
    ),
    SensitivitySpec(
        key="surface_abex",
        label="Surface ABEX fraction",
        category="Project-cost uncertainty",
        technologies=ALL_TECHS,
        overrides=(
            SensitivityOverride(
                override_scope="cost_hystories",
                parameter="surface_abex_fraction",
                low=0.10,
                default=0.20,
                high=0.30,
                unit="fraction",
            ),
        ),
        range_basis="literature",
        source="HyStories D7.2-1 ABEX range/reference treatment",
        note=(
            "Surface abandonment expenditure is represented as a fraction of the "
            "eligible surface cost base. The 10-30% range around the 20% reference "
            "is retained from the HyStories cost framework and tests its impact on "
            "the CEM-ready injection/withdrawal TCR coefficients."
        ),
    ),
    SensitivitySpec(
        key="subsurface_abex",
        label="Subsurface ABEX fraction",
        category="Project-cost uncertainty",
        technologies=ALL_TECHS,
        overrides=(
            SensitivityOverride(
                override_scope="cost_hystories",
                parameter="subsurface_abex_fraction",
                low=0.10,
                default=0.20,
                high=0.30,
                unit="fraction",
            ),
        ),
        range_basis="literature",
        source="HyStories D7.2-1 ABEX range/reference treatment",
        note=(
            "Subsurface abandonment expenditure is varied around the 20% "
            "reference to test its contribution to the storage-capacity TCR. "
            "As with surface ABEX, this is a project-cost allowance rather than a "
            "separate physical storage-performance uncertainty."
        ),
    ),
    SensitivitySpec(
        key="balance_of_plant",
        label="Balance-of-plant fraction",
        category="Project-cost uncertainty",
        technologies=ALL_TECHS,
        overrides=(
            SensitivityOverride(
                override_scope="cost_hystories",
                parameter="bop_fraction",
                low=0.025,
                default=0.05,
                high=0.10,
                unit="fraction",
            ),
        ),
        range_basis="screening",
        source="HyStories-derived 5% reference; surrounding values are screening",
        note=(
            "The 5% balance-of-plant allowance is retained from the present cost "
            "implementation. Halving/doubling it tests how a proportional surface "
            "allowance translates into final injection and withdrawal TCR rates. "
        ),
    ),

    # -------------------------------------------------------------------------
    # LRC cost basis
    # -------------------------------------------------------------------------
    SensitivitySpec(
        key="lrc_well_fixed_cost",
        label="LRC well fixed-cost basis",
        category="LRC cost basis",
        technologies=("lined_rock_cavern",),
        overrides=(
            SensitivityOverride(
                override_scope="cost_lrc",
                parameter="lrc_well_fixed_cost_eur_per_well",
                low=1_289_776.0805,
                default=2_579_552.1611,
                high=3_869_328.2416,
                unit="EUR/well",
            ),
        ),
        range_basis="screening",
        source="Huang-derived/indexed LRC cost basis",
        note=(
            "The reference is the indexed LRC well fixed-cost basis used by the "
            "current model. Low/high cases apply an explicit -50%/+50% screen to "
            "reflect uncertainty in transferring the source cost basis."
        ),
    ),
    SensitivitySpec(
        key="lrc_well_variable_cost",
        label="LRC well depth-cost basis",
        category="LRC cost basis",
        technologies=("lined_rock_cavern",),
        overrides=(
            SensitivityOverride(
                override_scope="cost_lrc",
                parameter="lrc_well_variable_cost_eur_per_m",
                low=760.7290,
                default=1_521.4581,
                high=2_282.1871,
                unit="EUR/m",
            ),
        ),
        range_basis="screening",
        source="Huang-derived/indexed LRC cost basis",
        note=(
            "The per-metre well cost is screened at +/-50% around the indexed "
            "reference because transferability of the original LRC well-cost "
            "equation is uncertain across sites and project scales."
        ),
    ),
    SensitivitySpec(
        key="lrc_mining_cost",
        label="LRC mining cost",
        category="LRC cost basis",
        technologies=("lined_rock_cavern",),
        overrides=(
            SensitivityOverride(
                override_scope="cost_lrc",
                parameter="lrc_mining_cost_eur_per_m3",
                low=50.1580,
                default=100.3159,
                high=150.4739,
                unit="EUR/m3",
            ),
        ),
        range_basis="screening",
        source="Huang-derived/indexed LRC excavation cost basis",
        note=(
            "Excavation/mining cost is varied by +/-50% around the indexed "
            "reference to test the sensitivity of the engineered-cavern cost to "
            "site- and market-dependent underground construction costs."
        ),
    ),
    SensitivitySpec(
        key="lrc_steel_cost",
        label="LRC steel lining cost",
        category="LRC cost basis",
        technologies=("lined_rock_cavern",),
        overrides=(
            SensitivityOverride(
                override_scope="cost_lrc",
                parameter="lrc_steel_lining_cost_eur_per_tonne",
                low=543.3779,
                default=1_086.7558,
                high=1_630.1337,
                unit="EUR/t",
            ),
        ),
        range_basis="screening",
        source="Huang-derived/indexed LRC steel cost basis",
        note=(
            "The steel unit-cost basis is screened at +/-50% around the indexed "
            "reference to represent commodity/fabrication uncertainty specific to "
            "the LRC lining calculation."
        ),
    ),
    SensitivitySpec(
        key="lrc_concrete_cost",
        label="LRC concrete lining cost",
        category="LRC cost basis",
        technologies=("lined_rock_cavern",),
        overrides=(
            SensitivityOverride(
                override_scope="cost_lrc",
                parameter="lrc_concrete_lining_cost_eur_per_tonne",
                low=30.0948,
                default=60.1896,
                high=90.2843,
                unit="EUR/t",
            ),
        ),
        range_basis="screening",
        source="Huang-derived/indexed LRC concrete cost basis",
        note=(
            "Concrete material cost is varied by +/-50% around the indexed "
            "reference. This is deliberately a cost-basis stress test rather than "
            "a probability distribution for future concrete prices."
        ),
    ),
    SensitivitySpec(
        key="lrc_lining_installation",
        label="LRC lining installation fraction",
        category="LRC cost basis",
        technologies=("lined_rock_cavern",),
        overrides=(
            SensitivityOverride(
                override_scope="cost_lrc",
                parameter="lrc_lining_installation_fraction",
                low=0.025,
                default=0.05,
                high=0.10,
                unit="fraction",
            ),
        ),
        range_basis="screening",
        source="Current LRC cost formulation; 5% reference with broad screen",
        note=(
            "The installation allowance applied to lining material cost is halved "
            "and doubled around the 5% reference to test the importance of this "
            "reduced-form implementation assumption."
        ),
    ),
)

# Candidate process assumptions deliberately not included in executable specs
# because they are currently hard-coded defaults of CompressionInput rather than
# model assumptions passed through the project constructors:
#   - polytropic_efficiency
#   - mechanical_efficiency
#   - motor_efficiency
#   - design_power_factor
# Promote them to GeneralDesignAssumptions first if a future screen is desired.


# =============================================================================
# Generic helpers
# =============================================================================

COMPONENT_GROUPS = ("Injection", "Storage", "Withdrawal")


def normalise_name(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "value"):
        value = value.value
    return str(value).lower().strip()


def safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def get_nested_attr(obj: Any, attr_path: str, default: Any = None) -> Any:
    current = obj
    for attr in attr_path.split("."):
        if current is None or not hasattr(current, attr):
            return default
        current = getattr(current, attr)
    return current


def is_close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)


def dataclass_field_names(instance: Any) -> set[str]:
    return {field.name for field in fields(instance)}


def resolve_range_value(value: RangeValue, technology: str) -> float | int:
    if isinstance(value, Mapping):
        if technology not in value:
            raise KeyError(f"No range value supplied for {technology!r}.")
        return value[technology]
    return value


def selected_technology_specs() -> tuple[TechnologySpec, ...]:
    selected = tuple(
        spec for spec in TECHNOLOGIES if spec.technology in RUN_TECHNOLOGIES
    )
    missing = set(RUN_TECHNOLOGIES) - {spec.technology for spec in selected}
    if missing:
        raise KeyError(f"Unknown RUN_TECHNOLOGIES entries: {sorted(missing)}")
    return selected


def selected_sensitivity_specs() -> tuple[SensitivitySpec, ...]:
    enabled = tuple(spec for spec in SENSITIVITY_SPECS if spec.enabled)
    if RUN_SENSITIVITIES is None:
        return enabled

    index = {spec.key: spec for spec in enabled}
    unknown = set(RUN_SENSITIVITIES) - set(index)
    if unknown:
        raise KeyError(f"Unknown/disabled RUN_SENSITIVITIES entries: {sorted(unknown)}")
    return tuple(index[key] for key in RUN_SENSITIVITIES)


def derive_reference_flows(spec: TechnologySpec) -> tuple[float, float]:
    storage_kwh = spec.default_storage_capacity_gwh_lhv * 1_000_000.0
    withdrawal_kw = storage_kwh / (spec.default_epr_days * 24.0)
    injection_kw = withdrawal_kw / spec.default_wtir
    return withdrawal_kw, injection_kw


def parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def reservoir_storage_class_to_technology(storage_class: str) -> str:
    key = str(storage_class).strip().upper()
    if key in RESERVOIR_STORAGE_CLASS_TO_TECHNOLOGY:
        return RESERVOIR_STORAGE_CLASS_TO_TECHNOLOGY[key]

    # Also accept the internal technology key directly for future/custom files.
    internal = str(storage_class).strip().lower()
    if internal in ALL_TECHS:
        return internal
    raise KeyError(
        f"Unknown reservoir storage_class {storage_class!r}. Known external "
        f"classes: {sorted(RESERVOIR_STORAGE_CLASS_TO_TECHNOLOGY)}; "
        f"internal keys: {sorted(ALL_TECHS)}"
    )


def load_reservoir_samples(
    path: Path = RESERVOIR_SAMPLES_CSV,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Reservoir-sample CSV not found: {path}. "
            "Set RESERVOIR_SAMPLES_CSV or copy the supplied file there."
        )

    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Reservoir-sample CSV has no header: {path}")

        required = {"storage_class", "site"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Reservoir-sample CSV is missing required columns: {sorted(missing)}"
            )

        rows: list[dict[str, Any]] = []
        for input_index, source_row in enumerate(reader):
            storage_class = str(source_row.get("storage_class", "")).strip().upper()
            site = str(source_row.get("site", "")).strip()
            if not storage_class or not site:
                continue

            if (
                RUN_RESERVOIR_STORAGE_CLASSES is not None
                and storage_class not in {x.upper() for x in RUN_RESERVOIR_STORAGE_CLASSES}
            ):
                continue
            if RUN_RESERVOIR_SITES is not None and site not in RUN_RESERVOIR_SITES:
                continue

            row = dict(source_row)
            row["storage_class"] = storage_class
            row["site"] = site
            row["technology"] = reservoir_storage_class_to_technology(storage_class)
            row["_input_index"] = input_index
            rows.append(row)

    return rows


# =============================================================================
# Assumption handling
# =============================================================================


def override_applies(
    override: SensitivityOverride,
    technology: TechnologySpec,
) -> bool:
    return (
        override.technologies is None
        or technology.technology in override.technologies
    )


def reference_override_applies(
    override: ReferenceOnlyOverride,
    technology: TechnologySpec,
) -> bool:
    return technology.technology in override.technologies


def get_scope_object(
    technology: TechnologySpec,
    override_scope: OverrideScope,
) -> Any:
    if override_scope == "design_general":
        return technology.design_assumptions.general
    if override_scope == "design_technology":
        return technology.design_assumptions.technology_specific
    if override_scope == "cost_hystories":
        return construct_hystories_cost_assumptions()
    if override_scope == "cost_lrc":
        return construct_lined_rock_cavern_cost_assumptions()
    raise ValueError(f"Unknown override scope: {override_scope}")


def resolve_override_internal_value(
    override: SensitivityOverride,
    technology: TechnologySpec,
    level: CaseLevel,
) -> float | int:
    raw = resolve_range_value(getattr(override, level), technology.technology)
    value: float | int = float(raw) * override.input_scale
    if override.integer:
        value = int(round(float(value)))
    return value


def resolve_reference_internal_value(
    override: ReferenceOnlyOverride,
    technology: TechnologySpec,
) -> float | int:
    raw = resolve_range_value(override.value, technology.technology)
    value: float | int = float(raw) * override.input_scale
    if override.integer:
        value = int(round(float(value)))
    return value


def internal_to_display(
    internal_value: float | int,
    input_scale: float,
) -> float:
    return float(internal_value) / input_scale


def _validate_parameter_exists(
    technology: TechnologySpec,
    override_scope: OverrideScope,
    parameter: str,
) -> None:
    obj = get_scope_object(technology, override_scope)
    if parameter not in dataclass_field_names(obj):
        raise AttributeError(
            f"{type(obj).__name__} has no field {parameter!r}."
        )


def _apply_parameter_value(
    *,
    technology: TechnologySpec,
    general: Any,
    technology_specific: Any,
    hystories_overrides: dict[str, object],
    lrc_overrides: dict[str, object],
    override_scope: OverrideScope,
    parameter: str,
    value: float | int,
) -> tuple[Any, Any, bool, bool]:
    """Apply one already-resolved value and return updated state.

    The two booleans indicate whether reservoir depth or temperature were
    explicitly changed, allowing the geothermal relationship to be applied once
    after all bundled overrides have been assembled.
    """

    depth_changed = False
    temperature_changed = False

    if override_scope == "design_general":
        if parameter not in dataclass_field_names(general):
            raise AttributeError(
                f"{type(general).__name__} has no field {parameter!r}."
            )
        general = replace(general, **{parameter: value})
        depth_changed = parameter == "reference_reservoir_depth_m"
        temperature_changed = parameter == "temperature_k"

    elif override_scope == "design_technology":
        if parameter not in dataclass_field_names(technology_specific):
            raise AttributeError(
                f"{type(technology_specific).__name__} has no field {parameter!r}."
            )
        technology_specific = replace(
            technology_specific,
            **{parameter: value},
        )

    elif override_scope == "cost_hystories":
        _validate_parameter_exists(technology, override_scope, parameter)
        hystories_overrides[parameter] = value

    elif override_scope == "cost_lrc":
        if technology.technology != "lined_rock_cavern":
            raise ValueError(
                f"LRC-specific cost parameter {parameter!r} supplied to "
                f"{technology.technology!r}."
            )
        _validate_parameter_exists(technology, override_scope, parameter)
        lrc_overrides[parameter] = value

    else:
        raise ValueError(f"Unknown override scope: {override_scope}")

    return general, technology_specific, depth_changed, temperature_changed


PRESSURE_COORDINATE_SENSITIVITIES = {
    "reservoir_pressure_level",
    "reservoir_pressure_ratio",
    "reservoir_abandonment_fraction",
}

PRESSURE_COORDINATE_PARAMETERS = {
    "reservoir_pressure_level": "minimum_operating_pressure_pa",
    "reservoir_pressure_ratio": "maximum_operating_pressure_pa",
    "reservoir_abandonment_fraction": "abandonment_pressure_pa",
}


def pressure_coordinate_multiplier(sensitivity: SensitivitySpec, level: CaseLevel) -> float:
    if sensitivity.key not in PRESSURE_COORDINATE_SENSITIVITIES:
        raise ValueError(f"Not a pressure-coordinate sensitivity: {sensitivity.key}")
    if len(sensitivity.overrides) != 1:
        raise ValueError(f"{sensitivity.key} requires exactly one factor marker")
    return float(getattr(sensitivity.overrides[0], level))


def apply_pressure_coordinates(
    general: Any, technology: TechnologySpec, sensitivity: SensitivitySpec, level: CaseLevel,
) -> Any:
    """Change one dimensionless pressure coordinate, preserving the others."""
    factor = pressure_coordinate_multiplier(sensitivity, level)
    pmin = float(general.minimum_operating_pressure_pa)
    pmax = float(general.maximum_operating_pressure_pa)
    if not (math.isfinite(pmin) and math.isfinite(pmax) and 0 < pmin < pmax):
        raise ValueError(f"Invalid reference pressure window for {technology.technology}: {pmin}, {pmax}")
    pab = None
    if technology.technology == "depleted_gas_field":
        pab = float(general.abandonment_pressure_pa)
        if not (math.isfinite(pab) and 0 < pab < pmin):
            raise ValueError(f"Invalid DGF reference abandonment: {pab}, {pmin}")

    if sensitivity.key == "reservoir_pressure_level":
        pmin_new, pmax_new = factor * pmin, factor * pmax
        pab_new = factor * pab if pab is not None else None
    elif sensitivity.key == "reservoir_pressure_ratio":
        # Dimensionless excess over unity, not direct multiplication of R.
        pmin_new = pmin
        pmax_new = pmin + factor * (pmax - pmin)
        pab_new = pab
    elif sensitivity.key == "reservoir_abandonment_fraction":
        if pab is None:
            raise ValueError("Abandonment sensitivity applies to DGF only")
        pmin_new, pmax_new, pab_new = pmin, pmax, factor * pab
    else:
        raise ValueError(f"Unknown pressure screen: {sensitivity.key}")

    validate_pressures(technology.technology, pmin_new, pmax_new, pab_new)
    updates = {
        "minimum_operating_pressure_pa": pmin_new,
        "maximum_operating_pressure_pa": pmax_new,
    }
    if pab_new is not None:
        updates["abandonment_pressure_pa"] = pab_new
    if sensitivity.key == "reservoir_pressure_level" and technology.technology == "lined_rock_cavern":
        # LRC's Pmin equals pipeline/interface pressure in the supplied model.
        updates["pipeline_pressure_pa"] = factor * float(general.pipeline_pressure_pa)
    return replace(general, **updates)


def validate_pressures(
    technology_name: str, pmin: float, pmax: float, pab: float | None,
) -> None:
    vals = (pmin, pmax) if pab is None else (pab, pmin, pmax)
    if not all(math.isfinite(v) for v in vals):
        raise ValueError(f"{technology_name}: nonfinite pressure values {vals}")
    if pmin <= 0 or pmax <= pmin:
        raise ValueError(f"{technology_name}: expected 0 < Pmin < Pmax, got {vals}")
    if pab is not None and not (0 < pab < pmin):
        raise ValueError(f"{technology_name}: expected 0 < Pab < Pmin < Pmax, got {vals}")


def build_case_configuration(
    technology: TechnologySpec,
    sensitivity: SensitivitySpec | None = None,
    level: Literal["low", "high"] | None = None,
) -> tuple[Any, dict[str, object] | None, dict[str, object] | None]:
    """Build a complete reference or sensitivity-case configuration.

    Every run starts from the same explicit paper reference case. Defaults from
    all sensitivity specs plus ``REFERENCE_ONLY_OVERRIDES`` are applied first.
    The requested factor's low/high bundle is then applied atomically on top.
    """

    assumptions = technology.design_assumptions
    general = assumptions.general
    technology_specific = assumptions.technology_specific
    hystories_overrides: dict[str, object] = {}
    lrc_overrides: dict[str, object] = {}

    depth_changed = False
    temperature_changed = False

    def apply(
        *,
        override_scope: OverrideScope,
        parameter: str,
        value: float | int,
    ) -> None:
        nonlocal general, technology_specific, depth_changed, temperature_changed
        (
            general,
            technology_specific,
            changed_depth,
            changed_temperature,
        ) = _apply_parameter_value(
            technology=technology,
            general=general,
            technology_specific=technology_specific,
            hystories_overrides=hystories_overrides,
            lrc_overrides=lrc_overrides,
            override_scope=override_scope,
            parameter=parameter,
            value=value,
        )
        depth_changed = depth_changed or changed_depth
        temperature_changed = temperature_changed or changed_temperature

    # 1) Baseline/reference values explicitly defined by the sensitivity table.
    for spec in SENSITIVITY_SPECS:
        if technology.technology not in spec.technologies:
            continue
        if spec.key in PRESSURE_COORDINATE_SENSITIVITIES:
            continue  # Pressure defaults are REFERENCE_ONLY_OVERRIDES.
        for override in spec.overrides:
            if not override_applies(override, technology):
                continue
            apply(
                override_scope=override.override_scope,
                parameter=override.parameter,
                value=resolve_override_internal_value(
                    override,
                    technology,
                    "default",
                ),
            )

    # 2) Reference values for parameters not sensitivity-screened.
    for override in REFERENCE_ONLY_OVERRIDES:
        if not reference_override_applies(override, technology):
            continue
        apply(
            override_scope=override.override_scope,
            parameter=override.parameter,
            value=resolve_reference_internal_value(override, technology),
        )

    # 3) Replace the selected factor's defaults with its low/high bundle.
    if sensitivity is not None:
        if level is None:
            raise ValueError("A low/high level is required for a sensitivity case.")
        if technology.technology not in sensitivity.technologies:
            raise ValueError(
                f"Sensitivity {sensitivity.key!r} does not apply to "
                f"{technology.technology!r}."
            )

        if sensitivity.key in PRESSURE_COORDINATE_SENSITIVITIES:
            general = apply_pressure_coordinates(general, technology, sensitivity, level)
        for override in (() if sensitivity.key in PRESSURE_COORDINATE_SENSITIVITIES else sensitivity.overrides):
            if not override_applies(override, technology):
                continue
            apply(
                override_scope=override.override_scope,
                parameter=override.parameter,
                value=resolve_override_internal_value(
                    override,
                    technology,
                    level,
                ),
            )

    # Keep depth and temperature physically consistent unless temperature was
    # explicitly overridden as part of a future factor.
    if depth_changed and not temperature_changed:
        general = replace(
            general,
            temperature_k=calculate_temperature_from_depth_k(
                float(general.reference_reservoir_depth_m)
            ),
        )

    assumptions = replace(
        assumptions,
        general=general,
        technology_specific=technology_specific,
    )

    return (
        assumptions,
        hystories_overrides or None,
        lrc_overrides or None,
    )


def build_reservoir_sample_configuration(
    technology: TechnologySpec,
    sample: Mapping[str, Any],
) -> tuple[Any, dict[str, object] | None, dict[str, object] | None, str]:
    """Apply one documented reservoir configuration to the paper reference case.

    Only non-blank recognised CSV columns are applied. Depth changes trigger the
    same geothermal temperature update as the ordinary sensitivity machinery.
    A zero abandonment pressure on a technology whose general assumptions do not
    expose that field is treated as a benign 'not applicable' convention.
    """

    assumptions, hystories_overrides, lrc_overrides = build_case_configuration(
        technology
    )
    general = assumptions.general
    applied: list[str] = []
    depth_changed = False

    for csv_column, (parameter, scale) in RESERVOIR_SAMPLE_PARAMETER_MAP.items():
        value = parse_optional_float(sample.get(csv_column))
        if value is None:
            continue

        if parameter not in dataclass_field_names(general):
            # SA/SC/LRC rows may use zero purely as a 'not applicable' marker.
            if csv_column == "model_abandonment_pressure_bar" and is_close(value, 0.0):
                continue
            raise AttributeError(
                f"{technology.technology}: reservoir sample {sample.get('site')!r} "
                f"supplies {csv_column!r}, but {type(general).__name__} has no "
                f"field {parameter!r}."
            )

        internal_value = value * scale
        general = replace(general, **{parameter: internal_value})
        applied.append(f"{parameter}={value:g}{' bar' if scale == 1e5 else ' m'}")
        depth_changed = depth_changed or parameter == "reference_reservoir_depth_m"

    if not applied:
        raise ValueError(
            f"Reservoir sample {sample.get('site')!r} contains no recognised numeric "
            "reservoir parameters."
        )

    if depth_changed and "temperature_k" in dataclass_field_names(general):
        general = replace(
            general,
            temperature_k=calculate_temperature_from_depth_k(
                float(general.reference_reservoir_depth_m)
            ),
        )

    assumptions = replace(assumptions, general=general)
    return assumptions, hystories_overrides, lrc_overrides, "; ".join(applied)


def sensitivity_case_definition(
    sensitivity: SensitivitySpec,
    technology: TechnologySpec,
    level: CaseLevel,
) -> str:
    """Human-readable constituent parameter values for CSV auditability."""

    parts: list[str] = []
    for override in sensitivity.overrides:
        if not override_applies(override, technology):
            continue
        value = resolve_override_internal_value(override, technology, level)
        display = internal_to_display(value, override.input_scale)
        parts.append(f"{override.parameter}={display:g} {override.unit}".rstrip())
    if sensitivity.key in PRESSURE_COORDINATE_SENSITIVITIES:
        cfg, _, _ = build_case_configuration(
            technology, sensitivity if level != "default" else None,
            level if level != "default" else None,
        )
        g = cfg.general
        parts = [f"factor={pressure_coordinate_multiplier(sensitivity, level):g} x",
                 f"Pmin={g.minimum_operating_pressure_pa / 1e5:.3f} bar",
                 f"Pmax={g.maximum_operating_pressure_pa / 1e5:.3f} bar"]
        if technology.technology == "depleted_gas_field":
            parts.append(f"Pab={g.abandonment_pressure_pa / 1e5:.3f} bar")
        if technology.technology == "lined_rock_cavern":
            parts.append(f"pipeline={g.pipeline_pressure_pa / 1e5:.3f} bar")
    return "; ".join(parts)


# =============================================================================
# Project and cost model execution
# =============================================================================


def construct_reference_project(
    technology: TechnologySpec,
    design_assumptions: Any,
) -> Any:
    withdrawal_kw, injection_kw = derive_reference_flows(technology)
    return technology.project_constructor(
        working_gas_capacity_kwh_lhv=(
            technology.default_storage_capacity_gwh_lhv * 1_000_000.0
        ),
        withdrawal_flow_kw_h2_lhv=withdrawal_kw,
        injection_flow_kw_h2_lhv=injection_kw,
        assumptions=design_assumptions,
        case_name=technology.case_name,
    )


def calculate_cost_breakdown(
    technology: TechnologySpec,
    project: Any,
    hystories_overrides: dict[str, object] | None = None,
    lrc_overrides: dict[str, object] | None = None,
) -> Any:
    if technology.technology == "lined_rock_cavern":
        return technology.cost_function(
            project,
            overrides=hystories_overrides,
            lrc_overrides=lrc_overrides,
        )
    if lrc_overrides:
        raise ValueError("LRC-specific cost overrides supplied to a non-LRC technology.")
    return technology.cost_function(project, overrides=hystories_overrides)


# =============================================================================
# Cost extraction
# =============================================================================


def get_cost_components(cost_breakdown: Any) -> tuple[Any, ...]:
    if hasattr(cost_breakdown, "components"):
        return tuple(cost_breakdown.components)
    if isinstance(cost_breakdown, (tuple, list)):
        return tuple(cost_breakdown)
    raise TypeError(f"Unsupported cost breakdown type: {type(cost_breakdown).__name__}")


def get_component_group(cost_component: Any) -> str | None:
    driver = normalise_name(cost_component.cost_driver)
    name = normalise_name(cost_component.name)

    if "injection" in driver or "injection" in name:
        return "Injection"
    if "withdrawal" in driver or "withdrawal" in name:
        return "Withdrawal"
    if "storage" in driver or "subsurface" in name:
        return "Storage"
    return None


def get_cost_type(cost_component: Any) -> str:
    return normalise_name(cost_component.cost_type)


def get_cost_rate(cost_component: Any) -> float:
    if cost_component.driver_value == 0:
        raise ValueError(f"Zero cost driver for component {cost_component.name!r}.")
    return float(cost_component.value_eur) / float(cost_component.driver_value)


def summarise_tcr_rates(cost_breakdown: Any) -> dict[str, float | None]:
    rates: dict[str, float] = defaultdict(float)
    present: set[str] = set()

    for component in get_cost_components(cost_breakdown):
        if get_cost_type(component) not in {"capex", "abex"}:
            continue
        group = get_component_group(component)
        if group is None:
            continue
        rates[group] += get_cost_rate(component)
        present.add(group)

    return {
        group: rates[group] if group in present else None
        for group in COMPONENT_GROUPS
    }


def total_project_tcr_eur(cost_breakdown: Any) -> float:
    return sum(
        float(component.value_eur)
        for component in get_cost_components(cost_breakdown)
        if get_cost_type(component) in {"capex", "abex"}
    )


# =============================================================================
# Diagnostics
# =============================================================================


def project_diagnostics(project: Any) -> dict[str, Any]:
    return {
        "number_well_heads": get_nested_attr(project, "wells.number_well_heads"),
        "number_caverns": get_nested_attr(project, "wells.number_caverns"),
        "number_production_wells": get_nested_attr(project, "wells.number_production_wells"),
        "number_observation_wells": get_nested_attr(project, "wells.number_observation_wells"),
        "required_storage_volume_m3": get_nested_attr(project, "inventory.required_storage_volume_m3"),
        "cushion_h2_mass_kg": get_nested_attr(project, "inventory.cushion_gas_h2_mass_kg"),
        "abandonment_methane_volume_sm3": get_nested_attr(
            project, "inventory.abandonment_gas_methane_volume_sm3"
        ),
        "compression_design_electric_power_kw": get_nested_attr(
            project, "compression.total_design_electric_power_kw"
        ),
        "compression_electricity_kwh_per_kwh_h2": get_nested_attr(
            project, "compression.electricity_kwh_per_kwh_h2_lhv"
        ),
        "purification_hydrogen_recovery": get_nested_attr(
            project, "purification.hydrogen_recovery"
        ),
        "purification_factor": get_nested_attr(project, "purification.purification_factor"),
        "reservoir_temperature_k": get_nested_attr(project, "wells.well_temperature_k"),
        "first_fill_duration_years": get_nested_attr(
            project, "porous_first_fill_process.first_gas_fill_duration_years"
        ),
        "leaching_duration_years": get_nested_attr(
            project, "salt_leaching_process.total_leaching_duration_years"
        ),
        "conversion_duration_years": get_nested_attr(
            project, "salt_conversion_process.total_conversion_duration_years"
        ),
    }


def print_aquifer_saturation_diagnostic(
    *,
    label: str,
    design_assumptions: Any,
    project: Any,
    cost_breakdown: Any,
) -> None:
    """Audit the aquifer saturation-to-inventory-to-cost calculation."""

    import uhs_costs.gas_properties.hydrogen as hydrogen

    general = design_assumptions.general
    inventory = project.inventory

    s_min = general.minimum_gas_saturation
    s_max = general.maximum_gas_saturation

    p_min = general.minimum_operating_pressure_pa
    p_max = general.maximum_operating_pressure_pa
    temperature = general.temperature_k

    rho_min = hydrogen.molar_density(
        pressure_pa=p_min,
        temperature_k=temperature,
    )
    rho_max = hydrogen.molar_density(
        pressure_pa=p_max,
        temperature_k=temperature,
    )

    # Independently reconstruct the expected cushion / working ratio.
    expected_ratio = (
        rho_min * s_min
        / (rho_max * s_max - rho_min * s_min)
    )

    actual_ratio = (
        inventory.cushion_gas_h2_mass_kg
        / inventory.working_gas_h2_mass_kg
    )

    assert math.isclose(
        actual_ratio,
        expected_ratio,
        rel_tol=1e-9,
    ), (
        f"{label}: cushion inventory does not match "
        f"the saturation-weighted density calculation."
    )

    storage_tcr = summarise_tcr_rates(cost_breakdown)["Storage"]

    # Find the underlying cushion-cost components.
    cushion_components = [
        component
        for component in get_cost_components(cost_breakdown)
        if "cushion" in normalise_name(component.name)
    ]

    cushion_capex_eur = sum(
        float(component.value_eur)
        for component in cushion_components
        if get_cost_type(component) == "capex"
    )

    print(
        f"{label:24s} "
        f"Smin={s_min:.2f} "
        f"Smax={s_max:.2f} "
        f"Vp={inventory.required_storage_volume_m3 / 1e6:7.2f} Mm3 "
        f"Cushion={inventory.cushion_gas_h2_mass_kg / 1e6:7.2f} kt "
        f"C/W={actual_ratio:6.3f} "
        f"Cushion CAPEX={cushion_capex_eur / 1e6:8.2f} MEUR "
        f"Storage TCR={storage_tcr:7.4f} EUR/kWh"
    )

    # Fail explicitly if the expected cost component cannot be found.
    if not cushion_components:
        print(
            "  WARNING: No cost component containing 'cushion' "
            "was found. Check the component naming."
        )



# =============================================================================
# Baseline and sensitivity runs
# =============================================================================


def run_default_case(technology: TechnologySpec) -> dict[str, Any]:
    design_assumptions, hystories_overrides, lrc_overrides = (
        build_case_configuration(technology)
    )
    project = construct_reference_project(
        technology,
        design_assumptions=design_assumptions,
    )
    breakdown = calculate_cost_breakdown(
        technology,
        project,
        hystories_overrides=hystories_overrides,
        lrc_overrides=lrc_overrides,
    )
    rates = summarise_tcr_rates(breakdown)

    if technology.technology == "aquifer":
        print_aquifer_saturation_diagnostic(
            label="Baseline",
            design_assumptions=design_assumptions,
            project=project,
            cost_breakdown=breakdown,
        )

    return {
        "project": project,
        "breakdown": breakdown,
        "rates": rates,
        "total_tcr_eur": total_project_tcr_eur(breakdown),
    }


def run_sensitivity_case(
    sensitivity: SensitivitySpec,
    technology: TechnologySpec,
    level: Literal["low", "high"],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    design_assumptions, hystories_overrides, lrc_overrides = (
        build_case_configuration(
            technology,
            sensitivity=sensitivity,
            level=level,
        )
    )

    project = construct_reference_project(
        technology,
        design_assumptions=design_assumptions,
    )
    breakdown = calculate_cost_breakdown(
        technology,
        project,
        hystories_overrides=hystories_overrides,
        lrc_overrides=lrc_overrides,
    )
    rates = summarise_tcr_rates(breakdown)
    total_tcr = total_project_tcr_eur(breakdown)

    if (
        technology.technology == "aquifer"
        and sensitivity.key in {
            "aquifer_min_saturation_design",
            "aquifer_max_saturation_design",
        }
    ):
        print_aquifer_saturation_diagnostic(
            label=f"{sensitivity.key.replace('aquifer_', '')} [{level}]",
            design_assumptions=design_assumptions,
            project=project,
            cost_breakdown=breakdown,
        )

    row: dict[str, Any] = {
        "sensitivity": sensitivity.key,
        "sensitivity_label": sensitivity.label,
        "category": sensitivity.category,
        "technology": technology.technology,
        "technology_label": technology.label,
        "level": level,
        "n_overrides": sum(
            1
            for override in sensitivity.overrides
            if override_applies(override, technology)
        ),
        "case_definition": sensitivity_case_definition(
            sensitivity,
            technology,
            level,
        ),
        "default_definition": sensitivity_case_definition(
            sensitivity,
            technology,
            "default",
        ),
        "range_basis": sensitivity.range_basis,
        "source": sensitivity.source,
        "note": sensitivity.note,
        "status": "ok",
        "error": "",
        "total_project_tcr_eur": total_tcr,
        "total_project_tcr_change_pct": (
            100.0 * (total_tcr / baseline["total_tcr_eur"] - 1.0)
        ),
    }

    for group in COMPONENT_GROUPS:
        key = group.lower()
        rate = rates[group]
        base_rate = baseline["rates"][group]
        row[f"{key}_tcr_rate"] = rate
        row[f"{key}_default_tcr_rate"] = base_rate
        row[f"{key}_tcr_change_pct"] = (
            None
            if rate is None or base_rate in {None, 0}
            else 100.0 * (rate / base_rate - 1.0)
        )

    row.update(project_diagnostics(project))
    return row


def failed_case_row(
    sensitivity: SensitivitySpec,
    technology: TechnologySpec,
    level: Literal["low", "high"],
    exc: Exception,
) -> dict[str, Any]:
    try:
        case_definition = sensitivity_case_definition(
            sensitivity,
            technology,
            level,
        )
        default_definition = sensitivity_case_definition(
            sensitivity,
            technology,
            "default",
        )
    except Exception:
        case_definition = ""
        default_definition = ""

    return {
        "sensitivity": sensitivity.key,
        "sensitivity_label": sensitivity.label,
        "category": sensitivity.category,
        "technology": technology.technology,
        "technology_label": technology.label,
        "level": level,
        "n_overrides": sum(
            1
            for override in sensitivity.overrides
            if override_applies(override, technology)
        ),
        "case_definition": case_definition,
        "default_definition": default_definition,
        "range_basis": sensitivity.range_basis,
        "source": sensitivity.source,
        "note": sensitivity.note,
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
    }


def run_experiment() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    techs = selected_technology_specs()
    sensitivities = selected_sensitivity_specs()

    baselines: dict[str, dict[str, Any]] = {}
    for technology in techs:
        print(f"Baseline: {technology.label}")
        baselines[technology.technology] = run_default_case(technology)

    rows: list[dict[str, Any]] = []
    for sensitivity in sensitivities:
        for technology in techs:
            if technology.technology not in sensitivity.technologies:
                continue

            for level in ("low", "high"):
                print(
                    f"{sensitivity.key}: {technology.label}: {level}",
                    flush=True,
                )
                try:
                    row = run_sensitivity_case(
                        sensitivity,
                        technology,
                        level,
                        baselines[technology.technology],
                    )
                except Exception as exc:
                    if FAIL_FAST:
                        raise
                    print(f"  FAILED -> {type(exc).__name__}: {exc}")
                    row = failed_case_row(sensitivity, technology, level, exc)
                rows.append(row)

    return rows, baselines


# =============================================================================
# Reservoir-sample runs
# =============================================================================


def _sample_metadata(sample: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in sample.items()
        if not str(key).startswith("_") and key != "technology"
    }


def run_reservoir_sample_case(
    sample: Mapping[str, Any],
    technology: TechnologySpec,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    (
        design_assumptions,
        hystories_overrides,
        lrc_overrides,
        sample_definition,
    ) = build_reservoir_sample_configuration(technology, sample)

    project = construct_reference_project(
        technology,
        design_assumptions=design_assumptions,
    )
    breakdown = calculate_cost_breakdown(
        technology,
        project,
        hystories_overrides=hystories_overrides,
        lrc_overrides=lrc_overrides,
    )
    rates = summarise_tcr_rates(breakdown)
    total_tcr = total_project_tcr_eur(breakdown)

    row: dict[str, Any] = {
        **_sample_metadata(sample),
        "technology": technology.technology,
        "technology_label": technology.label,
        "sample_definition": sample_definition,
        "status": "ok",
        "error": "",
        "total_project_tcr_eur": total_tcr,
        "total_project_default_tcr_eur": baseline["total_tcr_eur"],
        "total_project_tcr_change_pct": (
            100.0 * (total_tcr / baseline["total_tcr_eur"] - 1.0)
        ),
    }

    for group in COMPONENT_GROUPS:
        key = group.lower()
        rate = rates[group]
        base_rate = baseline["rates"][group]
        row[f"{key}_tcr_rate"] = rate
        row[f"{key}_default_tcr_rate"] = base_rate
        row[f"{key}_tcr_change_pct"] = (
            None
            if rate is None or base_rate in {None, 0}
            else 100.0 * (rate / base_rate - 1.0)
        )

    row.update(project_diagnostics(project))
    return row


def failed_reservoir_sample_row(
    sample: Mapping[str, Any],
    technology: TechnologySpec,
    exc: Exception,
) -> dict[str, Any]:
    return {
        **_sample_metadata(sample),
        "technology": technology.technology,
        "technology_label": technology.label,
        "sample_definition": "",
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
    }


def run_reservoir_sample_experiment(
    baselines: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not RUN_RESERVOIR_SAMPLE_SWEEP:
        return []

    samples = load_reservoir_samples()
    selected_names = {spec.technology for spec in selected_technology_specs()}
    tech_by_name = {spec.technology: spec for spec in TECHNOLOGIES}

    rows: list[dict[str, Any]] = []
    for sample in samples:
        technology_name = str(sample["technology"])
        if technology_name not in selected_names:
            continue

        technology = tech_by_name[technology_name]
        print(
            f"reservoir sample: {technology.label}: {sample['site']}",
            flush=True,
        )
        try:
            row = run_reservoir_sample_case(
                sample,
                technology,
                baselines[technology_name],
            )
        except Exception as exc:
            if FAIL_FAST:
                raise
            print(f"  FAILED -> {type(exc).__name__}: {exc}")
            row = failed_reservoir_sample_row(sample, technology, exc)
        rows.append(row)

    return rows


# =============================================================================
# Specification table
# =============================================================================


def build_specification_rows() -> list[dict[str, Any]]:
    """Return a long-form sensitivity specification table.

    Bundled sensitivities occupy multiple rows: one row per constituent model
    parameter and applicable technology. This keeps the executable low/default/
    high values auditable without flattening a correlated factor into one value.
    """

    rows: list[dict[str, Any]] = []
    tech_by_name = {spec.technology: spec for spec in TECHNOLOGIES}

    for sensitivity in SENSITIVITY_SPECS:
        for technology_name in sensitivity.technologies:
            technology = tech_by_name[technology_name]
            applicable = [
                override
                for override in sensitivity.overrides
                if override_applies(override, technology)
            ]

            if not applicable:
                rows.append(
                    {
                        "sensitivity": sensitivity.key,
                        "sensitivity_label": sensitivity.label,
                        "category": sensitivity.category,
                        "technology": technology_name,
                        "technology_label": technology.label,
                        "enabled": sensitivity.enabled,
                        "parameter": "",
                        "override_scope": "",
                        "low": None,
                        "default": None,
                        "high": None,
                        "unit": "",
                        "range_basis": sensitivity.range_basis,
                        "source": sensitivity.source,
                        "note": sensitivity.note,
                        "status": "invalid_spec",
                        "error": "No constituent override applies to this technology.",
                    }
                )
                continue

            for override in applicable:
                try:
                    _validate_parameter_exists(
                        technology,
                        override.override_scope,
                        override.parameter,
                    )
                    low_internal = resolve_override_internal_value(
                        override,
                        technology,
                        "low",
                    )
                    default_internal = resolve_override_internal_value(
                        override,
                        technology,
                        "default",
                    )
                    high_internal = resolve_override_internal_value(
                        override,
                        technology,
                        "high",
                    )
                    low_value = internal_to_display(
                        low_internal,
                        override.input_scale,
                    )
                    default_value = internal_to_display(
                        default_internal,
                        override.input_scale,
                    )
                    high_value = internal_to_display(
                        high_internal,
                        override.input_scale,
                    )
                    status = "ok"
                    error = ""
                except Exception as exc:
                    low_value = default_value = high_value = None
                    status = "invalid_spec"
                    error = f"{type(exc).__name__}: {exc}"

                rows.append(
                    {
                        "sensitivity": sensitivity.key,
                        "sensitivity_label": sensitivity.label,
                        "category": sensitivity.category,
                        "technology": technology_name,
                        "technology_label": technology.label,
                        "enabled": sensitivity.enabled,
                        "parameter": override.parameter,
                        "override_scope": override.override_scope,
                        "low": low_value,
                        "default": default_value,
                        "high": high_value,
                        "unit": override.unit,
                        "range_basis": sensitivity.range_basis,
                        "source": sensitivity.source,
                        "note": sensitivity.note,
                        "status": status,
                        "error": error,
                    }
                )

    # Include non-screened paper-reference values for completeness/auditability.
    for override in REFERENCE_ONLY_OVERRIDES:
        for technology_name in override.technologies:
            technology = tech_by_name[technology_name]
            try:
                _validate_parameter_exists(
                    technology,
                    override.override_scope,
                    override.parameter,
                )
                value_internal = resolve_reference_internal_value(
                    override,
                    technology,
                )
                default_value = internal_to_display(
                    value_internal,
                    override.input_scale,
                )
                status = "reference_only"
                error = ""
            except Exception as exc:
                default_value = None
                status = "invalid_spec"
                error = f"{type(exc).__name__}: {exc}"

            rows.append(
                {
                    "sensitivity": "reference_only",
                    "sensitivity_label": "Reference-only default",
                    "category": "Geological / storage design",
                    "technology": technology_name,
                    "technology_label": technology.label,
                    "enabled": False,
                    "parameter": override.parameter,
                    "override_scope": override.override_scope,
                    "low": None,
                    "default": default_value,
                    "high": None,
                    "unit": override.unit,
                    "range_basis": "reference only",
                    "source": "",
                    "note": override.note,
                    "status": status,
                    "error": error,
                }
            )

    return rows


# =============================================================================
# CSV output
# =============================================================================


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    # Preserve first-seen order while accommodating columns only present in
    # successful rows / diagnostics.
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Plotting
# =============================================================================


def technology_colours() -> dict[str, Any]:
    selected = selected_technology_specs()
    positions = np.linspace(PLASMA_MIN, PLASMA_MAX, len(selected))
    cmap = plt.get_cmap("plasma")
    return {
        spec.technology: cmap(position)
        for spec, position in zip(selected, positions, strict=True)
    }


def max_impact_by_sensitivity(rows: list[dict[str, Any]]) -> dict[str, float]:
    impact: dict[str, float] = defaultdict(float)
    for row in rows:
        if row.get("status") != "ok":
            continue
        for group in COMPONENT_GROUPS:
            value = row.get(f"{group.lower()}_tcr_change_pct")
            if value is None:
                continue
            impact[row["sensitivity"]] = max(impact[row["sensitivity"]], abs(float(value)))
    return impact


def ordered_specs_for_plot(rows: list[dict[str, Any]]) -> list[SensitivitySpec]:
    selected = list(selected_sensitivity_specs())
    impact = max_impact_by_sensitivity(rows)

    if PLOT_MATERIALITY_THRESHOLD_PERCENT is not None:
        threshold = float(PLOT_MATERIALITY_THRESHOLD_PERCENT)
        selected = [
            spec
            for spec in selected
            if impact.get(spec.key, 0.0) >= threshold
        ]

    output: list[SensitivitySpec] = []

    for category in CATEGORY_ORDER:
        category_specs = [
            spec for spec in selected
            if spec.category == category
        ]

        if SORT_WITHIN_CATEGORY_BY_MAX_IMPACT:
            category_specs.sort(
                key=lambda spec: impact.get(spec.key, 0.0),
                reverse=True,
            )

        output.extend(category_specs)

    return output


def rows_for(
    rows: list[dict[str, Any]],
    sensitivity_key: str,
    technology: str,
) -> dict[str, dict[str, Any]]:
    return {
        row["level"]: row
        for row in rows
        if row.get("sensitivity") == sensitivity_key
        and row.get("technology") == technology
        and row.get("status") == "ok"
    }


def spec_label_for_plot(spec: SensitivitySpec) -> str:
    if not SHOW_RANGE_IN_YLABEL:
        return spec.label

    # Only append a compact range for a genuinely single-parameter factor with
    # technology-invariant scalar values. Bundled/technology-specific cases are
    # documented in sensitivity_specification.csv instead.
    if len(spec.overrides) != 1:
        return spec.label
    override = spec.overrides[0]
    if (
        isinstance(override.low, Mapping)
        or isinstance(override.default, Mapping)
        or isinstance(override.high, Mapping)
    ):
        return spec.label
    return (
        f"{spec.label} "
        f"({float(override.low):g}--{float(override.high):g} {override.unit})"
    )


def plot_tcr_sensitivity_ranges(rows: list[dict[str, Any]]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    specs = ordered_specs_for_plot(rows)
    techs = selected_technology_specs()
    colours = technology_colours()

    # Build y coordinates with a small blank gap between categories.
    y_by_key: dict[str, float] = {}
    category_top_y: dict[str, float] = {}
    y = 0.0
    previous_category: str | None = None
    for spec in specs:
        if previous_category is not None and spec.category != previous_category:
            y += 0.65
        if spec.category not in category_top_y:
            category_top_y[spec.category] = y
        y_by_key[spec.key] = y
        y += 1.0
        previous_category = spec.category

    total_height_units = max(y, 1.0)
    # Portrait A4-ish manuscript page. Keep fixed rather than scaling without
    # bound so the screening figure remains intentionally compact.
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(8.27, 11.69),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    # Jitter technology observations within each sensitivity row.
    if len(techs) == 1:
        offsets = {techs[0].technology: 0.0}
    else:
        jitter = np.linspace(-0.24, 0.24, len(techs))
        offsets = {
            spec.technology: float(offset)
            for spec, offset in zip(techs, jitter, strict=True)
        }

    plotted_values: list[float] = []

    for panel_idx, group in enumerate(COMPONENT_GROUPS):
        ax = axes[panel_idx]
        value_key = f"{group.lower()}_tcr_change_pct"

        for sensitivity in specs:
            base_y = y_by_key[sensitivity.key]
            for technology in techs:
                if technology.technology not in sensitivity.technologies:
                    continue
                case_rows = rows_for(rows, sensitivity.key, technology.technology)
                if "low" not in case_rows or "high" not in case_rows:
                    continue

                low_value = case_rows["low"].get(value_key)
                high_value = case_rows["high"].get(value_key)
                if low_value is None or high_value is None:
                    continue
                low_value = float(low_value)
                high_value = float(high_value)
                plotted_values.extend((low_value, high_value))

                yy = base_y + offsets[technology.technology]
                colour = colours[technology.technology]

                ax.plot(
                    [low_value, high_value],
                    [yy, yy],
                    color=colour,
                    linewidth=1.35,
                    alpha=0.95,
                    zorder=2,
                )
                ax.scatter(
                    [low_value],
                    [yy],
                    marker=LOW_MARKER,
                    s=28,
                    color=colour,
                    zorder=3,
                )
                ax.scatter(
                    [high_value],
                    [yy],
                    marker=HIGH_MARKER,
                    s=28,
                    color=colour,
                    zorder=3,
                )

        ax.axvline(0.0, linewidth=0.9, color="0.35", alpha=0.8)
        ax.grid(axis="x", linewidth=0.4, alpha=0.25)
        ax.set_title(f"{group} TCR")
        ax.set_xlabel("Change in TCR [%]")

    # Y labels appear only on the first shared-y panel.
    yticks = [y_by_key[spec.key] for spec in specs]
    ylabels = [spec_label_for_plot(spec) for spec in specs]
    axes[0].set_yticks(yticks)
    axes[0].set_yticklabels(ylabels, fontsize=7)

    # Put the first sensitivity at the top.
    axes[0].set_ylim(total_height_units - 0.2, -0.7)

    # Category labels and separators. Draw on each panel; category text only on
    # the left panel to avoid visual repetition.
    category_specs: dict[str, list[SensitivitySpec]] = defaultdict(list)
    for spec in specs:
        category_specs[spec.category].append(spec)

    for category in CATEGORY_ORDER:
        members = category_specs.get(category)
        if not members:
            continue
        first_y = y_by_key[members[0].key]
        last_y = y_by_key[members[-1].key]

        axes[0].text(
            -0.03,
            first_y - 0.48,
            category,
            transform=axes[0].get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
            clip_on=False,
        )

        separator_y = last_y + 0.55
        for ax in axes:
            ax.axhline(separator_y, linewidth=0.45, color="0.75", alpha=0.7)

    # Shared symmetric x scale supports direct comparison across panels.
    if PLOT_X_LIMIT_PERCENT is not None:
        xlim = abs(float(PLOT_X_LIMIT_PERCENT))
    elif plotted_values:
        max_abs = max(abs(value) for value in plotted_values)
        # Rounded, slightly padded scale; enforce a small minimum so near-zero
        # screens remain readable.
        xlim = max(10.0, math.ceil(max_abs * 1.08 / 10.0) * 10.0)
    else:
        xlim = 10.0

    for ax in axes:
        ax.set_xlim(-xlim, xlim)

    # Technology legend uses line colour; low/high parameter-setting markers are
    # described separately so marker direction cannot be confused with cost sign.
    tech_handles = [
        Line2D(
            [0],
            [0],
            color=colours[tech.technology],
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=tech.label,
        )
        for tech in techs
    ]
    setting_handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            marker=LOW_MARKER,
            linestyle="None",
            markersize=6,
            label="Low parameter setting",
        ),
        Line2D(
            [0],
            [0],
            color="0.25",
            marker=HIGH_MARKER,
            linestyle="None",
            markersize=6,
            label="High parameter setting",
        ),
    ]

    fig.legend(
        handles=tech_handles + setting_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=6,
        frameon=False,
        fontsize=7.5,
    )

    fig.suptitle("Sensitivity of CEM-ready UHS capital-cost rates", y=0.995)
    fig.subplots_adjust(
        left=0.31,
        right=0.99,
        top=0.94,
        bottom=0.085,
        wspace=0.08,
    )
    fig.savefig(FIGURE_DIR / "tcr_sensitivity_ranges.pdf")
    plt.close(fig)


def plot_reservoir_sample_tcr_rates(rows: list[dict[str, Any]]) -> None:
    """Plot every documented reservoir sample using absolute component TCR rates.

    Each row is a site. The coloured circle is the site-specific configuration;
    the vertical tick is the ordinary paper reference case for that technology.
    Failed/infeasible samples remain visible as labelled rows.
    """

    if not rows:
        return

    tech_order = {spec.technology: i for i, spec in enumerate(TECHNOLOGIES)}
    input_order = {
        (row.get("technology"), row.get("site")): i
        for i, row in enumerate(rows)
    }

    def sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        tech = str(row.get("technology", ""))
        base = (tech_order.get(tech, 999),)
        if RESERVOIR_SAMPLE_SORT == "site":
            return base + (str(row.get("site", "")).lower(),)
        if RESERVOIR_SAMPLE_SORT == "storage_tcr":
            value = row.get("storage_tcr_rate")
            numeric = float(value) if value is not None else float("inf")
            return base + (numeric, str(row.get("site", "")).lower())
        return base + (input_order.get((tech, row.get("site")), 999999),)

    plot_rows = sorted(rows, key=sort_key)
    n = len(plot_rows)
    colours = technology_colours()

    component_meta = (
        ("injection", "Injection", "TCR rate [EUR/kW]"),
        ("storage", "Storage", "TCR rate [EUR/kWh]"),
        ("withdrawal", "Withdrawal", "TCR rate [EUR/kW]"),
    )

    fig_height = max(6.0, 0.34 * n + 2.4)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.5, fig_height),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    y_positions = np.arange(n, dtype=float)

    failed_y: list[float] = []
    for y, row in zip(y_positions, plot_rows, strict=True):
        technology = str(row["technology"])
        colour = colours[technology]
        if row.get("status") != "ok":
            failed_y.append(y)
            continue

        for ax, (key, _title, _xlabel) in zip(axes, component_meta, strict=True):
            sample_value = row.get(f"{key}_tcr_rate")
            default_value = row.get(f"{key}_default_tcr_rate")
            if sample_value is None:
                continue

            sample_value = float(sample_value)
            if default_value is not None:
                default_value = float(default_value)
                ax.plot(
                    [default_value, sample_value],
                    [y, y],
                    color=colour,
                    linewidth=1.0,
                    alpha=0.55,
                    zorder=1,
                )
                ax.plot(
                    default_value,
                    y,
                    marker="|",
                    color=colour,
                    markersize=8,
                    markeredgewidth=1.2,
                    linestyle="None",
                    zorder=2,
                )
            ax.plot(
                sample_value,
                y,
                marker="o",
                color=colour,
                markersize=4.5,
                linestyle="None",
                zorder=3,
            )

    labels = [
        f"{row.get('site', '')} [{row.get('storage_class', row.get('technology', ''))}]"
        for row in plot_rows
    ]
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(labels, fontsize=7.2)
    axes[0].invert_yaxis()

    for ax, (_key, title, xlabel) in zip(axes, component_meta, strict=True):
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.grid(axis="x", linewidth=0.45, alpha=0.35)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_xlim(left=0.0)

    # Technology-group separators and labels.
    last_tech: str | None = None
    for i, row in enumerate(plot_rows):
        tech = str(row["technology"])
        if last_tech is not None and tech != last_tech:
            separator = i - 0.5
            for ax in axes:
                ax.axhline(separator, linewidth=0.55, color="0.72", alpha=0.8)
        last_tech = tech

    # Keep failed samples visible even though no TCR coordinate exists.
    for y in failed_y:
        for ax in axes:
            xmin, xmax = ax.get_xlim()
            xpos = xmin + 0.02 * (xmax - xmin if xmax > xmin else 1.0)
            ax.text(
                xpos,
                y,
                "failed",
                fontsize=6.5,
                color="0.4",
                va="center",
                ha="left",
            )

    tech_names_present = {str(row["technology"]) for row in plot_rows}
    tech_handles = [
        Line2D(
            [0],
            [0],
            color=colours[tech.technology],
            marker="o",
            linewidth=1.2,
            markersize=4.5,
            label=tech.label,
        )
        for tech in TECHNOLOGIES
        if tech.technology in tech_names_present
    ]
    meaning_handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            marker="o",
            linestyle="None",
            markersize=4.5,
            label="Reservoir sample",
        ),
        Line2D(
            [0],
            [0],
            color="0.25",
            marker="|",
            linestyle="None",
            markersize=8,
            markeredgewidth=1.2,
            label="Paper reference",
        ),
    ]

    fig.legend(
        handles=tech_handles + meaning_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=min(6, len(tech_handles) + len(meaning_handles)),
        frameon=False,
        fontsize=7.5,
    )
    fig.suptitle("TCR rates across documented reservoir configurations", y=0.995)
    fig.subplots_adjust(
        left=0.30,
        right=0.99,
        top=0.955,
        bottom=0.07,
        wspace=0.12,
    )
    fig.savefig(FIGURE_DIR / "reservoir_sample_tcr_rates.pdf")
    plt.close(fig)


# =============================================================================
# Console summary
# =============================================================================


def print_screening_summary(rows: list[dict[str, Any]], top_n: int = 15) -> None:
    impact = max_impact_by_sensitivity(rows)
    spec_by_key = {spec.key: spec for spec in selected_sensitivity_specs()}

    ranking = sorted(impact.items(), key=lambda item: item[1], reverse=True)
    print("\nLargest component TCR sensitivities")
    print("=" * 50)
    for key, value in ranking[:top_n]:
        spec = spec_by_key[key]
        print(f"{value:7.1f}%  {spec.label}  [{spec.category}]")

    failures = [row for row in rows if row.get("status") == "failed"]
    if failures:
        print("\nFailed sensitivity cases")
        print("=" * 50)
        for row in failures:
            print(
                f"{row['sensitivity']} | {row['technology']} | {row['level']}: "
                f"{row['error']}"
            )


def print_reservoir_sample_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    print("\nReservoir-sample sweep")
    print("=" * 50)
    by_technology: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_technology[str(row["technology"])].append(row)

    tech_by_name = {spec.technology: spec for spec in TECHNOLOGIES}
    for technology, members in by_technology.items():
        ok = [row for row in members if row.get("status") == "ok"]
        failed = [row for row in members if row.get("status") != "ok"]
        label = tech_by_name[technology].label
        print(f"{label}: {len(ok)} successful, {len(failed)} failed")
        if ok:
            changes = [float(row["total_project_tcr_change_pct"]) for row in ok]
            print(
                "  total-project TCR change range: "
                f"{min(changes):.1f}% to {max(changes):.1f}%"
            )
        for row in failed:
            print(f"  FAILED {row.get('site')}: {row.get('error')}")


# =============================================================================
# Main
# =============================================================================


def validate_configuration() -> None:
    keys = [spec.key for spec in SENSITIVITY_SPECS]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"Duplicate sensitivity keys: {duplicates}")

    known_categories = set(CATEGORY_ORDER)
    unknown_categories = {
        spec.category
        for spec in SENSITIVITY_SPECS
        if spec.category not in known_categories
    }
    if unknown_categories:
        raise ValueError(
            f"Unknown sensitivity categories: {sorted(unknown_categories)}"
        )

    known_techs = set(ALL_TECHS)

    # Detect conflicting default assignments. The factor registry doubles as the
    # explicit paper reference configuration, so a parameter should have only one
    # default value per technology unless the duplicate resolves identically.
    defaults_seen: dict[tuple[str, OverrideScope, str], float | int] = {}

    for spec in SENSITIVITY_SPECS:
        unknown_techs = set(spec.technologies) - known_techs
        if unknown_techs:
            raise ValueError(
                f"Sensitivity {spec.key!r} references unknown technologies: "
                f"{sorted(unknown_techs)}"
            )
        if not spec.overrides:
            raise ValueError(f"Sensitivity {spec.key!r} has no overrides.")

        for override in spec.overrides:
            if override.input_scale <= 0:
                raise ValueError(
                    f"{spec.key}/{override.parameter}: input_scale must be positive."
                )

            override_techs = (
                set(spec.technologies)
                if override.technologies is None
                else set(override.technologies)
            )
            if not override_techs <= set(spec.technologies):
                raise ValueError(
                    f"{spec.key}/{override.parameter}: override technologies must "
                    "be a subset of the parent sensitivity technologies."
                )

            for technology_name in override_techs:
                technology = next(
                    spec_
                    for spec_ in TECHNOLOGIES
                    if spec_.technology == technology_name
                )
                _validate_parameter_exists(
                    technology,
                    override.override_scope,
                    override.parameter,
                )
                default_value = resolve_override_internal_value(
                    override,
                    technology,
                    "default",
                )
                if spec.key in PRESSURE_COORDINATE_SENSITIVITIES:
                    continue  # Coordinate markers are not physical pressure defaults.
                key = (
                    technology_name,
                    override.override_scope,
                    override.parameter,
                )
                previous = defaults_seen.get(key)
                if previous is not None and not is_close(
                    float(previous),
                    float(default_value),
                ):
                    raise ValueError(
                        "Conflicting reference defaults for "
                        f"{technology_name}/{override.parameter}: "
                        f"{previous} versus {default_value}."
                    )
                defaults_seen[key] = default_value

    for override in REFERENCE_ONLY_OVERRIDES:
        if override.input_scale <= 0:
            raise ValueError(
                f"reference-only {override.parameter}: input_scale must be positive."
            )
        unknown_techs = set(override.technologies) - known_techs
        if unknown_techs:
            raise ValueError(
                f"Reference-only override {override.parameter!r} references "
                f"unknown technologies: {sorted(unknown_techs)}"
            )

        for technology_name in override.technologies:
            technology = next(
                spec_
                for spec_ in TECHNOLOGIES
                if spec_.technology == technology_name
            )
            _validate_parameter_exists(
                technology,
                override.override_scope,
                override.parameter,
            )
            default_value = resolve_reference_internal_value(
                override,
                technology,
            )
            key = (
                technology_name,
                override.override_scope,
                override.parameter,
            )
            previous = defaults_seen.get(key)
            if previous is not None and not is_close(
                float(previous),
                float(default_value),
            ):
                raise ValueError(
                    "Conflicting reference-only default for "
                    f"{technology_name}/{override.parameter}: "
                    f"{previous} versus {default_value}."
                )
            defaults_seen[key] = default_value


def preflight_pressure_cases() -> None:
    """Reject invalid pressure-sensitivity configurations before any cost runs."""
    print("Pressure sensitivity preflight:")
    for tech in selected_technology_specs():
        for spec in selected_sensitivity_specs():
            if spec.key not in PRESSURE_COORDINATE_SENSITIVITIES or tech.technology not in spec.technologies:
                continue
            for level in ("low", "high"):
                cfg, _, _ = build_case_configuration(tech, spec, level)
                g = cfg.general
                pab = float(g.abandonment_pressure_pa) if tech.technology == "depleted_gas_field" else None
                validate_pressures(tech.technology, float(g.minimum_operating_pressure_pa),
                                   float(g.maximum_operating_pressure_pa), pab)
                print(f"  {tech.technology}/{spec.key}/{level}: "
                      f"Pmin={g.minimum_operating_pressure_pa/1e5:.2f} "
                      f"Pmax={g.maximum_operating_pressure_pa/1e5:.2f}" +
                      (f" Pab={pab/1e5:.2f}" if pab is not None else ""))


def main() -> None:
    validate_configuration()
    preflight_pressure_cases()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    specification_rows = build_specification_rows()
    write_csv(OUTPUT_SPEC_CSV, specification_rows)

    invalid_specs = [
        row
        for row in specification_rows
        if row["status"] == "invalid_spec" and row["enabled"]
    ]
    if invalid_specs:
        print("WARNING: one or more enabled sensitivity specifications do not match the current code:")
        for row in invalid_specs:
            print(
                f"  {row['sensitivity']} | {row['technology']}: {row['error']}"
            )
        raise RuntimeError("Invalid enabled sensitivity specifications.")

    rows, baselines = run_experiment()
    write_csv(OUTPUT_RESULTS_CSV, rows)
    failed = [row for row in rows if row.get("status") != "ok"]
    if failed:
        raise RuntimeError(f"Sensitivity sweep has {len(failed)} failed cases; see {OUTPUT_RESULTS_CSV}")
    plot_tcr_sensitivity_ranges(rows)
    print_screening_summary(rows)

    reservoir_rows = run_reservoir_sample_experiment(baselines)
    if reservoir_rows:
        write_csv(OUTPUT_RESERVOIR_SAMPLE_RESULTS_CSV, reservoir_rows)
        plot_reservoir_sample_tcr_rates(reservoir_rows)
        print_reservoir_sample_summary(reservoir_rows)

    print("\nSaved:")
    print(f"  {OUTPUT_RESULTS_CSV}")
    print(f"  {OUTPUT_SPEC_CSV}")
    print(f"  {FIGURE_DIR / 'tcr_sensitivity_ranges.pdf'}")
    if reservoir_rows:
        print(f"  {OUTPUT_RESERVOIR_SAMPLE_RESULTS_CSV}")
        print(f"  {FIGURE_DIR / 'reservoir_sample_tcr_rates.pdf'}")


if __name__ == "__main__":
    main()
