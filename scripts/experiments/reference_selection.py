"""Compare UHS design/economic medoids against an arithmetic-mean design.

Run from repository root:
    pixi run python scripts/experiments/reference_selection.py

Each site's cost model uses its own H2 reference working capacity from the
input CSV; only EPR and WTIR are held fixed within a technology class.
Design and economic medoids are observed cases, not synthetic
averages. Site coverage is not an availability-weighted European asset census.
Missing capacities/pressures are never silently imputed; hydrogen volume
equivalents from natural-gas working volume are not validated H2 capacities.
The synthetic design averages each source parameter independently, so partially
documented facilities contribute only to the parameters they report. It is a
comparison case, not a physically verified storage-site configuration.
"""

from __future__ import annotations

import argparse
import csv
import math
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

CLASSES = ("DGF", "SA", "SC", "LRC")
TECHS = {"DGF": "depleted_gas_field", "SA": "aquifer",
         "SC": "salt_cavern", "LRC": "lined_rock_cavern"}
COSTS = ("injection_eur_per_kw", "storage_eur_per_kwh", "withdrawal_eur_per_kw")
CSV_FIELDS = ("Storage class", "Site", "Country", "H2 reference working GWh",
              "Cavern volume maximum m3", "Reference depth m",
              "Abandonment pressure bar", "Minimum pressure bar",
              "Min. Op. Pressure", "Maximum pressure bar", "Notes and sources")

# Only EPR (days) and WTIR (withdrawal/injection ratio) are fixed. Each site's
# capacity E comes from its "H2 reference working GWh" CSV field.
REFERENCE_RATIOS = {"DGF": (66., 2.), "SA": (66., 2.),
                    "SC": (12., 2.), "LRC": (7.5, 2.)}
CAPACITY_SWEEP_FACTORS = (0.5, 1.5)
DESIGN_RANGES = {
    "DGF": ((33., 66., 100.), (1., 2., 3.)),
    "SA": ((33., 66., 100.), (1., 2., 3.)),
    "SC": ((6., 12., 18.), (1., 2., 3.)),
    "LRC": ((5., 7.5, 10.), (1., 2., 3.)),
}

# Reference settings from the old exogenous cost-sensitivity registry. The
# paired site depth/pressures override the geological defaults afterwards.
GENERAL_ALL = {"pipeline_pressure_pa": 55e5,
               "gas_plant_to_nearest_wellhead_distance_km": 2.,
               "drilling_complexity_index": 1.}
GENERAL_BY_CLASS = {
    "DGF": {"recovery_factor": .80, "purification_factor": 1.5,
            "purification_hydrogen_recovery": .8996},
    "SA": {"recovery_factor": .90, "purification_factor": .33,
           "purification_hydrogen_recovery": .95,
           "minimum_gas_saturation": .60, "maximum_gas_saturation": .80},
    "SC": {"recovery_factor": .9983},
    "LRC": {"recovery_factor": 1.},
}
TECHNOLOGY_BY_CLASS = {
    "DGF": {"number_production_well_heads": 24,
            "number_observation_well_heads": 6},
    "SA": {"number_production_well_heads": 24,
           "number_observation_well_heads": 6},
    "SC": {"working_gas_volume_per_cavern_sm3": 31.25e6,
           "maximum_withdrawal_flow_per_cavern_million_sm3_per_day": 2.8,
           "fresh_water_pipeline_length_km": 15.,
           "brine_disposal_pipeline_length_km": 30.,
           "debrining_flowrate_per_cavern_m3_per_hour": 200.,
           "number_leaching_pumps": 4},
    "LRC": {"maximum_h2_mass_per_well_kg": 3e6,
            "maximum_cavern_radius_m": 20., "cavern_height_m": 100.,
            "effective_volume_fraction": .95,
            "steel_lining_thickness_m": .015,
            "concrete_lining_thickness_m": .50},
}
HYSTORIES_COST_OVERRIDES = {
    "hydrogen_cost_eur_per_kg": 6.71,
    "cost_of_electricity_eur_per_mwh": 80.,
    "material_cost_factor_injection": 1.5,
    "material_cost_factor_withdrawal": 1.5,
    "surface_contingency_fraction": .20,
    "subsurface_contingency_fraction": .20,
    "surface_abex_fraction": .20,
    "subsurface_abex_fraction": .20,
    "bop_fraction": .05,
}
LRC_COST_OVERRIDES = {
    "lrc_well_fixed_cost_eur_per_well": 2_579_552.1611,
    "lrc_well_variable_cost_eur_per_m": 1_521.4581,
    "lrc_mining_cost_eur_per_m3": 100.3159,
    "lrc_steel_lining_cost_eur_per_tonne": 1_086.7558,
    "lrc_concrete_lining_cost_eur_per_tonne": 60.1896,
    "lrc_lining_installation_fraction": .05,
}


@dataclass(frozen=True)
class Site:
    row_index: int
    cls: str
    name: str
    country: str
    depth_m: float | None
    abandonment_bar: float | None
    pmin_bar: float | None
    pmax_bar: float | None
    h2_reference_gwh: float | None
    cavern_max_m3: float | None
    notes: str
    min_source: str
    raw_pmin_bar: float | None = None

    def vector(self) -> tuple[float, ...]:
        # The site configuration includes scale as well as geological properties.
        vals = [self.h2_reference_gwh, self.depth_m,
                self.pmin_bar, self.pmax_bar]
        if self.cls == "DGF":
            vals.insert(1, self.abandonment_bar)
        if any(x is None for x in vals):
            raise ValueError(f"Incomplete site: {self.cls}/{self.name}")
        return tuple(float(x) for x in vals)


def number(value: str | None) -> float | None:
    s = (value or "").strip().replace(",", "")
    if not s:
        return None
    val = float(s)
    if not math.isfinite(val):
        raise ValueError(f"Nonfinite numeric CSV value {value!r}")
    return val


def read_sites(path: Path) -> list[Site]:
    if not path.is_file():
        raise FileNotFoundError(path)
    sites = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(CSV_FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f"Missing CSV column(s): {set(CSV_FIELDS) - set(reader.fieldnames or [])}")
        for i, row in enumerate(reader, start=2):
            cls = row["Storage class"].strip().upper()
            if cls not in CLASSES:
                raise ValueError(f"CSV row {i}: unknown class {cls!r}")
            explicit_min = number(row["Minimum pressure bar"])
            chosen_min = number(row["Min. Op. Pressure"])
            if chosen_min is None:
                chosen_min = explicit_min
                source = "raw minimum; operating field unavailable"
            elif explicit_min is None:
                source = "operating field; raw minimum missing (potentially imputed)"
            else:
                source = "operating field; raw minimum also available"
            sites.append(Site(
                row_index=i, cls=cls, name=row["Site"].strip(),
                country=row["Country"].strip(),
                depth_m=number(row["Reference depth m"]),
                abandonment_bar=number(row["Abandonment pressure bar"]),
                pmin_bar=chosen_min, pmax_bar=number(row["Maximum pressure bar"]),
                h2_reference_gwh=number(row["H2 reference working GWh"]),
                cavern_max_m3=number(row["Cavern volume maximum m3"]),
                notes=row["Notes and sources"].strip(), min_source=source,
                raw_pmin_bar=explicit_min,
            ))
    return sites


def site_issue(site: Site, min_lrc_volume: float) -> str:
    if site.cls == "LRC" and site.cavern_max_m3 is not None:
        if site.cavern_max_m3 < min_lrc_volume:
            return f"pilot cavern under {min_lrc_volume:g} m3"
    required = [("H2 reference working GWh", site.h2_reference_gwh),
                ("depth", site.depth_m), ("pmin", site.pmin_bar),
                ("pmax", site.pmax_bar)]
    if site.cls == "DGF":
        required.append(("abandonment", site.abandonment_bar))
    missing = [key for key, value in required if value is None]
    if missing:
        return "missing " + ", ".join(missing)
    if site.h2_reference_gwh <= 0:
        return "nonpositive H2 reference working GWh"
    if site.depth_m <= 0 or not 0 < site.pmin_bar < site.pmax_bar:
        return "invalid depth or pressure order"
    pipeline_bar = GENERAL_ALL["pipeline_pressure_pa"] / 1e5
    if site.pmax_bar <= pipeline_bar:
        return (f"pmax {site.pmax_bar:g} bar does not exceed "
                f"assumed pipeline pressure {pipeline_bar:g} bar")
    if site.cls == "DGF" and not 0 < site.abandonment_bar < site.pmin_bar:
        return "invalid DGF abandonment-pressure order"
    return ""


def site_fields(site: Site) -> dict[str, Any]:
    return {"storage_class": site.cls, "site": site.name,
            "country": site.country, "csv_row": site.row_index,
            "depth_m": site.depth_m,
            "abandonment_pressure_bar": site.abandonment_bar,
            "minimum_pressure_bar": site.pmin_bar,
            "maximum_pressure_bar": site.pmax_bar,
            "E_gwh": site.h2_reference_gwh,
            "cavern_maximum_volume_m3": site.cavern_max_m3,
            "min_pressure_source": site.min_source, "notes_and_sources": site.notes}


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_models() -> dict[str, tuple[Any, Any, Any]]:
    """Load the current package without importing either superseded script."""
    from uhs_costs.design.aquifer import ASSUMPTIONS as sa_ass, construct_project as sa_build
    from uhs_costs.design.depleted_gas_field import ASSUMPTIONS as dgf_ass, construct_project as dgf_build
    from uhs_costs.design.lined_rock_cavern import ASSUMPTIONS as lrc_ass, construct_project as lrc_build
    from uhs_costs.design.salt_cavern import ASSUMPTIONS as sc_ass, construct_project as sc_build
    from uhs_costs.cost_model.aquifer import calculate_cost_components as sa_cost
    from uhs_costs.cost_model.depleted_gas_field import calculate_depleted_gas_field_cost_components as dgf_cost
    from uhs_costs.cost_model.lined_rock_cavern import calculate_lined_rock_cavern_cost_components as lrc_cost
    from uhs_costs.cost_model.salt_cavern import calculate_salt_cavern_cost_components as sc_cost
    from uhs_costs.design.default_design_assumptions import calculate_temperature_from_depth_k
    return {
        "DGF": (dgf_ass, dgf_build, dgf_cost),
        "SA": (sa_ass, sa_build, sa_cost),
        "SC": (sc_ass, sc_build, sc_cost),
        "LRC": (lrc_ass, lrc_build, lrc_cost),
        "temperature_fn": calculate_temperature_from_depth_k,
    }


def component_rates(breakdown: Any) -> dict[str, float]:
    """Component CAPEX+ABEX/driver, matching the previous study definition."""
    totals = defaultdict(float)
    components = getattr(breakdown, "components", None)
    if components is None:
        if isinstance(breakdown, (tuple, list)):
            components = breakdown
        else:
            raise TypeError(f"Unknown cost breakdown: {type(breakdown).__name__}")
    for part in components:
        category = getattr(part.cost_type, "value", part.cost_type)
        if str(category).strip().lower() not in {"capex", "abex"}:
            continue
        driver = str(getattr(part.cost_driver, "value", part.cost_driver)).lower()
        name = str(part.name).lower()
        if "injection" in driver or "injection" in name:
            group = COSTS[0]
        elif "withdrawal" in driver or "withdrawal" in name:
            group = COSTS[2]
        elif "storage" in driver or "subsurface" in name:
            group = COSTS[1]
        else:
            # A positive CAPEX/ABEX with unallocated component is a QA issue.
            if float(part.value_eur) != 0:
                raise ValueError(f"Unclassified TCR component {part.name!r}, driver {driver!r}")
            continue
        denominator = float(part.driver_value)
        if not math.isfinite(denominator) or denominator <= 0:
            raise ValueError(f"Invalid component driver {part.name!r}: {denominator}")
        rate = float(part.value_eur) / denominator
        if not math.isfinite(rate) or rate < 0:
            raise ValueError(f"Invalid component rate {part.name!r}: {rate}")
        totals[group] += rate
    if set(totals) != set(COSTS) or any(totals[c] <= 0 for c in COSTS):
        raise ValueError(f"Missing/nonpositive component TCR: {dict(totals)}")
    return dict(totals)


def model_costs(site: Site, models: dict, facility: tuple[float, float, float]) -> dict[str, float]:
    E_gwh, EPR_days, WTIR = facility
    if not (E_gwh > 0 and EPR_days > 0 and WTIR > 0):
        raise ValueError(f"Nonpositive E/EPR/WTIR: {facility}")
    assumptions, constructor, cost_function = models[site.cls]
    general_updates = {
        **GENERAL_ALL, **GENERAL_BY_CLASS[site.cls],
        "reference_reservoir_depth_m": site.depth_m,
        "minimum_operating_pressure_pa": site.pmin_bar * 1e5,
        "maximum_operating_pressure_pa": site.pmax_bar * 1e5,
        "temperature_k": models["temperature_fn"](site.depth_m),
    }
    if site.cls == "DGF":
        general_updates["abandonment_pressure_pa"] = site.abandonment_bar * 1e5
    # Fail fast if package field names changed: no silent loss of overrides.
    updated_general = replace(assumptions.general, **general_updates)
    updated_tech = replace(assumptions.technology_specific,
                           **TECHNOLOGY_BY_CLASS[site.cls])
    updated_assumptions = replace(assumptions, general=updated_general,
                                  technology_specific=updated_tech)
    E_kwh = E_gwh * 1_000_000.
    withdrawal_kw = E_kwh / (EPR_days * 24.)
    injection_kw = withdrawal_kw / WTIR
    project = constructor(
        working_gas_capacity_kwh_lhv=E_kwh,
        withdrawal_flow_kw_h2_lhv=withdrawal_kw,
        injection_flow_kw_h2_lhv=injection_kw,
        assumptions=updated_assumptions, case_name=TECHS[site.cls],
    )
    if site.cls == "LRC":
        result = cost_function(project, overrides=HYSTORIES_COST_OVERRIDES,
                               lrc_overrides=LRC_COST_OVERRIDES)
    else:
        result = cost_function(project, overrides=HYSTORIES_COST_OVERRIDES)
    return component_rates(result)


def reference_facility(site: Site) -> tuple[float, float, float]:
    """Use the site's CSV working capacity with class-fixed EPR and WTIR."""
    if (site.h2_reference_gwh is None or not math.isfinite(site.h2_reference_gwh)
            or site.h2_reference_gwh <= 0):
        raise ValueError(f"Missing/invalid H2 working capacity: {site.cls}/{site.name}")
    return (site.h2_reference_gwh, *REFERENCE_RATIOS[site.cls])


def weighted_median(values: list[float], weights: list[float]) -> float:
    """Weighted Q50 with interpolation between cumulative-weight midpoints."""
    pairs = sorted(zip(values, weights))
    if not pairs or sum(weights) <= 0:
        raise ValueError("Empty/invalid weighted median")
    total = sum(weights)
    points = []
    cumulative = 0.
    for value, weight in pairs:
        if weight <= 0:
            raise ValueError("Nonpositive Q50 weight")
        points.append(((cumulative + .5 * weight) / total, value))
        cumulative += weight
    if .5 <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= .5 <= x1:
            alpha = (.5-x0)/(x1-x0)
            return y0 + alpha*(y1-y0)
    return points[-1][1]


def medoid_ranking(vectors: list[tuple[float, ...]], weights: list[float]
                   ) -> list[tuple[float, int]]:
    """All-pairs Euclidean medoid in robustly normalised log-coordinate space.

    Coordinate log scaling handles physical units and proportional differences.
    Sample IQR normalisation avoids depth or one cost channel dominating merely
    by numerical range. A zero-IQR axis falls back to its finite log range.
    """
    if not vectors or any(len(v) != len(vectors[0]) for v in vectors):
        raise ValueError("Inconsistent/empty medoid vectors")
    if len(vectors) != len(weights) or any(w <= 0 for w in weights):
        raise ValueError("Invalid medoid weights")
    if any(x <= 0 or not math.isfinite(x) for v in vectors for x in v):
        raise ValueError("All medoid features must be finite and strictly positive")
    log_v = [[math.log(x) for x in vector] for vector in vectors]
    axes = list(zip(*log_v))
    def quantile(v: tuple[float, ...], q: float) -> float:
        s = sorted(v)
        t = (len(s) - 1) * q
        i = math.floor(t)
        return s[i] if i == len(s) - 1 else s[i] * (1 - (t-i)) + s[i+1] * (t-i)
    scales = []
    for axis in axes:
        iqr = quantile(axis, .75) - quantile(axis, .25)
        scale = iqr or (max(axis) - min(axis)) or 1.
        scales.append(scale)
    scores = []
    for i, row in enumerate(log_v):
        score = 0.
        for j, other in enumerate(log_v):
            distance = math.sqrt(sum(((a-b)/s)**2
                                     for a, b, s in zip(row, other, scales)))
            score += weights[j] * distance
        scores.append((score / sum(weights), i))
    return sorted(scores, key=lambda item: (item[0], item[1]))


def group_key(site: Site) -> tuple[str, str, str]:
    """Normalise country abbreviations and common transliterated site names."""
    countries = {"de": "germany", "fr": "france", "nl": "netherlands",
                 "at": "austria", "se": "sweden", "uk": "united kingdom"}
    country = site.country.strip().lower()
    country = countries.get(country, country)
    # German transliteration: Hähnlein == Haehnlein, both appear in the CSV.
    site_name = site.name.lower().translate(str.maketrans(
        {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}))
    site_name = unicodedata.normalize("NFKD", site_name)
    site_name = "".join(ch for ch in site_name if ch.isalnum())
    return site.cls, country, site_name


def site_weights(sites: list[Site]) -> list[float]:
    """Repeated facility configurations share unit weight; not capacity-weighted."""
    keys = [group_key(s) for s in sites]
    counts = {key: keys.count(key) for key in set(keys)}
    return [1. / counts[key] for key in keys]


def average_design(cls: str, sites: list[Site], min_lrc_volume: float
                   ) -> tuple[Site, dict[str, int]]:
    """Mean each available source field, with one vote per distinct facility.

    The population includes partially documented sites. For minimum pressure,
    use only the *reported* minimum, not the 55-bar pipeline fallback. The
    separate resulting means may not describe a real geological configuration;
    validate the combination before passing it to the cost model.
    """
    eligible = [s for s in sites if s.cls == cls and not (
        cls == "LRC" and s.cavern_max_m3 is not None
        and s.cavern_max_m3 < min_lrc_volume)]
    if not eligible:
        raise ValueError(f"No source sites for synthetic {cls} design")
    fields = ("h2_reference_gwh", "depth_m", "raw_pmin_bar", "pmax_bar")
    if cls == "DGF":
        fields += ("abandonment_bar",)
    values: dict[str, float] = {}
    counts: dict[str, int] = {}
    for field in fields:
        by_facility: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for site in eligible:
            value = getattr(site, field)
            if value is None:
                continue
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid {field} for {cls}/{site.name}: {value}")
            by_facility[group_key(site)].append(value)
        if not by_facility:
            raise ValueError(f"No documented {field} values to average for {cls}")
        # Repeated entries for one facility are first consolidated; no site is
        # weighted by how many duplicate records happen to be in the CSV.
        site_means = [sum(xs) / len(xs) for xs in by_facility.values()]
        values[field] = sum(site_means) / len(site_means)
        counts[field] = len(site_means)
    result = Site(
        row_index=0, cls=cls, name="Arithmetic mean (synthetic)",
        country="Synthetic comparison", depth_m=values["depth_m"],
        abandonment_bar=values.get("abandonment_bar"),
        pmin_bar=values["raw_pmin_bar"], pmax_bar=values["pmax_bar"],
        h2_reference_gwh=values["h2_reference_gwh"], cavern_max_m3=None,
        notes="Independent source-field arithmetic means; not a real site.",
        min_source="Mean of reported minima only (no 55-bar fallback)",
        raw_pmin_bar=values["raw_pmin_bar"],
    )
    issue = site_issue(result, min_lrc_volume)
    if issue:
        raise ValueError(
            f"Synthetic {cls} means are not a valid cost-model configuration: "
            f"{issue}; values={values}. Do not silently repair these means."
        )
    return result, counts


def evaluate_class(cls: str, sites: list[Site], models: dict,
                   quality: dict[int, dict], min_sites: int,
                   min_lrc_volume: float,
                   ) -> tuple[dict, list[dict], list[dict], list[Site], dict | None]:
    selected = [s for s in sites if s.cls == cls and not quality[s.row_index]["reason"]]
    successful: list[Site] = []
    rates: list[dict[str, float]] = []
    for site in selected:
        try:
            result = model_costs(site, models, reference_facility(site))
        except Exception as exc:
            quality[site.row_index]["status"] = "model_failure"
            quality[site.row_index]["reason"] = f"{type(exc).__name__}: {exc}"
        else:
            quality[site.row_index]["status"] = "model_success"
            successful.append(site)
            rates.append(result)
    n = len(successful)
    n_groups = len({group_key(s) for s in successful})
    summary: dict[str, Any] = {
        "storage_class": cls, "n_total": sum(s.cls == cls for s in sites),
        "n_eligible": len(selected), "n_model_success": n,
        "n_distinct_facilities": n_groups,
        "n_model_failure": len(selected)-n, "minimum_cohort_size": min_sites,
        "capacity_basis": "site H2 reference working GWh (CSV)",
        "min_E_gwh": min((s.h2_reference_gwh for s in successful), default=None),
        "max_E_gwh": max((s.h2_reference_gwh for s in successful), default=None),
        "EPR_days": REFERENCE_RATIOS[cls][0],
        "WTIR": REFERENCE_RATIOS[cls][1],
        "status": "underpowered" if n_groups < min_sites else "ok",
    }
    if n_groups < min_sites:
        summary["issue"] = (f"Only {n_groups} independent facilities ({n} rows); "
                            f"need >= {min_sites} for a medoid comparison")
        return summary, [], [], successful, None
    weights = site_weights(successful)
    geo = medoid_ranking([s.vector() for s in successful], weights)
    econ = medoid_ranking([tuple(r[c] for c in COSTS) for r in rates], weights)
    geo_i, econ_i = geo[0][1], econ[0][1]
    gscore = dict((i, score) for score, i in geo)
    escore = dict((i, score) for score, i in econ)
    medians = {c: weighted_median([r[c] for r in rates], weights) for c in COSTS}
    synthetic_site, mean_counts = average_design(cls, sites, min_lrc_volume)
    try:
        synthetic_rates = model_costs(synthetic_site, models,
                                      reference_facility(synthetic_site))
    except Exception as exc:
        raise RuntimeError(
            f"Synthetic {cls} arithmetic-mean design could not be costed: "
            f"{type(exc).__name__}: {exc}; source counts={mean_counts}"
        ) from exc
    synthetic = {**site_fields(synthetic_site), **synthetic_rates,
                 "EPR_days": REFERENCE_RATIOS[cls][0],
                 "WTIR": REFERENCE_RATIOS[cls][1],
                 **{f"n_source_{field}": count
                    for field, count in mean_counts.items()}}
    rows = []
    for i, (site, result) in enumerate(zip(successful, rates)):
        rows.append({**site_fields(site), **result,
                     "group_weight": weights[i],
                     "design_medoid_score": gscore[i],
                     "economic_medoid_score": escore[i],
                     "is_design_medoid": i == geo_i,
                     "is_economic_medoid": i == econ_i})
    summary.update({"design_medoid_site": successful[geo_i].name,
                    "design_medoid_row": successful[geo_i].row_index,
                    "economic_medoid_site": successful[econ_i].name,
                    "economic_medoid_row": successful[econ_i].row_index,
                    "same_medoid": geo_i == econ_i,
                    "issue": ""})
    comparisons = []
    for c in COSTS:
        x = rates[geo_i][c]
        y = rates[econ_i][c]
        target = medians[c]
        synthetic_rate = synthetic_rates[c]
        comparisons.append({"storage_class": cls, "component": c,
                            "median_site_cost_rate": target,
                            "design_medoid_rate": x,
                            "economic_medoid_rate": y,
                            "synthetic_mean_rate": synthetic_rate,
                            "design_vs_econ_pct": 100*(x/y - 1),
                            "design_vs_median_pct": 100*(x/target - 1),
                            "econ_vs_median_pct": 100*(y/target - 1),
                            "synthetic_vs_median_pct": 100*(synthetic_rate/target - 1),
                            "synthetic_vs_econ_pct": 100*(synthetic_rate/y - 1)})
    return summary, rows, comparisons, [successful[geo_i], successful[econ_i]], synthetic


def sweep_design(cls: str, medoids: list[Site], models: dict,
                 n_steps: int, quality: dict[int, dict]
                 ) -> list[dict[str, Any]]:
    """OAT E/EPR/WTIR screens about each selected site's CSV working capacity."""
    result = []
    for parameter_index, parameter in enumerate(("E_gwh", "EPR_days", "WTIR")):
        for role, site in zip(("design_medoid", "economic_medoid"), medoids):
            baseline = reference_facility(site)
            if parameter_index == 0:
                low = baseline[0] * CAPACITY_SWEEP_FACTORS[0]
                ref = baseline[0]
                high = baseline[0] * CAPACITY_SWEEP_FACTORS[1]
            else:
                low, ref, high = DESIGN_RANGES[cls][parameter_index-1]
            if not 0 < low <= ref <= high:
                raise ValueError(f"Invalid OAT range for {cls}/{site.name}: {(low, ref, high)}")
            if low == high:
                continue
            values = [low + i*(high-low)/(n_steps-1) for i in range(n_steps)]
            if not any(math.isclose(v, ref) for v in values):
                values.append(ref)
            for value in sorted(values):
                facility = list(baseline)
                facility[parameter_index] = value
                record = {"storage_class": cls, "selected_role": role,
                          "selected_site": site.name, "csv_row": site.row_index,
                          "varied_parameter": parameter, "parameter_value": value,
                          "reference_E_gwh": baseline[0],
                          "tested_E_gwh": facility[0],
                          "tested_EPR_days": facility[1],
                          "tested_WTIR": facility[2]}
                try:
                    record.update(model_costs(site, models, tuple(facility)))
                    record["status"] = "ok"
                    record["error"] = ""
                except Exception as exc:
                    record["status"] = "model_failure"
                    record["error"] = f"{type(exc).__name__}: {exc}"
                result.append(record)
    return result


def plot_cost_distribution(cls: str, rows: list[dict[str, Any]],
                           comparisons: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plasma = plt.get_cmap("plasma")
    colors = {"design": plasma(.13), "economic": plasma(.53),
              "synthetic": plasma(.83)}
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.7), layout="constrained")
    for ax, metric, comp in zip(axes, COSTS, comparisons):
        values = sorted(float(row[metric]) for row in rows)
        point_colors = [plasma(.22 + .56*i/max(1, len(values)-1))
                        for i in range(len(values))]
        ax.scatter(range(len(values)), values, s=23, alpha=.70,
                   c='grey', label="Evaluated sites", zorder=2)
        ax.axhline(comp["median_site_cost_rate"], color="0.3", ls="--", lw=1.2)
        ax.axhline(comp["design_medoid_rate"], color=colors["design"], lw=2.0)
        ax.axhline(comp["economic_medoid_rate"], color=colors["economic"], lw=2.0)
        ax.axhline(comp["synthetic_mean_rate"], color=colors["synthetic"],
                   lw=2.0, ls="-.")
        labels = {COSTS[0]: "Injection [EUR/kW]", COSTS[1]: "Storage [EUR/kWh]",
                  COSTS[2]: "Withdrawal [EUR/kW]"}
        ax.set_title(labels[metric])
        ax.set_xlabel("Candidate rank (ascending TCR)")
        ax.set_xlim(-.5, len(values)-.5)
        ax.grid(axis="y", color=".9", lw=.8)
    fig.suptitle(f"{cls}: cost distribution, medoids, and arithmetic-mean design")
    fig.legend(handles=[Line2D([0], [0], color="0.3", ls="--", label="Site median"),
                        Line2D([0], [0], color=colors["design"], lw=2, label="Design medoid"),
                        Line2D([0], [0], color=colors["economic"], lw=2, label="Economic medoid"),
                        Line2D([0], [0], color=colors["synthetic"], lw=2, ls="-.",
                               label="Arithmetic-mean design")],
               ncol=4, loc="lower center", bbox_to_anchor=(.5, -.10), frameon=False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_sweeps(cls: str, sweep: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt
    plasma = plt.get_cmap("plasma")
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), layout="constrained")
    roles = [("design_medoid", plasma(.13)),
             ("economic_medoid", plasma(.53))]
    names = ("E_gwh", "EPR_days", "WTIR")
    for j, parameter in enumerate(names):
        for k, cost in enumerate(COSTS):
            ax = axes[k][j]
            for role, color in roles:
                sampled = sorted((r for r in sweep if r["varied_parameter"] == parameter
                                  and r["selected_role"] == role and r["status"] == "ok"),
                                 key=lambda r: r["parameter_value"])
                if sampled:
                    ax.plot([r["parameter_value"] for r in sampled],
                            [r[cost] for r in sampled], "-o", markersize=3,
                            color=color, label=role.replace("_", " ").title())
            ax.grid(color=".9", lw=.8)
            if k == 0:
                ax.set_title(parameter.replace("_", " "))
            if j == 0:
                ax.set_ylabel({COSTS[0]: "Injection [EUR/kW]",
                               COSTS[1]: "Storage [EUR/kWh]",
                               COSTS[2]: "Withdrawal [EUR/kW]"}[cost])
            if k == 2:
                ax.set_xlabel(parameter.replace("_", " "))
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{cls}: design variation at selected site configurations (OAT)")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("data/resources/uhs_reference_sites.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/reference_selection"))
    parser.add_argument("--classes", nargs="+", choices=CLASSES,
                        default=list(CLASSES))
    parser.add_argument("--min-sites", type=int, default=3,
                        help="Minimum complete costed site count before reporting medoids")
    parser.add_argument("--min-lrc-cavern-m3", type=float, default=5000.,
                        help="Exclude demonstrator-sized LRC pilot caverns")
    parser.add_argument("--require-raw-pmin", action="store_true",
                        help="Exclude rows whose minimum operating pressure exists only as an inferred value")
    parser.add_argument("--sweep-points", type=int, default=9)
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate and tabulate CSV without importing uhs_costs")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--allow-underpowered", action="store_true",
                        help="Write partial results without nonzero exit for thin classes")
    parser.add_argument("--allow-model-failures", action="store_true",
                        help="Write partial results without nonzero exit for failed model runs")
    args = parser.parse_args()
    if args.min_sites < 3 or args.sweep_points < 3 or args.min_lrc_cavern_m3 <= 0:
        parser.error("Need min-sites >=3, sweep-points >=3, LRC cavern threshold >0")
    sites = [s for s in read_sites(args.input) if s.cls in args.classes]
    if not sites:
        raise RuntimeError("No candidates for requested classes")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    quality = {}
    for site in sites:
        problem = site_issue(site, args.min_lrc_cavern_m3)
        if not problem and args.require_raw_pmin and "raw minimum missing" in site.min_source:
            problem = "operating minimum is inferred; --require-raw-pmin is set"
        quality[site.row_index] = {
            **site_fields(site), "status": "incomplete_or_excluded" if problem else "eligible_for_costing",
            "reason": problem,
        }
    print(f"Read {len(sites)} site rows from {args.input}")
    for cls in args.classes:
        all_count = sum(s.cls == cls for s in sites)
        complete = sum(s.cls == cls and not quality[s.row_index]["reason"] for s in sites)
        print(f"  {cls}: {complete}/{all_count} complete capacity/geology configurations")

    summaries: list[dict] = []
    ranked: list[dict] = []
    comparisons: list[dict] = []
    sensitivity: list[dict] = []
    synthetic_designs: list[dict] = []
    if not args.validate_only:
        models = load_models()
        for cls in args.classes:
            summary, class_rows, comp, medoids, synthetic = evaluate_class(
                cls, sites, models, quality, args.min_sites,
                args.min_lrc_cavern_m3)
            summaries.append(summary)
            ranked.extend(class_rows)
            comparisons.extend(comp)
            print(f"  {cls} result: {summary['status']}; "
                  f"costed {summary['n_model_success']} candidates")
            if summary["status"] != "ok":
                print(f"    WARNING: {summary['issue']}")
                continue
            print(f"    Design Medoid: {summary['design_medoid_site']}; "
                  f"Economic Medoid: {summary['economic_medoid_site']}")
            if synthetic is None:
                raise RuntimeError(f"Missing synthetic mean despite eligible {cls} cohort")
            synthetic_designs.append(synthetic)
            print("    Arithmetic-mean design (independent available-source means): "
                  f"E={synthetic['E_gwh']:.2f} GWh, "
                  f"depth={synthetic['depth_m']:.1f} m, "
                  f"pmin={synthetic['minimum_pressure_bar']:.1f} bar, "
                  f"pmax={synthetic['maximum_pressure_bar']:.1f} bar" +
                  (f", pab={synthetic['abandonment_pressure_bar']:.1f} bar"
                   if cls == "DGF" else ""))
            print("      Rates: " + ", ".join(
                f"{key}={synthetic[key]:.4g}" for key in COSTS))
            print("      Available-site counts per mean: " + ", ".join(
                f"{field}={synthetic[f'n_source_{field}']}"
                for field in (("h2_reference_gwh", "depth_m", "raw_pmin_bar",
                               "pmax_bar", "abandonment_bar") if cls == "DGF"
                              else ("h2_reference_gwh", "depth_m",
                                    "raw_pmin_bar", "pmax_bar"))))
            sweep = sweep_design(cls, medoids, models, args.sweep_points, quality)
            sensitivity.extend(sweep)
            if not args.no_plots:
                plot_cost_distribution(cls, class_rows, comp,
                                       args.output_dir / f"{cls.lower()}_cost_distribution.pdf")
                plot_sweeps(cls, sweep,
                            args.output_dir / f"{cls.lower()}_design_sensitivity.pdf")

    quality_rows = [quality[s.row_index] for s in sites]
    quality_cols = list(quality_rows[0])
    write_csv(args.output_dir / "candidate_quality.csv", quality_rows, quality_cols)
    if args.validate_only:
        print("CSV-only validation complete; cost functions not executed.")
        return

    summary_cols = ["storage_class", "status", "issue", "n_total", "n_eligible",
                    "n_model_success", "n_distinct_facilities", "n_model_failure",
                    "minimum_cohort_size",
                    "capacity_basis", "min_E_gwh", "max_E_gwh",
                    "EPR_days", "WTIR", "design_medoid_site",
                    "design_medoid_row", "economic_medoid_site",
                    "economic_medoid_row", "same_medoid"]
    write_csv(args.output_dir / "selection_summary.csv", summaries, summary_cols)
    if ranked:
        write_csv(args.output_dir / "candidate_cost_rates.csv", ranked,
                  list(ranked[0]))
        selected_rows = []
        for r in ranked:
            if r["is_design_medoid"] or r["is_economic_medoid"]:
                selected_rows.append(r)
        write_csv(args.output_dir / "selected_references.csv", selected_rows,
                  list(ranked[0]))
    comparison_cols = ["storage_class", "component", "median_site_cost_rate",
                       "design_medoid_rate", "economic_medoid_rate",
                       "synthetic_mean_rate", "design_vs_econ_pct",
                       "design_vs_median_pct", "econ_vs_median_pct",
                       "synthetic_vs_median_pct", "synthetic_vs_econ_pct"]
    write_csv(args.output_dir / "medoid_cost_comparison.csv", comparisons,
              comparison_cols)
    if synthetic_designs:
        mean_columns = list(synthetic_designs[0])
        for item in synthetic_designs[1:]:
            mean_columns.extend(col for col in item if col not in mean_columns)
        write_csv(args.output_dir / "synthetic_mean_designs.csv",
                  synthetic_designs, mean_columns)
    if sensitivity:
        write_csv(args.output_dir / "selected_design_sensitivity.csv", sensitivity,
                  list(sensitivity[0]))
    failed = [r for r in quality_rows if r["status"] == "model_failure"]
    underpowered = [r for r in summaries if r["status"] == "underpowered"]
    sweeps_failed = [r for r in sensitivity if r["status"] == "model_failure"]
    if failed or sweeps_failed:
        print(f"WARNING: {len(failed)} baseline model failures and "
              f"{len(sweeps_failed)} sweep failures; see diagnostic CSVs")
    if underpowered:
        print("WARNING: underpowered classes: " + ", ".join(r["storage_class"] for r in underpowered))
    print("Saved diagnostics and results:", args.output_dir)
    if (failed or sweeps_failed) and not args.allow_model_failures:
        raise RuntimeError("Model failures present; inspect outputs or use --allow-model-failures")
    if underpowered and not args.allow_underpowered:
        raise RuntimeError("Insufficient complete sites; inspect outputs or use --allow-underpowered")


if __name__ == "__main__":
    main()
