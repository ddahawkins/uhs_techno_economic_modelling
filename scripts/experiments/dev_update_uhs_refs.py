"""One-time, auditable NG-to-H2 working-capacity conversion for UHS sites.

From the repository root::

    pixi run python -m scripts.experiments.update_uhs_capacity

Inputs:  data/resources/uhs_reference_sites.csv
Outputs: data/resources/uhs_reference_sites_h2_adjusted.csv
         outputs/reference_selection/capacity_conversion_audit.csv

Never edits the input. Preserves published H2 estimates (HyUnder etc.). For
reported NG working Sm3 with depth and a pressure window, it corrects the old
same-standard-volume approximation using CoolProp real-gas Z for pure methane
and H2. This assumes the same reservoir volume, temperature and operating
pressure window, but does NOT validate geochemistry, deliverability or well
integrity for H2. Natural-gas composition is approximated as pure methane.

An NG TWh-only conversion is off by default because it requires an assumed
gas calorific value and a reporting basis. Enable --infer-from-twh knowingly.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Callable


H2_STANDARD_DENSITY_KG_M3 = 0.0852  # Provided study standard, at 15 C, 101325 Pa
H2_LHV_KWH_KG = 33.33
H2_KWH_PER_STANDARD_M3 = H2_STANDARD_DENSITY_KG_M3 * H2_LHV_KWH_KG
MARKER = "[UHS_H2_Z_ADJUSTMENT_V1]"

FIELDS = (
    "Storage class", "Site", "Country", "H2 reference working GWh",
    "NG working million Sm3", "NG working TWh", "Reference depth m",
    "Minimum pressure bar", "Min. Op. Pressure", "Maximum pressure bar",
    "Notes and sources",
)

# These entries have a separate published H2-based estimate rather than the
# original direct NG-Sm3 x 2.839716 kWh/Sm3 approximation. Protect them even if
# the source note is edited. Any *other* mismatch is also preserved and flagged.
PRESERVE_PUBLISHED_H2 = {
    ("LRC", "skallen"),
    ("SA", "hähnlein"),
    ("SA", "stenlille"),
    ("DGF", "roden"),
    ("DGF", "rubensdorf"),
    ("SC", "clemens"),
    ("SC", "moss bluff"),
    ("SC", "teesside"),
}


def as_float(text: str | None) -> float | None:
    raw = (text or "").strip().replace(",", "")
    if not raw:
        return None
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite source number: {text!r}")
    return value


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        missing = set(FIELDS) - set(fieldnames)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        return fieldnames, list(reader)


def safe_positive(value: float, label: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and >0, found {value}")
    return value


def real_gas_volume_ratio(pmin_bar: float, pmax_bar: float, temperature_k: float,
                          z: Callable[[str, float, float], float]) -> tuple[float, dict]:
    """Standard H2 WGV / standard CH4 WGV at fixed physical volume and P,T.

    Real gas: n_working = V/R/T * (Pmax/Zmax - Pmin/Zmin).
    The absolute-volume and temperature factors cancel for the two gases.
    Pressure must be ABSOLUTE bar; no 1 bar offset is automatically applied.
    """
    pmin_bar = safe_positive(pmin_bar, "pmin [bar(a)]")
    pmax_bar = safe_positive(pmax_bar, "pmax [bar(a)]")
    safe_positive(temperature_k, "temperature [K]")
    if pmax_bar <= pmin_bar:
        raise ValueError(f"pmax={pmax_bar:g} must exceed pmin={pmin_bar:g}")
    props = {}
    for gas in ("Hydrogen", "Methane"):
        zmin = safe_positive(float(z(gas, pmin_bar*1e5, temperature_k)),
                             f"Z {gas} at pmin")
        zmax = safe_positive(float(z(gas, pmax_bar*1e5, temperature_k)),
                             f"Z {gas} at pmax")
        delta = pmax_bar/zmax - pmin_bar/zmin
        if delta <= 0 or not math.isfinite(delta):
            raise ValueError(f"Nonpositive usable pressure-density interval: {gas}")
        props[gas] = {"zmin": zmin, "zmax": zmax, "delta": delta}
    ratio = props["Hydrogen"]["delta"] / props["Methane"]["delta"]
    safe_positive(ratio, "H2/CH4 standard-volume ratio")
    return ratio, props


def original_equivalent_gwh(ng_million_sm3: float) -> float:
    return ng_million_sm3 * H2_KWH_PER_STANDARD_M3


def published_h2(row: dict[str, str]) -> bool:
    name = (row["Storage class"].strip().upper(), row["Site"].strip().lower())
    notes = row["Notes and sources"].lower()
    return (name in PRESERVE_PUBLISHED_H2 or
            "h2 reference gwh: hyunder possible working-gas estimate" in notes or
            "h2 reference gwh: site-specific" in notes)


def source_pressure(row: dict[str, str]) -> tuple[float | None, str]:
    raw = as_float(row["Minimum pressure bar"])
    operating = as_float(row["Min. Op. Pressure"])
    if operating is not None:
        if raw is None and math.isclose(operating, 55., abs_tol=1e-9):
            origin = "Min. Op. Pressure=55 bar; assumed pipeline-interface minimum"
        elif raw is None:
            origin = "Min. Op. Pressure only; original reservoir minimum unavailable"
        else:
            origin = "Min. Op. Pressure; raw minimum also available"
        return operating, origin
    return raw, "Raw minimum pressure (operating field empty)"


def convert_rows(rows: list[dict[str, str]], *,
                 z: Callable[[str, float, float], float],
                 temperature_at_depth: Callable[[float], float],
                 infer_from_twh: bool, methane_gcv_kwh_sm3: float,
                 ) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    revised = [r.copy() for r in rows]
    audit: list[dict[str, str]] = []
    unexpected: list[str] = []

    for rownum, (source, target) in enumerate(zip(rows, revised), start=2):
        cls = source["Storage class"].strip().upper()
        site = source["Site"].strip()
        h2_old = as_float(source["H2 reference working GWh"])
        ng_volume = as_float(source["NG working million Sm3"])
        ng_twh = as_float(source["NG working TWh"])
        pmin, pmin_origin = source_pressure(source)
        pmax = as_float(source["Maximum pressure bar"])
        depth = as_float(source["Reference depth m"])
        baseline = {
            "csv_row": str(rownum), "storage_class": cls, "site": site,
            "status": "", "basis": "", "old_h2_gwh": "" if h2_old is None else str(h2_old),
            "new_h2_gwh": "", "ng_wgv_million_sm3_used": "",
            "ng_wgv_source": "", "ng_twh": "" if ng_twh is None else str(ng_twh),
            "depth_m": "" if depth is None else str(depth),
            "pmin_bar": "" if pmin is None else str(pmin),
            "pmax_bar": "" if pmax is None else str(pmax),
            "temperature_k": "", "z_h2_min": "", "z_h2_max": "",
            "z_methane_min": "", "z_methane_max": "",
            "standard_volume_ratio_h2_to_methane": "", "reason": "",
            "min_pressure_basis": pmin_origin,
        }
        audit.append(baseline)
        if MARKER in source["Notes and sources"]:
            baseline["status"] = "already_adjusted"
            continue
        if published_h2(source) and h2_old is not None:
            baseline["status"] = "preserved_published_h2"
            baseline["reason"] = "Separate literature H2 estimate; not a simple NG volume equivalent"
            continue
        if ng_volume is not None and h2_old is not None:
            prior = original_equivalent_gwh(ng_volume)
            if not math.isclose(h2_old, prior, rel_tol=.015, abs_tol=.06):
                baseline["status"] = "preserved_unexplained_h2_difference"
                baseline["reason"] = (f"Existing H2 {h2_old:g} GWh differs from "
                                      f"direct NG volume equivalent {prior:g} GWh")
                unexpected.append(f"{cls}/{site}: existing H2 not a direct NG equivalent")
                continue
        if ng_volume is not None:
            basis = "reported_ng_working_volume"
            volume_used = ng_volume
        elif infer_from_twh and ng_twh is not None:
            # 1 TWh = 1e9 kWh; 1 million Sm3 = 1e6 Sm3.
            basis = "ng_twh_inverted_assuming_methane_gcv"
            volume_used = ng_twh * 1000. / methane_gcv_kwh_sm3
        else:
            baseline["status"] = "no_direct_ng_volume"
            baseline["reason"] = ("NG TWh-only; use --infer-from-twh for an explicit proxy"
                                  if ng_twh is not None else "No NG working volume or TWh")
            continue
        baseline["basis"] = basis
        baseline["ng_wgv_million_sm3_used"] = f"{volume_used:.9g}"
        baseline["ng_wgv_source"] = ("reported" if ng_volume is not None
                                      else f"TWh / {methane_gcv_kwh_sm3:g} kWh/Sm3 GCV")
        if depth is None or pmin is None or pmax is None:
            baseline["status"] = "insufficient_pressure_or_depth"
            baseline["reason"] = "Need depth, operating minimum and maximum pressure"
            continue
        try:
            safe_positive(volume_used, "NG working standard volume")
            safe_positive(depth, "reservoir depth")
            temperature = safe_positive(float(temperature_at_depth(depth)),
                                        "reservoir temperature")
            ratio, props = real_gas_volume_ratio(pmin, pmax, temperature, z)
            revised_gwh = original_equivalent_gwh(volume_used) * ratio
            safe_positive(revised_gwh, "adjusted H2 working GWh")
        except Exception as exc:
            baseline["status"] = "calculation_error"
            baseline["reason"] = f"{type(exc).__name__}: {exc}"
            unexpected.append(f"{cls}/{site}: {baseline['reason']}")
            continue
        baseline.update({
            "status": "updated", "new_h2_gwh": f"{revised_gwh:.8g}",
            "temperature_k": f"{temperature:.8g}",
            "z_h2_min": f"{props['Hydrogen']['zmin']:.8g}",
            "z_h2_max": f"{props['Hydrogen']['zmax']:.8g}",
            "z_methane_min": f"{props['Methane']['zmin']:.8g}",
            "z_methane_max": f"{props['Methane']['zmax']:.8g}",
            "standard_volume_ratio_h2_to_methane": f"{ratio:.8g}",
        })
        target["H2 reference working GWh"] = f"{revised_gwh:,.2f}"
        note = target["Notes and sources"]
        note = note.replace(
            "H2 reference GWh: NG standard working volume × 0.0852 kg/m3 × "
            "33.33 kWh/kg; VOLUME-EQUIVALENT SCREENING ONLY, NOT a physically "
            "modelled H2 working capacity.",
            "Former H2 reference GWh: simple NG standard-volume equivalent "
            "(superseded by Z-adjusted estimate).",
        )
        note = note.replace(
            "H2 screening equivalent=NG standard volume x 0.0852 kg/m3 x "
            "33.33 kWh/kg; NOT site-validated H2 working capacity",
            "Former simple H2 volume-equivalent screening (superseded)",
        )
        comment = (
            f"{MARKER} H2 WGV={revised_gwh:.2f} GWh LHV estimated from "
            f"{volume_used:.6g} million Sm3 methane working volume "
            f"({basis}); ratio=(pmax/Z_H2,max-pmin/Z_H2,min)/"
            f"(pmax/Z_CH4,max-pmin/Z_CH4,min)={ratio:.5f}; "
            f"Pmin={pmin:g} bar(a), Pmax={pmax:g} bar(a), "
            f"T={temperature:.2f} K from depth={depth:g} m; "
            f"{pmin_origin}; pure-CH4 real-gas proxy, not validated H2 capacity."
        )
        if basis == "ng_twh_inverted_assuming_methane_gcv":
            comment += (f" NG working volume estimated from TWh assuming pure methane "
                        f"gross calorific value={methane_gcv_kwh_sm3:g} kWh/Sm3 "
                        f"at 15 C; source TWh basis and gas mix unverified.")
        target["Notes and sources"] = f"{note} | {comment}"
    return revised, audit, unexpected


def write_csv(path: Path, fieldnames: list[str], records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("data/resources/uhs_reference_sites.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/resources/uhs_reference_sites_h2_adjusted.csv"))
    parser.add_argument("--audit", type=Path,
                        default=Path("outputs/reference_selection/capacity_conversion_audit.csv"))
    parser.add_argument("--infer-from-twh", action="store_true",
                        help="Estimate missing NG Sm3 from NG TWh using assumed methane GCV")
    parser.add_argument("--methane-gcv-kwh-sm3", type=float, default=37.652/3.6,
                        help="Pure methane GCV basis: IEA 37.652 MJ/Sm3 at 15 C")
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly permit replacing a previous output and audit")
    parser.add_argument("--allow-calculation-errors", action="store_true")
    args = parser.parse_args()
    if args.input.resolve() in (args.output.resolve(), args.audit.resolve()):
        parser.error("Never overwrite the original input; output and audit must be separate")
    if args.output.resolve() == args.audit.resolve():
        parser.error("Output and audit paths must be distinct")
    if not args.input.is_file():
        parser.error(f"Input does not exist: {args.input}")
    if (args.output.exists() or args.audit.exists()) and not args.overwrite:
        parser.error("Output/audit already exists; choose a new path or use --overwrite")
    safe_positive(args.methane_gcv_kwh_sm3, "methane GCV")

    from CoolProp.CoolProp import PropsSI
    from uhs_costs.design.default_design_assumptions import calculate_temperature_from_depth_k

    def z(gas: str, pressure_pa: float, temp_k: float) -> float:
        return float(PropsSI("Z", "P", pressure_pa, "T", temp_k, gas))

    fieldnames, rows = read_csv(args.input)
    revised, audit, errors = convert_rows(
        rows, z=z, temperature_at_depth=calculate_temperature_from_depth_k,
        infer_from_twh=args.infer_from_twh,
        methane_gcv_kwh_sm3=args.methane_gcv_kwh_sm3,
    )
    if errors and not args.allow_calculation_errors:
        print("ERROR: Conversion ambiguities/failures; no files written:")
        for issue in errors:
            print(" -", issue)
        raise RuntimeError("Resolve conversion problems or use --allow-calculation-errors")
    write_csv(args.output, fieldnames, revised)
    write_csv(args.audit, list(audit[0]), audit)
    from collections import Counter
    statuses = Counter(r["status"] for r in audit)
    print("Conversion statuses:")
    for status, count in sorted(statuses.items()):
        print(f"  {status}: {count}")
    print("Revised CSV:", args.output)
    print("Audit:", args.audit)
    if errors:
        print("WARNING: conversion issues: " + "; ".join(errors))


if __name__ == "__main__":
    main()
