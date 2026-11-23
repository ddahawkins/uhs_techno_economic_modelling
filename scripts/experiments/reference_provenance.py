"""Record and validate the detailed UHS reference used by a CEM cost catalogue.

This is a small *reference* manifest, not a second cost-rate table. Actual CEM
investment and operating costs are read from each saved Calliope NetCDF file.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path


def model_source_fingerprint(root: Path) -> str:
    """Fingerprint detailed design, physical/cost code and evaluator sources."""
    root = root.resolve()
    paths = list((root / "src" / "uhs_costs").rglob("*.py"))
    paths += [
        root / "scripts" / "experiments" / "cost_rates.py",
        root / "scripts" / "experiments" / "linearisation_sensitivity.py",
    ]
    paths = sorted(set(paths))
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError("Incomplete UHS detailed-model source tree")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode("utf8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def reference_manifest(root: Path, samples) -> dict:
    from scripts.experiments.cost_rates import get_technology

    entries = {}
    for sample in samples:
        tech = str(sample.technology)
        spec = get_technology(tech)
        design_module = importlib.import_module(spec.project_constructor.__module__)
        if not hasattr(design_module, "REFERENCE_SITE"):
            raise ValueError(f"{tech}: design module has no REFERENCE_SITE")
        entry = {
            "reference_site": str(design_module.REFERENCE_SITE),
            "reference_energy_gwh": float(sample.storage_reference_gwh_lhv),
            "reference_epr_days": float(sample.epr_days),
            "reference_wtir": float(sample.wtir),
            "design_file_reference_energy_gwh": float(spec.default_storage_capacity_gwh_lhv),
            "design_file_reference_epr_days": float(spec.default_epr_days),
            "design_file_reference_wtir": float(spec.default_wtir),
            "epr_domain_days": [float(sample.epr_interval_days.lower), float(sample.epr_interval_days.upper)],
            "wtir_domain": [float(sample.wtir_interval.lower), float(sample.wtir_interval.upper)],
        }
        if tech in entries:
            other = entries[tech]
            if other != entry:
                raise ValueError(f"{tech}: one manifest cannot represent multiple reference designs")
        else:
            entries[tech] = entry
    if not entries:
        raise ValueError("No cost samples for reference manifest")
    return {
        "schema_version": 1,
        "source_model_sha256": model_source_fingerprint(root),
        "technologies": entries,
    }


def write_reference_manifest(root: Path, samples, path: Path) -> None:
    manifest = reference_manifest(root, samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never silently overwrite provenance for a different experiment.
    payload = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf8") != payload:
        raise FileExistsError(f"Reference provenance exists but differs: {path}")
    path.write_text(payload, encoding="utf8")


def validate_reference_manifest(root: Path, manifest: dict, module) -> dict:
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported or missing reference manifest schema")
    expected = manifest.get("source_model_sha256")
    actual = model_source_fingerprint(root)
    if not isinstance(expected, str) or expected != actual:
        raise ValueError(
            "Detailed-model source differs from the original cost-generation source. "
            "Use its archived revision; do not reprice against a different site/model."
        )
    entries = manifest.get("technologies")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Reference manifest lacks technology references")
    for tech, values in entries.items():
        spec = module.get_technology(tech)
        design = importlib.import_module(spec.project_constructor.__module__)
        if str(getattr(design, "REFERENCE_SITE", "")) != values["reference_site"]:
            raise ValueError(f"{tech}: reference site has changed")
        if not math.isclose(spec.default_storage_capacity_gwh_lhv, values["design_file_reference_energy_gwh"], rel_tol=1e-10):
            raise ValueError(f"{tech}: reference capacity has changed")
        if not math.isclose(spec.default_epr_days, values["design_file_reference_epr_days"], rel_tol=1e-10):
            raise ValueError(f"{tech}: reference EPR has changed")
        if not math.isclose(spec.default_wtir, values["design_file_reference_wtir"], rel_tol=1e-10):
            raise ValueError(f"{tech}: reference WTIR has changed")
        for key in ("reference_energy_gwh", "reference_epr_days", "reference_wtir"):
            value = values.get(key)
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{tech}: invalid {key} in manifest")
    return entries
