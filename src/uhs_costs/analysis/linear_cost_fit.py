"""Shared zero-intercept, two-driver regression for UHS capital costs.

This fits *one* linear cost function from a collection of detailed projects:

    C = a * own_capacity + b * secondary_capacity

The coefficients are independent. In particular, the regression is **not**
forced to pass through any reference or sample-centre observation. Nor does it
add a fixed intercept, which would require extra care for optional investment
in a continuous capacity-expansion model.
"""

from __future__ import annotations

import numpy as np


def fit_two_driver_cost(
    own_capacity: np.ndarray,
    secondary_capacity: np.ndarray,
    detailed_cost: np.ndarray,
) -> tuple[float, float]:
    """Fit ``cost ~= a * own + b * secondary`` by ordinary least squares.

    The two capacity columns are normalised internally so kWh and kW may be
    mixed without harming numerical conditioning. The returned coefficients
    retain their original EUR/kWh or EUR/kW units.

    Raises ValueError when the training designs cannot identify both drivers.
    """

    own = np.asarray(own_capacity, dtype=float).reshape(-1)
    secondary = np.asarray(secondary_capacity, dtype=float).reshape(-1)
    costs = np.asarray(detailed_cost, dtype=float).reshape(-1)

    if own.size != secondary.size or own.size != costs.size:
        raise ValueError("Capacity and cost arrays must have identical lengths.")
    if own.size < 2:
        raise ValueError("At least two detailed designs are required for a two-driver fit.")
    if not all(np.isfinite(a).all() for a in (own, secondary, costs)):
        raise ValueError("Two-driver regression inputs must be finite.")

    design = np.column_stack((own, secondary))
    scales = np.linalg.norm(design, axis=0)
    if np.any(scales == 0):
        raise ValueError("Both capacity drivers must vary over nonzero designs.")

    normalised = design / scales
    coefficients, _, rank, _ = np.linalg.lstsq(normalised, costs, rcond=None)
    if rank != 2:
        raise ValueError(
            "Cannot identify both cost drivers: training capacities are collinear. "
            "Vary EPR and/or WTIR independently of storage capacity."
        )

    a, b = coefficients / scales
    return float(a), float(b)
