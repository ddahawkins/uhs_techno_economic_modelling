"""Shared least-squares fits for homogeneous UHS capacity-cost approximations.

The fitted expressions have no constant project intercept. All coefficients
multiply installed capacities, so zero installed capacity implies zero cost
and the expressions remain linear in the CEM.

The objective is expressed per unit of the source component's own capacity.
This avoids an implicit preference for high-capacity samples when fitting
reference-energy designs across a grid of EPR and WTIR values.
"""

from __future__ import annotations

import numpy as np


def _fit_capacity_cost(
    own_capacity: np.ndarray,
    other_capacities: tuple[np.ndarray, ...],
    detailed_cost: np.ndarray,
) -> tuple[float, ...]:
    """Fit cost = sum(coefficient_i * capacity_i) via unit-rate least squares.

    The own-capacity term becomes the intercept of the *unit-rate regression*;
    it is NOT an intercept in absolute costs. Capacity columns are scaled for
    numerical conditioning. A rank-deficient design fails rather than silently
    generating unstable coefficients.
    """
    own = np.asarray(own_capacity, dtype=float).reshape(-1)
    other = [np.asarray(values, dtype=float).reshape(-1) for values in other_capacities]
    costs = np.asarray(detailed_cost, dtype=float).reshape(-1)
    n_coefficients = 1 + len(other)

    if own.size < n_coefficients or any(column.size != own.size for column in (*other, costs)):
        raise ValueError(f"At least {n_coefficients} equally sized observations are required")
    if not all(np.isfinite(column).all() for column in (own, *other, costs)):
        raise ValueError("Capacity and cost inputs must be finite")
    if np.any(own <= 0) or any(np.any(column <= 0) for column in other):
        raise ValueError("All training capacities must be positive")

    design = np.column_stack((np.ones_like(own), *(column / own for column in other)))
    target = costs / own
    scales = np.linalg.norm(design, axis=0)
    if np.any(scales == 0):
        raise ValueError("Zero-norm capacity predictor")
    normalised = design / scales
    coefficients, _, rank, _ = np.linalg.lstsq(normalised, target, rcond=None)
    if rank != n_coefficients:
        raise ValueError(
            f"Cannot identify {n_coefficients} independent cost drivers; "
            "vary EPR and WTIR independently across the training grid"
        )
    return tuple(float(value) for value in coefficients / scales)


def fit_two_driver_cost(
    own_capacity: np.ndarray,
    secondary_capacity: np.ndarray,
    detailed_cost: np.ndarray,
) -> tuple[float, float]:
    """Original two-driver unit-rate regression, retained without changes."""
    own = np.asarray(own_capacity, dtype=float).reshape(-1)
    secondary = np.asarray(secondary_capacity, dtype=float).reshape(-1)
    costs = np.asarray(detailed_cost, dtype=float).reshape(-1)

    if own.size != secondary.size or own.size != costs.size:
        raise ValueError("Capacity and cost arrays must have identical lengths.")
    if own.size < 2:
        raise ValueError("At least two detailed designs are required for a two-driver fit.")
    if not all(np.isfinite(a).all() for a in (own, secondary, costs)):
        raise ValueError("Two-driver regression inputs must be finite.")
    if np.any(own <= 0):
        raise ValueError("The own capacity must be positive for unit-rate fitting.")

    design = np.column_stack((np.ones_like(own), secondary / own))
    target_rate = costs / own
    scales = np.linalg.norm(design, axis=0)
    if np.any(scales == 0):
        raise ValueError("Both cost drivers must have nonzero training observations.")

    normalised = design / scales
    coefficients, _, rank, _ = np.linalg.lstsq(normalised, target_rate, rcond=None)
    if rank != 2:
        raise ValueError(
            "Cannot identify both cost drivers: training capacities are collinear. "
            "Vary EPR and/or WTIR independently of storage capacity."
        )

    a, b = coefficients / scales
    return float(a), float(b)


def fit_three_driver_cost(
    own_capacity: np.ndarray,
    secondary_capacity: np.ndarray,
    tertiary_capacity: np.ndarray,
    detailed_cost: np.ndarray,
) -> tuple[float, float, float]:
    """Fit three independent capacity coefficients without a project intercept."""
    a, b, c = _fit_capacity_cost(
        own_capacity, (secondary_capacity, tertiary_capacity), detailed_cost
    )
    return a, b, c
