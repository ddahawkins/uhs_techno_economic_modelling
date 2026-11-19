"""HyStories subsurface OPEX equations.

This module contains atomic HyStories-derived surface OPEX components.
Functions return EUR/year unless stated otherwise.
"""

from __future__ import annotations


def _validate_non_negative(value: float, name: str) -> None:
    """Validate that a numeric value is non-negative."""
    if value < 0:
        raise ValueError(f"{name} cannot be negative.")


#----------------------------------------------------------------------------------------------------
# 
#                               fixed opex
#
#----------------------------------------------------------------------------------------------------

def fixed_opex_fraction_of_epc_cost_eur_per_year(
    subsurface_epc_cost_eur: float,
    fixed_opex_fraction_of_epc: float = 0.03, #hystories default
) -> float:
    """HyStories fixed subsurface OPEX component proportional to EPC.

    Returns EUR/year.
    """
    _validate_non_negative(subsurface_epc_cost_eur, "subsurface_epc_cost_eur")
    _validate_non_negative(
        fixed_opex_fraction_of_epc,
        "fixed_opex_fraction_of_epc",
    )

    return fixed_opex_fraction_of_epc * subsurface_epc_cost_eur







