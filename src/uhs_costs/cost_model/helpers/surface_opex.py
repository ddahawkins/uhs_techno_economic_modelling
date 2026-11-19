"""HyStories surface OPEX equations.

This module contains atomic HyStories-derived surface OPEX components.
Functions return EUR/year unless stated otherwise.

The HyStories simplified variable-OPEX correlation is interpreted as::

    rate = 0.746 * COE * (ln(tau) + 0.50 + K_purif)

where ``ln(tau)`` is assigned to injection/compression and
``0.50 + K_purif`` is assigned to withdrawal. HyStories also states that
lubricants, waste disposal, and chemicals are assumed to equal 4% of the
associated electricity cost. This module therefore permits the compression
term to be decomposed into an electricity share and a residual ancillary share.
"""

from __future__ import annotations

from math import log

from uhs_costs.constants import HYDROGEN_LHV_KWH_PER_KG, HYSTORIES_COST_INFLATION


HYSTORIES_VARIABLE_OPEX_COEFFICIENT = 0.746
HYSTORIES_WITHDRAWAL_BASE_TERM = 0.50


def _validate_non_negative(value: float, name: str) -> None:
    """Validate that a numeric value is non-negative."""
    if value < 0:
        raise ValueError(f"{name} cannot be negative.")


def _validate_compression_ratio(compression_ratio: float) -> None:
    if compression_ratio <= 1:
        raise ValueError("compression_ratio must be > 1.")


def _eur_per_tonne_h2_to_eur_per_kwh_h2_lhv(value: float) -> float:
    """Convert EUR/tH2 to EUR/kWh-H2 (LHV)."""
    return value / (1000 * HYDROGEN_LHV_KWH_PER_KG)


# --------------------------------------------------------------------------------------------------
# Fixed OPEX
# --------------------------------------------------------------------------------------------------


def fixed_opex_base_cost_eur_per_year(fixed_cost_k_eur: float = 2100) -> float:
    """HyStories fixed surface OPEX base component, related to employee wages.

    Returns EUR/year.

    Component:
        2100 kEUR/year
    """
    return fixed_cost_k_eur * 1000 * HYSTORIES_COST_INFLATION


def fixed_opex_fraction_of_epc_cost_eur_per_year(
    surface_epc_cost_eur: float,
    fixed_opex_fraction_of_epc: float = 0.04,
) -> float:
    """HyStories fixed surface OPEX component proportional to EPC.

    Returns EUR/year.
    """
    _validate_non_negative(surface_epc_cost_eur, "surface_epc_cost_eur")
    _validate_non_negative(
        fixed_opex_fraction_of_epc,
        "fixed_opex_fraction_of_epc",
    )

    return fixed_opex_fraction_of_epc * surface_epc_cost_eur


# --------------------------------------------------------------------------------------------------
# Variable OPEX
# --------------------------------------------------------------------------------------------------


def variable_opex_rate_injection_compression_total_eur_per_kwh_h2_lhv(
    cost_of_electricity_eur_per_mwh: float,
    compression_ratio: float,
) -> float:
    """Full HyStories compression-related injection variable OPEX.

    This is the ``ln(tau)`` part of the simplified HyStories variable-OPEX
    correlation and contains both the electricity-related cost and the small
    ancillary allowance tied to that electricity cost.

    Returns EUR/kWh-H2 (LHV) injected.
    """
    _validate_non_negative(
        cost_of_electricity_eur_per_mwh,
        "cost_of_electricity_eur_per_mwh",
    )
    _validate_compression_ratio(compression_ratio)

    rate_eur_per_tonne_h2 = (
        HYSTORIES_VARIABLE_OPEX_COEFFICIENT
        * cost_of_electricity_eur_per_mwh
        * log(compression_ratio)
    )
    return _eur_per_tonne_h2_to_eur_per_kwh_h2_lhv(rate_eur_per_tonne_h2) #no escalation, electricity prices already updated


def variable_opex_rate_injection_compression_electricity_eur_per_kwh_h2_lhv(
    cost_of_electricity_eur_per_mwh: float,
    compression_ratio: float,
    ancillary_cost_fraction_of_electricity: float = 0.04,
) -> float:
    """Electricity share of HyStories compression-related injection OPEX.

    HyStories states that lubricants, waste disposal, and chemicals are assumed
    to equal a fraction (4% by default) of electricity cost. Under the working
    interpretation that the published 0.746 coefficient contains both shares,
    the electricity-only share is the full compression term divided by
    ``1 + ancillary_cost_fraction_of_electricity``.

    Returns EUR/kWh-H2 (LHV) injected.
    """
    _validate_non_negative(
        ancillary_cost_fraction_of_electricity,
        "ancillary_cost_fraction_of_electricity",
    )

    total = variable_opex_rate_injection_compression_total_eur_per_kwh_h2_lhv(
        cost_of_electricity_eur_per_mwh=cost_of_electricity_eur_per_mwh,
        compression_ratio=compression_ratio,
    )
    return total / (1 + ancillary_cost_fraction_of_electricity) #no escalation, electricity prices already updated


def variable_opex_rate_injection_compression_ancillary_eur_per_kwh_h2_lhv(
    cost_of_electricity_eur_per_mwh: float,
    compression_ratio: float,
    ancillary_cost_fraction_of_electricity: float = 0.04,
) -> float:
    """Non-electric ancillary share of compression-related injection OPEX.

    Represents the HyStories allowance for lubricants, waste disposal, and
    chemicals associated with compression. It is retained even when compressor
    electricity is represented explicitly in a downstream CEM.

    Returns EUR/kWh-H2 (LHV) injected.
    """
    electricity = (
        variable_opex_rate_injection_compression_electricity_eur_per_kwh_h2_lhv(
            cost_of_electricity_eur_per_mwh=cost_of_electricity_eur_per_mwh,
            compression_ratio=compression_ratio,
            ancillary_cost_fraction_of_electricity=(
                ancillary_cost_fraction_of_electricity
            ),
        )
    )
    return ancillary_cost_fraction_of_electricity * electricity #no escalation, electricity prices already updated


def variable_opex_rate_injection_eur_per_kwh_h2_lhv(
    cost_of_electricity_eur_per_mwh: float,
    compression_ratio: float,
    include_compression_electricity_cost: bool = True,
    ancillary_cost_fraction_of_electricity: float = 0.04,
) -> float:
    """Injection variable OPEX under the selected compression-energy treatment.

    Parameters
    ----------
    include_compression_electricity_cost:
        If True, return the full HyStories ``ln(tau)`` compression-related
        variable OPEX. If False, remove the electricity share and retain only
        the ancillary 4%-of-electricity allowance so compressor electricity can
        be represented explicitly in a downstream CEM.

    Returns
    -------
    float
        EUR/kWh-H2 (LHV) injected.
    """
    if include_compression_electricity_cost:
        return variable_opex_rate_injection_compression_total_eur_per_kwh_h2_lhv(
            cost_of_electricity_eur_per_mwh=cost_of_electricity_eur_per_mwh,
            compression_ratio=compression_ratio,
        )

    return variable_opex_rate_injection_compression_ancillary_eur_per_kwh_h2_lhv(
        cost_of_electricity_eur_per_mwh=cost_of_electricity_eur_per_mwh,
        compression_ratio=compression_ratio,
        ancillary_cost_fraction_of_electricity=(
            ancillary_cost_fraction_of_electricity
        ), 
    ) #no escalation, electricity prices already updated


def variable_opex_rate_withdrawal_eur_per_kwh_h2_lhv(
    cost_of_electricity_eur_per_mwh: float,
    purification_coefficient: float,
) -> float:
    """HyStories withdrawal-side variable OPEX per cycled hydrogen quantity.

    The withdrawal side is interpreted from HyStories Figure 12 as containing
    the complete base term (0.50) plus the purification term ``K_purif``::

        withdrawal rate = 0.746 * COE * (0.50 + K_purif)

    Returns EUR/kWh-H2 (LHV) withdrawn.
    """
    _validate_non_negative(
        cost_of_electricity_eur_per_mwh,
        "cost_of_electricity_eur_per_mwh",
    )
    _validate_non_negative(purification_coefficient, "purification_coefficient")

    rate_eur_per_tonne_h2 = (
        HYSTORIES_VARIABLE_OPEX_COEFFICIENT
        * cost_of_electricity_eur_per_mwh
        * (HYSTORIES_WITHDRAWAL_BASE_TERM + purification_coefficient)
    )
    return _eur_per_tonne_h2_to_eur_per_kwh_h2_lhv(rate_eur_per_tonne_h2) #no escalation, electricity prices already updated
