"""HyStories cost equations and decomposed EPC components.

This module implements HyStories-derived cost equations in both:
1. original aggregated form, where appropriate;
2. decomposed component form, for mapping costs to storage, injection,
   withdrawal, compression, and fixed-cost categories.

The goal is not always to reproduce HyStories exactly, but to make the
equation components transparent and reusable in decomposed techno-economic
models.
"""

#----------------------------------------------------------------------------------------------------
# 
#                                               EPC1
#
#----------------------------------------------------------------------------------------------------


def epc1_compression_cost_eur(
    total_installed_compression_brake_power_mw: float,
    material_cost_factor_injection: float = 0.0,
) -> float:
    """HyStories EPC1 compression-related component.

    Component:
        8655 * (1 + MCFi * 0.14) * TICBP

    Returns EUR, assuming HyStories equation is expressed in kEUR.
    """
    if total_installed_compression_brake_power_mw <= 0:
        raise ValueError("total_installed_compression_brake_power_mw must be positive.")

    cost_keur = (
        8655
        * (1 + material_cost_factor_injection * 0.14)
        * total_installed_compression_brake_power_mw
    )

    return cost_keur * 1000

def epc1_fixed_cost_eur() -> float:
    """HyStories EPC1 fixed process-plant component."""
    return 20_700 * 1000

def epc1_withdrawal_cost_eur(
    withdrawal_flow_million_sm3_per_day: float,
    material_cost_factor_withdrawal: float,
    total_installed_withdrawal_compression_brake_power_mw: float | None = None,
) -> float:
    """HyStories EPC1 withdrawal-flow-related component.

    Component:
        9100 * (1 + MCFw * 0.11) * Qw^0.643
    """
    if withdrawal_flow_million_sm3_per_day <= 0:
        raise ValueError("withdrawal_flow_million_sm3_per_day must be positive.")

    cost_keur = (
        9100
        * (1 + material_cost_factor_withdrawal * 0.11)
        * withdrawal_flow_million_sm3_per_day**0.643
    )

    #Optional compression cost for withdrawal where pipeline pressure > min pressure
    if total_installed_withdrawal_compression_brake_power_mw:
        cost_keur += (
                8655
                * (1 + material_cost_factor_withdrawal * 0.14)
                * total_installed_withdrawal_compression_brake_power_mw
            )
        print("additional withdrawal compression cost:", (
                8655
                * (1 + material_cost_factor_withdrawal * 0.14)
                * total_installed_withdrawal_compression_brake_power_mw
            ))

    return cost_keur * 1000


#----------------------------------------------------------------------------------------------------
# 
#                                EPC2 (Wellpads and downstream equipment)
#
#----------------------------------------------------------------------------------------------------

def epc2_fieldlines_cost_eur(
        withdrawal_flow_million_sm3_per_day: float,
        operating_pressure_ratio: float,
        material_cost_factor_withdrawal: float,
        number_well_heads: int,
) -> float:
    """HyStories EPC2 costs associated with piping and field lines."""
    return number_well_heads*(58.25*(1+0.742*material_cost_factor_withdrawal)*operating_pressure_ratio*withdrawal_flow_million_sm3_per_day)*1000

def epc2_instrumentation_cost_eur(
        material_cost_factor_withdrawal: float,
        number_well_heads: int,
) -> float:
    """HyStories EPC2 costs associated with wellhead separators, instrumentation, and valves"""
    return number_well_heads*(1605*(1+0.476*material_cost_factor_withdrawal))*1000

#----------------------------------------------------------------------------------------------------
# 
#                                EPC3 (Well interconnection)
#
#----------------------------------------------------------------------------------------------------

def epc3_well_interconnection_cost_eur(
    withdrawal_flow_million_sm3_per_day: float,
    operating_pressure_ratio: float,
    material_cost_factor_withdrawal: float,
    additional_field_line_length_km: float,
) -> float:
    """Original HyStories EPC3 cost for well interconnection."""
    return 117*additional_field_line_length_km*(1+0.743*material_cost_factor_withdrawal)*(operating_pressure_ratio*withdrawal_flow_million_sm3_per_day+1.9)*1000

#----------------------------------------------------------------------------------------------------
# 
#                                EPC4 (Hydrogen purification)
#
#----------------------------------------------------------------------------------------------------

def epc4_purification_cost_eur(
    withdrawal_flow_million_sm3_per_day: float,
    purification_factor: float,

) -> float:
    """Original HyStories EPC4 cost for hydrogen purification."""
    return purification_factor*42500*withdrawal_flow_million_sm3_per_day**0.65 *1000

#----------------------------------------------------------------------------------------------------
# 
#                                EPC5 (Balance of Plant)
#
#----------------------------------------------------------------------------------------------------

def epc5_fixed_cost_eur() -> float:
    """ HyStories EPC5 fixed BOP cost."""
    return 8000*1000

def epc5_proportional_cost_eur(
        cost: float,
        factor_bop: float = 0.05 #default from HyStories
) -> float:
    """ HyStories EPC5 BOP cost proportional to wider EPCs."""
    return factor_bop*cost


#----------------------------------------------------------------------------------------------------
# 
#                                Contingencies
#
#----------------------------------------------------------------------------------------------------

def contingency_cost_eur(
    base_cost_eur: float,
    contingency_fraction: float = 0.20,
) -> float:
    """HyStories surface contingency cost.

    Returns EUR.
    """
    _validate_non_negative(base_cost_eur, "base_cost_eur")
    _validate_non_negative(contingency_fraction, "contingency_fraction")

    return contingency_fraction * base_cost_eur

def _validate_non_negative(value: float, name: str) -> None:
    """Validate that a numeric value is non-negative."""
    if value < 0:
        raise ValueError(f"{name} cannot be negative.")