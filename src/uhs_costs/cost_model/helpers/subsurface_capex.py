"""HyStories subsurface cost equations.

This module contains atomic HyStories-derived subsurface cost components.

The functions return EUR, even where the HyStories equations are reported in kEUR.
They do not apply contingency unless explicitly stated.

The goal is to support both:
1. original HyStories-style aggregation; and
2. decomposed allocation across storage, injection, withdrawal, fixed, and other CEM-relevant categories.
"""

from __future__ import annotations

from math import ceil

from uhs_costs.constants import HYSTORIES_COST_INFLATION

#----------------------------------------------------------------------------------------------------
# 
#                                               Helpers
#
#----------------------------------------------------------------------------------------------------


def _validate_positive(value: float, name: str) -> None:
    """Validate that a numeric value is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_non_negative(value: float, name: str) -> None:
    """Validate that a numeric value is non-negative."""
    if value < 0:
        raise ValueError(f"{name} cannot be negative.")


def _validate_positive_int(value: int, name: str) -> None:
    """Validate that an integer value is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    
#----------------------------------------------------------------------------------------------------
# 
#                                               EPC4 Solution Mining 
#                                                   (salt caverns)
#
#----------------------------------------------------------------------------------------------------

def epc4_salt_development_drilling_and_leaching_completion_cost_eur(
    number_well_heads: int,
    last_cemented_casing_shoe_m: float,
    material_cost_factor_withdrawal: float = 1.0,
    drilling_complexity_index: float = 1.0,
) -> float:
    """HyStories EPC4 Salt: development drilling and leaching completion.

    Returns EUR.
    """
    _validate_positive_int(number_well_heads, "number_well_heads")
    _validate_positive(last_cemented_casing_shoe_m, "last_cemented_casing_shoe_m")
    _validate_non_negative(material_cost_factor_withdrawal, "material_cost_factor_withdrawal")
    _validate_positive(drilling_complexity_index, "drilling_complexity_index")

    cost_keur = (
        number_well_heads
        * (
            1781
            + 196 * material_cost_factor_withdrawal
            + 86 * (18 + 12 * last_cemented_casing_shoe_m / 500)
        )
        * drilling_complexity_index
    )

    return cost_keur * 1000*HYSTORIES_COST_INFLATION

#----------------------------------------------------------------------------------------------------
# 
#                                               EPC1 Leaching Capex
#                                                   (salt caverns)
#
#----------------------------------------------------------------------------------------------------

def epc1_salt_leaching_facilities_fixed_cost_eur() -> float:
    """HyStories EPC1 Salt fixed component for leaching facilities.

    Returns EUR.
    """
    return 50_000 * 1000*HYSTORIES_COST_INFLATION


def epc1_salt_leaching_facilities_wellhead_cost_eur(
    number_well_heads: int,
) -> float:
    """HyStories EPC1 Salt wellhead/cavern-scaled component.

    Returns EUR.
    """
    _validate_positive_int(number_well_heads, "number_well_heads")
    return 2350 * number_well_heads * 1000 *HYSTORIES_COST_INFLATION


def epc1_salt_leaching_facilities_pipeline_cost_eur(
    fresh_water_pipeline_length_km: float,
    brine_disposal_pipeline_length_km: float,
) -> float:
    """HyStories EPC1 Salt pipeline-scaled component.

    Returns EUR.
    """
    _validate_non_negative(fresh_water_pipeline_length_km, "fresh_water_pipeline_length_km")
    _validate_non_negative(brine_disposal_pipeline_length_km, "brine_disposal_pipeline_length_km")

    cost_keur = 640 * (
        fresh_water_pipeline_length_km
        + brine_disposal_pipeline_length_km
    )

    return cost_keur * 1000*HYSTORIES_COST_INFLATION

#----------------------------------------------------------------------------------------------------
# 
#                                               EPC2 Leaching
#                                               (salt caverns)
#
#----------------------------------------------------------------------------------------------------

def epc2_salt_leaching_cost_eur(
    number_caverns: int,
    leaching_duration_per_cavern_months: float,
    total_leaching_duration_years: float,
    cost_of_electricity_eur_per_mwh: float = 60.0,
) -> float:
    """HyStories EPC2 Salt leaching operation and maintenance cost.

    Returns EUR.
    """
    _validate_positive_int(number_caverns, "number_caverns")
    _validate_positive(
        leaching_duration_per_cavern_months,
        "leaching_duration_per_cavern_months",
    )
    _validate_positive(
        total_leaching_duration_years,
        "total_leaching_duration_years",
    )
    _validate_positive(
        cost_of_electricity_eur_per_mwh,
        "cost_of_electricity_eur_per_mwh",
    )

    cost_keur = (
        (
            total_leaching_duration_years * (28 * number_caverns + 9500)
            + number_caverns * (
                87.5 * 1.4 * leaching_duration_per_cavern_months - 420
            )
        ) * HYSTORIES_COST_INFLATION
        + number_caverns * 87.5
        * (cost_of_electricity_eur_per_mwh / 60)
        * leaching_duration_per_cavern_months
    )

    return cost_keur * 1000



#----------------------------------------------------------------------------------------------------
# 
#                                               EPC3 Debrining and First Fill
#                                                   (salt caverns)
#
#----------------------------------------------------------------------------------------------------



def epc3_salt_conversion_time_scaled_cost_eur(
    total_conversion_duration_years: float,
) -> float:
    """HyStories EPC3 Salt time-scaled conversion component.

    Returns EUR.
    """
    _validate_positive(total_conversion_duration_years, "total_conversion_duration_years")
    return 6750 * total_conversion_duration_years * 1000 *HYSTORIES_COST_INFLATION


def epc3_salt_conversion_fixed_cost_eur() -> float:
    """HyStories EPC3 Salt fixed conversion component.

    Returns EUR.
    """
    return 1700 * 1000 *HYSTORIES_COST_INFLATION


def epc3_salt_conversion_cavern_scaled_cost_eur(
    number_well_heads: int,
    free_gas_volume_per_cavern_thousand_m3: float,
    cost_of_electricity_eur_per_mwh: float = 60.0,
) -> float:
    """HyStories EPC3 Salt cavern-scaled conversion component.

    Returns EUR.
    """
    _validate_positive_int(number_well_heads, "number_well_heads")
    _validate_positive(free_gas_volume_per_cavern_thousand_m3, "free_gas_volume_per_cavern_thousand_m3")
    _validate_positive(cost_of_electricity_eur_per_mwh, "cost_of_electricity_eur_per_mwh")

    cost_keur = number_well_heads * (
        1.42 * (cost_of_electricity_eur_per_mwh / 60)
        * free_gas_volume_per_cavern_thousand_m3
        + 2780 * HYSTORIES_COST_INFLATION
    )

    return cost_keur * 1000

#----------------------------------------------------------------------------------------------------
# 
#                                               EPC4 Drilling
#                                               (DGF / Aquifers)
#
#----------------------------------------------------------------------------------------------------

def epc4_porous_production_well_drilling_cost_eur(
    number_production_wells: int,
    last_cemented_casing_shoe_m: float,
    material_cost_factor_withdrawal: float = 0.0,
    drilling_complexity_index: float = 1.0,
) -> float:
    """HyStories EPC4 Porous production-well drilling component.

    Returns EUR.
    """
    _validate_positive_int(number_production_wells, "number_production_wells")
    _validate_positive(last_cemented_casing_shoe_m, "last_cemented_casing_shoe_m")
    _validate_non_negative(material_cost_factor_withdrawal, "material_cost_factor_withdrawal")
    _validate_positive(drilling_complexity_index, "drilling_complexity_index")

    cost_keur = (
        number_production_wells
        * (
            1018
            + 960 * material_cost_factor_withdrawal
            + 86 * (19 + 12 * last_cemented_casing_shoe_m / 600)
        )
        * drilling_complexity_index
    )

    return cost_keur * 1000 *HYSTORIES_COST_INFLATION


def epc4_porous_observation_well_drilling_cost_eur(
    number_observation_wells: int,
    last_cemented_casing_shoe_m: float,
    material_cost_factor_withdrawal: float = 0.0,
    drilling_complexity_index: float = 1.0,
) -> float:
    """HyStories EPC4 Porous observation-well drilling component.

    Returns EUR.
    """
    _validate_positive_int(number_observation_wells, "number_observation_wells")
    _validate_positive(last_cemented_casing_shoe_m, "last_cemented_casing_shoe_m")
    _validate_non_negative(material_cost_factor_withdrawal, "material_cost_factor_withdrawal")
    _validate_positive(drilling_complexity_index, "drilling_complexity_index")

    cost_keur = (
        number_observation_wells
        * (
            628
            + 618 * material_cost_factor_withdrawal
            + 46 * (21 + 6 * last_cemented_casing_shoe_m / 600)
        )
        * drilling_complexity_index
    )

    return cost_keur * 1000 *HYSTORIES_COST_INFLATION

#----------------------------------------------------------------------------------------------------
# 
#                                               EPC3 First Fill
#                                               (DGF / Aquifers)
#
#----------------------------------------------------------------------------------------------------


def epc3_porous_first_gas_fill_cost_eur(
    first_gas_fill_duration_years: float,
    cost_of_electricity_eur_per_mwh: float = 60.0,
) -> float:
    """HyStories EPC3 Porous first gas fill cost.

    Returns EUR.

    Note: based on the visible HyStories formula:
    EPC3_Porous [k€] = 2400 * dFGF + 2100 * COE / 60.
    """
    _validate_positive(first_gas_fill_duration_years, "first_gas_fill_duration_years")
    _validate_positive(cost_of_electricity_eur_per_mwh, "cost_of_electricity_eur_per_mwh")

    cost_keur = (
        2400 * first_gas_fill_duration_years * HYSTORIES_COST_INFLATION
        + 2100 * cost_of_electricity_eur_per_mwh / 60
    )

    return cost_keur * 1000

#----------------------------------------------------------------------------------------------------
# 
#                                               CG 
#                                         (Cushion Gas)
#
#----------------------------------------------------------------------------------------------------

#redundant but kept here for legacy transparency
# def cushion_gas_cost_eur(
#     cushion_gas_to_total_gas_ratio: float,
#     working_gas_volume_million_sm3: float,
#     hydrogen_cost_eur_per_kg: float = 2.0,
#     hydrogen_tonnes_per_million_sm3: float = 85.0,
# ) -> float:
#     """HyStories cushion gas cost.

#     Returns EUR.
#     """
#     if not 0 <= cushion_gas_to_total_gas_ratio < 1:
#         raise ValueError("cushion_gas_to_total_gas_ratio must be in [0, 1).")

#     _validate_positive(working_gas_volume_million_sm3, "working_gas_volume_million_sm3")
#     _validate_positive(hydrogen_cost_eur_per_kg, "hydrogen_cost_eur_per_kg")
#     _validate_positive(hydrogen_tonnes_per_million_sm3, "hydrogen_tonnes_per_million_sm3")

#     cushion_gas_volume_million_sm3 = (
#         cushion_gas_to_total_gas_ratio
#         / (1 - cushion_gas_to_total_gas_ratio)
#         * working_gas_volume_million_sm3
#     )

#     hydrogen_kg = (
#         cushion_gas_volume_million_sm3
#         * hydrogen_tonnes_per_million_sm3
#         * 1000
#     )

#     return hydrogen_kg * hydrogen_cost_eur_per_kg

def cushion_gas_cost_eur_simple(
    cushion_gas_mass_kg: float,
    hydrogen_cost_eur_per_kg: float = 2.0,
) -> float:
    """HyStories cushion gas cost, but refactored given cushion gas mass is calculated by the inventory model directly.

    Returns EUR.
    """
    # ensure cushion gas only include h2 component of injection and does not account for methane already in the field
    _validate_positive(cushion_gas_mass_kg, "cushion_gas_mass_kg")
    _validate_positive(hydrogen_cost_eur_per_kg, "hydrogen_cost_eur_per_kg")

    
    return cushion_gas_mass_kg * hydrogen_cost_eur_per_kg #no cost escalation necessary as h2 prices are 2025

#----------------------------------------------------------------------------------------------------
# 
#                                Contingencies
#
#----------------------------------------------------------------------------------------------------

def contingency_cost_eur(
    base_cost_eur: float,
    contingency_fraction: float = 0.20,
) -> float:
    """HyStories subsurface contingency cost.

    Returns EUR.
    """
    _validate_non_negative(base_cost_eur, "base_cost_eur")
    _validate_non_negative(contingency_fraction, "contingency_fraction")

    return contingency_fraction * base_cost_eur


#----------------------------------------------------------------------------------------------------
# 
#                                Lined Rock Cavern Cost Formulae
#                   Based on Huang et al. doi.org/10.1016/j.apenergy.2025.126564.
#
#----------------------------------------------------------------------------------------------------



def lrc_well_capex_eur(
    number_well_heads: int,
    well_depth_m: float,
    fixed_cost_eur_per_well: float,
    variable_cost_eur_per_m: float,
    drilling_complexity_index: float = 1.0,
) -> float:
    return number_well_heads * (
        fixed_cost_eur_per_well
        + variable_cost_eur_per_m * well_depth_m * drilling_complexity_index
    ) #no cost escalation necessary, prices already adjusted


def lrc_mining_capex_eur(
    excavation_volume_m3: float,
    mining_cost_eur_per_m3: float,
) -> float:
    return excavation_volume_m3 * mining_cost_eur_per_m3 #no cost escalation necessary, prices already adjusted


def lrc_lining_capex_eur(
    steel_lining_mass_tonnes: float,
    concrete_lining_mass_tonnes: float,
    steel_lining_cost_eur_per_tonne: float,
    concrete_lining_cost_eur_per_tonne: float,
    installation_fraction: float,
) -> float:
    material_cost_eur = (
        steel_lining_mass_tonnes * steel_lining_cost_eur_per_tonne
        + concrete_lining_mass_tonnes * concrete_lining_cost_eur_per_tonne
    )

    return material_cost_eur * (1 + installation_fraction) #no cost escalation necessary, prices already adjusted


def lrc_drainage_capex_eur(
    tunnel_drainage_length_m: float,
    total_cavern_drainage_length_m: float,
    tunnel_drainage_cost_eur_per_m: float,
    cavern_drainage_cost_eur_per_m,
) -> float:
    return (
        tunnel_drainage_length_m * tunnel_drainage_cost_eur_per_m
        + total_cavern_drainage_length_m * cavern_drainage_cost_eur_per_m
    ) #no cost escalation necessary, prices already adjusted