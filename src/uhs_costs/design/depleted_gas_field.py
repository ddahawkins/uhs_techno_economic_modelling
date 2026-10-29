"""Depleted Gas Field design constructor.

This module defines DGF-specific design defaults and constructs
complete StorageProject objects from the lower-level design constructors.
"""

from __future__ import annotations

from math import ceil

from uhs_costs.design.helpers.project import StorageProject, StorageTechnology
from uhs_costs.design.helpers.storage_inventory import construct_storage_inventory, StorageInventory
from uhs_costs.design.helpers.storage_flows import construct_storage_flows, StorageFlows
from uhs_costs.design.helpers.storage_pressures import construct_storage_pressures
from uhs_costs.design.helpers.well_design import construct_well_design
from uhs_costs.design.helpers.site_development import (
    construct_drilling_design,
    construct_field_interconnection_design,
    construct_porous_first_fill_process
)
from uhs_costs.design.helpers.purification import construct_purification
from uhs_costs.design.helpers.compression_model import (
    CompressionInput,
    calculate_compression,
)

from uhs_costs.design.default_design_assumptions import HyStoriesDesignAssumptions, construct_hystories_design_assumptions


# ----------------------------------------------------------------------------------------------------
#
#                               Depleted Gas Field Design Assumptions
#
# ----------------------------------------------------------------------------------------------------

ASSUMPTIONS =  construct_hystories_design_assumptions(
    storage_technology=StorageTechnology.DEPLETED_GAS_FIELD,
    general_overrides={
        'reference_reservoir_depth_m': 1100, #from reference_selection script
        'minimum_operating_pressure_pa': 77.4e5, #from reference_selection script
        'maximum_operating_pressure_pa': 142.2e5, #from reference_selection script
        'abandonment_pressure_pa': 68.65e5, #from reference_selection script
    }
)

# ----------------------------------------------------------------------------------------------------
#
#                                       DESIGN CONSTRUCTOR
#
# ----------------------------------------------------------------------------------------------------

def construct_project(
    working_gas_capacity_kwh_lhv: float,
    withdrawal_flow_kw_h2_lhv: float,
    injection_flow_kw_h2_lhv: float,
    assumptions: HyStoriesDesignAssumptions = ASSUMPTIONS,
    case_name: str | None = None,


) -> StorageProject:
    """Construct a complete DGF StorageProject.

    Primary user-facing inputs are energy capacity and injection/withdrawal
    rates. Technology-specific defaults are supplied by this module and can be
    overridden through keyword arguments.
    """
    #construct StorageInventory
    inventory = construct_storage_inventory(
        working_gas_capacity_kwh_lhv=working_gas_capacity_kwh_lhv,
        temperature_k=assumptions.general.temperature_k,
        maximum_pressure_pa=assumptions.general.maximum_operating_pressure_pa,
        minimum_pressure_pa=assumptions.general.minimum_operating_pressure_pa,
        abandonment_pressure_pa=assumptions.general.abandonment_pressure_pa,
        recovery_factor=assumptions.general.recovery_factor
    )

    #construct StorageFlows
    flows = construct_storage_flows(
        withdrawal_flow_kw_h2_lhv=withdrawal_flow_kw_h2_lhv,
        injection_flow_kw_h2_lhv=injection_flow_kw_h2_lhv,
    )

    #construct StoragePressures
    pressures = construct_storage_pressures(
        maximum_operating_pressure_pa=assumptions.general.maximum_operating_pressure_pa,
        minimum_operating_pressure_pa=assumptions.general.minimum_operating_pressure_pa,
        abandonment_pressure_pa=assumptions.general.abandonment_pressure_pa,
        pipeline_pressure_pa=assumptions.general.pipeline_pressure_pa,
    )

    purification = construct_purification(
        purification_factor=assumptions.general.purification_factor,
        hydrogen_recovery=assumptions.general.purification_hydrogen_recovery,
    )

    #determine number of production well heads and observation well heads
    number_observation_well_heads = assumptions.technology_specific.number_observation_well_heads  #this is not a dynamic function for HyStories TODO: implement dynamic function from excel?
    number_production_well_heads = assumptions.technology_specific.number_production_well_heads
    number_well_heads =  number_observation_well_heads + number_production_well_heads

    #construct WellDesign
    wells = construct_well_design(
        number_well_heads=number_well_heads,
        well_temperature_k=assumptions.general.temperature_k,
        number_caverns=None,
        number_observation_wells=number_observation_well_heads,
        number_production_wells=number_production_well_heads
    )

    #construct DrillingDesign
    drilling = construct_drilling_design(
        last_cemented_casing_shoe_m=assumptions.general.reference_reservoir_depth_m,
        drilling_complexity_index=assumptions.general.drilling_complexity_index,
    )

    #construct FieldInterconnectionDesign
    field_interconnection = construct_field_interconnection_design(
        gas_plant_to_nearest_wellhead_distance_km=(
            assumptions.general.gas_plant_to_nearest_wellhead_distance_km
        ),
    )

    porous_first_fill_process = construct_porous_first_fill_process(
        working_gas_volume_million_sm3=inventory.working_gas_h2_volume_sm3 / 1_000_000,
        cushion_gas_volume_million_sm3=inventory.cushion_gas_h2_volume_sm3  / 1_000_000,
        injection_flow_million_sm3_per_day=flows.injection_flow_million_sm3_per_day,
        injection_availability_factor=assumptions.general.compressor_availability_factor
    )

    #construct CompressionResult
    compression = calculate_compression(
        CompressionInput(
            inlet_pressure_pa=assumptions.general.pipeline_pressure_pa,
            outlet_pressure_pa=assumptions.general.maximum_operating_pressure_pa,
            inlet_temperature_k=assumptions.general.temperature_k,
            mass_flow_kg_s=flows.injection_flow_kg_per_s,
            method=assumptions.general.compression_method,
        )
    )

    if assumptions.general.minimum_operating_pressure_pa < assumptions.general.pipeline_pressure_pa:
            withdrawal_compression = calculate_compression(
                CompressionInput(
                    inlet_pressure_pa=assumptions.general.minimum_operating_pressure_pa,
                    outlet_pressure_pa=assumptions.general.pipeline_pressure_pa,
                    inlet_temperature_k=assumptions.general.temperature_k,
                    mass_flow_kg_s=flows.withdrawal_flow_kg_per_s,
                    method=assumptions.general.compression_method,
                )
        )
    else:
        withdrawal_compression = None

    #return fully defined storage project
    return StorageProject(
        technology=StorageTechnology.DEPLETED_GAS_FIELD,
        case_name=case_name,
        inventory=inventory,
        flows=flows,
        pressures=pressures,
        wells=wells,
        drilling=drilling,
        field_interconnection=field_interconnection,
        salt_leaching=None, #n/a for DGFs
        salt_leaching_process=None, #n/a for DGFs
        salt_conversion_process=None, #n/a for DGFs
        porous_first_fill_process=porous_first_fill_process, 
        compression=compression,
        withdrawal_compression=withdrawal_compression if withdrawal_compression else None,
        purification=purification
    )