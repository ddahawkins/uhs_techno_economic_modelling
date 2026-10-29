"""Salt cavern design constructor.

This module defines salt-cavern-specific design defaults and constructs
complete StorageProject objects from the lower-level design constructors.
"""

from __future__ import annotations

from math import ceil

from uhs_costs.design.helpers.project import StorageProject, StorageTechnology
from uhs_costs.design.helpers.storage_inventory import construct_storage_inventory
from uhs_costs.design.helpers.storage_flows import construct_storage_flows
from uhs_costs.design.helpers.storage_pressures import construct_storage_pressures
from uhs_costs.design.helpers.well_design import construct_well_design
from uhs_costs.design.helpers.site_development import (
    construct_drilling_design,
    construct_field_interconnection_design,
    construct_salt_leaching_design,
    construct_salt_conversion_process,
    construct_salt_leaching_process

)
from uhs_costs.design.helpers.purification import (
    construct_purification,
    gross_withdrawal_from_net,
)
from uhs_costs.design.helpers.compression_model import (
    CompressionInput,
    calculate_compression,
)

from uhs_costs.design.default_design_assumptions import HyStoriesDesignAssumptions, construct_hystories_design_assumptions


# ----------------------------------------------------------------------------------------------------
#
#                               Salt cavern Design Assumptions
#
# ----------------------------------------------------------------------------------------------------

ASSUMPTIONS =  construct_hystories_design_assumptions(
    storage_technology=StorageTechnology.SALT_CAVERN,
    general_overrides={
        'reference_reservoir_depth_m': 1000,
        'minimum_operating_pressure_pa': 60e5,
        'maximum_operating_pressure_pa': 190e5,
        'abandonment_pressure_pa': 0,
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
    """Construct a complete salt cavern StorageProject.

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

    # Public/project withdrawal flow is usable post-purification H2.
    # Physical cavern deliverability must be sized against the larger gross
    # pre-purification withdrawal where hydrogen recovery is below 100%.
    gross_withdrawal_flow_million_sm3_per_day = gross_withdrawal_from_net(
        flows.withdrawal_flow_million_sm3_per_day,
        purification.hydrogen_recovery,
    )

    #determine number of caverns, number of well heads, and actual per cavern withdrawal requirements
    number_caverns_for_storage_capacity_requirements = ceil(inventory.working_gas_h2_volume_sm3 / assumptions.technology_specific.working_gas_volume_per_cavern_sm3)
    number_caverns_for_withdrawal_requirements = ceil(gross_withdrawal_flow_million_sm3_per_day / assumptions.technology_specific.maximum_withdrawal_flow_per_cavern_million_sm3_per_day)
    number_caverns = max(number_caverns_for_storage_capacity_requirements, number_caverns_for_withdrawal_requirements)
    # withdrawal_flow_per_cavern_million_sm3_per_day = flows.withdrawal_flow_million_sm3_per_day / number_caverns
    
    number_well_heads = number_caverns  #HyStories assumes one well per cavern

    #construct WellDesign
    wells = construct_well_design(
        well_temperature_k=assumptions.general.temperature_k,
        number_well_heads=number_well_heads,
        number_caverns=number_caverns,
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

    #construct SaltLeachingDesign
    salt_leaching = construct_salt_leaching_design(
        fresh_water_pipeline_length_km=assumptions.technology_specific.fresh_water_pipeline_length_km,
        brine_disposal_pipeline_length_km=assumptions.technology_specific.brine_disposal_pipeline_length_km,
        free_gas_volume_per_cavern_thousand_m3=(
            inventory.required_storage_volume_m3 / wells.number_caverns / 1000
        ),
    )

    salt_leaching_process = construct_salt_leaching_process(
        free_gas_volume_per_cavern_thousand_m3=(
            salt_leaching.free_gas_volume_per_cavern_thousand_m3
        ),
        number_caverns=wells.number_caverns,
        number_working_leaching_pumps=assumptions.technology_specific.number_leaching_pumps,
    )

    salt_conversion_process = construct_salt_conversion_process(
        number_well_heads=wells.number_well_heads,
        free_gas_volume_per_cavern_thousand_m3=(
            salt_leaching.free_gas_volume_per_cavern_thousand_m3
        ),
        debrining_flowrate_per_cavern_m3_per_hour=(
            assumptions.technology_specific.debrining_flowrate_per_cavern_m3_per_hour
        ),
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
        technology=StorageTechnology.SALT_CAVERN,
        case_name=case_name,
        inventory=inventory,
        flows=flows,
        pressures=pressures,
        wells=wells,
        drilling=drilling,
        field_interconnection=field_interconnection,
        salt_leaching=salt_leaching,
        salt_leaching_process=salt_leaching_process,
        salt_conversion_process=salt_conversion_process,
        porous_first_fill_process=None, #n/a for salt caverns
        compression=compression,
        withdrawal_compression=withdrawal_compression if withdrawal_compression else None,
        purification=purification
    )
