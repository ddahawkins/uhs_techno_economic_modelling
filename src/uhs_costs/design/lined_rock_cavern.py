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
    construct_lined_rock_cavern_geometry,
    construct_lined_rock_cavern_drainage_design,
    construct_lined_rock_cavern_lining_design,
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
#                               Salt cavern Design Assumptions
#
# ----------------------------------------------------------------------------------------------------

ASSUMPTIONS =  construct_hystories_design_assumptions(
    storage_technology=StorageTechnology.LINED_ROCK_CAVERN,
    general_overrides={
        'reference_reservoir_depth_m': 150, # from reference_selection script
        'minimum_operating_pressure_pa': 20e5, # from reference_selection script
        'maximum_operating_pressure_pa': 150e5, # from reference_selection script
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
    """Construct a complete LRC StorageProject.

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

    lrc_geometry = construct_lined_rock_cavern_geometry(
        required_effective_storage_volume_m3=inventory.required_storage_volume_m3,
        maximum_cavern_radius_m=assumptions.technology_specific.maximum_cavern_radius_m,
        cavern_height_m=assumptions.technology_specific.cavern_height_m,
        effective_volume_fraction=assumptions.technology_specific.effective_volume_fraction,
    )

    lrc_lining = construct_lined_rock_cavern_lining_design(
        geometry=lrc_geometry,
        steel_lining_thickness_m=assumptions.technology_specific.steel_lining_thickness_m,
        concrete_lining_thickness_m=assumptions.technology_specific.concrete_lining_thickness_m,
        steel_density_kg_per_m3=assumptions.technology_specific.steel_density_kg_per_m3,
        concrete_density_kg_per_m3=assumptions.technology_specific.concrete_density_kg_per_m3,
    )

    lrc_drainage = construct_lined_rock_cavern_drainage_design(
        geometry=lrc_geometry,
        tunnel_drainage_length_m=assumptions.technology_specific.tunnel_drainage_length_m,
        cavern_drainage_length_per_cavern_m=(
            assumptions.technology_specific.cavern_drainage_length_per_cavern_m
        ),
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

    #determine number of caverns, number of well heads, and actual per cavern withdrawal requirements
    number_well_heads = max(
        lrc_geometry.number_caverns,
        ceil(
            inventory.working_gas_h2_mass_kg
            / assumptions.technology_specific.maximum_h2_mass_per_well_kg
        ),
    )

    number_caverns = number_well_heads #assumes 1 WH per cavern
    number_production_wells = number_well_heads
    number_observation_wells = 1 #based on Huang et al. which assumes a single observation tunnel.

    #construct WellDesign
    wells = construct_well_design(
        well_temperature_k=assumptions.general.temperature_k,
        number_well_heads=number_well_heads,
        number_caverns=number_caverns,
        number_observation_wells=number_observation_wells,
        number_production_wells=number_production_wells
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

    lrc_first_fill_process = construct_porous_first_fill_process(
        working_gas_volume_million_sm3=inventory.working_gas_h2_volume_sm3 / 1_000_000,
        cushion_gas_volume_million_sm3=inventory.cushion_gas_h2_volume_sm3  / 1_000_000,
        injection_flow_million_sm3_per_day=flows.injection_flow_million_sm3_per_day,
        injection_availability_factor=assumptions.general.compressor_availability_factor
    )

    #return fully defined storage project
    return StorageProject(
        technology=StorageTechnology.LINED_ROCK_CAVERN,
        case_name=case_name,
        inventory=inventory,
        flows=flows,
        pressures=pressures,
        wells=wells,
        drilling=drilling,
        field_interconnection=field_interconnection,
        salt_leaching=None,
        salt_leaching_process=None,
        salt_conversion_process=None,
        porous_first_fill_process=lrc_first_fill_process,
        lrc_geometry=lrc_geometry,
        lrc_lining=lrc_lining,
        lrc_drainage=lrc_drainage,
        compression=compression,
        withdrawal_compression=withdrawal_compression if withdrawal_compression else None,
        purification=purification
    )
