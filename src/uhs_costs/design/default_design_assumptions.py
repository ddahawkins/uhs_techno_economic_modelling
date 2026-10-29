from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Mapping, TypeVar

from uhs_costs.design.helpers.project import StorageTechnology
from uhs_costs.design.helpers.compression_model import CompressionMethod


# =============================================================================
# Assumption dataclasses
# =============================================================================


@dataclass(frozen=True)
class GeneralDesignAssumptions:
    """Technology-independent design and physical assumptions."""

    # Physical characteristics
    temperature_k: float | None = None

    # Design pressure constraints
    minimum_operating_pressure_pa: float = 0
    maximum_operating_pressure_pa: float = 0
    abandonment_pressure_pa: float = 0

    # Pipeline assumptions
    pipeline_pressure_pa: float = 5.5e6

    # Wells / field lines
    drilling_complexity_index: float = 1.0

    # Surface interconnection
    # HyStories assumes the gas plant is 500 m from the nearest wellhead.
    # EPC3 applies only to distance beyond this baseline.
    gas_plant_to_nearest_wellhead_distance_km: float = 2.0

    # Purification
    # purification_factor is the HyStories economic cost scalar.
    # purification_hydrogen_recovery is the physical fraction of gross
    # withdrawal delivered as usable H2 to the downstream system.
    purification_factor: float = 0.0
    purification_hydrogen_recovery: float = 1.0

    #reference depth
    reference_reservoir_depth_m: float | None = None

    # Compression
    compression_method: CompressionMethod = CompressionMethod.HGSM_POLYTROPIC
    compressor_availability_factor: float = 0.9 #source=https://doi.org/10.1002/ente.202500552

    # recovery factor
    recovery_factor: float = 1

    #variable-volume gas storages (Aquifers)
    minimum_gas_saturation: float = 0.6 #source=https://doi.org/10.1016/j.ijhydene.2021.09.174
    maximum_gas_saturation: float = 0.8 #source=https://doi.org/10.1016/j.est.2024.112916


@dataclass(frozen=True)
class SaltCavernDesignAssumptions:
    """Technology-specific design assumptions for salt caverns."""

    # Cavern constraints
    working_gas_volume_per_cavern_sm3: float = 31_250_000.0
    maximum_withdrawal_flow_per_cavern_million_sm3_per_day: float = 2.79

    # Drilling constraints
    # last_cemented_casing_shoe_m: float = 1_000.0   # replaced by reference_reservoir_depth_m

    # Leaching constraints
    fresh_water_pipeline_length_km: float = 15.0
    brine_disposal_pipeline_length_km: float = 30.0
    debrining_flowrate_per_cavern_m3_per_hour: float = 200.0
    distance_nearest_gas_plant_km: float = 2.0
    number_leaching_pumps: int = 4


@dataclass(frozen=True)
class DepletedGasFieldDesignAssumptions:
    """Technology-specific design assumptions for depleted gas fields. """

    #wells
    number_existing_wells: int = 0

    # Drilling constraints
    # last_cemented_casing_shoe_m: float = 1_200.0 #https://publica-rest.fraunhofer.de/server/api/core/bitstreams/841ea3eb-07e5-4df6-a897-10e0daba18d7/content
    
    # flow rate constraints
    maximum_withdrawal_flow_site_million_sm3_per_day: float = 8.25

    #well heads
    number_observation_well_heads: int = 6
    number_production_well_heads: int = 24

@dataclass(frozen=True)
class AquiferDesignAssumptions:
    """Technology-specific design assumptions for aquifers. """

    #wells
    number_existing_wells: int = 0

    # Drilling constraints
    # last_cemented_casing_shoe_m: float = 1_300.0 #Sized based on Suliszewo reservoir, https://doi.org/10.1016/j.ijhydene.2022.09.284
    
    # flow rate constraints
    maximum_withdrawal_flow_site_million_sm3_per_day: float = 8.52 #derived from 8.4kg/s limit suggested by https://doi.org/10.1016/j.ijhydene.2022.09.284 

    #well heads
    number_observation_well_heads: int = 6 #taking Hystories default assumption
    number_production_well_heads: int = 24

@dataclass(frozen=True)
class LinedRockCavernDesignAssumptions:
    """Technology-specific design assumptions for lined rock caverns."""

    # Well / cavern constraints
    maximum_h2_mass_per_well_kg: float = 3_000_000.0

    # Cavern geometry
    maximum_cavern_radius_m: float = 20.0
    cavern_height_m: float = 100.0
    effective_volume_fraction: float = 0.95

    # Lining design
    steel_lining_thickness_m: float = 0.015
    concrete_lining_thickness_m: float = 0.5
    steel_density_kg_per_m3: float = 7850.0
    concrete_density_kg_per_m3: float = 2500.0

    # Drainage design
    tunnel_drainage_length_m: float = 760.0
    cavern_drainage_length_per_cavern_m: float = 100.0


TechnologySpecificDesignAssumptions = (
    SaltCavernDesignAssumptions | DepletedGasFieldDesignAssumptions | AquiferDesignAssumptions | LinedRockCavernDesignAssumptions
)


@dataclass(frozen=True)
class HyStoriesDesignAssumptions:
    """Complete HyStories design assumptions.

    This wrapper keeps general assumptions separate from assumptions that are
    only meaningful for a specific storage technology.
    """

    storage_technology: StorageTechnology
    general: GeneralDesignAssumptions
    technology_specific: TechnologySpecificDesignAssumptions


# =============================================================================
# Constructors
# =============================================================================


def construct_hystories_design_assumptions(
    storage_technology: StorageTechnology,
    general_overrides: Mapping[str, Any] | None = None,
    technology_overrides: Mapping[str, Any] | None = None,
) -> HyStoriesDesignAssumptions:
    """Construct default HyStories design assumptions for a storage technology.

    Parameters
    ----------
    storage_technology:
        Storage technology for which assumptions should be constructed.

    general_overrides:
        Optional overrides for technology-independent assumptions.

    technology_overrides:
        Optional overrides for technology-specific assumptions.

    Returns
    -------
    HyStoriesDesignAssumptions
        Complete design assumptions object.
    """

    general = construct_general_design_assumptions(
        storage_technology=storage_technology,
        overrides=general_overrides,
    )

    technology_specific = construct_technology_specific_design_assumptions(
        storage_technology=storage_technology,
        overrides=technology_overrides,
    )

    return HyStoriesDesignAssumptions(
        storage_technology=storage_technology,
        general=general,
        technology_specific=technology_specific,
    )


def construct_general_design_assumptions(
    storage_technology: StorageTechnology,
    overrides: Mapping[str, Any] | None = None,
) -> GeneralDesignAssumptions:
    """Construct general assumptions, including technology-dependent defaults."""

    if storage_technology == StorageTechnology.SALT_CAVERN:
        assumptions = GeneralDesignAssumptions(
            purification_factor=0.0,
            purification_hydrogen_recovery=1.0,
            reference_reservoir_depth_m=1_000.0,
            recovery_factor=0.9983
        )

    elif storage_technology == StorageTechnology.LINED_ROCK_CAVERN:
        assumptions = GeneralDesignAssumptions(
            purification_factor=0.0,
            purification_hydrogen_recovery=1.0,
            reference_reservoir_depth_m=150.0,
            recovery_factor=1
        )

    elif storage_technology ==  StorageTechnology.DEPLETED_GAS_FIELD:
        assumptions = GeneralDesignAssumptions(
            purification_factor=1.5,
            purification_hydrogen_recovery=0.8996,
            reference_reservoir_depth_m=1956.0,
            recovery_factor=0.8
        )
    elif storage_technology == StorageTechnology.AQUIFER:
        assumptions = GeneralDesignAssumptions(
            purification_factor=0.33,
            purification_hydrogen_recovery=0.95,
            reference_reservoir_depth_m=1162.0, 
            recovery_factor=0.9  
        )

    else:
        raise ValueError(
            f"{storage_technology} is not a valid input for storage_technology"
        )
    
    assumptions = apply_dataclass_overrides(assumptions, overrides)

    if assumptions.temperature_k in {0, None}:
        if assumptions.reference_reservoir_depth_m in {0, None}:
            raise ValueError(
                "Either temperature_k or reference_reservoir_depth_m must be assigned."
            )

        assumptions = replace(
            assumptions,
            temperature_k=calculate_temperature_from_depth_k(
                depth_m=assumptions.reference_reservoir_depth_m,
                surface_temperature_k=283.15,
                geothermal_gradient_k_per_km=30.0,
            ),
        )

    return assumptions


def construct_technology_specific_design_assumptions(
    storage_technology: StorageTechnology,
    overrides: Mapping[str, Any] | None = None,
) -> TechnologySpecificDesignAssumptions:
    """Construct technology-specific design assumptions."""

    if storage_technology == StorageTechnology.SALT_CAVERN:
        assumptions = SaltCavernDesignAssumptions()
    elif storage_technology ==StorageTechnology.DEPLETED_GAS_FIELD :
        assumptions = DepletedGasFieldDesignAssumptions()
    elif storage_technology ==StorageTechnology.AQUIFER:
        assumptions = AquiferDesignAssumptions()

    elif storage_technology == StorageTechnology.LINED_ROCK_CAVERN:
        assumptions = LinedRockCavernDesignAssumptions()

    else:
        raise ValueError(
            f"{storage_technology} is not a valid input for storage_technology"
        )

    return apply_dataclass_overrides(assumptions, overrides)


# =============================================================================
# Override helper
# =============================================================================


T = TypeVar("T")


def apply_dataclass_overrides(
    instance: T,
    overrides: Mapping[str, Any] | None = None,
) -> T:
    """Apply user overrides to a frozen dataclass instance.

    This preserves immutability by returning a new instance rather than mutating
    the original one.

    Parameters
    ----------
    instance:
        Dataclass instance to update.

    overrides:
        Mapping of field names to replacement values.

    Returns
    -------
    T
        Updated dataclass instance.

    Raises
    ------
    ValueError
        If any override key is not a valid field on the dataclass.
    """

    if overrides is None:
        return instance

    valid_field_names = {field.name for field in fields(instance)}
    override_field_names = set(overrides)

    invalid_field_names = override_field_names - valid_field_names

    if invalid_field_names:
        invalid_fields = ", ".join(sorted(invalid_field_names))
        class_name = type(instance).__name__

        raise ValueError(
            f"Invalid override field(s) for {class_name}: {invalid_fields}"
        )

    return replace(instance, **overrides)

def calculate_temperature_from_depth_k(
    depth_m: float,
    surface_temperature_k: float = 283.15,
    geothermal_gradient_k_per_km: float = 30.0,
) -> float:
    """Estimate subsurface temperature from depth using a linear geothermal gradient.

    This is a screening-level approximation suitable when site-specific
    reservoir temperature data are unavailable.

    Parameters
    ----------
    depth_m:
        Storage depth below ground level in metres.

    surface_temperature_k:
        Reference near-surface temperature in degrees Kelvin.

    geothermal_gradient_k_per_km:
        Geothermal gradient in degrees Kelvin per kilometre.

    Returns
    -------
    float
        Estimated subsurface temperature in Kelvin.
    """

    if depth_m < 0:
        raise ValueError("depth_m must be non-negative.")

    temperature_k = (
        surface_temperature_k
        + geothermal_gradient_k_per_km * depth_m / 1000.0
    )

    return temperature_k