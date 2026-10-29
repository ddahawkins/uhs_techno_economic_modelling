from dataclasses import dataclass
from enum import StrEnum

from uhs_costs.design.helpers.storage_inventory import StorageInventory
from uhs_costs.design.helpers.storage_flows import StorageFlows
from uhs_costs.design.helpers.storage_pressures import StoragePressures
from uhs_costs.design.helpers.well_design import WellDesign
from uhs_costs.design.helpers.site_development import (
    DrillingDesign, 
    FieldInterconnectionDesign,
    SaltLeachingDesign, 
    SaltLeachingProcess,
    SaltConversionProcess,
    PorousFirstFillProcess,
    LinedRockCavernGeometry,
    LinedRockCavernDrainageDesign,
    LinedRockCavernLiningDesign,

    )
from uhs_costs.design.helpers.compression_model import CompressionResult
from uhs_costs.design.helpers.purification import Purification



class StorageTechnology(StrEnum):
    SALT_CAVERN = "salt_cavern"
    DEPLETED_GAS_FIELD = "depleted_gas_field"
    AQUIFER = "aquifer"
    LINED_ROCK_CAVERN = "lined_rock_cavern"

@dataclass(frozen=True)
class StorageProject:
    technology: StorageTechnology
    case_name: str | None
    inventory: StorageInventory 
    flows: StorageFlows
    pressures: StoragePressures
    wells: WellDesign | None = None
    drilling: DrillingDesign | None = None
    field_interconnection: FieldInterconnectionDesign | None = None
    salt_leaching: SaltLeachingDesign | None = None
    salt_leaching_process: SaltLeachingProcess | None = None
    salt_conversion_process: SaltConversionProcess | None = None
    porous_first_fill_process: PorousFirstFillProcess | None = None
    lrc_geometry: LinedRockCavernGeometry | None = None
    lrc_lining: LinedRockCavernLiningDesign | None = None
    lrc_drainage: LinedRockCavernDrainageDesign | None = None
    compression: CompressionResult | None = None
    withdrawal_compression: CompressionResult | None = None
    purification: Purification | None = None


