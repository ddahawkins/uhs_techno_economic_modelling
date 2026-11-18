from dataclasses import dataclass, fields, replace
from typing import TypeVar, Mapping, Any

from uhs_costs.design.helpers.project import StorageTechnology
from uhs_costs.constants import DEFAULT_USD_TO_EUR, DEFAULT_CEPCI_2025, DEFAULT_CEPCI_2014


@dataclass(frozen=True)
class HyStoriesCostAssumptions:
    """Economic and cost-model assumptions for HyStories-derived equations."""

    # Currency/cost basis
    target_currency: str = "EUR"
    target_year: int = 2025

    # Material cost factors
    material_cost_factor_injection: float = 1
    material_cost_factor_withdrawal: float = 1

    # Energy and commodity prices
    #
    # The electricity price is retained as a reference cost for construction /
    # commissioning energy and for reduced-form operational OPEX correlations.
    # Operational compression electricity can instead be left out of UHS OPEX
    # and represented explicitly in a downstream CEM.
    cost_of_electricity_eur_per_mwh: float = 80.0
    hydrogen_cost_eur_per_kg: float = 6.71

    # OPEX
    surface_fixed_component_of_fixed_opex: float = 2100.0
    surface_fixed_opex_fraction_of_epc: float = 0.04
    subsurface_fixed_opex_fraction_of_wells_capex: float = 0.03

    # HyStories variable surface OPEX treatment.
    #
    # True:
    #   Keep compression electricity inside the injection variable-OPEX rate
    #   (reduced-form / standalone mode).
    # False:
    #   Remove the compression-electricity share from injection OPEX so a CEM
    #   can represent the compressor electricity demand explicitly. The
    #   non-electric ancillary share (lubricants, waste, chemicals) is retained.
    include_compression_electricity_cost_in_opex: bool = False

    # HyStories states that lubricants, waste disposal, and chemicals are
    # assumed to equal 4% of electricity cost. We use this to separate the
    # compression-related HyStories OPEX correlation into electricity and
    # residual ancillary cost when the electricity is externalised to a CEM.
    compression_ancillary_cost_fraction_of_electricity: float = 0.04


    # Contingency / BOP / indirects
    surface_contingency_fraction: float = 0.20
    subsurface_contingency_fraction: float = 0.20
    bop_fraction: float = 0.05

    #abex
    surface_abex_fraction: float = 0.20
    subsurface_abex_fraction: float = 0.20

@dataclass(frozen=True)
class LinedRockCavernCostAssumptions:
    """Economic and cost-model assumptions for LRC equations based on Huang et al. doi.org/10.1016/j.apenergy.2025.126564.
    Converted to EUR 2025 from USD 2014."""

    # LRC subsurface CAPEX assumptions, based on Huang et al and raised to EUR 2025. Huang's paper is from 2025, but parameters are from 2014 sources.

    lrc_well_fixed_cost_eur_per_well: float = 2.16e6 * DEFAULT_USD_TO_EUR * (DEFAULT_CEPCI_2025/DEFAULT_CEPCI_2014)
    lrc_well_variable_cost_eur_per_m: float = 1274.0 * DEFAULT_USD_TO_EUR * (DEFAULT_CEPCI_2025/DEFAULT_CEPCI_2014)

    lrc_mining_cost_eur_per_m3: float = 84.0 * DEFAULT_USD_TO_EUR * (DEFAULT_CEPCI_2025/DEFAULT_CEPCI_2014)

    lrc_steel_lining_cost_eur_per_tonne: float = 910.0 * DEFAULT_USD_TO_EUR * (DEFAULT_CEPCI_2025/DEFAULT_CEPCI_2014)
    lrc_concrete_lining_cost_eur_per_tonne: float = 50.4 * DEFAULT_USD_TO_EUR * (DEFAULT_CEPCI_2025/DEFAULT_CEPCI_2014)
    lrc_lining_installation_fraction: float = 0.05

    lrc_tunnel_drainage_cost_eur_per_m: float = 9.0 * DEFAULT_USD_TO_EUR * (DEFAULT_CEPCI_2025/DEFAULT_CEPCI_2014)
    lrc_cavern_drainage_cost_eur_per_m: float = 870.0 * DEFAULT_USD_TO_EUR * (DEFAULT_CEPCI_2025/DEFAULT_CEPCI_2014)

    
def construct_hystories_cost_assumptions(
    overrides: dict[str, object] | None = None,
) -> HyStoriesCostAssumptions:
    assumptions = HyStoriesCostAssumptions()
    return apply_dataclass_overrides(assumptions, overrides)


def construct_lined_rock_cavern_cost_assumptions(
    overrides: dict[str, object] | None = None,
) -> LinedRockCavernCostAssumptions:
    assumptions = LinedRockCavernCostAssumptions()
    return apply_dataclass_overrides(assumptions, overrides)



T = TypeVar("T")

def apply_dataclass_overrides(
    instance: T,
    overrides: Mapping[str, Any] | None = None,
) -> T:
    """Apply overrides to a frozen dataclass instance."""

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