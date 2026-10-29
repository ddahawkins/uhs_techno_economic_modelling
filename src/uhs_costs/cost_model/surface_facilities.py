
from uhs_costs.cost_model.helpers import abex, surface_capex,surface_opex
from uhs_costs.design.helpers.project import StorageProject
from uhs_costs.design.helpers.purification import gross_withdrawal_from_net
from uhs_costs.cost_model.default_cost_assumptions import  HyStoriesCostAssumptions
from uhs_costs.cost_model.helpers.cost_components import (
    CostComponent,
    CostType,
    HyStoriesGroup,
    CostDriver,
    AllocationMethod,
    CostUnit,
    fixed_component_cost_allocation
)



def calculate_surface_cost_components(
    project: StorageProject,
    assumptions: HyStoriesCostAssumptions,
    components: list[CostComponent] = None,
) -> list[CostComponent]:
    """Calculate decomposed HyStories-derived surface facility cost components."""

    # -------------- index of objects within project --------------
    # project.compression
    # project.pressures
    # project.flows
    # project.inventory
    # project.wells
    # project.drilling
    # project.field_interconnection
    # project.salt_leaching
    # project.salt_leaching_process
    # project.salt_conversion_process
    # project.purification
    

    if components is None:
        components: list[CostComponent] = []

    # project.flows uses the CEM/system-interface basis. Withdrawal is usable
    # post-purification H2. HyStories surface equipment must process gross
    # pre-purification flow, so derive that quantity only for internal cost
    # equations. CostComponent drivers remain net so returned rates are CEM-ready.
    gross_withdrawal_flow_million_sm3_per_day = gross_withdrawal_from_net(
        project.flows.withdrawal_flow_million_sm3_per_day,
        project.purification.hydrogen_recovery,
    )
 
    # ---------------------------------------------------------------------------------------------------------------
    #
    #                                                   Surface EPC1
    #
    # ---------------------------------------------------------------------------------------------------------------

    surface_epc1_injection = surface_capex.epc1_compression_cost_eur(
        total_installed_compression_brake_power_mw=( project.compression.total_design_brake_power_kw / 1000 ),
        material_cost_factor_injection=assumptions.material_cost_factor_injection,
    )

    withdrawal_compresion_power_mw = project.withdrawal_compression.total_design_brake_power_kw / 1000 if project.withdrawal_compression else None
    surface_epc1_withdrawal = surface_capex.epc1_withdrawal_cost_eur(
        withdrawal_flow_million_sm3_per_day=gross_withdrawal_flow_million_sm3_per_day,
        material_cost_factor_withdrawal=assumptions.material_cost_factor_withdrawal,
        total_installed_withdrawal_compression_brake_power_mw=( withdrawal_compresion_power_mw )
    )

    surface_epc1_fixed = surface_capex.epc1_fixed_cost_eur()

    components.append(
        CostComponent(
            name="surface_epc1_injection",
            value_eur=surface_epc1_injection,
            cost_type=CostType.CAPEX,
            hystories_group=HyStoriesGroup.SURFACE,
            cost_driver=CostDriver.INJECTION,
            driver_value=project.flows.injection_flow_kw_h2_lhv,
            cost_unit=CostUnit.EUR_PER_KW,
            notes="HyStories EPC1 compression component mapped to injection capacity.",
        )
    )

    components.append(
        CostComponent(
            name="surface_epc1_withdrawal",
            value_eur=surface_epc1_withdrawal,
            cost_type=CostType.CAPEX,
            hystories_group=HyStoriesGroup.SURFACE,
            cost_driver=CostDriver.WITHDRAWAL,
            driver_value=project.flows.withdrawal_flow_kw_h2_lhv,
            cost_unit=CostUnit.EUR_PER_KW,
            notes="HyStories EPC1 withdrawal component mapped to withdrawal capacity",
        )
    )

    components.extend(
        fixed_component_cost_allocation(
            name="surface_epc1_fixed",
            value_eur=surface_epc1_fixed,
            cost_type=CostType.CAPEX,
            hystories_group=HyStoriesGroup.SURFACE,
            cost_unit=CostUnit.EUR_PER_KW,
            cost_drivers_and_values=(
                (CostDriver.WITHDRAWAL,project.flows.withdrawal_flow_kw_h2_lhv),
                (CostDriver.INJECTION,project.flows.injection_flow_kw_h2_lhv),
            ),
            allocation_method=AllocationMethod.EQUAL_SPLIT
        )
    )

    # ---------------------------------------------------------------------------------------------------------------
    #
    #                                                   Surface EPC2
    #
    # ---------------------------------------------------------------------------------------------------------------

    surface_epc2_fieldlines = surface_capex.epc2_fieldlines_cost_eur(
        withdrawal_flow_million_sm3_per_day=gross_withdrawal_flow_million_sm3_per_day,
        operating_pressure_ratio=project.pressures.operating_pressure_ratio,
        material_cost_factor_withdrawal=assumptions.material_cost_factor_withdrawal,
        number_well_heads=project.wells.number_well_heads
    )

    surface_epc2_instrumentation = surface_capex.epc2_instrumentation_cost_eur(
        material_cost_factor_withdrawal=assumptions.material_cost_factor_withdrawal,
        number_well_heads=project.wells.number_well_heads
    )

    components.append(
        CostComponent(
            name="surface_epc2_fieldlines",
            value_eur=surface_epc2_fieldlines,
            cost_type=CostType.CAPEX,
            hystories_group=HyStoriesGroup.SURFACE,
            cost_driver=CostDriver.WITHDRAWAL,
            driver_value=project.flows.withdrawal_flow_kw_h2_lhv,
            cost_unit=CostUnit.EUR_PER_KW,
            notes="HyStories EPC2 fieldlines component mapped to withdrawal capacity",
        )
    )

    components.append(
        CostComponent(
            name="surface_epc2_instrumentation",
            value_eur=surface_epc2_instrumentation,
            cost_type=CostType.CAPEX,
            hystories_group=HyStoriesGroup.SURFACE,
            cost_driver=CostDriver.WITHDRAWAL,
            driver_value=project.flows.withdrawal_flow_kw_h2_lhv,
            cost_unit=CostUnit.EUR_PER_KW,
            notes="HyStories EPC2 instrumentation component mapped to withdrawal capacity",
        )
    )

    # ---------------------------------------------------------------------------------------------------------------
    #
    #                                                   Surface EPC3
    #
    # ---------------------------------------------------------------------------------------------------------------

    surface_epc3_well_interconnection = surface_capex.epc3_well_interconnection_cost_eur(
        withdrawal_flow_million_sm3_per_day=gross_withdrawal_flow_million_sm3_per_day,
        operating_pressure_ratio=project.pressures.operating_pressure_ratio,
        material_cost_factor_withdrawal=assumptions.material_cost_factor_withdrawal,
        additional_field_line_length_km=project.field_interconnection.additional_field_line_length_km,
    )

    components.append(
        CostComponent(
            name="surface_epc3_well_interconnection",
            value_eur=surface_epc3_well_interconnection,
            cost_type=CostType.CAPEX,
            hystories_group=HyStoriesGroup.SURFACE,
            cost_driver=CostDriver.WITHDRAWAL,
            driver_value=project.flows.withdrawal_flow_kw_h2_lhv,
            cost_unit=CostUnit.EUR_PER_KW,
            notes="HyStories EPC3 well interconnection component mapped to withdrawal capacity",
        )
    )

    # ---------------------------------------------------------------------------------------------------------------
    #
    #                                                   Surface EPC4
    #
    # ---------------------------------------------------------------------------------------------------------------

    surface_epc4_purification = surface_capex.epc4_purification_cost_eur(
        withdrawal_flow_million_sm3_per_day=gross_withdrawal_flow_million_sm3_per_day,
        purification_factor=project.purification.purification_factor
    )

    components.append(
        CostComponent(
            name="surface_epc4_purification",
            value_eur=surface_epc4_purification,
            cost_type=CostType.CAPEX,
            hystories_group=HyStoriesGroup.SURFACE,
            cost_driver=CostDriver.WITHDRAWAL,
            driver_value=project.flows.withdrawal_flow_kw_h2_lhv,
            cost_unit=CostUnit.EUR_PER_KW,
            notes="HyStories EPC4 purification component mapped to withdrawal capacity",
        )
    )

    # ---------------------------------------------------------------------------------------------------------------
    #
    #                                            Surface Balance of Plant
    #
    # ---------------------------------------------------------------------------------------------------------------

    surface_epc5_fixed = surface_capex.epc5_fixed_cost_eur()

    surface_epc5_withdrawal = surface_capex.epc5_proportional_cost_eur(
        cost = sum(
            component.value_eur
            for component in components
            if (
                component.hystories_group == HyStoriesGroup.SURFACE
                and component.cost_type == CostType.CAPEX
                and component.cost_driver == CostDriver.WITHDRAWAL
            )
            ),
            factor_bop=assumptions.bop_fraction
    )

    surface_epc5_injection = surface_capex.epc5_proportional_cost_eur(
        cost = sum(
            component.value_eur
            for component in components
            if (
                component.hystories_group == HyStoriesGroup.SURFACE
                and component.cost_type == CostType.CAPEX
                and component.cost_driver == CostDriver.INJECTION
            )
            ),
        factor_bop=assumptions.bop_fraction
    )

    #fixed components (wells) are allowed for via the above sum loop

    components.append(
        CostComponent(
            name="surface_epc5_withdrawal",
            value_eur=surface_epc5_withdrawal,
            cost_type=CostType.CAPEX,
            hystories_group=HyStoriesGroup.SURFACE,
            cost_driver=CostDriver.WITHDRAWAL,
            driver_value=project.flows.withdrawal_flow_kw_h2_lhv,
            cost_unit=CostUnit.EUR_PER_KW,
            notes="HyStories EPC5 BOP mapped to withdrawal capacity",
        )
    )

    components.append(
        CostComponent(
            name="surface_epc5_injection",
            value_eur=surface_epc5_injection,
            cost_type=CostType.CAPEX,
            hystories_group=HyStoriesGroup.SURFACE,
            cost_driver=CostDriver.INJECTION,
            driver_value=project.flows.injection_flow_kw_h2_lhv,
            cost_unit=CostUnit.EUR_PER_KW,
            notes="HyStories EPC5 BOP mapped to injection capacity",
        )
    )

    components.extend(
        fixed_component_cost_allocation(
            name="surface_epc5_fixed_bop_component",
            value_eur=surface_epc5_fixed,
            cost_type=CostType.CAPEX,
            hystories_group=HyStoriesGroup.SURFACE,
            cost_unit=CostUnit.EUR_PER_KW,
            cost_drivers_and_values=(
                (CostDriver.WITHDRAWAL,project.flows.withdrawal_flow_kw_h2_lhv),
                (CostDriver.INJECTION,project.flows.injection_flow_kw_h2_lhv),
            ),
            allocation_method=AllocationMethod.EQUAL_SPLIT
        )
    )

    # ---------------------------------------------------------------------------------------------------------------
    #
    #                                                   Surface Contingency
    #
    # ---------------------------------------------------------------------------------------------------------------


    surface_withdrawal_base_eur = sum(
            component.value_eur
            for component in components
            if (
                component.hystories_group == HyStoriesGroup.SURFACE
                and component.cost_type == CostType.CAPEX
                and component.cost_driver == CostDriver.WITHDRAWAL
            )
        )
    surface_injection_base_eur = sum(
            component.value_eur
            for component in components
            if (
                component.hystories_group == HyStoriesGroup.SURFACE
                and component.cost_type == CostType.CAPEX
                and component.cost_driver == CostDriver.INJECTION
            )
        )

    surface_contingency_withdrawal = surface_capex.contingency_cost_eur(
        base_cost_eur=surface_withdrawal_base_eur,
        contingency_fraction=assumptions.surface_contingency_fraction
    )

    components.append(
        CostComponent(
            name="surface_contingency_withdrawal",
            value_eur=surface_contingency_withdrawal,
            cost_type=CostType.CAPEX,
            hystories_group=HyStoriesGroup.SURFACE,
            cost_driver=CostDriver.WITHDRAWAL,
            driver_value=project.flows.withdrawal_flow_kw_h2_lhv,
            cost_unit=CostUnit.EUR_PER_KW,
            notes="HyStories Withdrawal component of surface Contingencies mapped to withdrawal capacity",
        )
    )

    surface_contingency_injection = surface_capex.contingency_cost_eur(
        base_cost_eur=surface_injection_base_eur,
        contingency_fraction=assumptions.surface_contingency_fraction
    )

    components.append(
        CostComponent(
            name="surface_contingency_injection",
            value_eur=surface_contingency_injection,
            cost_type=CostType.CAPEX,
            hystories_group=HyStoriesGroup.SURFACE,
            cost_driver=CostDriver.INJECTION,
            driver_value=project.flows.injection_flow_kw_h2_lhv,
            cost_unit=CostUnit.EUR_PER_KW,
            notes="HyStories injection component of surface Contingencies mapped to injection capacity",
        )
    )

    # ---------------------------------------------------------------------------------------------------------------
    #
    #                                                   Surface ABEX
    #
    # ---------------------------------------------------------------------------------------------------------------

    surface_abex_withdrawal_base_eur = sum(
            component.value_eur
            for component in components
            if (
                component.hystories_group == HyStoriesGroup.SURFACE
                and component.cost_type == CostType.CAPEX
                and component.cost_driver == CostDriver.WITHDRAWAL
            )
        )
    surface_abex_injection_base_eur = sum(
            component.value_eur
            for component in components
            if (
                component.hystories_group == HyStoriesGroup.SURFACE
                and component.cost_type == CostType.CAPEX
                and component.cost_driver == CostDriver.INJECTION
            )
        )
    
    surface_abex_withdrawal = abex.abex_cost_eur(
        epc_cost_eur=surface_abex_withdrawal_base_eur,
        abex_fraction=assumptions.surface_abex_fraction
    )

    components.append(
        CostComponent(
            name="surface_abex_withdrawal",
            value_eur=surface_abex_withdrawal,
            cost_type=CostType.ABEX,
            hystories_group=HyStoriesGroup.SURFACE,
            cost_driver=CostDriver.WITHDRAWAL,
            driver_value=project.flows.withdrawal_flow_kw_h2_lhv,
            cost_unit=CostUnit.EUR_PER_KW,
            notes="HyStories Withdrawal component of surface ABEX mapped to withdrawal capacity",
        )
    )

    surface_abex_injection = abex.abex_cost_eur(
        epc_cost_eur=surface_abex_injection_base_eur,
        abex_fraction=assumptions.surface_abex_fraction
    )

    components.append(
        CostComponent(
            name="surface_abex_injection",
            value_eur=surface_abex_injection,
            cost_type=CostType.ABEX,
            hystories_group=HyStoriesGroup.SURFACE,
            cost_driver=CostDriver.INJECTION,
            driver_value=project.flows.injection_flow_kw_h2_lhv,
            cost_unit=CostUnit.EUR_PER_KW,
            notes="HyStories injection component of surface ABEX mapped to injection capacity",
        )
    )

    # ---------------------------------------------------------------------------------------------------------------
    #
    #                                                   Surface OPEX
    #
    # ---------------------------------------------------------------------------------------------------------------

    # ---------------------- Fixed opex ----------------------

    surface_fixed_opex_employees_fixed = surface_opex.fixed_opex_base_cost_eur_per_year(
        fixed_cost_k_eur=assumptions.surface_fixed_component_of_fixed_opex
    )

    components.extend(
        fixed_component_cost_allocation(
            name="surface_fixed_opex_employees_fixed",
            value_eur=surface_fixed_opex_employees_fixed,
            cost_type=CostType.FIXED_OPEX,
            hystories_group=HyStoriesGroup.SURFACE,
            cost_unit=CostUnit.EUR_PER_KW_YEAR,
            cost_drivers_and_values=(
                (CostDriver.WITHDRAWAL,project.flows.withdrawal_flow_kw_h2_lhv),
                (CostDriver.INJECTION,project.flows.injection_flow_kw_h2_lhv),
            ),
            allocation_method=AllocationMethod.EQUAL_SPLIT
        )
    )

    surface_base_capex_injection = sum(
            component.value_eur
            for component in components
            if (
                component.hystories_group == HyStoriesGroup.SURFACE
                and component.cost_type == CostType.CAPEX
                and component.cost_driver == CostDriver.INJECTION
                and component.name != "surface_contingency_injection"
            )
        )
    
    surface_base_capex_withdrawal = sum(
            component.value_eur
            for component in components
            if (
                component.hystories_group == HyStoriesGroup.SURFACE
                and component.cost_type == CostType.CAPEX
                and component.cost_driver == CostDriver.WITHDRAWAL
                and component.name != "surface_contingency_withdrawal"
            )
        )

    surface_fixed_opex_fraction_of_capex_injection = surface_opex.fixed_opex_fraction_of_epc_cost_eur_per_year(
        surface_epc_cost_eur=surface_base_capex_injection,
        fixed_opex_fraction_of_epc=assumptions.surface_fixed_opex_fraction_of_epc
    )

    components.append(
        CostComponent(
            name="surface_fixed_opex_fraction_of_capex_injection",
            value_eur=surface_fixed_opex_fraction_of_capex_injection,
            cost_type=CostType.FIXED_OPEX,
            hystories_group=HyStoriesGroup.SURFACE,
            cost_driver=CostDriver.INJECTION,
            driver_value=project.flows.injection_flow_kw_h2_lhv,
            cost_unit=CostUnit.EUR_PER_KW_YEAR,
            notes="HyStories injection component of surface fixed opex mapped to injection capacity, euro/year/kW_injection",
        )
    )

    surface_fixed_opex_fraction_of_capex_withdrawal = surface_opex.fixed_opex_fraction_of_epc_cost_eur_per_year(
        surface_epc_cost_eur=surface_base_capex_withdrawal,
        fixed_opex_fraction_of_epc=assumptions.surface_fixed_opex_fraction_of_epc
    )

    components.append(
        CostComponent(
            name="surface_fixed_opex_fraction_of_capex_withdrawal",
            value_eur=surface_fixed_opex_fraction_of_capex_withdrawal,
            cost_type=CostType.FIXED_OPEX,
            hystories_group=HyStoriesGroup.SURFACE,
            cost_driver=CostDriver.WITHDRAWAL,
            driver_value=project.flows.withdrawal_flow_kw_h2_lhv,
            cost_unit=CostUnit.EUR_PER_KW_YEAR,
            notes="HyStories withdrawal component of surface fixed opex mapped to withdrawal capacity, euro/year/kW_withdrawal",
        )
    )

    # ---------------------- Variable opex ----------------------

    surface_variable_opex_injection = surface_opex.variable_opex_rate_injection_eur_per_kwh_h2_lhv(
        cost_of_electricity_eur_per_mwh=assumptions.cost_of_electricity_eur_per_mwh,
        compression_ratio=project.compression.overall_pressure_ratio,
        include_compression_electricity_cost=(
            assumptions.include_compression_electricity_cost_in_opex
        ),
        ancillary_cost_fraction_of_electricity=(
            assumptions.compression_ancillary_cost_fraction_of_electricity
        ),
    )

    components.append(
        CostComponent(
            name="surface_variable_opex_injection",
            value_eur=surface_variable_opex_injection,
            cost_type=CostType.VARIABLE_OPEX,
            hystories_group=HyStoriesGroup.SURFACE,
            cost_driver=CostDriver.INJECTION_THROUGHPUT_KWH,
            driver_value=1,
            cost_unit=CostUnit.EUR_PER_KWH,
            notes=(
                "HyStories injection compression-related variable OPEX mapped "
                "to injection throughput. Compression electricity is "
                + (
                    "included in this cost rate."
                    if assumptions.include_compression_electricity_cost_in_opex
                    else (
                        "excluded for explicit downstream CEM treatment; only "
                        "the ancillary lubricants/waste/chemicals allowance is retained."
                    )
                )
            ),
        )
    )

    surface_variable_opex_withdrawal_gross = surface_opex.variable_opex_rate_withdrawal_eur_per_kwh_h2_lhv(
        cost_of_electricity_eur_per_mwh=assumptions.cost_of_electricity_eur_per_mwh,
        purification_coefficient=project.purification.purification_factor
    )

    # HyStories' withdrawal rate is on gross throughput. Calliope applies
    # cost_flow_out to usable post-efficiency output, so return a net-basis
    # CEM-ready rate.
    surface_variable_opex_withdrawal = gross_withdrawal_from_net(
        surface_variable_opex_withdrawal_gross,
        project.purification.hydrogen_recovery,
    )

    components.append(
        CostComponent(
            name="surface_variable_opex_withdrawal",
            value_eur=surface_variable_opex_withdrawal,
            cost_type=CostType.VARIABLE_OPEX,
            hystories_group=HyStoriesGroup.SURFACE,
            cost_driver=CostDriver.WITHDRAWAL_THROUGHPUT_KWH,
            driver_value=1,
            cost_unit=CostUnit.EUR_PER_KWH,
            notes=(
                "HyStories withdrawal variable OPEX normalised to usable "
                "post-purification H2 throughput for direct CEM use."
            ),
        )
    )

    return components



    

