"""Model simulating building energy system."""

import os
import yaml
from linopy import Model, Variable, Constraint
import xarray as xr
import numpy as np
import pandas as pd
from typing import Iterable



def run_model(
        solar_capacity: float | None,
        battery_capacity: float | None,
        solar_year: Iterable[int] = None,
        load_year: Iterable[int] = None,
        mean_load: Iterable[float] = None,
        battery_efficiency: Iterable[float] = None,
        battery_cost: Iterable[float] = None,
        scenario_weightings: Iterable[float] = None,
        defaults_file: str = os.path.join('resources','defaults.yaml'),
        progress=False,
        env=None
    ):
    """Build and run building energy model for scenarios with specified parameter values.
    If parameter values are not specified, the model will be run using default values.

    The model can either be run for a given design but providing values for `solar_capacity`
    and `battery_capacity`, or the model can be run in design mode by providing `None` for
    these values. In design mode, the model will determine the optimal capacity for the
    solar and battery systems by including them as decision variables in the Linear Program.
    In both the simulation and design modes, the model is a Linear (Stochastic) Program,
    and operation is performed by optimising the operational variables (which assumes perfect)
    foresight for control.

    Args:
        solar_capacity (float | None): Solar PV capacity in kW. If None, the model will be
            run in design mode, and the optimal capacity will be determined.
        battery_capacity (float | None): Battery capacity in kWh. If None, the model will be
            run in design mode, and the optimal capacity will be determined.
        solar_year (Iterable[int], optional): Years of solar data to be used for each scenario.
            Defaults to None. In which case default value is used.
        load_year (Iterable[int], optional): Years of load data to be used for each scenario.
            Defaults to None. In which case default value is used.
        mean_load (Iterable[float], optional): Mean building load to be used for each scenario.
            Defaults to None. In which case default value is used.
        battery_efficiency (Iterable[float], optional): Efficiency of battery storage to be used
            for each scenario. Defaults to None. In which case default value is used.
        battery_cost (Iterable[float], optional): Cost of battery storage to be used for each scenario.
            Defaults to None. In which case default value is used.
        scenario_weightings (Iterable[float], optional): Weightings of scenarios. Defaults to None.
            In which case all scenarios are equally weighted.
        defaults_file (str, optional): YAML file containing default values to use. Defaults to 'defaults.yaml'.
        progress (bool, optional): Whether to print logs during optimisation. Defaults to False.
        env (gurobipy.Env, optional): Gurobi environment to use for optimisation. Defaults to None.
            If None, HiGHS will be used instead.

    Returns:
        results:
            Dictionary containing the following keys:
            - 'solar_capacity': Optimal solar capacity in kW.
            - 'battery_capacity': Optimal battery capacity in kWh.
            - 'total': Total cost of the system (CAPEX + OPEX). In £/yr.
            - 'capex': Capital expenditure of the system. In £/yr.
            - 'opex': Operational expenditure of the system. In £/yr.
            - 'scenario_costs': List of dictionaries containing the costs for each scenario.
                Each dictionary contains the following keys:
                - 'solar': Cost of solar capacity in £/yr.
                - 'battery': Cost of battery capacity in £/yr.
                - 'elec': Cost of electricity in £/yr.
    """

    ## Validate inputs
    ## ===============
    input_lengths = set([len(v) for v in [solar_year, load_year, mean_load, battery_efficiency] if v is not None])
    assert len(input_lengths) <= 1, \
        f'All input lists must be of the same length (i.e. same no. of scenarios). Lengths were: {input_lengths}'
    M = [i for i in input_lengths][0] if len(input_lengths) > 0 else 1

    if scenario_weightings is None:
        scenario_weightings = np.ones(M) / M
    else:
        assert len(scenario_weightings) == M, 'Scenario weightings must be the same length as the number of scenarios.'
        np.isclose(np.sum(scenario_weightings), 1.0, rtol=1e-3),\
            f"Scenario weightings must sum to 1.  Currently sum to {np.sum(scenario_weightings)}"

    if (solar_capacity is None) and (battery_capacity is None):
        design = True
    else:
        design = False
        assert (solar_capacity >= 0) and (battery_capacity >= 0), 'Capacities must be non-negative'

    ## Load data
    ## =========
    with open(defaults_file, 'r') as f:
        defaults = yaml.safe_load(f)

    # is there a programmatic way to do this?
    if solar_year is None:
        solar_year = [defaults['variables']['solar_year'] for _ in range(M)]
    if load_year is None:
        load_year = [defaults['variables']['load_year'] for _ in range(M)]
    if mean_load is None:
        mean_load = [defaults['variables']['mean_load'] for _ in range(M)]
    if battery_efficiency is None:
        battery_efficiency = [defaults['variables']['battery_efficiency'] for _ in range(M)]
    if battery_cost is None:
        battery_cost = [defaults['variables']['battery_cost'] for _ in range(M)]

    T = 8760
    solar_series = [pd.read_csv(os.path.join(defaults['data_dir'],f'solar_{year}.csv')).values.flatten()[:T]/1000 for year in solar_year]
    load_series = [pd.read_csv(os.path.join(defaults['data_dir'],f'load_{year}.csv')).values.flatten()[:T] for year in load_year]
    for m,s in enumerate(load_series): load_series[m] = s * (mean_load[m] / np.mean(s))
    elec_prices = pd.read_csv(os.path.join(defaults['data_dir'],'pricing.csv'))['Electricity Pricing [£/kWh]'].values.flatten()[:T]

    ## Build model
    ## ===========
    model: Model = Model(force_dim_names=True)

    # Capacity variables
    solar_cap: Variable = model.add_variables(lower=0, name='solar_capacity')
    battery_cap: Variable = model.add_variables(lower=0, name='battery_capacity')

    if not design:
        model.add_constraints(solar_cap, '=', solar_capacity, name='solar_capacity')
        model.add_constraints(battery_cap, '=', battery_capacity, name='wind_capacity')
    else:
        pass
        # model.add_constraints(solar_cap, '<=', defaults['static']['solar_limit'], name='solar_max_capacity')
        # model.add_constraints(battery_cap, '<=', defaults['static']['battery_limit'], name='battery_max_capacity')

    # save vars
    scen_obj_contrs = []
    scenario_objectives = []

    ## Scenarios
    for m in range(M):

        load = xr.DataArray(load_series[m], coords=[pd.RangeIndex(T,name='time')])
        solar = xr.DataArray(solar_series[m], coords=[pd.RangeIndex(T,name='time')]) * solar_cap
        elec_prices = xr.DataArray(elec_prices, coords=[pd.RangeIndex(T,name='time')])

        ## Dynamics
        battery_energy = 0 # net energy flow *into* batteries

        P_max = defaults['static']['discharge_ratio']*battery_cap
        E_min = (1-defaults['static']['depth_of_discharge'])*battery_cap
        eta = battery_efficiency[m]

        # Dynamics decision variables
        SOC: Variable = model.add_variables(lower=0, name=f'SOC_s{m}', coords=[pd.RangeIndex(T,name='time')])
        Ein: Variable = model.add_variables(lower=0, name=f'Ein_s{m}', coords=[pd.RangeIndex(T,name='time')])
        Eout: Variable = model.add_variables(lower=0, name=f'Eout_s{m}', coords=[pd.RangeIndex(T,name='time')])

        # Dynamics constraints
        model.add_constraints(defaults['static']['initial_SoC']*battery_cap.at[0] + -1*SOC.at[0] + np.sqrt(eta)*Ein.at[0] - 1/np.sqrt(eta)*Eout.at[0], '=', 0, name=f'SOC_init_s{m}')
        model.add_constraints(SOC[:-1] - SOC[1:] + np.sqrt(eta)*Ein[1:] - 1/np.sqrt(eta)*Eout[1:], '=', 0, name=f'SOC_series_s{m}')

        model.add_constraints(Ein, '<=', P_max, name=f'Pin_max_s{m}')
        model.add_constraints(Eout, '<=', P_max, name=f'Pout_max_s{m}')

        model.add_constraints(SOC, '<=', battery_cap, name=f'SOC_max_s{m}')
        model.add_constraints(SOC, '>=', E_min, name=f'SOC_min_s{m}')

        battery_energy += (Ein - Eout)

        supplied_energy = -1*solar + battery_energy
        grid_energy = supplied_energy + load

        model.add_constraints(grid_energy, '<=', defaults['static']['grid_cap'], name=f'grid_energ_pos_s{m}')
        model.add_constraints(grid_energy, '>=', -1*defaults['static']['grid_cap'], name=f'grid_energy_neg_s{m}')


        ## Objective
        pos_grid_energy = model.add_variables(lower=0, name=f'pos_grid_energy_s{m}', coords=[pd.RangeIndex(T,name='time')])
        model.add_constraints(pos_grid_energy, '>=', grid_energy, name=f'pos_grid_energy_s{m}')

        scen_obj_contrs.append({
            'solar': defaults['static']['solar_cost'] * solar_cap,
            'battery': battery_cost[m] * battery_cap,
            # this electricity cost term is some serious jank to make different buy & sell prices work when you can't have
            # constants in the objective, and because have slack variables on two sides doesn't work (i.e. can't also have neg_grid_energy)
            'elec': (1-defaults['static']['sell_price']) * pos_grid_energy @ elec_prices + defaults['static']['sell_price'] * supplied_energy @ elec_prices,
            #'elec': supplied_energy @ elec_prices # electricity cost without plant usage (constants not allowed in objective)
            #'carbon': pos_grid_energy @ carbon_intensity * carbon_price
        })
        scenario_objectives.append(sum(scen_obj_contrs[m].values()))

    # Overall objective
    scenario_objectives = np.array(scenario_objectives)
    objective = scenario_weightings @ scenario_objectives

    model.add_objective(objective, sense='min')

    ## Solve model
    ## ===========

    if env is None:
        solver_name = 'highs'
        solver_kwargs = {
            'output_flag': progress,
            'log_to_console': progress
        }
    else:
        solver_name = 'gurobi'
        solver_kwargs = {
            'env': env,
            'Method': 2
        }

    model.solve(progress=progress, solver_name=solver_name, **solver_kwargs)

    ## Post-process results
    ## ====================
    for m in range(M): # correct objective values by adding constants back in
        scen_obj_contrs[m]['elec'] += (xr.DataArray(load_series[m], coords=[pd.RangeIndex(T,name='time')]) @ elec_prices) * defaults['static']['sell_price']
        # the trick is to think about the sum of the original cost term and this correction term and what the price looks like if grid_energy is positive or negative

    capex = np.mean([float(c['solar'].solution.values + c['battery'].solution.values) for c in scen_obj_contrs])
    opex = np.mean([c['elec'].solution.values for c in scen_obj_contrs])

    results = {
        'solar_capacity': float(solar_cap.solution.values),
        'battery_capacity': float(battery_cap.solution.values),
        'total': capex + opex,
        'capex': capex,
        'opex': opex,
        'scenario_costs': [{
            k: float(v.solution.values) for k, v in scen_obj_contrs[m].items()
        } for m in range(M)],
    }

    return results



if __name__ == '__main__':

    print(run_model(None,None))
    print(run_model(0,0))
    print(run_model(800,500,progress=True))
    print(run_model(800,500,solar_year=[2012,2013,2014,2015,2016],progress=True))
    #print(run_model(None,None,solar_year=[2012,2013,2014,2015,2016],progress=True))