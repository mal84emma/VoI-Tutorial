import numpy as np
from tabulate import tabulate

def fmt_design_results(results):
    """Display design results."""

    out = tabulate(
            [
                ['Solar capacity','kWp',round(results['solar_capacity'],1)],
                ['Battery capacity','kWh',round(results['battery_capacity'],1)],
                ['Total cost','£k/yr',round(results['total']/1e3,1)],
                ['CAPEX','£k/yr',round(results['capex']/1e3,1)],
                ['OPEX','£k/yr',round(results['opex']/1e3,1)]
            ],
            headers=['Parameter', 'Unit', 'Value']
        )

    return out

def fmt_voi_results(prior_results, posterior_results):
    """Display VoI results."""

    prior_cost = prior_results['total']
    preposterior_cost = np.mean([r['total'] for r in posterior_results])

    out = f"""
    Prior cost: £{prior_cost/1e3:.2f}/yr
    Posterior cost: £{preposterior_cost/1e3:.2f}/yr
    VoI: £{(prior_cost - preposterior_cost)/1e3:.2f}/yr
    VoI percent: {(prior_cost - preposterior_cost) / prior_cost * 100:.2f}%
    """

    return out