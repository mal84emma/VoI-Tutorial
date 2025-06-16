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