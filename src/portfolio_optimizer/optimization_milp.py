from scipy.optimize import linprog
import numpy as np
import pulp

def milp_rebalancing(
    mean_returns,
    current_weights,
    max_weight=0.35,
    transaction_cost=0.001,
    K=4
):

    assets = list(mean_returns.index)

    model = pulp.LpProblem(
        "Portfolio_Rebalancing",
        pulp.LpMaximize
    )

    # Decision variables
    x = pulp.LpVariable.dicts(
        "weight",
        assets,
        lowBound=0,
        upBound=1
    )

    buy = pulp.LpVariable.dicts(
        "buy",
        assets,
        lowBound=0
    )

    sell = pulp.LpVariable.dicts(
        "sell",
        assets,
        lowBound=0
    )

    y = pulp.LpVariable.dicts(
        "trade",
        assets,
        cat="Binary"
    )

    # Objective
    model += pulp.lpSum(
        mean_returns[a] * x[a]
        - transaction_cost * buy[a]
        - transaction_cost * sell[a]
        for a in assets
    )

    # Sum weights = 1
    model += pulp.lpSum(x[a] for a in assets) == 1

    # Asset constraints
    for a in assets:

        # Max weight
        model += x[a] <= max_weight

        # Rebalancing equation
        model += (
            x[a]
            - buy[a]
            + sell[a]
            == current_weights[a]
        )

        # Trade activation
        model += buy[a] + sell[a] <= y[a]

    # Maximum number of trades
    model += pulp.lpSum(y[a] for a in assets) <= K

    model.solve()

    weights = np.array([
        x[a].varValue for a in assets
    ])

    buys = np.array([
        buy[a].varValue for a in assets
    ])

    sells = np.array([
        sell[a].varValue for a in assets
    ])

    traded = np.array([
        y[a].varValue for a in assets
    ])

    portfolio_return = np.dot(mean_returns, weights)

    return (
        weights,
        buys,
        sells,
        traded,
        portfolio_return
    )