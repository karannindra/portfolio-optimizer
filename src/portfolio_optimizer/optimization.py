from scipy.optimize import linprog
import numpy as np
import pulp


def baseline_lp(mean_returns, max_weight=0.35):

    n_assets = len(mean_returns)

    c = -mean_returns.values

    A = np.eye(n_assets)
    b = np.ones(n_assets) * max_weight

    Aeq = np.ones((1, n_assets))
    beq = [1]

    bounds = [(0, 1)] * n_assets

    result = linprog(
        c,
        A_ub=A,
        b_ub=b,
        A_eq=Aeq,
        b_eq=beq,
        bounds=bounds,
        method="highs"
    )

    weights = result.x

    portfolio_return = np.dot(mean_returns, weights)

    return weights, portfolio_return

def risk_constrained_lp(
    mean_returns,
    risk,
    risk_limit=0.18,
    max_weight=0.35
):

    n_assets = len(mean_returns)

    # Objective function
    c = -mean_returns.values

    # Weight constraints
    A_weights = np.eye(n_assets)
    b_weights = np.ones(n_assets) * max_weight

    # Risk constraint
    A_risk = np.array([risk.values])
    b_risk = np.array([risk_limit])

    # Combine constraints
    A_ub = np.vstack([A_weights, A_risk])
    b_ub = np.concatenate([b_weights, b_risk])

    # Equality constraint
    A_eq = np.ones((1, n_assets))
    b_eq = [1]

    # Bounds
    bounds = [(0, 1)] * n_assets

    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs"
    )

    weights = result.x

    portfolio_return = np.dot(mean_returns, weights)
    portfolio_risk = np.dot(risk, weights)

    return weights, portfolio_return, portfolio_risk

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