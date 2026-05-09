from scipy.optimize import linprog
import numpy as np


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