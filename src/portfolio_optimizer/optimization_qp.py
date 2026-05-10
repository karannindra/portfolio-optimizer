import numpy as np
from scipy.optimize import minimize


def portfolio_return(weights, mean_returns):

    return np.dot(weights, mean_returns)


def portfolio_risk(weights, cov_matrix):

    return np.sqrt(
        weights.T @ cov_matrix @ weights
    )


def negative_sharpe_ratio(
    weights,
    mean_returns,
    cov_matrix,
    risk_free_rate
):

    p_return = portfolio_return(
        weights,
        mean_returns
    )

    p_risk = portfolio_risk(
        weights,
        cov_matrix
    )

    sharpe = (
        p_return - risk_free_rate
    ) / p_risk

    # Minimize negative Sharpe
    return -sharpe


def sharpe_optimization(
    mean_returns,
    cov_matrix,
    risk_free_rate=0.02,
    max_weight=0.35
):

    n_assets = len(mean_returns)

    # Initial equal weights
    initial_weights = np.ones(n_assets) / n_assets

    # Constraint: weights sum to 1
    constraints = ({
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1
    })

    # Bounds for each asset
    bounds = tuple(
        (0, max_weight)
        for _ in range(n_assets)
    )

    result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(
            mean_returns,
            cov_matrix,
            risk_free_rate
        ),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = result.x

    optimal_return = portfolio_return(
        optimal_weights,
        mean_returns
    )

    optimal_risk = portfolio_risk(
        optimal_weights,
        cov_matrix
    )

    optimal_sharpe = (
        optimal_return - risk_free_rate
    ) / optimal_risk

    return (
        optimal_weights,
        optimal_return,
        optimal_risk,
        optimal_sharpe
    )