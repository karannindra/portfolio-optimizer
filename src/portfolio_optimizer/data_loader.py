import yfinance as yf
import pandas as pd
import numpy as np
from portfolio_optimizer import config


def download_prices():
    """
    Download adjusted closing prices for a list of tickers.
    """

    data = yf.download(
        config.TICKERS,
        start=config.START_DATE,
        end=config.END_DATE,
        auto_adjust=False
    )["Close"]

    return data

def compute_returns(price_data):

    returns = price_data.pct_change().dropna()

    mean_returns = returns.mean() * 252
    risk = returns.std() * np.sqrt(252)

    return returns, mean_returns, risk

def compute_covariance_matrix(returns):

    cov_matrix = returns.cov() * 252

    return cov_matrix