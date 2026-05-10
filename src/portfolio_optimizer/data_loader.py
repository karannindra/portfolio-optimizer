import yfinance as yf
import pandas as pd
import numpy as np


def download_prices(tickers, start_date, end_date):
    """
    Download adjusted closing prices for a list of tickers.
    """

    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
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