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
        end=end_date
    )["Close"]

    return data