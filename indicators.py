"""
indicators.py
-------------
Technical indicator calculations: moving averages, Bollinger Bands, RSI.
Pure pandas/numpy functions with no Streamlit dependency, so they can be
unit-tested or reused outside the dashboard (e.g. in a notebook or a
future non-Streamlit version of this project).
"""

import numpy as np
import pandas as pd


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Adds SMA-50, SMA-200 and EMA-20 columns."""
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Adds Bollinger Band upper/mid/lower columns."""
    df["Bollinger_Mid"] = df["Close"].rolling(window=window).mean()
    rolling_std = df["Close"].rolling(window=window).std()
    df["Bollinger_Upper"] = df["Bollinger_Mid"] + (rolling_std * num_std)
    df["Bollinger_Lower"] = df["Bollinger_Mid"] - (rolling_std * num_std)
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Adds a Relative Strength Index (RSI) column."""
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))
    return df


def add_daily_return(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a Daily_Return_% column (percentage change day over day)."""
    df["Daily_Return_%"] = df["Close"].pct_change() * 100
    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper: adds every indicator this project uses."""
    df = add_moving_averages(df)
    df = add_bollinger_bands(df)
    df = add_rsi(df)
    df = add_daily_return(df)
    return df
