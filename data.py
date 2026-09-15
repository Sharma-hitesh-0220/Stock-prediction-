"""
data.py
-------
Everything related to fetching data: stock price history from yfinance,
native-currency detection, and live FX conversion rates.

Keeping this isolated means if Yahoo Finance ever changes its API, or you
want to swap in a different data provider later, this is the only file
you need to touch.
"""

import pandas as pd
import streamlit as st
import yfinance as yf

CURRENCY_SYMBOLS = {
    "USD": "$", "INR": "₹", "EUR": "€", "GBP": "£", "JPY": "¥", "USDT": "₮",
}

CURRENCY_OPTIONS = ["USD", "INR", "EUR", "GBP", "JPY", "USDT"]

PRESET_TICKERS = ["AAPL", "TSLA", "MSFT", "RELIANCE.NS", "TCS.NS", "NVDA", "GOOGL"]


@st.cache_data(show_spinner=False)
def load_data(ticker: str, start, end) -> pd.DataFrame:
    """Downloads OHLCV history for a ticker between start and end dates."""
    df = yf.download(ticker, start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def get_native_currency(ticker: str) -> str:
    """Best-effort detection of the currency a ticker is quoted in."""
    try:
        info = yf.Ticker(ticker).fast_info
        cur = getattr(info, "currency", None)
        if not cur and isinstance(info, dict):
            cur = info.get("currency")
        if cur:
            return cur.upper()
    except Exception:
        pass
    return "USD"


@st.cache_data(show_spinner=False, ttl=1800)
def get_fx_rate(from_currency: str, to_currency: str) -> float:
    """Fetch a live conversion rate. USDT is treated as a USD-pegged stablecoin."""
    a = "USD" if from_currency == "USDT" else from_currency
    b = "USD" if to_currency == "USDT" else to_currency
    if a == b:
        return 1.0
    try:
        pair = yf.download(f"{a}{b}=X", period="5d", progress=False)
        if not pair.empty:
            return float(pair["Close"].dropna().iloc[-1])
        inverse = yf.download(f"{b}{a}=X", period="5d", progress=False)
        if not inverse.empty:
            return 1.0 / float(inverse["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return 1.0  # graceful fallback — no conversion rather than a crash


def fmt(value: float, currency: str) -> str:
    """Formats a numeric value with the correct currency symbol."""
    symbol = CURRENCY_SYMBOLS.get(currency, currency + " ")
    return f"{symbol}{value:,.2f}"
