"""
model.py
--------
The machine learning piece: trains a Random Forest Regressor to predict
next-day closing price, evaluates it on a chronological (no-shuffle)
train/test split, and produces a forward forecast plus a simple
BUY / SELL / HOLD signal.

Kept separate so the ML approach could be swapped (e.g. back to Linear
Regression, or to XGBoost) without touching any Streamlit/UI code.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume", "SMA_50", "EMA_20"]


@dataclass
class PredictionResult:
    """Bundle of everything the UI layer needs to render the prediction section."""
    mae: float
    rmse: float
    r2: float
    test_dates: pd.Index
    y_test: np.ndarray
    predictions: np.ndarray
    next_day_pred_native: float
    signal: str  # "BUY" | "SELL" | "HOLD / NEUTRAL"


def train_and_predict(
    df: pd.DataFrame,
    n_estimators: int = 100,
    max_depth: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
) -> PredictionResult | None:
    """
    Trains a Random Forest on historical data (features -> next day's Close)
    and returns a PredictionResult, or None if there isn't enough data.
    """
    model_df = df[FEATURE_COLUMNS].copy()
    model_df["Target"] = model_df["Close"].shift(-1)
    model_df.dropna(inplace=True)

    if len(model_df) <= 30:
        return None

    X = model_df[FEATURE_COLUMNS].values
    y = model_df["Target"].values

    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(np.mean((y_test - predictions) ** 2)))
    r2 = r2_score(y_test, predictions)

    # Refit on ALL available data to forecast the next, truly-unseen day.
    final_model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
    )
    final_model.fit(X, y)
    last_row = df[FEATURE_COLUMNS].iloc[[-1]].values
    next_day_pred_native = float(final_model.predict(last_row)[0])

    return PredictionResult(
        mae=mae,
        rmse=rmse,
        r2=r2,
        test_dates=model_df.index[split_idx:],
        y_test=y_test,
        predictions=predictions,
        next_day_pred_native=next_day_pred_native,
        signal="",  # filled in by decision_signal() once currency conversion is known
    )


def decision_signal(expected_change_pct: float) -> tuple[str, str, str]:
    """
    Maps a predicted % change to a (signal, css_class, arrow) tuple.
    Threshold is +/-1% — deliberately simple and easy to tune.
    """
    if expected_change_pct > 1.0:
        return "BUY", "ps-buy", "▲"
    elif expected_change_pct < -1.0:
        return "SELL", "ps-sell", "▼"
    return "HOLD / NEUTRAL", "ps-hold", "▬"
