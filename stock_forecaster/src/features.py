import numpy as np
import pandas as pd

from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands


def create_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # --------------------------------------------------
    # BASIC RETURNS
    # --------------------------------------------------

    df["return_1d"] = df["Close"].pct_change(1)
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_10d"] = df["Close"].pct_change(10)
    df["return_20d"] = df["Close"].pct_change(20)
    df["return_60d"] = df["Close"].pct_change(60)

    # --------------------------------------------------
    # PRICE STRUCTURE
    # --------------------------------------------------

    df["high_low_range"] = (
        (df["High"] - df["Low"]) / df["Close"]
    )

    df["open_close_change"] = (
        (df["Close"] - df["Open"]) / df["Open"]
    )

    # --------------------------------------------------
    # MOVING AVERAGES
    # --------------------------------------------------

    sma_periods = [5, 10, 20, 50, 100, 200]

    for period in sma_periods:

        sma = SMAIndicator(
            close=df["Close"],
            window=period,
        ).sma_indicator()

        df[f"sma_{period}"] = sma

        df[f"close_sma_{period}_ratio"] = (
            df["Close"] / sma - 1
        )

    # --------------------------------------------------
    # EMA
    # --------------------------------------------------

    ema12 = EMAIndicator(
        close=df["Close"],
        window=12,
    ).ema_indicator()

    ema26 = EMAIndicator(
        close=df["Close"],
        window=26,
    ).ema_indicator()

    df["ema_12"] = ema12
    df["ema_26"] = ema26

    df["ema_12_ratio"] = df["Close"] / ema12 - 1
    df["ema_26_ratio"] = df["Close"] / ema26 - 1

    # --------------------------------------------------
    # RSI
    # --------------------------------------------------

    rsi = RSIIndicator(
        close=df["Close"],
        window=14,
    )

    df["rsi_14"] = rsi.rsi()

    # --------------------------------------------------
    # MACD
    # --------------------------------------------------

    macd = MACD(
        close=df["Close"],
        window_slow=26,
        window_fast=12,
        window_sign=9,
    )

    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # Normalize MACD
    df["macd_ratio"] = df["macd"] / df["Close"]
    df["macd_signal_ratio"] = (
        df["macd_signal"] / df["Close"]
    )

    # --------------------------------------------------
    # BOLLINGER BANDS
    # --------------------------------------------------

    bollinger = BollingerBands(
        close=df["Close"],
        window=20,
        window_dev=2,
    )

    upper = bollinger.bollinger_hband()
    lower = bollinger.bollinger_lband()

    df["bollinger_width"] = (
        (upper - lower) / df["Close"]
    )

    df["bollinger_position"] = (
        (df["Close"] - lower) /
        (upper - lower)
    )

    # --------------------------------------------------
    # ATR
    # --------------------------------------------------

    atr = AverageTrueRange(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=14,
    ).average_true_range()

    df["atr_ratio"] = atr / df["Close"]

    # --------------------------------------------------
    # STOCHASTIC
    # --------------------------------------------------

    stochastic = StochasticOscillator(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=14,
        smooth_window=3,
    )

    df["stoch_k"] = stochastic.stoch()
    df["stoch_d"] = stochastic.stoch_signal()

    # --------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------

    df["volatility_5d"] = (
        df["return_1d"].rolling(5).std()
    )

    df["volatility_10d"] = (
        df["return_1d"].rolling(10).std()
    )

    df["volatility_20d"] = (
        df["return_1d"].rolling(20).std()
    )

    df["volatility_60d"] = (
        df["return_1d"].rolling(60).std()
    )

    # --------------------------------------------------
    # VOLUME
    # --------------------------------------------------

    df["volume_ma_5"] = (
        df["Volume"].rolling(5).mean()
    )

    df["volume_ma_20"] = (
        df["Volume"].rolling(20).mean()
    )

    df["volume_ratio_5"] = (
        df["Volume"] / df["volume_ma_5"]
    )

    df["volume_ratio_20"] = (
        df["Volume"] / df["volume_ma_20"]
    )

    # --------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------

    df["momentum_5"] = (
        df["Close"] / df["Close"].shift(5) - 1
    )

    df["momentum_10"] = (
        df["Close"] / df["Close"].shift(10) - 1
    )

    df["momentum_20"] = (
        df["Close"] / df["Close"].shift(20) - 1
    )

    df["momentum_60"] = (
        df["Close"] / df["Close"].shift(60) - 1
    )

    # --------------------------------------------------
    # 52 WEEK POSITION
    # --------------------------------------------------

    high_252 = df["High"].rolling(252).max()
    low_252 = df["Low"].rolling(252).min()

    df["distance_52w_high"] = (
        df["Close"] / high_252 - 1
    )

    df["distance_52w_low"] = (
        df["Close"] / low_252 - 1
    )

    df["position_52w"] = (
        (df["Close"] - low_252) /
        (high_252 - low_252)
    )

    # --------------------------------------------------
    # CLEAN INF
    # --------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return df