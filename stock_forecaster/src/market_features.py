import numpy as np
import pandas as pd

from ta.momentum import RSIIndicator


def create_market_features(
    market_df: pd.DataFrame,
) -> pd.DataFrame:
    df = market_df.copy()

    # -----------------------------------------
    # MARKET RETURNS
    # -----------------------------------------

    for period in [
        1,
        5,
        10,
        20,
        60,
    ]:
        df[f"market_return_{period}d"] = (
            df["Close"].pct_change(period)
        )

    # -----------------------------------------
    # RSI
    # -----------------------------------------

    df["market_rsi_14"] = (
        RSIIndicator(
            close=df["Close"],
            window=14,
        ).rsi()
    )

    # -----------------------------------------
    # VOLATILITY
    # -----------------------------------------

    daily_return = (
        df["Close"].pct_change()
    )

    df["market_volatility_10d"] = (
        daily_return
        .rolling(10)
        .std()
    )

    df["market_volatility_20d"] = (
        daily_return
        .rolling(20)
        .std()
    )

    df["market_volatility_60d"] = (
        daily_return
        .rolling(60)
        .std()
    )

    # -----------------------------------------
    # SMA DISTANCE
    # -----------------------------------------

    for period in [
        20,
        50,
        200,
    ]:
        sma = (
            df["Close"]
            .rolling(period)
            .mean()
        )

        df[
            f"market_sma_{period}_distance"
        ] = (
            df["Close"] / sma - 1
        )

    # -----------------------------------------
    # MARKET TREND FLAGS
    # -----------------------------------------

    sma20 = (
        df["Close"]
        .rolling(20)
        .mean()
    )

    sma50 = (
        df["Close"]
        .rolling(50)
        .mean()
    )

    sma200 = (
        df["Close"]
        .rolling(200)
        .mean()
    )

    df["market_above_sma20"] = (
        df["Close"] > sma20
    ).astype(float)

    df["market_above_sma50"] = (
        df["Close"] > sma50
    ).astype(float)

    df["market_above_sma200"] = (
        df["Close"] > sma200
    ).astype(float)

    # -----------------------------------------
    # 52 WEEK POSITION
    # -----------------------------------------

    high_252 = (
        df["High"]
        .rolling(252)
        .max()
    )

    low_252 = (
        df["Low"]
        .rolling(252)
        .min()
    )

    df["market_52w_position"] = (
        (df["Close"] - low_252)
        / (high_252 - low_252)
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Sadece market feature'larını döndür.
    columns = [
        col
        for col in df.columns
        if col.startswith("market_")
    ]

    return df[columns]