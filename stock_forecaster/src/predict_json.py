"""
BorsaBot Finans arayüzü için JSON köprüsü.

predict.py içindeki model/feature mantığını aynen kullanır, sonucu stdout'a
tek satır JSON olarak yazar. Streamlit tarafı bu betiği stock_forecaster'ın
kendi .venv'i ile alt süreç olarak çağırır (kütüphane sürümleri farklı).

Kullanım:
    .venv/bin/python src/predict_json.py THYAO
"""

import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_DIR)                    # predict.py göreli yollar kullanıyor
sys.path.insert(0, str(PROJECT_DIR / "src"))

import pandas as pd  # noqa: E402

from predict import (  # noqa: E402
    HORIZONS,
    MODEL_DIR,
    build_forecast_data,
    build_latest_row,
    load_calibration,
    load_stock,
)

# Arayüzde gösterilecek ham feature'lar
FEATURE_KEYS = [
    "rsi_14", "macd", "macd_signal", "stoch_k", "stoch_d",
    "return_20d", "return_60d", "close_sma_50_ratio", "close_sma_200_ratio",
    "volume_ratio_20", "volatility_20d", "position_52w",
    "market_return_20d", "market_return_60d",
    "market_sma_50_distance", "market_sma_200_distance",
]


def model_evaluation() -> list:
    """Modellerin geçmiş out-of-sample başarı ölçümleri."""
    path = MODEL_DIR / "direction_model_evaluation.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return [
        {
            "horizon": int(r["Horizon"]),
            "auc": float(r["Overall_OOS_AUC"]),
            "accuracy": float(r["Overall_Accuracy"]),
            "worst_year_auc": float(r["Worst_Year_AUC"]),
            "best_year_auc": float(r["Best_Year_AUC"]),
            "years_above_050": int(r["Years_Above_050"]),
        }
        for _, r in df.iterrows()
    ]


def price_history(ticker: str, days: int = 180) -> list:
    """Grafik için son kapanışlar."""
    df = load_stock(ticker).tail(days)
    return [
        {"date": idx.strftime("%Y-%m-%d"), "close": float(row["Close"])}
        for idx, row in df.iterrows()
    ]


def available_tickers() -> list:
    raw_dir = PROJECT_DIR / "data" / "raw"
    skip = {"XU100", "gpr_daily"}
    return sorted(
        p.stem for p in raw_dir.glob("*.csv")
        if p.stem not in skip and (MODEL_DIR / "direction_30d.joblib").exists()
    )


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Sembol verilmedi."}))
        sys.exit(1)

    if sys.argv[1] == "--list":
        print(json.dumps({"tickers": available_tickers()}))
        return

    ticker = sys.argv[1].upper().replace(".IS", "")

    try:
        latest_row, latest_date, latest_close = build_latest_row(ticker)
        calibration = load_calibration()
        forecasts = build_forecast_data(latest_row, calibration, latest_close)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)

    row = latest_row.iloc[0]
    features = {}
    for key in FEATURE_KEYS:
        if key in row.index:
            try:
                features[key] = float(row[key])
            except (TypeError, ValueError):
                pass

    result = {
        "ticker": ticker,
        "date": pd.Timestamp(latest_date).strftime("%Y-%m-%d"),
        "close": float(latest_close),
        "market_regime": str(row.get("MarketRegime", "")),
        "horizons": HORIZONS,
        "forecasts": forecasts,
        "features": features,
        "evaluation": model_evaluation(),
        "history": price_history(ticker),
    }
    print(json.dumps(result, default=float))


if __name__ == "__main__":
    main()
