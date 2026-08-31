from pathlib import Path

import pandas as pd
import yfinance as yf


OUTPUT_PATH = Path("data/raw/XU100.csv")


def main():
    print("=" * 70)
    print("XU100 MARKET DATA DOWNLOADER")
    print("=" * 70)

    df = yf.download(
        "XU100.IS",
        start="2010-01-01",
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise RuntimeError(
            "XU100 verisi indirilemedi."
        )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    wanted = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]

    available = [
        col
        for col in wanted
        if col in df.columns
    ]

    df = df[available].copy()

    df.index.name = "Date"
    df = df.sort_index()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH
    )

    print()
    print("Satır:", len(df))
    print(
        "Tarih:",
        df.index.min(),
        "→",
        df.index.max(),
    )
    print("Kaydedildi:", OUTPUT_PATH)


if __name__ == "__main__":
    main()