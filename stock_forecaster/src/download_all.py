from pathlib import Path

import pandas as pd
import yfinance as yf


DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)


TICKERS = [
    "THYAO.IS",
    "ASELS.IS",
    "TUPRS.IS",
    "AKBNK.IS",
    "GARAN.IS",
    "YKBNK.IS",
    "ISCTR.IS",
    "FROTO.IS",
    "TOASO.IS",
    "KCHOL.IS",
    "SAHOL.IS",
    "SISE.IS",
    "EREGL.IS",
    "BIMAS.IS",
    "MGROS.IS",
    "TCELL.IS",
    "TTKOM.IS",
    "ENKAI.IS",
    "PETKM.IS",
    "PGSUS.IS",
    "ARCLK.IS",
    "VESTL.IS",
    "KOZAL.IS",
    "KOZAA.IS",
    "SASA.IS",
    "HEKTS.IS",
    "GUBRF.IS",
    "OYAKC.IS",
    "EKGYO.IS",
    "ULKER.IS",
]


def clean_ticker_name(ticker: str) -> str:
    return ticker.replace(".IS", "")


def download_stock(
    ticker: str,
    start: str = "2010-01-01",
) -> pd.DataFrame | None:
    print()
    print("-" * 70)
    print(f"{ticker} indiriliyor...")
    print("-" * 70)

    try:
        df = yf.download(
            ticker,
            start=start,
            auto_adjust=False,
            progress=False,
        )

        if df.empty:
            print(f"{ticker}: veri bulunamadı.")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        wanted_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
        ]

        available_columns = [
            col
            for col in wanted_columns
            if col in df.columns
        ]

        df = df[
            available_columns
        ].copy()

        df.index.name = "Date"
        df = df.sort_index()

        ticker_name = clean_ticker_name(
            ticker
        )

        output_path = (
            DATA_DIR
            / f"{ticker_name}.csv"
        )

        df.to_csv(
            output_path
        )

        print(
            f"{ticker_name}: "
            f"{len(df)} satır"
        )

        print(
            f"Kaydedildi: "
            f"{output_path}"
        )

        return df

    except Exception as exc:
        print(
            f"{ticker} indirilemedi: "
            f"{exc}"
        )

        return None


def main():
    successful = []
    failed = []

    print()
    print("=" * 70)
    print("BIST MULTI-STOCK DOWNLOADER")
    print("=" * 70)

    for ticker in TICKERS:
        result = download_stock(
            ticker=ticker,
            start="2010-01-01",
        )

        if result is None:
            failed.append(
                ticker
            )
        else:
            successful.append(
                ticker
            )

    print()
    print("=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)

    print(
        f"Başarılı: "
        f"{len(successful)}"
    )

    print(
        f"Başarısız: "
        f"{len(failed)}"
    )

    if failed:
        print()
        print(
            "Başarısız hisseler:"
        )

        for ticker in failed:
            print(
                "-",
                ticker,
            )


if __name__ == "__main__":
    main()