from pathlib import Path

import numpy as np
import pandas as pd

from features import create_features
from market_features import create_market_features


# ============================================================
# CONFIG
# ============================================================

RAW_DATA_DIR = Path("data/raw")

MARKET_PATH = Path(
    "data/raw/XU100.csv"
)

OUTPUT_PATH = Path(
    "data/multi_stock_dataset.csv"
)

HORIZONS = [
    10,
    30,
    60,
    120,
    180,
]

UP_THRESHOLD = 0.02
DOWN_THRESHOLD = -0.02


# ============================================================
# TARGETS
# ============================================================

def create_targets(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    for horizon in HORIZONS:

        future_close = (
            df["Close"]
            .shift(-horizon)
        )

        target_return = (
            future_close
            / df["Close"]
            - 1
        )

        # ----------------------------------------------------
        # REGRESSION TARGET
        # ----------------------------------------------------

        df[
            f"target_return_{horizon}d"
        ] = target_return

        # ----------------------------------------------------
        # BINARY TARGET
        #
        # 1 = UP
        # 0 = DOWN
        # NaN = geleceği henüz bilinmiyor
        # ----------------------------------------------------

        df[
            f"target_up_{horizon}d"
        ] = (
            target_return
            .gt(0)
            .where(
                target_return.notna()
            )
        )

        # ----------------------------------------------------
        # 3-CLASS TARGET
        #
        #  1 = UP
        #  0 = NEUTRAL
        # -1 = DOWN
        # ----------------------------------------------------

        class_target = pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

        valid_mask = (
            target_return.notna()
        )

        # UP
        class_target.loc[
            valid_mask
            & (
                target_return
                > UP_THRESHOLD
            )
        ] = 1

        # DOWN
        class_target.loc[
            valid_mask
            & (
                target_return
                < DOWN_THRESHOLD
            )
        ] = -1

        # NEUTRAL
        class_target.loc[
            valid_mask
            & (
                target_return
                >= DOWN_THRESHOLD
            )
            & (
                target_return
                <= UP_THRESHOLD
            )
        ] = 0

        df[
            f"target_class_{horizon}d"
        ] = class_target

    return df


# ============================================================
# SINGLE STOCK BUILDER
# ============================================================

def build_single_stock(
    csv_path: Path,
) -> pd.DataFrame:

    ticker = (
        csv_path
        .stem
        .upper()
    )

    print()
    print(
        f"{ticker} işleniyor..."
    )

    df = pd.read_csv(
        csv_path,
        parse_dates=["Date"],
        index_col="Date",
    )

    df = df.sort_index()

    # --------------------------------------------------------
    # STOCK FEATURES
    # --------------------------------------------------------

    df = create_features(
        df
    )

    # --------------------------------------------------------
    # TARGETS
    # --------------------------------------------------------

    df = create_targets(
        df
    )

    # --------------------------------------------------------
    # TICKER
    # --------------------------------------------------------

    df["Ticker"] = ticker

    return df


# ============================================================
# MARKET DATA
# ============================================================

def load_market_features():

    if not MARKET_PATH.exists():

        raise FileNotFoundError(
            "XU100 verisi bulunamadı.\n"
            "Önce şu komutu çalıştır:\n"
            "python src/download_market.py"
        )

    print()
    print("=" * 70)
    print("MARKET FEATURES HAZIRLANIYOR")
    print("=" * 70)

    market_df = pd.read_csv(
        MARKET_PATH,
        parse_dates=["Date"],
        index_col="Date",
    )

    market_df = (
        market_df
        .sort_index()
    )

    market_features = (
        create_market_features(
            market_df
        )
    )

    market_features = (
        market_features
        .reset_index()
    )

    print()
    print(
        "Market feature sayısı:",
        len(
            market_features.columns
        ) - 1,
    )

    print(
        "XU100 tarih aralığı:",
        market_features[
            "Date"
        ].min(),
        "→",
        market_features[
            "Date"
        ].max(),
    )

    return market_features


# ============================================================
# ADD MARKET FEATURES
# ============================================================

def add_market_features(
    stock_df: pd.DataFrame,
    market_features: pd.DataFrame,
) -> pd.DataFrame:

    # Date index -> column
    stock_df = (
        stock_df
        .reset_index()
    )

    # --------------------------------------------------------
    # MERGE XU100
    # --------------------------------------------------------

    stock_df = stock_df.merge(
        market_features,
        on="Date",
        how="left",
    )

    # --------------------------------------------------------
    # RELATIVE STRENGTH
    #
    # Stock return - XU100 return
    # --------------------------------------------------------

    periods = [
        5,
        10,
        20,
        60,
    ]

    for period in periods:

        stock_return_col = (
            f"return_{period}d"
        )

        market_return_col = (
            f"market_return_{period}d"
        )

        relative_col = (
            f"relative_strength_{period}d"
        )

        if (
            stock_return_col
            in stock_df.columns
            and market_return_col
            in stock_df.columns
        ):

            stock_df[
                relative_col
            ] = (
                stock_df[
                    stock_return_col
                ]
                - stock_df[
                    market_return_col
                ]
            )

    stock_df = (
        stock_df
        .set_index("Date")
    )

    return stock_df


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("MULTI-STOCK + MARKET DATASET BUILDER")
    print("=" * 70)

    # --------------------------------------------------------
    # MARKET
    # --------------------------------------------------------

    market_features = (
        load_market_features()
    )

    # --------------------------------------------------------
    # STOCK CSV FILES
    # --------------------------------------------------------

    csv_files = sorted(
        [
            path
            for path
            in RAW_DATA_DIR.glob(
                "*.csv"
            )
            if path.name.upper()
            != "XU100.CSV"
        ]
    )

    if not csv_files:

        raise FileNotFoundError(
            "data/raw klasöründe "
            "hisse CSV dosyası bulunamadı."
        )

    print()
    print(
        "Bulunan hisse dosyası:",
        len(csv_files),
    )

    datasets = []

    successful = []
    failed = []

    # --------------------------------------------------------
    # BUILD EACH STOCK
    # --------------------------------------------------------

    for csv_path in csv_files:

        ticker = (
            csv_path
            .stem
            .upper()
        )

        try:

            stock_df = (
                build_single_stock(
                    csv_path
                )
            )

            stock_df = (
                add_market_features(
                    stock_df,
                    market_features,
                )
            )

            datasets.append(
                stock_df
            )

            successful.append(
                ticker
            )

            print(
                f"  -> "
                f"{len(stock_df)} satır"
            )

        except Exception as exc:

            failed.append(
                ticker
            )

            print(
                f"  -> HATA: {exc}"
            )

    # --------------------------------------------------------
    # CHECK
    # --------------------------------------------------------

    if not datasets:

        raise RuntimeError(
            "Hiçbir hisse dataset'i "
            "oluşturulamadı."
        )

    # --------------------------------------------------------
    # CONCAT
    # --------------------------------------------------------

    combined = pd.concat(
        datasets,
        axis=0,
    )

    combined = (
        combined
        .reset_index()
        .sort_values(
            [
                "Date",
                "Ticker",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("DATASET HAZIR")
    print("=" * 70)

    print()
    print(
        "Toplam satır:",
        len(combined),
    )

    print(
        "Hisse sayısı:",
        combined[
            "Ticker"
        ].nunique(),
    )

    print(
        "Toplam kolon:",
        len(
            combined.columns
        ),
    )

    print()

    print(
        "Tarih aralığı:"
    )

    print(
        combined[
            "Date"
        ].min(),
        "→",
        combined[
            "Date"
        ].max(),
    )

    # --------------------------------------------------------
    # FEATURE COUNTS
    # --------------------------------------------------------

    market_columns = [
        col
        for col
        in combined.columns
        if col.startswith(
            "market_"
        )
    ]

    relative_columns = [
        col
        for col
        in combined.columns
        if col.startswith(
            "relative_strength_"
        )
    ]

    print()
    print(
        "Market feature sayısı:",
        len(
            market_columns
        ),
    )

    print(
        "Relative strength feature sayısı:",
        len(
            relative_columns
        ),
    )

    # --------------------------------------------------------
    # MARKET FEATURES
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MARKET FEATURES")
    print("=" * 70)

    for col in market_columns:

        print(
            "-",
            col,
        )

    # --------------------------------------------------------
    # RELATIVE FEATURES
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RELATIVE STRENGTH FEATURES")
    print("=" * 70)

    for col in relative_columns:

        print(
            "-",
            col,
        )

    # --------------------------------------------------------
    # STOCK COUNTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("HİSSE BAŞINA SATIR")
    print("=" * 70)

    counts = (
        combined
        .groupby(
            "Ticker"
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    print(
        counts
    )

    # --------------------------------------------------------
    # SUCCESS / FAIL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BUILD SUMMARY")
    print("=" * 70)

    print(
        "Başarılı:",
        len(
            successful
        ),
    )

    print(
        "Başarısız:",
        len(
            failed
        ),
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

    print()
    print(
        "Kaydedildi:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()