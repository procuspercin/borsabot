from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PREDICTIONS_PATH = Path(
    "models/direction_walk_forward_predictions.csv"
)

DATASET_PATH = Path(
    "data/multi_stock_dataset.csv"
)

OUTPUT_PATH = Path(
    "models/expected_return_calibration.csv"
)

HORIZONS = [
    10,
    30,
    60,
    120,
    180,
]

BIN_WIDTH = 0.05

MIN_SAMPLES = 100


# ============================================================
# HELPERS
# ============================================================

def probability_bin(probability):
    probability = float(
        np.clip(
            probability,
            0.0,
            0.999999,
        )
    )

    lower = (
        np.floor(
            probability / BIN_WIDTH
        )
        * BIN_WIDTH
    )

    return round(
        float(lower),
        4,
    )


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions():

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction dosyası bulunamadı: "
            f"{PREDICTIONS_PATH}"
        )

    df = pd.read_csv(
        PREDICTIONS_PATH,
        low_memory=False,
    )

    required = [
        "Date",
        "Ticker",
        "Horizon",
        "actual",
        "P_UP",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Prediction dosyasında eksik kolonlar: "
            f"{missing}"
        )

    df = df.rename(
        columns={
            "P_UP": "probability",
        }
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df["Ticker"] = (
        df["Ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["Horizon"] = pd.to_numeric(
        df["Horizon"],
        errors="coerce",
    )

    df["probability"] = pd.to_numeric(
        df["probability"],
        errors="coerce",
    )

    df["actual"] = pd.to_numeric(
        df["actual"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "Date",
            "Ticker",
            "Horizon",
            "probability",
            "actual",
        ]
    )

    df["Horizon"] = (
        df["Horizon"]
        .astype(int)
    )

    df["actual"] = (
        df["actual"]
        .astype(int)
    )

    return df


# ============================================================
# LOAD REALIZED RETURNS
# ============================================================

def load_returns():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset bulunamadı: "
            f"{DATASET_PATH}"
        )

    return_columns = [
        f"target_return_{h}d"
        for h in HORIZONS
    ]

    usecols = [
        "Date",
        "Ticker",
        *return_columns,
    ]

    print(
        "Gerçekleşen getiriler dataset'ten okunuyor..."
    )

    df = pd.read_csv(
        DATASET_PATH,
        usecols=usecols,
        low_memory=False,
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df["Ticker"] = (
        df["Ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    for col in return_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    return df


# ============================================================
# MERGE PREDICTIONS + RETURNS
# ============================================================

def attach_realized_returns(
    predictions,
    returns_df,
):

    print(
        "Prediction'lar gerçekleşen getirilerle "
        "eşleştiriliyor..."
    )

    merged = predictions.merge(
        returns_df,
        on=[
            "Date",
            "Ticker",
        ],
        how="left",
        validate="many_to_one",
    )

    def pick_return(row):
        horizon = int(
            row["Horizon"]
        )

        return row[
            f"target_return_{horizon}d"
        ]

    merged["ActualReturn"] = (
        merged.apply(
            pick_return,
            axis=1,
        )
    )

    before = len(
        merged
    )

    merged = merged.dropna(
        subset=[
            "ActualReturn",
        ]
    )

    after = len(
        merged
    )

    print(
        f"Eşleşen prediction: "
        f"{after:,} / {before:,}"
    )

    if after == 0:
        raise RuntimeError(
            "Hiçbir prediction gerçek getiriyle "
            "eşleştirilemedi."
        )

    calculated_actual = (
        merged["ActualReturn"] > 0
    ).astype(int)

    agreement = (
        calculated_actual
        == merged["actual"]
    ).mean()

    print(
        f"Direction/return kontrolü: "
        f"%{agreement * 100:.2f} eşleşiyor"
    )

    if agreement < 0.99:
        print()
        print(
            "UYARI: Direction target ile return işareti "
            "tam eşleşmiyor."
        )

    return merged


# ============================================================
# BUILD CALIBRATION
# ============================================================

def build_calibration(df):

    rows = []

    for horizon in HORIZONS:

        horizon_df = (
            df[
                df["Horizon"]
                == horizon
            ]
            .copy()
        )

        if horizon_df.empty:
            continue

        horizon_df["BinLower"] = (
            horizon_df["probability"]
            .apply(
                probability_bin
            )
        )

        print()
        print("=" * 110)
        print(
            f"{horizon} İŞLEM GÜNÜ"
        )
        print("=" * 110)

        print(
            f"{'P(UP)':<14}"
            f"{'N':>8}"
            f"{'Ort P':>12}"
            f"{'Gerçek UP':>14}"
            f"{'Ort Değişim':>16}"
            f"{'Medyan':>12}"
            f"{'Orta %50':>24}"
        )

        print(
            "-" * 110
        )

        grouped = (
            horizon_df
            .groupby(
                "BinLower",
                observed=True,
            )
        )

        for bin_lower, group in grouped:

            bin_upper = min(
                bin_lower + BIN_WIDTH,
                1.0,
            )

            samples = len(
                group
            )

            mean_probability = (
                group["probability"]
                .mean()
            )

            actual_up_rate = (
                (
                    group["ActualReturn"]
                    > 0
                )
                .mean()
            )

            mean_return = (
                group["ActualReturn"]
                .mean()
            )

            median_return = (
                group["ActualReturn"]
                .median()
            )

            p25_return = (
                group["ActualReturn"]
                .quantile(0.25)
            )

            p75_return = (
                group["ActualReturn"]
                .quantile(0.75)
            )

            std_return = (
                group["ActualReturn"]
                .std()
            )

            reliable = (
                samples >= MIN_SAMPLES
            )

            rows.append(
                {
                    "Horizon": horizon,
                    "BinLower": bin_lower,
                    "BinUpper": bin_upper,
                    "Samples": samples,
                    "MeanProbability": mean_probability,
                    "ActualUpRate": actual_up_rate,
                    "MeanReturn": mean_return,
                    "MedianReturn": median_return,
                    "P25Return": p25_return,
                    "P75Return": p75_return,
                    "StdReturn": std_return,
                    "Reliable": reliable,
                }
            )

            status = (
                ""
                if reliable
                else "  [AZ ÖRNEK]"
            )

            interval_text = (
                f"%{p25_return * 100:+.1f} → "
                f"%{p75_return * 100:+.1f}"
            )

            print(
                f"{bin_lower:.2f}–{bin_upper:.2f}"
                f"{samples:>8}"
                f"%{mean_probability * 100:>10.2f}"
                f"%{actual_up_rate * 100:>12.2f}"
                f"%{mean_return * 100:>14.2f}"
                f"%{median_return * 100:>10.2f}"
                f"{interval_text:>24}"
                f"{status}"
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(calibration):

    print()
    print()
    print("=" * 110)
    print(
        "KALİBRASYON ÖZETİ"
    )
    print("=" * 110)

    for horizon in HORIZONS:

        data = calibration[
            (
                calibration["Horizon"]
                == horizon
            )
            &
            (
                calibration["Reliable"]
                == True
            )
        ].copy()

        if data.empty:
            print()
            print(
                f"{horizon}d: "
                f"yeterli örnek yok."
            )
            continue

        print()
        print(
            f"{horizon} İŞLEM GÜNÜ"
        )

        print(
            "-" * 50
        )

        strongest = (
            data.sort_values(
                "ActualUpRate",
                ascending=False,
            )
            .iloc[0]
        )

        print(
            "Geçmişte en yüksek gerçek yükseliş oranı:"
        )

        print(
            f"  P(UP): "
            f"{strongest['BinLower']:.2f}"
            f"–"
            f"{strongest['BinUpper']:.2f}"
        )

        print(
            f"  Örnek: "
            f"{int(strongest['Samples']):,}"
        )

        print(
            f"  Gerçek yükseliş: "
            f"%{strongest['ActualUpRate'] * 100:.2f}"
        )

        print(
            f"  Ortalama değişim: "
            f"%{strongest['MeanReturn'] * 100:+.2f}"
        )

        print(
            f"  Medyan değişim: "
            f"%{strongest['MedianReturn'] * 100:+.2f}"
        )

        print(
            f"  Orta %50 sonuç: "
            f"%{strongest['P25Return'] * 100:+.2f}"
            f" → "
            f"%{strongest['P75Return'] * 100:+.2f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 110)
    print(
        "STOCK FORECASTER — EXPECTED RETURN CALIBRATION"
    )
    print("=" * 110)

    print()
    print(
        "OOS prediction:",
        PREDICTIONS_PATH,
    )

    print(
        "Dataset       :",
        DATASET_PATH,
    )

    predictions = (
        load_predictions()
    )

    print()
    print(
        f"OOS prediction sayısı: "
        f"{len(predictions):,}"
    )

    print(
        "Horizons:",
        sorted(
            predictions[
                "Horizon"
            ]
            .unique()
            .tolist()
        ),
    )

    returns_df = (
        load_returns()
    )

    merged = (
        attach_realized_returns(
            predictions,
            returns_df,
        )
    )

    calibration = (
        build_calibration(
            merged
        )
    )

    if calibration.empty:
        raise RuntimeError(
            "Calibration tablosu üretilemedi."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    calibration.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print_summary(
        calibration
    )

    print()
    print("=" * 110)
    print(
        "TAMAMLANDI"
    )
    print("=" * 110)

    print(
        "Calibration kaydedildi:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()