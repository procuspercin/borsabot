from pathlib import Path

import numpy as np
import pandas as pd


GPR_PATH = Path(
    "data/raw/gpr_daily.csv"
)


def detect_column(
    df,
    candidates,
):
    for column in candidates:
        if column in df.columns:
            return column

    return None


def load_gpr():
    if not GPR_PATH.exists():
        raise FileNotFoundError(
            f"GPR dosyası bulunamadı: {GPR_PATH}"
        )

    df = pd.read_csv(
        GPR_PATH,
        low_memory=False,
    )

    date_col = detect_column(
        df,
        [
            "date",
            "Date",
            "DATE",
        ],
    )

    if date_col is None:
        raise ValueError(
            "GPR tarih kolonu bulunamadı."
        )

    gpr_col = detect_column(
        df,
        [
            "GPRD",
            "GPR",
            "gpr",
            "GPRC",
        ],
    )

    threat_col = detect_column(
        df,
        [
            "GPRD_Threats",
            "GPR_Threats",
            "Threats",
            "threats",
        ],
    )

    acts_col = detect_column(
        df,
        [
            "GPRD_Acts",
            "GPR_Acts",
            "Acts",
            "acts",
        ],
    )

    if gpr_col is None:
        raise ValueError(
            "Ana GPR kolonu bulunamadı."
        )

    rename_map = {
        date_col: "Date",
        gpr_col: "gpr_level",
    }

    if threat_col is not None:
        rename_map[
            threat_col
        ] = "gpr_threats"

    if acts_col is not None:
        rename_map[
            acts_col
        ] = "gpr_acts"

    df = df.rename(
        columns=rename_map
    )

    keep_columns = [
        "Date",
        "gpr_level",
    ]

    if "gpr_threats" in df.columns:
        keep_columns.append(
            "gpr_threats"
        )

    if "gpr_acts" in df.columns:
        keep_columns.append(
            "gpr_acts"
        )

    df = df[
        keep_columns
    ].copy()

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    numeric_columns = [
        column
        for column in df.columns
        if column != "Date"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df.dropna(
            subset=[
                "Date",
                "gpr_level",
            ]
        )
        .sort_values("Date")
        .drop_duplicates(
            subset=["Date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return df


def create_gpr_features():
    df = load_gpr()

    # -----------------------------------------
    # CHANGE FEATURES
    # -----------------------------------------

    df["gpr_change_5d"] = (
        df["gpr_level"]
        .pct_change(5)
    )

    df["gpr_change_20d"] = (
        df["gpr_level"]
        .pct_change(20)
    )

    # -----------------------------------------
    # Z-SCORE
    # -----------------------------------------

    rolling_mean = (
        df["gpr_level"]
        .rolling(60)
        .mean()
    )

    rolling_std = (
        df["gpr_level"]
        .rolling(60)
        .std()
    )

    df["gpr_zscore_60d"] = (
        (
            df["gpr_level"]
            - rolling_mean
        )
        / rolling_std
    )

    # -----------------------------------------
    # EXTREME FLAG
    # -----------------------------------------

    df["gpr_extreme"] = (
        df["gpr_zscore_60d"]
        >= 2.0
    ).astype(float)

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return df