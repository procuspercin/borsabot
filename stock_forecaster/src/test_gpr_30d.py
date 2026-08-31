from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from gpr_features import create_gpr_features


# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = Path("data/multi_stock_dataset.csv")

HORIZON = 30
TRAIN_WINDOW = 1260

TEST_TRADING_DAYS = 5
TEST_STEP_DAYS = 5

TEST_YEARS = [
    2016,
    2017,
    2018,
    2019,
    2020,
    2021,
    2022,
    2023,
    2024,
    2025,
    2026,
]

N_ESTIMATORS = 150
RANDOM_STATE = 42


MARKET_FEATURES = [
    "market_return_20d",
    "market_return_60d",
    "market_sma_50_distance",
    "market_sma_200_distance",
    "market_volatility_20d",
]


GPR_FEATURES = [
    "gpr_level",
    "gpr_change_5d",
    "gpr_change_20d",
    "gpr_zscore_60d",
    "gpr_extreme",
]


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    print("Dataset yükleniyor...")

    df = pd.read_csv(
        DATASET_PATH,
        parse_dates=["Date"],
        low_memory=False,
    )

    df = (
        df
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )

    target_up = f"target_up_{HORIZON}d"
    target_return = f"target_return_{HORIZON}d"

    for column in [
        target_up,
        target_return,
    ]:

        if column not in df.columns:
            raise ValueError(
                f"Eksik kolon: {column}"
            )

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df


# ============================================================
# MARKET REGIME
# ============================================================

def add_market_regime(df):

    df = df.copy()

    bullish = (
        (df["market_return_20d"] > 0)
        & (df["market_return_60d"] > 0)
        & (df["market_sma_50_distance"] > 0)
        & (df["market_sma_200_distance"] > 0)
    )

    bearish = (
        (df["market_return_20d"] < 0)
        & (df["market_return_60d"] < 0)
        & (df["market_sma_50_distance"] < 0)
        & (df["market_sma_200_distance"] < 0)
    )

    df["MarketRegime"] = "NEUTRAL"

    df.loc[
        bullish,
        "MarketRegime",
    ] = "BULL"

    df.loc[
        bearish,
        "MarketRegime",
    ] = "BEAR"

    return df


# ============================================================
# TARGET END DATE
# ============================================================

def add_target_end_date(df):

    df = df.copy()

    df["TargetEndDate"] = (
        df.groupby(
            "Ticker",
            sort=False,
        )["Date"]
        .shift(-HORIZON)
    )

    return df


# ============================================================
# GPR
# ============================================================

def add_gpr(df):

    print("GPR feature'ları ekleniyor...")

    gpr = create_gpr_features()

    gpr = (
        gpr
        .sort_values("Date")
        .reset_index(drop=True)
    )

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Her BIST işlem gününe o tarihte bilinen
    # en son GPR değerini bağla.
    df = pd.merge_asof(
        df,
        gpr,
        on="Date",
        direction="backward",
    )

    return df


# ============================================================
# STOCK FEATURES
# ============================================================

def get_stock_features(df):

    excluded_columns = {
        "Date",
        "Ticker",
        "MarketRegime",
        "TargetEndDate",

        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",

        "sma_5",
        "sma_10",
        "sma_20",
        "sma_50",
        "sma_100",
        "sma_200",

        "ema_12",
        "ema_26",

        "volume_ma_5",
        "volume_ma_20",

        "macd",
        "macd_signal",
        "macd_hist",
    }

    features = []

    for column in df.columns:

        if column.startswith("target_"):
            continue

        if column.startswith("market_"):
            continue

        # GPR'yi stock feature olarak
        # yanlışlıkla ikinci kez ekleme.
        if column.startswith("gpr_"):
            continue

        if column.startswith(
            "relative_strength_"
        ):
            continue

        if column in excluded_columns:
            continue

        features.append(column)

    return features


# ============================================================
# PREPARE
# ============================================================

def prepare_xy(
    df,
    numeric_features,
):

    feature_columns = (
        numeric_features
        + [
            "Ticker",
            "MarketRegime",
        ]
    )

    X = (
        df[feature_columns]
        .copy()
    )

    X[numeric_features] = (
        X[numeric_features]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    y = pd.to_numeric(
        df[f"target_up_{HORIZON}d"],
        errors="coerce",
    )

    valid = (
        X[numeric_features]
        .notna()
        .all(axis=1)
        & y.notna()
    )

    X = (
        X.loc[valid]
        .reset_index(drop=True)
    )

    y = (
        y.loc[valid]
        .astype(int)
        .reset_index(drop=True)
    )

    metadata = (
        df.loc[valid]
        .reset_index(drop=True)
    )

    return X, y, metadata


# ============================================================
# MODEL
# ============================================================

def create_model(
    numeric_features,
):

    categorical_features = [
        "Ticker",
        "MarketRegime",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                numeric_features,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_features,
            ),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=10,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


# ============================================================
# AUC
# ============================================================

def safe_auc(
    y_true,
    probability,
):

    if len(
        np.unique(y_true)
    ) < 2:

        return np.nan

    return roc_auc_score(
        y_true,
        probability,
    )


# ============================================================
# WALK FORWARD
# ============================================================

def run_test(
    df,
    numeric_features,
    model_name,
):

    prediction_frames = []

    all_dates = (
        df["Date"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    test_dates_all = (
        all_dates[
            all_dates.dt.year.isin(
                TEST_YEARS
            )
        ]
        .reset_index(drop=True)
    )

    block_count = 0

    for test_index in range(
        0,
        len(test_dates_all),
        TEST_STEP_DAYS,
    ):

        test_dates = (
            test_dates_all.iloc[
                test_index:
                test_index
                + TEST_TRADING_DAYS
            ]
        )

        if len(test_dates) < TEST_TRADING_DAYS:
            break

        # Yıl sınırını aşan blok olmasın.
        if (
            test_dates.iloc[0].year
            != test_dates.iloc[-1].year
        ):
            continue

        test_start = test_dates.iloc[0]
        test_end = test_dates.iloc[-1]

        eligible_train = (
            df[
                (
                    df["TargetEndDate"]
                    < test_start
                )
                &
                (
                    df[
                        f"target_up_{HORIZON}d"
                    ].notna()
                )
            ]
            .copy()
        )

        if eligible_train.empty:
            continue

        eligible_dates = (
            eligible_train["Date"]
            .drop_duplicates()
            .sort_values()
        )

        if (
            len(eligible_dates)
            < TRAIN_WINDOW
        ):
            continue

        train_dates = (
            eligible_dates.iloc[
                -TRAIN_WINDOW:
            ]
        )

        train_df = (
            eligible_train[
                eligible_train["Date"]
                .isin(train_dates)
            ]
            .copy()
        )

        test_df = (
            df[
                df["Date"]
                .isin(test_dates)
            ]
            .copy()
        )

        test_df = (
            test_df[
                test_df[
                    f"target_up_{HORIZON}d"
                ].notna()
            ]
            .copy()
        )

        if test_df.empty:
            continue

        (
            X_train,
            y_train,
            _,
        ) = prepare_xy(
            train_df,
            numeric_features,
        )

        (
            X_test,
            y_test,
            test_meta,
        ) = prepare_xy(
            test_df,
            numeric_features,
        )

        if len(X_train) < 100:
            continue

        if len(X_test) == 0:
            continue

        if y_train.nunique() < 2:
            continue

        block_count += 1

        print(
            f"\r{model_name:<6} "
            f"| Block {block_count:>3} "
            f"| {test_start.date()} "
            f"→ {test_end.date()}",
            end="",
            flush=True,
        )

        model = create_model(
            numeric_features
        )

        model.fit(
            X_train,
            y_train,
        )

        probability = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        prediction = (
            probability >= 0.50
        ).astype(int)

        frame = pd.DataFrame(
            {
                "Model": model_name,
                "Date":
                    test_meta[
                        "Date"
                    ].values,
                "Ticker":
                    test_meta[
                        "Ticker"
                    ].values,
                "actual":
                    y_test.values,
                "prediction":
                    prediction,
                "probability":
                    probability,
            }
        )

        frame["Year"] = (
            pd.to_datetime(
                frame["Date"]
            ).dt.year
        )

        prediction_frames.append(
            frame
        )

    print()

    if not prediction_frames:
        return None

    return pd.concat(
        prediction_frames,
        ignore_index=True,
    )


# ============================================================
# SUMMARY
# ============================================================

def summarize(
    predictions,
):

    rows = []

    for (
        model_name,
        year
    ), group in predictions.groupby(
        ["Model", "Year"]
    ):

        rows.append(
            {
                "Model":
                    model_name,

                "Year":
                    year,

                "Samples":
                    len(group),

                "Accuracy":
                    accuracy_score(
                        group["actual"],
                        group[
                            "prediction"
                        ],
                    ),

                "BalancedAccuracy":
                    balanced_accuracy_score(
                        group["actual"],
                        group[
                            "prediction"
                        ],
                    ),

                "AUC":
                    safe_auc(
                        group["actual"]
                        .values,
                        group[
                            "probability"
                        ].values,
                    ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 100)
    print(
        "30D GPR A/B TEST — "
        "1260 DAY TRAIN WINDOW"
    )
    print("=" * 100)

    print()
    print(
        "Test years:",
        TEST_YEARS,
    )

    print(
        "Train window:",
        TRAIN_WINDOW,
    )

    df = load_dataset()

    df = add_market_regime(df)
    df = add_target_end_date(df)

    # Stock feature listesini GPR eklenmeden
    # mevcut dataset yapısından alıyoruz.
    stock_features = (
        get_stock_features(df)
    )

    base_features = (
        stock_features
        + MARKET_FEATURES
    )

    print()
    print(
        "BASE numeric:",
        len(base_features),
    )

    # --------------------------------------------------------
    # ADD GPR
    # --------------------------------------------------------

    df = add_gpr(df)

    gpr_features = (
        base_features
        + GPR_FEATURES
    )

    print(
        "GPR numeric :",
        len(gpr_features),
    )

    print()
    print(
        "GPR tarih coverage:"
    )

    print(
        df.loc[
            df["gpr_level"].notna(),
            "Date",
        ].min(),
        "→",
        df.loc[
            df["gpr_level"].notna(),
            "Date",
        ].max(),
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Aynı satırları BASE ve GPR'de kullan.
    # --------------------------------------------------------

    common_valid = (
        df[
            gpr_features
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .notna()
        .all(axis=1)
    )

    common_df = (
        df.loc[
            common_valid
        ]
        .copy()
    )

    print()
    print(
        "Ortak valid rows:",
        f"{len(common_df):,}",
    )

    # --------------------------------------------------------
    # BASE
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("BASE MODEL")
    print("=" * 100)

    base_predictions = run_test(
        common_df,
        base_features,
        "BASE",
    )

    # --------------------------------------------------------
    # GPR
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("GPR MODEL")
    print("=" * 100)

    gpr_predictions = run_test(
        common_df,
        gpr_features,
        "GPR",
    )

    if (
        base_predictions is None
        or gpr_predictions is None
    ):
        raise RuntimeError(
            "Prediction üretilemedi."
        )

    predictions = pd.concat(
        [
            base_predictions,
            gpr_predictions,
        ],
        ignore_index=True,
    )

    summary = summarize(
        predictions
    )

    print()
    print("=" * 100)
    print("SONUÇ")
    print("=" * 100)
    print()

    display = summary.copy()

    display["Accuracy"] = (
        display["Accuracy"]
        * 100
    )

    display[
        "BalancedAccuracy"
    ] = (
        display[
            "BalancedAccuracy"
        ]
        * 100
    )

    print(
        display.to_string(
            index=False,
            formatters={
                "Accuracy":
                    lambda x:
                    f"%{x:.2f}",

                "BalancedAccuracy":
                    lambda x:
                    f"%{x:.2f}",

                "AUC":
                    lambda x:
                    f"{x:.4f}",
            },
        )
    )

    # --------------------------------------------------------
    # DELTA
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("GPR ETKİSİ")
    print("=" * 100)
    print()

    for year in TEST_YEARS:

        year_rows = (
            summary[
                summary["Year"]
                == year
            ]
            .set_index("Model")
        )

        if (
            "BASE"
            not in year_rows.index
            or "GPR"
            not in year_rows.index
        ):
            continue

        base_auc = (
            year_rows.loc[
                "BASE",
                "AUC",
            ]
        )

        gpr_auc = (
            year_rows.loc[
                "GPR",
                "AUC",
            ]
        )

        delta = (
            gpr_auc
            - base_auc
        )

        print(
            f"{year}: "
            f"{base_auc:.4f} "
            f"→ {gpr_auc:.4f} "
            f"({delta:+.4f})"
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    summary.to_csv(
        "data/gpr_ab_test_30d_summary.csv",
        index=False,
    )

    predictions.to_csv(
        "data/gpr_ab_test_30d_predictions.csv",
        index=False,
    )

    print()
    print("Kaydedildi:")
    print(
        "data/gpr_ab_test_30d_summary.csv"
    )
    print(
        "data/gpr_ab_test_30d_predictions.csv"
    )


if __name__ == "__main__":
    main()