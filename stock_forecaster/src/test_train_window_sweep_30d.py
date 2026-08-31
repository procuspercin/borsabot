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


# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = Path(
    "data/multi_stock_dataset.csv"
)

SUMMARY_PATH = Path(
    "data/train_window_sweep_30d_summary.csv"
)

YEARLY_PATH = Path(
    "data/train_window_sweep_30d_yearly.csv"
)

PREDICTIONS_PATH = Path(
    "data/train_window_sweep_30d_predictions.csv"
)

HORIZON = 30

TEST_TRADING_DAYS = 5
TEST_STEP_DAYS = 5

WARMUP_TRADING_DAYS = 252

TRAIN_WINDOWS = [756, 1008]

N_ESTIMATORS = 150
RANDOM_STATE = 42


# ============================================================
# MARKET FEATURES
# ============================================================

MARKET_FEATURES = [
    "market_return_20d",
    "market_return_60d",
    "market_sma_50_distance",
    "market_sma_200_distance",
    "market_volatility_20d",
]


# ============================================================
# LOAD
# ============================================================

def load_dataset():

    print()
    print("Dataset yükleniyor...")

    df = pd.read_csv(
        DATASET_PATH,
        parse_dates=["Date"],
        low_memory=False,
    )

    df = (
        df
        .sort_values(
            [
                "Date",
                "Ticker",
            ]
        )
        .reset_index(drop=True)
    )

    target_up = (
        f"target_up_{HORIZON}d"
    )

    target_return = (
        f"target_return_{HORIZON}d"
    )

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
# FEATURES
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

        if column.startswith(
            "relative_strength_"
        ):
            continue

        if column in excluded_columns:
            continue

        features.append(
            column
        )

    return features


def build_valid_mask(
    df,
    numeric_features,
):

    X = (
        df[
            numeric_features
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    return (
        X.notna()
        .all(axis=1)
    )


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
        df[
            feature_columns
        ]
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
        df[
            f"target_up_{HORIZON}d"
        ],
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
        df.loc[
            valid
        ]
        .reset_index(drop=True)
    )

    return (
        X,
        y,
        metadata,
    )


# ============================================================
# METRIC
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
# SINGLE TRAIN WINDOW
# ============================================================

def run_window(
    valid_df,
    valid_dates,
    numeric_features,
    train_window,
):

    prediction_frames = []

    block_count = 0

    for test_index in range(
        0,
        len(valid_dates),
        TEST_STEP_DAYS,
    ):

        test_dates = (
            valid_dates.iloc[
                test_index:
                test_index
                + TEST_TRADING_DAYS
            ]
        )

        if len(
            test_dates
        ) < TEST_TRADING_DAYS:

            break

        test_start = (
            test_dates.iloc[0]
        )

        test_end = (
            test_dates.iloc[-1]
        )

        # ----------------------------------------------------
        # ONLY LABELS KNOWN BEFORE TEST
        # ----------------------------------------------------

        eligible_train = (
            valid_df[
                (
                    valid_df[
                        "TargetEndDate"
                    ]
                    < test_start
                )
                &
                (
                    valid_df[
                        f"target_up_{HORIZON}d"
                    ]
                    .notna()
                )
            ]
            .copy()
        )

        if eligible_train.empty:
            continue

        eligible_dates = (
            eligible_train[
                "Date"
            ]
            .drop_duplicates()
            .sort_values()
        )

        if len(
            eligible_dates
        ) < train_window:

            continue

        train_dates = (
            eligible_dates
            .iloc[
                -train_window:
            ]
        )

        train_df = (
            eligible_train[
                eligible_train[
                    "Date"
                ].isin(
                    train_dates
                )
            ]
            .copy()
        )

        test_df = (
            valid_df[
                valid_df[
                    "Date"
                ].isin(
                    test_dates
                )
            ]
            .copy()
        )

        test_df = (
            test_df[
                test_df[
                    f"target_up_{HORIZON}d"
                ]
                .notna()
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

        if len(
            X_train
        ) < 100:

            continue

        if len(
            X_test
        ) == 0:

            continue

        if y_train.nunique() < 2:
            continue

        block_count += 1

        print(
            f"\rWindow {train_window:>4} "
            f"| Block {block_count:>4} "
            f"| Test {test_start.date()} "
            f"→ {test_end.date()}",
            end="",
            flush=True,
        )

        model = create_model(
            numeric_features,
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
                "TrainWindow":
                    train_window,

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

                "actual_return_30d":
                    test_meta[
                        f"target_return_{HORIZON}d"
                    ].values,

                "TestStart":
                    test_start,

                "TestEnd":
                    test_end,
            }
        )

        frame["Year"] = (
            pd.to_datetime(
                frame["Date"]
            )
            .dt.year
        )

        prediction_frames.append(
            frame
        )

    print()

    if not prediction_frames:
        return None

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    return predictions


# ============================================================
# SUMMARY
# ============================================================

def summarize_window(
    predictions,
    train_window,
):

    accuracy = accuracy_score(
        predictions["actual"],
        predictions["prediction"],
    )

    balanced = balanced_accuracy_score(
        predictions["actual"],
        predictions["prediction"],
    )

    auc = safe_auc(
        predictions["actual"].values,
        predictions[
            "probability"
        ].values,
    )

    yearly_aucs = []

    yearly_rows = []

    for year, group in predictions.groupby(
        "Year"
    ):

        year_auc = safe_auc(
            group["actual"].values,
            group["probability"].values,
        )

        year_acc = accuracy_score(
            group["actual"],
            group["prediction"],
        )

        year_bal = (
            balanced_accuracy_score(
                group["actual"],
                group["prediction"],
            )
        )

        yearly_aucs.append(
            year_auc
        )

        yearly_rows.append(
            {
                "TrainWindow":
                    train_window,

                "Year":
                    year,

                "Samples":
                    len(group),

                "Accuracy":
                    year_acc,

                "BalancedAccuracy":
                    year_bal,

                "AUC":
                    year_auc,

                "ActualUpRate":
                    group[
                        "actual"
                    ].mean(),
            }
        )

    yearly_aucs = np.array(
        [
            value
            for value in yearly_aucs
            if not pd.isna(value)
        ]
    )

    summary = {
        "TrainWindow":
            train_window,

        "Samples":
            len(predictions),

        "Accuracy":
            accuracy,

        "BalancedAccuracy":
            balanced,

        "OverallAUC":
            auc,

        "MeanYearlyAUC":
            yearly_aucs.mean(),

        "MedianYearlyAUC":
            np.median(
                yearly_aucs
            ),

        "WorstYearAUC":
            yearly_aucs.min(),

        "BestYearAUC":
            yearly_aucs.max(),

        "YearsAbove050":
            int(
                np.sum(
                    yearly_aucs
                    > 0.50
                )
            ),

        "YearsAbove055":
            int(
                np.sum(
                    yearly_aucs
                    > 0.55
                )
            ),

        "YearCount":
            len(
                yearly_aucs
            ),
    }

    return (
        summary,
        yearly_rows,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 110)
    print(
        "30D TRAIN WINDOW SWEEP — "
        "WEEKLY WALK-FORWARD"
    )
    print("=" * 110)

    print()
    print(
        "Train windows:",
        TRAIN_WINDOWS,
    )

    df = load_dataset()

    df = add_market_regime(
        df
    )

    df = add_target_end_date(
        df
    )

    stock_features = (
        get_stock_features(
            df
        )
    )

    numeric_features = (
        stock_features
        + MARKET_FEATURES
    )

    print()
    print(
        "Stock feature sayısı :",
        len(stock_features),
    )

    print(
        "Market feature sayısı:",
        len(MARKET_FEATURES),
    )

    print(
        "Toplam numeric       :",
        len(numeric_features),
    )

    trading_dates = (
        df[
            "Date"
        ]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    first_candidate_date = (
        trading_dates.iloc[
            WARMUP_TRADING_DAYS
        ]
    )

    feature_mask = (
        build_valid_mask(
            df,
            numeric_features,
        )
    )

    valid_df = (
        df.loc[
            feature_mask
        ]
        .copy()
    )

    valid_df = (
        valid_df[
            valid_df[
                "Date"
            ]
            >= first_candidate_date
        ]
        .copy()
    )

    valid_dates = (
        valid_df[
            "Date"
        ]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    print()
    print(
        "İlk valid tarih:",
        valid_dates.iloc[0].date(),
    )

    print(
        "Son valid tarih:",
        valid_dates.iloc[-1].date(),
    )

    all_prediction_frames = []

    summary_rows = []

    all_yearly_rows = []

    # ========================================================
    # WINDOWS
    # ========================================================

    for train_window in TRAIN_WINDOWS:

        print()
        print()
        print("=" * 110)
        print(
            f"TRAIN WINDOW: "
            f"{train_window} İŞLEM GÜNÜ"
        )
        print("=" * 110)

        predictions = run_window(
            valid_df,
            valid_dates,
            numeric_features,
            train_window,
        )

        if predictions is None:
            print(
                "Prediction üretilemedi."
            )
            continue

        (
            summary,
            yearly_rows,
        ) = summarize_window(
            predictions,
            train_window,
        )

        summary_rows.append(
            summary
        )

        all_yearly_rows.extend(
            yearly_rows
        )

        all_prediction_frames.append(
            predictions
        )

        print()
        print(
            f"Samples       : "
            f"{summary['Samples']:,}"
        )

        print(
            f"Accuracy      : "
            f"%{summary['Accuracy'] * 100:.2f}"
        )

        print(
            f"Bal Accuracy  : "
            f"%{summary['BalancedAccuracy'] * 100:.2f}"
        )

        print(
            f"OOS AUC       : "
            f"{summary['OverallAUC']:.4f}"
        )

        print(
            f"Mean year AUC : "
            f"{summary['MeanYearlyAUC']:.4f}"
        )

        print(
            f"Median AUC    : "
            f"{summary['MedianYearlyAUC']:.4f}"
        )

        print(
            f"Worst year    : "
            f"{summary['WorstYearAUC']:.4f}"
        )

        print(
            f"Best year     : "
            f"{summary['BestYearAUC']:.4f}"
        )

        print(
            f"AUC > .50     : "
            f"{summary['YearsAbove050']}"
            f"/{summary['YearCount']}"
        )

        print(
            f"AUC > .55     : "
            f"{summary['YearsAbove055']}"
            f"/{summary['YearCount']}"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    summary_df = (
        pd.DataFrame(
            summary_rows
        )
        .sort_values(
            "OverallAUC",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    yearly_df = pd.DataFrame(
        all_yearly_rows
    )

    predictions_df = pd.concat(
        all_prediction_frames,
        ignore_index=True,
    )

    print()
    print()
    print("=" * 110)
    print(
        "TRAIN WINDOW KARŞILAŞTIRMASI"
    )
    print("=" * 110)

    print()

    print(
        f"{'Window':>8}"
        f"{'Samples':>12}"
        f"{'Acc':>10}"
        f"{'BalAcc':>10}"
        f"{'OOS AUC':>10}"
        f"{'MeanAUC':>10}"
        f"{'Median':>10}"
        f"{'Worst':>10}"
        f"{'>.50':>8}"
    )

    print(
        "-" * 98
    )

    for _, row in summary_df.iterrows():

        print(
            f"{int(row['TrainWindow']):>8}"
            f"{int(row['Samples']):>12,}"
            f"%{row['Accuracy'] * 100:>8.2f}"
            f"%{row['BalancedAccuracy'] * 100:>8.2f}"
            f"{row['OverallAUC']:>10.4f}"
            f"{row['MeanYearlyAUC']:>10.4f}"
            f"{row['MedianYearlyAUC']:>10.4f}"
            f"{row['WorstYearAUC']:>10.4f}"
            f"{str(int(row['YearsAbove050'])) + '/' + str(int(row['YearCount'])):>8}"
        )

    # ========================================================
    # BEST WINDOW YEARLY
    # ========================================================

    best_window = int(
        summary_df.iloc[0][
            "TrainWindow"
        ]
    )

    print()
    print("=" * 110)
    print(
        f"EN İYİ WINDOW: "
        f"{best_window} İŞLEM GÜNÜ"
    )
    print("=" * 110)

    best_yearly = (
        yearly_df[
            yearly_df[
                "TrainWindow"
            ]
            == best_window
        ]
        .sort_values(
            "Year"
        )
    )

    print()

    print(
        best_yearly[
            [
                "Year",
                "Samples",
                "Accuracy",
                "BalancedAccuracy",
                "AUC",
                "ActualUpRate",
            ]
        ]
        .to_string(
            index=False,
            formatters={
                "Accuracy":
                    lambda x:
                    f"%{x * 100:.2f}",

                "BalancedAccuracy":
                    lambda x:
                    f"%{x * 100:.2f}",

                "AUC":
                    lambda x:
                    f"{x:.4f}",

                "ActualUpRate":
                    lambda x:
                    f"%{x * 100:.2f}",
            }
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    yearly_df.to_csv(
        YEARLY_PATH,
        index=False,
    )

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print()
    print("=" * 110)
    print(
        "KAYDEDİLDİ"
    )
    print("=" * 110)

    print()
    print(
        SUMMARY_PATH
    )

    print(
        YEARLY_PATH
    )

    print(
        PREDICTIONS_PATH
    )


if __name__ == "__main__":
    main()