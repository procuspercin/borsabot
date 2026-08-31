from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
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

PREDICTIONS_PATH = Path(
    "data/rolling_3w1w_30d_predictions.csv"
)

BLOCKS_PATH = Path(
    "data/rolling_3w1w_30d_blocks.csv"
)

CONTEXT_PATH = Path(
    "data/rolling_3w1w_30d_context_summary.csv"
)

HORIZON = 30

# 3 hafta
TRAIN_TRADING_DAYS = 15

# 1 hafta
TEST_TRADING_DAYS = 5

# İlk feature'ların oluşması için
WARMUP_TRADING_DAYS = 252

# Test haftalarını 5 işlem günü ileri kaydır.
# Böylece her hafta yeni OOS test yapıyoruz.
TEST_STEP_DAYS = 5

# Bu deney çok sayıda model eğiteceği için
# production'daki 400 yerine 150 kullanıyoruz.
# Mantık aynı RandomForest.
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
# SAME STOCK FEATURES AS PRODUCTION
# ============================================================

def get_stock_features(df):

    excluded_columns = {
        "Date",
        "Ticker",
        "MarketRegime",
        "TargetEndDate",

        # Raw OHLCV
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",

        # Absolute MA
        "sma_5",
        "sma_10",
        "sma_20",
        "sma_50",
        "sma_100",
        "sma_200",

        # Absolute EMA
        "ema_12",
        "ema_26",

        # Raw volume averages
        "volume_ma_5",
        "volume_ma_20",

        # Absolute MACD
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


# ============================================================
# TARGET END DATE
# ============================================================

def add_target_end_date(df):

    """
    Bir satırın 30d target'ının hangi tarihte
    gerçekten öğrenilebilir hale geldiğini belirler.

    Örnek:
    Date = 2024-01-02

    target_up_30d ancak yaklaşık 30 işlem günü
    sonraki fiyat oluşunca bilinebilir.

    Böylece test tarihindeki modele gelecekteki
    label'ın sızmasını engelliyoruz.
    """

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
# VALID FEATURE ROWS
# ============================================================

def build_valid_mask(
    df,
    numeric_features,
):

    X = df[
        numeric_features
    ].replace(
        [np.inf, -np.inf],
        np.nan,
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
# PREPARE X/Y
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
# METRICS
# ============================================================

def safe_auc(
    y_true,
    probabilities,
):

    if len(
        np.unique(y_true)
    ) < 2:

        return np.nan

    return roc_auc_score(
        y_true,
        probabilities,
    )


# ============================================================
# RSI BAND
# ============================================================

def rsi_band(value):

    if pd.isna(value):
        return "UNKNOWN"

    if value < 30:
        return "<30"

    if value < 45:
        return "30-45"

    if value < 55:
        return "45-55"

    if value < 70:
        return "55-70"

    return "70+"


# ============================================================
# MARKET CONDITION
# ============================================================

def market_condition(row):

    r20 = row[
        "market_return_20d"
    ]

    r60 = row[
        "market_return_60d"
    ]

    if (
        r20 > 0
        and r60 > 0
    ):
        return "20d_UP_60d_UP"

    if (
        r20 < 0
        and r60 < 0
    ):
        return "20d_DOWN_60d_DOWN"

    if (
        r20 > 0
        and r60 < 0
    ):
        return "20d_UP_60d_DOWN"

    return "20d_DOWN_60d_UP"


# ============================================================
# FAILURE CONTEXT SUMMARY
# ============================================================

def context_summary(
    predictions,
):

    rows = []

    group_specs = [
        (
            "MarketRegime",
            "MarketRegime",
        ),
        (
            "RSIBand",
            "RSIBand",
        ),
        (
            "MarketDirection",
            "MarketDirection",
        ),
        (
            "Ticker",
            "Ticker",
        ),
        (
            "Year",
            "Year",
        ),
    ]

    for context_name, column in group_specs:

        for value, group in predictions.groupby(
            column,
            dropna=False,
        ):

            if len(group) < 20:
                continue

            auc = safe_auc(
                group["actual"].values,
                group["probability"].values,
            )

            accuracy = accuracy_score(
                group["actual"],
                group["prediction"],
            )

            balanced = balanced_accuracy_score(
                group["actual"],
                group["prediction"],
            )

            rows.append(
                {
                    "Context":
                        context_name,

                    "Value":
                        str(value),

                    "Samples":
                        len(group),

                    "Accuracy":
                        accuracy,

                    "BalancedAccuracy":
                        balanced,

                    "AUC":
                        auc,

                    "ActualUpRate":
                        group[
                            "actual"
                        ].mean(),

                    "MeanProbability":
                        group[
                            "probability"
                        ].mean(),

                    "MeanReturn30d":
                        group[
                            "actual_return_30d"
                        ].mean(),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN ROLLING TEST
# ============================================================

def main():

    print()
    print("=" * 110)
    print(
        "30D — 3 HAFTA TRAIN / 1 HAFTA TEST "
        "ROLLING HISTORICAL SIMULATION"
    )
    print("=" * 110)

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

    # --------------------------------------------------------
    # GLOBAL TRADING DATES
    # --------------------------------------------------------

    trading_dates = (
        df["Date"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if len(
        trading_dates
    ) <= WARMUP_TRADING_DAYS:

        raise RuntimeError(
            "252 işlem günlük warm-up için "
            "yeterli tarih yok."
        )

    warmup_end_date = (
        trading_dates.iloc[
            WARMUP_TRADING_DAYS - 1
        ]
    )

    first_candidate_date = (
        trading_dates.iloc[
            WARMUP_TRADING_DAYS
        ]
    )

    print()
    print(
        "İlk ham tarih       :",
        trading_dates.iloc[0].date(),
    )

    print(
        "252. işlem günü     :",
        warmup_end_date.date(),
    )

    print(
        "İlk aday test tarihi:",
        first_candidate_date.date(),
    )

    # --------------------------------------------------------
    # VALID FEATURE DATES
    # --------------------------------------------------------

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

    valid_df = valid_df[
        valid_df["Date"]
        >= first_candidate_date
    ].copy()

    valid_dates = (
        valid_df["Date"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    print(
        "Tüm feature'ların hazır "
        "olduğu ilk tarih:",
        valid_dates.iloc[0].date(),
    )

    print(
        "Son tarih:",
        valid_dates.iloc[-1].date(),
    )

    # --------------------------------------------------------
    # ROLLING TEST
    # --------------------------------------------------------

    prediction_frames = []

    block_rows = []

    block_id = 0

    # Her test haftasını 5 gün ileri taşıyoruz.
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
        # LABEL'I TESTTEN ÖNCE BİLİNEN SATIRLAR
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

        eligible_train_dates = (
            eligible_train["Date"]
            .drop_duplicates()
            .sort_values()
        )

        if len(
            eligible_train_dates
        ) < TRAIN_TRADING_DAYS:

            continue

        # Sonucu bilinen EN GÜNCEL 15 işlem günü.
        train_dates = (
            eligible_train_dates
            .iloc[
                -TRAIN_TRADING_DAYS:
            ]
        )

        train_start = (
            train_dates.iloc[0]
        )

        train_end = (
            train_dates.iloc[-1]
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

        # Test target'ı henüz gelecekte olabilir,
        # ama historical dataset'te sonucu bildiğimiz için
        # sadece evaluation amacıyla kullanıyoruz.
        test_df = test_df[
            test_df[
                f"target_up_{HORIZON}d"
            ].notna()
        ].copy()

        if test_df.empty:
            # Dataset'in son 30 günü civarında
            # doğal olarak target olmayacak.
            continue

        (
            X_train,
            y_train,
            train_meta,
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

        block_id += 1

        print(
            f"\rBlok {block_id:>4} "
            f"| Train {train_start.date()} "
            f"→ {train_end.date()} "
            f"| Test {test_start.date()} "
            f"→ {test_end.date()}",
            end="",
            flush=True,
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # BLOCK METRICS
        # ----------------------------------------------------

        block_accuracy = (
            accuracy_score(
                y_test,
                prediction,
            )
        )

        block_balanced = (
            balanced_accuracy_score(
                y_test,
                prediction,
            )
        )

        block_auc = safe_auc(
            y_test.values,
            probability,
        )

        cm = confusion_matrix(
            y_test,
            prediction,
            labels=[
                0,
                1,
            ],
        )

        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        result = pd.DataFrame(
            {
                "BlockID":
                    block_id,

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

                "TrainStart":
                    train_start,

                "TrainEnd":
                    train_end,

                "TestStart":
                    test_start,

                "TestEnd":
                    test_end,
            }
        )

        # ----------------------------------------------------
        # FAILURE CONTEXT
        # ----------------------------------------------------

        context_columns = [
            "rsi_14",
            "return_20d",
            "return_60d",
            "volatility_20d",
            "volume_ratio_20",
            "close_sma_50_ratio",
            "close_sma_200_ratio",
            "market_return_20d",
            "market_return_60d",
            "market_volatility_20d",
            "market_sma_50_distance",
            "market_sma_200_distance",
            "MarketRegime",
        ]

        for column in context_columns:

            if column in test_meta.columns:

                result[column] = (
                    test_meta[
                        column
                    ].values
                )

        result["Correct"] = (
            result["actual"]
            == result["prediction"]
        ).astype(int)

        result["Year"] = (
            pd.to_datetime(
                result["Date"]
            )
            .dt.year
        )

        result["Month"] = (
            pd.to_datetime(
                result["Date"]
            )
            .dt.to_period("M")
            .astype(str)
        )

        result["RSIBand"] = (
            result["rsi_14"]
            .apply(
                rsi_band
            )
        )

        result["MarketDirection"] = (
            result.apply(
                market_condition,
                axis=1,
            )
        )

        prediction_frames.append(
            result
        )

        # ----------------------------------------------------
        # BLOCK CONTEXT
        # ----------------------------------------------------

        block_rows.append(
            {
                "BlockID":
                    block_id,

                "TrainStart":
                    train_start,

                "TrainEnd":
                    train_end,

                "TestStart":
                    test_start,

                "TestEnd":
                    test_end,

                "TrainSamples":
                    len(
                        y_train
                    ),

                "TestSamples":
                    len(
                        y_test
                    ),

                "TrainUPRate":
                    y_train.mean(),

                "TestUPRate":
                    y_test.mean(),

                "Accuracy":
                    block_accuracy,

                "BalancedAccuracy":
                    block_balanced,

                "AUC":
                    block_auc,

                "TN":
                    int(
                        cm[0, 0]
                    ),

                "FP":
                    int(
                        cm[0, 1]
                    ),

                "FN":
                    int(
                        cm[1, 0]
                    ),

                "TP":
                    int(
                        cm[1, 1]
                    ),

                "MarketReturn20d":
                    test_meta[
                        "market_return_20d"
                    ].mean(),

                "MarketReturn60d":
                    test_meta[
                        "market_return_60d"
                    ].mean(),

                "MarketVolatility20d":
                    test_meta[
                        "market_volatility_20d"
                    ].mean(),

                "MeanRSI":
                    test_meta[
                        "rsi_14"
                    ].mean(),

                "MeanStockReturn20d":
                    test_meta[
                        "return_20d"
                    ].mean(),

                "MeanStockVolatility20d":
                    test_meta[
                        "volatility_20d"
                    ].mean(),

                "MeanProbability":
                    probability.mean(),
            }
        )

    print()

    if not prediction_frames:

        raise RuntimeError(
            "Hiç rolling prediction üretilemedi."
        )

    # ========================================================
    # COMBINE
    # ========================================================

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    blocks = pd.DataFrame(
        block_rows
    )

    # ========================================================
    # OVERALL
    # ========================================================

    overall_accuracy = (
        accuracy_score(
            predictions["actual"],
            predictions["prediction"],
        )
    )

    overall_balanced = (
        balanced_accuracy_score(
            predictions["actual"],
            predictions["prediction"],
        )
    )

    overall_auc = safe_auc(
        predictions["actual"].values,
        predictions[
            "probability"
        ].values,
    )

    print()
    print()
    print("=" * 110)
    print(
        "OVERALL OUT-OF-SAMPLE"
    )
    print("=" * 110)

    print()
    print(
        "Test başlangıcı       :",
        predictions[
            "Date"
        ].min(),
    )

    print(
        "Test bitişi           :",
        predictions[
            "Date"
        ].max(),
    )

    print(
        f"Toplam blok           : "
        f"{len(blocks):,}"
    )

    print(
        f"Toplam tahmin         : "
        f"{len(predictions):,}"
    )

    print(
        f"Accuracy              : "
        f"%{overall_accuracy * 100:.2f}"
    )

    print(
        f"Balanced Accuracy     : "
        f"%{overall_balanced * 100:.2f}"
    )

    print(
        f"ROC AUC               : "
        f"{overall_auc:.4f}"
    )

    print(
        f"Gerçek UP oranı       : "
        f"%{predictions['actual'].mean() * 100:.2f}"
    )

    print(
        f"Ortalama model skoru  : "
        f"%{predictions['probability'].mean() * 100:.2f}"
    )

    # ========================================================
    # YEARLY
    # ========================================================

    print()
    print("=" * 110)
    print(
        "YILLARA GÖRE"
    )
    print("=" * 110)

    print()

    print(
        f"{'Yıl':<8}"
        f"{'N':>10}"
        f"{'Acc':>12}"
        f"{'BalAcc':>12}"
        f"{'AUC':>10}"
        f"{'UP Base':>12}"
    )

    print(
        "-" * 64
    )

    for year, group in predictions.groupby(
        "Year"
    ):

        auc = safe_auc(
            group["actual"].values,
            group["probability"].values,
        )

        acc = accuracy_score(
            group["actual"],
            group["prediction"],
        )

        bal = balanced_accuracy_score(
            group["actual"],
            group["prediction"],
        )

        print(
            f"{year:<8}"
            f"{len(group):>10}"
            f"%{acc * 100:>10.2f}"
            f"%{bal * 100:>10.2f}"
            f"{auc:>10.4f}"
            f"%{group['actual'].mean() * 100:>10.2f}"
        )

    # ========================================================
    # BLOCK STABILITY
    # ========================================================

    print()
    print("=" * 110)
    print(
        "BLOK STABILITY"
    )
    print("=" * 110)

    valid_block_auc = (
        blocks["AUC"]
        .dropna()
    )

    print()
    print(
        f"Ortalama blok AUC     : "
        f"{valid_block_auc.mean():.4f}"
    )

    print(
        f"Medyan blok AUC       : "
        f"{valid_block_auc.median():.4f}"
    )

    print(
        f"En kötü blok AUC      : "
        f"{valid_block_auc.min():.4f}"
    )

    print(
        f"En iyi blok AUC       : "
        f"{valid_block_auc.max():.4f}"
    )

    print(
        f"AUC > 0.50 blok       : "
        f"{(valid_block_auc > 0.50).sum()}"
        f"/{len(valid_block_auc)}"
    )

    print(
        f"AUC < 0.45 blok       : "
        f"{(valid_block_auc < 0.45).sum()}"
        f"/{len(valid_block_auc)}"
    )

    # ========================================================
    # WORST BLOCKS
    # ========================================================

    print()
    print("=" * 110)
    print(
        "EN KÖTÜ 15 TEST HAFTASI"
    )
    print("=" * 110)

    worst = (
        blocks
        .dropna(
            subset=["AUC"]
        )
        .sort_values(
            "AUC"
        )
        .head(15)
    )

    print()

    columns = [
        "TestStart",
        "TestEnd",
        "AUC",
        "Accuracy",
        "BalancedAccuracy",
        "TestUPRate",
        "MarketReturn20d",
        "MarketReturn60d",
        "MarketVolatility20d",
        "MeanRSI",
    ]

    print(
        worst[
            columns
        ].to_string(
            index=False,
            formatters={
                "AUC":
                    lambda x:
                    f"{x:.4f}",

                "Accuracy":
                    lambda x:
                    f"%{x * 100:.2f}",

                "BalancedAccuracy":
                    lambda x:
                    f"%{x * 100:.2f}",

                "TestUPRate":
                    lambda x:
                    f"%{x * 100:.2f}",

                "MarketReturn20d":
                    lambda x:
                    f"%{x * 100:+.2f}",

                "MarketReturn60d":
                    lambda x:
                    f"%{x * 100:+.2f}",

                "MarketVolatility20d":
                    lambda x:
                    f"%{x * 100:.2f}",

                "MeanRSI":
                    lambda x:
                    f"{x:.1f}",
            }
        )
    )

    # ========================================================
    # CONTEXT SUMMARY
    # ========================================================

    context = context_summary(
        predictions
    )

    print()
    print("=" * 110)
    print(
        "MODELİN ÇÖKTÜĞÜ CONTEXTLER"
    )
    print("=" * 110)

    bad_contexts = (
        context[
            context["Samples"]
            >= 100
        ]
        .dropna(
            subset=["AUC"]
        )
        .sort_values(
            "AUC"
        )
        .head(20)
    )

    print()

    print(
        bad_contexts[
            [
                "Context",
                "Value",
                "Samples",
                "Accuracy",
                "BalancedAccuracy",
                "AUC",
                "ActualUpRate",
                "MeanReturn30d",
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

                "MeanReturn30d":
                    lambda x:
                    f"%{x * 100:+.2f}",
            }
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    blocks.to_csv(
        BLOCKS_PATH,
        index=False,
    )

    context.to_csv(
        CONTEXT_PATH,
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
        PREDICTIONS_PATH
    )

    print(
        BLOCKS_PATH
    )

    print(
        CONTEXT_PATH
    )


if __name__ == "__main__":
    main()