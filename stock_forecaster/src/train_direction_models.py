from pathlib import Path
import json

import joblib
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

MODEL_DIR = Path(
    "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

HORIZONS = [
    10,
    30,
    60,
    120,
    180,
]

TRAIN_YEARS = 5

TEST_YEARS = [
    2020,
    2021,
    2022,
    2023,
    2024,
    2025,
    2026,
]


# ============================================================
# SELECTED MARKET FEATURES
# ============================================================

MARKET_FEATURES = [
    "market_return_20d",
    "market_return_60d",
    "market_sma_50_distance",
    "market_sma_200_distance",
    "market_volatility_20d",
]


# ============================================================
# LOAD DATASET
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

    for horizon in HORIZONS:
        target_col = (
            f"target_up_{horizon}d"
        )

        if target_col not in df.columns:
            raise ValueError(
                f"Eksik target: {target_col}"
            )

        df[target_col] = pd.to_numeric(
            df[target_col],
            errors="coerce",
        )

    return df


# ============================================================
# MARKET REGIME
# ============================================================

def add_market_regime(df):
    df = df.copy()

    required = [
        "market_return_20d",
        "market_return_60d",
        "market_sma_50_distance",
        "market_sma_200_distance",
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Eksik market feature: {col}"
            )

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
# STOCK FEATURES
# ============================================================

def get_stock_features(df):
    excluded_columns = {
        "Date",
        "Ticker",
        "MarketRegime",

        # Raw price / volume
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",

        # Absolute moving averages
        "sma_5",
        "sma_10",
        "sma_20",
        "sma_50",
        "sma_100",
        "sma_200",

        # Absolute EMA
        "ema_12",
        "ema_26",

        # Absolute volume averages
        "volume_ma_5",
        "volume_ma_20",

        # Absolute MACD
        "macd",
        "macd_signal",
        "macd_hist",
    }

    features = []

    for col in df.columns:

        if col.startswith("target_"):
            continue

        if col.startswith("market_"):
            continue

        if col.startswith(
            "relative_strength_"
        ):
            continue

        if col in excluded_columns:
            continue

        features.append(col)

    return features


# ============================================================
# PURGE
# ============================================================

def purge_train_boundary(
    train_df,
    horizon,
):
    """
    Walk-forward test sırasında train setinin son
    horizon kadar satırını hisse bazında çıkarır.

    Böylece train target'larının test dönemindeki
    fiyatlardan faydalanması engellenir.
    """

    parts = []

    for ticker, group in train_df.groupby(
        "Ticker",
        sort=False,
    ):
        group = (
            group
            .sort_values("Date")
            .copy()
        )

        if len(group) <= horizon:
            continue

        group = (
            group
            .iloc[:-horizon]
            .copy()
        )

        parts.append(group)

    if not parts:
        return train_df.iloc[
            0:0
        ].copy()

    result = pd.concat(
        parts,
        ignore_index=True,
    )

    result = (
        result
        .sort_values(
            [
                "Date",
                "Ticker",
            ]
        )
        .reset_index(drop=True)
    )

    return result


# ============================================================
# PREPARE X / Y
# ============================================================

def prepare_xy(
    df,
    stock_features,
    horizon,
):
    target_col = (
        f"target_up_{horizon}d"
    )

    numeric_features = (
        stock_features
        + MARKET_FEATURES
    )

    categorical_features = [
        "Ticker",
        "MarketRegime",
    ]

    feature_columns = (
        numeric_features
        + categorical_features
    )

    X = df[
        feature_columns
    ].copy()

    y = df[
        target_col
    ].copy()

    metadata = df[
        [
            "Date",
            "Ticker",
        ]
    ].copy()

    X[numeric_features] = (
        X[numeric_features]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    valid_mask = (
        X[numeric_features]
        .notna()
        .all(axis=1)
        & y.notna()
    )

    X = (
        X.loc[
            valid_mask
        ]
        .reset_index(drop=True)
    )

    y = (
        y.loc[
            valid_mask
        ]
        .astype(int)
        .reset_index(drop=True)
    )

    metadata = (
        metadata.loc[
            valid_mask
        ]
        .reset_index(drop=True)
    )

    return (
        X,
        y,
        metadata,
        numeric_features,
        categorical_features,
    )


# ============================================================
# MODEL PIPELINE
# ============================================================

def create_model(
    numeric_features,
    categorical_features,
):
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
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model = Pipeline(
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

    return model


# ============================================================
# WALK FORWARD — SINGLE YEAR
# ============================================================

def evaluate_year(
    df,
    stock_features,
    horizon,
    year,
):
    train_start = pd.Timestamp(
        f"{year - TRAIN_YEARS}-01-01"
    )

    train_end = pd.Timestamp(
        f"{year - 1}-12-31"
    )

    test_start = pd.Timestamp(
        f"{year}-01-01"
    )

    test_end = pd.Timestamp(
        f"{year}-12-31"
    )

    train_df = df[
        (df["Date"] >= train_start)
        & (df["Date"] <= train_end)
    ].copy()

    test_df = df[
        (df["Date"] >= test_start)
        & (df["Date"] <= test_end)
    ].copy()

    train_df = purge_train_boundary(
        train_df,
        horizon,
    )

    (
        X_train,
        y_train,
        _,
        numeric_features,
        categorical_features,
    ) = prepare_xy(
        train_df,
        stock_features,
        horizon,
    )

    (
        X_test,
        y_test,
        metadata,
        _,
        _,
    ) = prepare_xy(
        test_df,
        stock_features,
        horizon,
    )

    if len(X_train) == 0:
        return None

    if len(X_test) == 0:
        return None

    if y_train.nunique() < 2:
        return None

    if y_test.nunique() < 2:
        return None

    model = create_model(
        numeric_features,
        categorical_features,
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

    auc = roc_auc_score(
        y_test,
        probability,
    )

    accuracy = accuracy_score(
        y_test,
        prediction,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            prediction,
        )
    )

    predictions = metadata.copy()

    predictions["actual"] = (
        y_test.values
    )

    predictions["P_UP"] = (
        probability
    )

    predictions[
        "prediction"
    ] = prediction

    predictions[
        "TestYear"
    ] = year

    predictions[
        "Horizon"
    ] = horizon

    return {
        "year": year,
        "auc": auc,
        "accuracy": accuracy,
        "balanced_accuracy":
            balanced_accuracy,
        "samples": len(y_test),
        "predictions": predictions,
    }


# ============================================================
# WALK FORWARD — HORIZON
# ============================================================

def evaluate_horizon(
    df,
    stock_features,
    horizon,
):
    print()
    print("=" * 100)
    print(
        f"WALK-FORWARD — {horizon}D"
    )
    print("=" * 100)

    results = []
    prediction_frames = []

    for year in TEST_YEARS:
        print()
        print(
            f"{horizon}d / {year} "
            "eğitiliyor..."
        )

        result = evaluate_year(
            df,
            stock_features,
            horizon,
            year,
        )

        if result is None:
            print(
                "  Sonuç üretilemedi."
            )
            continue

        results.append(
            result
        )

        prediction_frames.append(
            result["predictions"]
        )

        print(
            f"  Samples: {result['samples']}"
            f" | AUC: {result['auc']:.4f}"
            f" | Acc: %{result['accuracy'] * 100:.2f}"
            f" | BalAcc: "
            f"%{result['balanced_accuracy'] * 100:.2f}"
        )

    if not results:
        return None

    all_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    overall_auc = roc_auc_score(
        all_predictions["actual"],
        all_predictions["P_UP"],
    )

    overall_accuracy = accuracy_score(
        all_predictions["actual"],
        all_predictions["prediction"],
    )

    overall_balanced = (
        balanced_accuracy_score(
            all_predictions["actual"],
            all_predictions["prediction"],
        )
    )

    auc_values = np.array(
        [
            result["auc"]
            for result in results
        ]
    )

    print()
    print("-" * 100)

    print(
        f"{horizon}D OVERALL OOS"
    )

    print("-" * 100)

    print(
        f"ROC AUC           : "
        f"{overall_auc:.4f}"
    )

    print(
        f"Accuracy          : "
        f"%{overall_accuracy * 100:.2f}"
    )

    print(
        f"Balanced Accuracy : "
        f"%{overall_balanced * 100:.2f}"
    )

    print(
        f"Mean yearly AUC   : "
        f"{auc_values.mean():.4f}"
    )

    print(
        f"Median yearly AUC : "
        f"{np.median(auc_values):.4f}"
    )

    print(
        f"AUC > 0.50        : "
        f"{np.sum(auc_values > 0.50)}"
        f"/{len(auc_values)}"
    )

    return {
        "horizon": horizon,
        "overall_auc": overall_auc,
        "overall_accuracy":
            overall_accuracy,
        "overall_balanced":
            overall_balanced,
        "mean_auc":
            auc_values.mean(),
        "median_auc":
            np.median(
                auc_values
            ),
        "worst_auc":
            auc_values.min(),
        "best_auc":
            auc_values.max(),
        "years_above_50":
            int(
                np.sum(
                    auc_values > 0.50
                )
            ),
        "year_count":
            len(
                auc_values
            ),
        "year_results":
            results,
        "predictions":
            all_predictions,
    }


# ============================================================
# FINAL MODEL
# ============================================================

def train_final_model(
    df,
    stock_features,
    horizon,
):
    """
    Walk-forward değerlendirmeden sonra,
    geleceği gerçekten bilinen bütün satırlarla
    production modelini eğitir.
    """

    target_col = (
        f"target_up_{horizon}d"
    )

    final_df = df[
        df[target_col].notna()
    ].copy()

    (
        X,
        y,
        metadata,
        numeric_features,
        categorical_features,
    ) = prepare_xy(
        final_df,
        stock_features,
        horizon,
    )

    if len(X) == 0:
        raise RuntimeError(
            f"{horizon}d final training "
            "verisi bulunamadı."
        )

    model = create_model(
        numeric_features,
        categorical_features,
    )

    print()
    print(
        f"{horizon}d final model "
        f"{len(X)} sample ile eğitiliyor..."
    )

    model.fit(
        X,
        y,
    )

    model_path = (
        MODEL_DIR
        / f"direction_{horizon}d.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    metadata_path = (
        MODEL_DIR
        / f"direction_{horizon}d_metadata.json"
    )

    metadata_json = {
        "horizon": horizon,
        "training_samples":
            int(len(X)),
        "training_start":
            str(
                metadata[
                    "Date"
                ].min().date()
            ),
        "training_end":
            str(
                metadata[
                    "Date"
                ].max().date()
            ),
        "positive_rate":
            float(
                y.mean()
            ),
        "stock_features":
            stock_features,
        "market_features":
            MARKET_FEATURES,
        "categorical_features":
            categorical_features,
        "model_type":
            "RandomForestClassifier",
        "ticker_count":
            int(
                metadata[
                    "Ticker"
                ].nunique()
            ),
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata_json,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "Model kaydedildi:",
        model_path,
    )

    print(
        "Metadata kaydedildi:",
        metadata_path,
    )

    return model_path


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 100)
    print(
        "STOCK FORECASTER — "
        "MULTI-HORIZON DIRECTION TRAINING"
    )
    print("=" * 100)

    print()
    print(
        "Horizons:",
        HORIZONS,
    )

    print(
        "Training window:",
        TRAIN_YEARS,
        "yıl",
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_dataset()

    df = add_market_regime(
        df
    )

    stock_features = (
        get_stock_features(
            df
        )
    )

    print()
    print(
        "Stock feature sayısı:",
        len(stock_features),
    )

    print(
        "Market feature sayısı:",
        len(MARKET_FEATURES),
    )

    print(
        "Ticker sayısı:",
        df["Ticker"].nunique(),
    )

    print()

    print(
        "Market regime dağılımı:"
    )

    print(
        df[
            "MarketRegime"
        ]
        .value_counts(
            normalize=True
        )
        .mul(100)
        .round(2)
    )

    # --------------------------------------------------------
    # WALK FORWARD
    # --------------------------------------------------------

    evaluation_results = []

    prediction_frames = []

    for horizon in HORIZONS:
        result = evaluate_horizon(
            df,
            stock_features,
            horizon,
        )

        if result is None:
            continue

        evaluation_results.append(
            result
        )

        prediction_frames.append(
            result["predictions"]
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print()
    print("=" * 100)
    print("DIRECTION MODEL ÖZETİ")
    print("=" * 100)

    print()

    print(
        f"{'Horizon':<10}"
        f"{'OOS AUC':>12}"
        f"{'Mean AUC':>12}"
        f"{'Median':>12}"
        f"{'Worst':>12}"
        f"{'Best':>12}"
        f"{'>0.50':>10}"
    )

    print(
        "-" * 80
    )

    for result in evaluation_results:
        print(
            f"{str(result['horizon']) + 'd':<10}"
            f"{result['overall_auc']:>12.4f}"
            f"{result['mean_auc']:>12.4f}"
            f"{result['median_auc']:>12.4f}"
            f"{result['worst_auc']:>12.4f}"
            f"{result['best_auc']:>12.4f}"
            f"{str(result['years_above_50']) + '/' + str(result['year_count']):>10}"
        )

    # --------------------------------------------------------
    # SAVE EVALUATION SUMMARY
    # --------------------------------------------------------

    summary_rows = []

    for result in evaluation_results:
        summary_rows.append(
            {
                "Horizon":
                    result["horizon"],
                "Overall_OOS_AUC":
                    result[
                        "overall_auc"
                    ],
                "Overall_Accuracy":
                    result[
                        "overall_accuracy"
                    ],
                "Overall_BalancedAccuracy":
                    result[
                        "overall_balanced"
                    ],
                "Mean_Yearly_AUC":
                    result[
                        "mean_auc"
                    ],
                "Median_Yearly_AUC":
                    result[
                        "median_auc"
                    ],
                "Worst_Year_AUC":
                    result[
                        "worst_auc"
                    ],
                "Best_Year_AUC":
                    result[
                        "best_auc"
                    ],
                "Years_Above_050":
                    result[
                        "years_above_50"
                    ],
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        MODEL_DIR
        / "direction_model_evaluation.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE WALK-FORWARD PREDICTIONS
    # --------------------------------------------------------

    if prediction_frames:
        all_predictions = pd.concat(
            prediction_frames,
            ignore_index=True,
        )

        prediction_path = (
            MODEL_DIR
            / "direction_walk_forward_predictions.csv"
        )

        all_predictions.to_csv(
            prediction_path,
            index=False,
        )

    # --------------------------------------------------------
    # FINAL PRODUCTION MODELS
    # --------------------------------------------------------

    print()
    print()
    print("=" * 100)
    print("FINAL MODELLER EĞİTİLİYOR")
    print("=" * 100)

    final_models = []

    for horizon in HORIZONS:
        model_path = train_final_model(
            df,
            stock_features,
            horizon,
        )

        final_models.append(
            model_path
        )

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print()
    print()
    print("=" * 100)
    print("TAMAMLANDI")
    print("=" * 100)

    print()

    print(
        "Evaluation:",
        summary_path,
    )

    print()
    print(
        "Kaydedilen direction modelleri:"
    )

    for model_path in final_models:
        print(
            "-",
            model_path,
        )


if __name__ == "__main__":
    main()