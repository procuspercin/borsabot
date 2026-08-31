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

        frame["Month"] = (
            pd.to_datetime(
                frame["Date"]
            ).dt.month
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
# 2026 MONTHLY SUMMARY
# ============================================================

def summarize_2026_monthly(predictions):

    rows = []

    monthly = predictions[predictions["Year"] == 2026].copy()

    for (model_name, month), group in monthly.groupby(["Model", "Month"]):

        rows.append(
            {
                "Model": model_name,
                "Year": 2026,
                "Month": int(month),
                "Samples": len(group),
                "Accuracy": accuracy_score(
                    group["actual"],
                    group["prediction"],
                ),
                "BalancedAccuracy": balanced_accuracy_score(
                    group["actual"],
                    group["prediction"],
                ),
                "AUC": safe_auc(
                    group["actual"].values,
                    group["probability"].values,
                ),
            }
        )

    return pd.DataFrame(rows)


def build_2026_monthly_comparison(monthly_summary):

    if monthly_summary.empty:
        return pd.DataFrame()

    base = (
        monthly_summary[monthly_summary["Model"] == "BASE"]
        .set_index("Month")
    )

    gpr = (
        monthly_summary[monthly_summary["Model"] == "GPR"]
        .set_index("Month")
    )

    common_months = base.index.intersection(gpr.index)

    rows = []

    for month in sorted(common_months):
        rows.append(
            {
                "Year": 2026,
                "Month": int(month),
                "Samples": int(base.loc[month, "Samples"]),
                "BASE_Accuracy": base.loc[month, "Accuracy"],
                "GPR_Accuracy": gpr.loc[month, "Accuracy"],
                "Delta_Accuracy": (
                    gpr.loc[month, "Accuracy"]
                    - base.loc[month, "Accuracy"]
                ),
                "BASE_BalancedAccuracy": base.loc[month, "BalancedAccuracy"],
                "GPR_BalancedAccuracy": gpr.loc[month, "BalancedAccuracy"],
                "Delta_BalancedAccuracy": (
                    gpr.loc[month, "BalancedAccuracy"]
                    - base.loc[month, "BalancedAccuracy"]
                ),
                "BASE_AUC": base.loc[month, "AUC"],
                "GPR_AUC": gpr.loc[month, "AUC"],
                "Delta_AUC": (
                    gpr.loc[month, "AUC"]
                    - base.loc[month, "AUC"]
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
        "2026 MONTHLY BREAKDOWN — "
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
    # 2026 MONTHLY BREAKDOWN
    # --------------------------------------------------------

    monthly_summary = summarize_2026_monthly(
        predictions
    )

    monthly_comparison = build_2026_monthly_comparison(
        monthly_summary
    )

    print()
    print("=" * 100)
    print("2026 AYLIK BASE vs GPR")
    print("=" * 100)
    print()

    if monthly_comparison.empty:
        print("2026 aylık sonuç üretilemedi.")
    else:
        month_names = {
            1: "Ocak",
            2: "Şubat",
            3: "Mart",
            4: "Nisan",
            5: "Mayıs",
            6: "Haziran",
            7: "Temmuz",
            8: "Ağustos",
            9: "Eylül",
            10: "Ekim",
            11: "Kasım",
            12: "Aralık",
        }

        monthly_display = monthly_comparison.copy()
        monthly_display["Ay"] = monthly_display["Month"].map(month_names)

        monthly_display = monthly_display[
            [
                "Ay",
                "Samples",
                "BASE_Accuracy",
                "GPR_Accuracy",
                "Delta_Accuracy",
                "BASE_BalancedAccuracy",
                "GPR_BalancedAccuracy",
                "Delta_BalancedAccuracy",
                "BASE_AUC",
                "GPR_AUC",
                "Delta_AUC",
            ]
        ]

        print(
            monthly_display.to_string(
                index=False,
                formatters={
                    "BASE_Accuracy": lambda x: f"%{x * 100:.2f}",
                    "GPR_Accuracy": lambda x: f"%{x * 100:.2f}",
                    "Delta_Accuracy": lambda x: f"{x * 100:+.2f} puan",
                    "BASE_BalancedAccuracy": lambda x: f"%{x * 100:.2f}",
                    "GPR_BalancedAccuracy": lambda x: f"%{x * 100:.2f}",
                    "Delta_BalancedAccuracy": lambda x: f"{x * 100:+.2f} puan",
                    "BASE_AUC": lambda x: f"{x:.4f}",
                    "GPR_AUC": lambda x: f"{x:.4f}",
                    "Delta_AUC": lambda x: f"{x:+.4f}",
                },
            )
        )

        print()
        print("2026 AYLIK GPR AUC ETKİSİ")
        print("-" * 100)

        for _, row in monthly_comparison.iterrows():
            month_name = month_names[int(row["Month"])]
            print(
                f"{month_name:<8}: "
                f"{row['BASE_AUC']:.4f} "
                f"→ {row['GPR_AUC']:.4f} "
                f"({row['Delta_AUC']:+.4f})"
            )

    # --------------------------------------------------------

    # ============================================================
    # BASE MODEL — 2016-2026 MONTHLY BREAKDOWN
    # ============================================================

    base_all = predictions[predictions["Model"] == "BASE"].copy()
    base_all["Date"] = pd.to_datetime(base_all["Date"])
    base_all["Year"] = base_all["Date"].dt.year
    base_all["Month"] = base_all["Date"].dt.month

    month_names = {
        1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan",
        5: "Mayıs", 6: "Haziran", 7: "Temmuz", 8: "Ağustos",
        9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık",
    }

    monthly_rows = []
    for (year, month), group in base_all.groupby(["Year", "Month"]):
        monthly_rows.append({
            "Year": int(year), "Month": int(month),
            "Ay": month_names[int(month)], "Samples": len(group),
            "Accuracy": accuracy_score(group["actual"], group["prediction"]),
            "BalancedAccuracy": balanced_accuracy_score(group["actual"], group["prediction"]),
            "AUC": safe_auc(group["actual"].values, group["probability"].values),
            "ActualUpRate": group["actual"].mean(),
            "PredictedUpRate": group["prediction"].mean(),
            "MeanProbability": group["probability"].mean(),
        })

    monthly_base = pd.DataFrame(monthly_rows).sort_values(["Year", "Month"]).reset_index(drop=True)

    print()
    print("=" * 100)
    print("BASE MODEL — 2016-2026 AYLIK PERFORMANS")
    print("=" * 100)
    print()

    d = monthly_base.copy()
    for c in ["Accuracy", "BalancedAccuracy", "ActualUpRate", "PredictedUpRate", "MeanProbability"]:
        d[c] *= 100
    print(d.to_string(index=False, formatters={
        "Accuracy": lambda x: f"%{x:.2f}",
        "BalancedAccuracy": lambda x: f"%{x:.2f}",
        "AUC": lambda x: f"{x:.4f}",
        "ActualUpRate": lambda x: f"%{x:.2f}",
        "PredictedUpRate": lambda x: f"%{x:.2f}",
        "MeanProbability": lambda x: f"%{x:.2f}",
    }))

    auc_pivot = monthly_base.pivot(index="Year", columns="Month", values="AUC")
    auc_pivot = auc_pivot.rename(columns=month_names)

    print()
    print("=" * 100)
    print("BASE AUC MATRİSİ — YIL x AY")
    print("=" * 100)
    print()
    print(auc_pivot.round(3).to_string())

    stability_rows = []
    for month, group in monthly_base.groupby("Month"):
        a = group["AUC"].dropna()
        stability_rows.append({
            "Month": int(month), "Ay": month_names[int(month)],
            "Years": len(a), "MeanAUC": a.mean(), "MedianAUC": a.median(),
            "MinAUC": a.min(), "MaxAUC": a.max(),
            "AUC_gt_050_Rate": (a > 0.50).mean(),
            "AUC_gt_055_Rate": (a > 0.55).mean(),
            "MeanBalancedAccuracy": group["BalancedAccuracy"].mean(),
        })

    monthly_stability = pd.DataFrame(stability_rows).sort_values("Month")

    print()
    print("=" * 100)
    print("TAKVİM AYINA GÖRE TUTARLILIK — 2016-2026")
    print("=" * 100)
    print()
    s = monthly_stability.copy()
    for c in ["AUC_gt_050_Rate", "AUC_gt_055_Rate", "MeanBalancedAccuracy"]:
        s[c] *= 100
    print(s.to_string(index=False, formatters={
        "MeanAUC": lambda x: f"{x:.4f}",
        "MedianAUC": lambda x: f"{x:.4f}",
        "MinAUC": lambda x: f"{x:.4f}",
        "MaxAUC": lambda x: f"{x:.4f}",
        "AUC_gt_050_Rate": lambda x: f"%{x:.1f}",
        "AUC_gt_055_Rate": lambda x: f"%{x:.1f}",
        "MeanBalancedAccuracy": lambda x: f"%{x:.2f}",
    }))

    monthly_base.to_csv("data/base_30d_monthly_2016_2026.csv", index=False)
    auc_pivot.to_csv("data/base_30d_auc_matrix_2016_2026.csv")
    monthly_stability.to_csv("data/base_30d_monthly_stability_2016_2026.csv", index=False)


    # ============================================================
    # BASE MODEL — REGIME ANALYSIS
    # ============================================================

    print()
    print("=" * 100)
    print("BASE MODEL — REGIME ANALYSIS — 2016-2026")
    print("=" * 100)

    base_regime = predictions[predictions["Model"] == "BASE"].copy()
    base_regime["Date"] = pd.to_datetime(base_regime["Date"])

    regime_columns = [
        "Date", "Ticker", "MarketRegime",
        "market_return_20d", "market_return_60d",
        "market_volatility_20d", "market_sma_200_distance",
    ]

    regime_meta = (
        common_df[regime_columns]
        .drop_duplicates(["Date", "Ticker"])
        .copy()
    )

    base_regime = base_regime.merge(
        regime_meta,
        on=["Date", "Ticker"],
        how="left",
        validate="many_to_one",
    )

    def regime_metrics(group):
        return pd.Series({
            "Samples": len(group),
            "Accuracy": accuracy_score(group["actual"], group["prediction"]),
            "BalancedAccuracy": balanced_accuracy_score(group["actual"], group["prediction"]),
            "AUC": safe_auc(group["actual"].values, group["probability"].values),
            "ActualUpRate": group["actual"].mean(),
            "PredictedUpRate": group["prediction"].mean(),
            "MeanProbability": group["probability"].mean(),
        })

    def print_regime_table(title, table):
        print()
        print("-" * 100)
        print(title)
        print("-" * 100)
        print()
        d = table.copy()
        for col in ["Accuracy", "BalancedAccuracy", "ActualUpRate", "PredictedUpRate", "MeanProbability"]:
            if col in d.columns:
                d[col] = d[col] * 100
        print(d.to_string(
            index=False,
            formatters={
                "Accuracy": lambda x: f"%{x:.2f}",
                "BalancedAccuracy": lambda x: f"%{x:.2f}",
                "AUC": lambda x: f"{x:.4f}" if pd.notna(x) else "NaN",
                "ActualUpRate": lambda x: f"%{x:.2f}",
                "PredictedUpRate": lambda x: f"%{x:.2f}",
                "MeanProbability": lambda x: f"%{x:.2f}",
            },
        ))

    # 1) BULL / NEUTRAL / BEAR
    market_regime_summary = (
        base_regime.dropna(subset=["MarketRegime"])
        .groupby("MarketRegime", observed=False)
        .apply(regime_metrics)
        .reset_index()
    )
    print_regime_table("1) MARKET REGIME — BULL / NEUTRAL / BEAR", market_regime_summary)

    # 2) 20D momentum
    base_regime["Momentum20Bucket"] = pd.cut(
        base_regime["market_return_20d"],
        bins=[-np.inf, -0.05, 0.05, np.inf],
        labels=["NEGATIVE (<-5%)", "NEUTRAL (-5%..+5%)", "POSITIVE (>+5%)"],
        include_lowest=True,
    )
    momentum20_summary = (
        base_regime.dropna(subset=["Momentum20Bucket"])
        .groupby("Momentum20Bucket", observed=False)
        .apply(regime_metrics)
        .reset_index()
    )
    print_regime_table("2) MARKET 20D MOMENTUM", momentum20_summary)

    # 3) 60D momentum
    base_regime["Momentum60Bucket"] = pd.cut(
        base_regime["market_return_60d"],
        bins=[-np.inf, -0.10, 0.10, np.inf],
        labels=["NEGATIVE (<-10%)", "NEUTRAL (-10%..+10%)", "POSITIVE (>+10%)"],
        include_lowest=True,
    )
    momentum60_summary = (
        base_regime.dropna(subset=["Momentum60Bucket"])
        .groupby("Momentum60Bucket", observed=False)
        .apply(regime_metrics)
        .reset_index()
    )
    print_regime_table("3) MARKET 60D MOMENTUM", momentum60_summary)

    # 4) Volatility terciles
    vol_valid = base_regime["market_volatility_20d"].dropna()
    if len(vol_valid):
        vol_q33 = vol_valid.quantile(1/3)
        vol_q67 = vol_valid.quantile(2/3)
        base_regime["VolatilityBucket"] = pd.cut(
            base_regime["market_volatility_20d"],
            bins=[-np.inf, vol_q33, vol_q67, np.inf],
            labels=["LOW", "MEDIUM", "HIGH"],
            include_lowest=True,
        )
        volatility_summary = (
            base_regime.dropna(subset=["VolatilityBucket"])
            .groupby("VolatilityBucket", observed=False)
            .apply(regime_metrics)
            .reset_index()
        )
        print()
        print(f"Volatility cutoffs: LOW <= {vol_q33:.6f}, MEDIUM <= {vol_q67:.6f}, HIGH > {vol_q67:.6f}")
        print_regime_table("4) MARKET VOLATILITY 20D", volatility_summary)
    else:
        volatility_summary = pd.DataFrame()

    # 5) SMA200 distance
    base_regime["SMA200Bucket"] = pd.cut(
        base_regime["market_sma_200_distance"],
        bins=[-np.inf, -0.10, -0.02, 0.02, 0.10, np.inf],
        labels=[
            "FAR BELOW (<-10%)",
            "BELOW (-10%..-2%)",
            "NEAR (-2%..+2%)",
            "ABOVE (+2%..+10%)",
            "FAR ABOVE (>+10%)",
        ],
        include_lowest=True,
    )
    sma200_summary = (
        base_regime.dropna(subset=["SMA200Bucket"])
        .groupby("SMA200Bucket", observed=False)
        .apply(regime_metrics)
        .reset_index()
    )
    print_regime_table("5) MARKET SMA200 DISTANCE", sma200_summary)

    # 6) Combined MarketRegime x Volatility
    if "VolatilityBucket" in base_regime.columns:
        combined_summary = (
            base_regime.dropna(subset=["MarketRegime", "VolatilityBucket"])
            .groupby(["MarketRegime", "VolatilityBucket"], observed=False)
            .apply(regime_metrics)
            .reset_index()
        )
        combined_summary = combined_summary[combined_summary["Samples"] >= 300].copy()
        combined_summary = combined_summary.sort_values(["AUC", "BalancedAccuracy"], ascending=False)
        print_regime_table("6) COMBINED — MARKET REGIME x VOLATILITY (Samples >= 300)", combined_summary)
    else:
        combined_summary = pd.DataFrame()

    # Global comparison
    frames = []

    def add_comp(table, typ, label_col):
        if table is None or table.empty:
            return
        t = table.copy()
        t["RegimeType"] = typ
        t["Regime"] = t[label_col].astype(str)
        frames.append(t[[
            "RegimeType", "Regime", "Samples", "Accuracy",
            "BalancedAccuracy", "AUC", "ActualUpRate",
            "PredictedUpRate", "MeanProbability"
        ]])

    add_comp(market_regime_summary, "MarketRegime", "MarketRegime")
    add_comp(momentum20_summary, "Momentum20", "Momentum20Bucket")
    add_comp(momentum60_summary, "Momentum60", "Momentum60Bucket")
    add_comp(volatility_summary, "Volatility20", "VolatilityBucket")
    add_comp(sma200_summary, "SMA200Distance", "SMA200Bucket")

    regime_comparison = pd.concat(frames, ignore_index=True)
    regime_comparison = regime_comparison[regime_comparison["Samples"] >= 500].copy()

    print()
    print("=" * 100)
    print("EN GÜÇLÜ BASE MODEL KOŞULLARI")
    print("=" * 100)
    print()
    top10 = regime_comparison.sort_values(["AUC", "BalancedAccuracy"], ascending=False).head(10)
    print(top10[["RegimeType", "Regime", "Samples", "AUC", "BalancedAccuracy"]].to_string(
        index=False,
        formatters={
            "AUC": lambda x: f"{x:.4f}",
            "BalancedAccuracy": lambda x: f"%{100*x:.2f}",
        },
    ))

    print()
    print("=" * 100)
    print("EN ZAYIF BASE MODEL KOŞULLARI")
    print("=" * 100)
    print()
    bottom10 = regime_comparison.sort_values(["AUC", "BalancedAccuracy"], ascending=True).head(10)
    print(bottom10[["RegimeType", "Regime", "Samples", "AUC", "BalancedAccuracy"]].to_string(
        index=False,
        formatters={
            "AUC": lambda x: f"{x:.4f}",
            "BalancedAccuracy": lambda x: f"%{100*x:.2f}",
        },
    ))

    market_regime_summary.to_csv("data/base_30d_regime_market.csv", index=False)
    momentum20_summary.to_csv("data/base_30d_regime_momentum20.csv", index=False)
    momentum60_summary.to_csv("data/base_30d_regime_momentum60.csv", index=False)
    if not volatility_summary.empty:
        volatility_summary.to_csv("data/base_30d_regime_volatility20.csv", index=False)
    sma200_summary.to_csv("data/base_30d_regime_sma200.csv", index=False)
    if not combined_summary.empty:
        combined_summary.to_csv("data/base_30d_regime_combined.csv", index=False)
    regime_comparison.to_csv("data/base_30d_regime_comparison.csv", index=False)
    base_regime.to_csv("data/base_30d_regime_predictions_enriched.csv", index=False)

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

    monthly_summary.to_csv(
        "data/gpr_ab_test_30d_2026_monthly_summary.csv",
        index=False,
    )

    monthly_comparison.to_csv(
        "data/gpr_ab_test_30d_2026_monthly_comparison.csv",
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
    print(
        "data/gpr_ab_test_30d_2026_monthly_summary.csv"
    )
    print(
        "data/gpr_ab_test_30d_2026_monthly_comparison.csv"
    )


if __name__ == "__main__":
    main()