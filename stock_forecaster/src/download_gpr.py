from pathlib import Path
import pandas as pd

URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"

OUTPUT = Path("data/raw/gpr_daily.csv")


def main():
    print("GPR verisi indiriliyor...")

    df = pd.read_excel(URL)

    print("\nKolonlar:")
    for c in df.columns:
        print("-", c)

    # Tarih kolonunu bul
    date_col = None

    for candidate in [
        "date",
        "Date",
        "DATE",
    ]:
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:
        raise ValueError(
            "Tarih kolonu bulunamadı."
        )

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce",
    )

    df = (
        df.dropna(subset=[date_col])
        .sort_values(date_col)
        .reset_index(drop=True)
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print("Kaydedildi:", OUTPUT)
    print(
        "Tarih aralığı:",
        df[date_col].min(),
        "→",
        df[date_col].max(),
    )


if __name__ == "__main__":
    main()