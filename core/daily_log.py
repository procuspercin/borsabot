"""
Günlük Kayıt / Excel Tablosu modülü.

- SQLite ile kalıcı saklama (data/daily_log.db).
- Aynı (tarih, hisse) çifti için tekrar eklenmez (PRIMARY KEY).
- Eski tarihler asla silinmez veya üzerine yazılmaz (overwrite seçeneği opsiyoneldir).
- Excel dosyasından hisse kodlarını esnek parse eder.
"""

from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from zoneinfo import ZoneInfo  # Py3.9+
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore

# ------------------------------------------------------------------ #
#  Sabitler
# ------------------------------------------------------------------ #

ISTANBUL_TZ = ZoneInfo("Europe/Istanbul") if ZoneInfo else None

# Proje köküne göre veri klasörü
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "daily_log.db"

# Excel dosyasının varsayılan yolu (proje köküne taşındı)
DEFAULT_EXCEL_PATH = PROJECT_ROOT / "Başlıksız e-tablo.xlsx"

# Her zaman ekstra eklenecek hisseler
ALWAYS_INCLUDE = ["SISE", "ASELS"]

# Excel'de hisse kodu olamayacak token'ları filtrelemek için kara liste
_BLACKLIST = {
    "NAN", "NONE", "TRUE", "FALSE", "AÇILIŞ", "KAPANIŞ",
    "MACD", "RSI", "FIB", "ATR", "ADX", "ICHI", "STOCH",
}

# Tablo kolonları (DB ve UI için ortak referans)
COLUMNS = [
    "Tarih",
    "Hisse",
    "Kapanış",
    "Açılış",
    "Yüksek",
    "Düşük",
    "MA Sinyali",
    "MACD Sinyali",
    "RSI",
    "Bollinger Sinyali",
    "Stokastik",
    "Ichimoku",
    "Genel Sinyal",
    "Kaydedilme Zamanı",
]

_DB_COLUMNS = [
    "tarih", "hisse", "kapanis", "acilis", "yuksek", "dusuk",
    "ma_sinyal", "macd_sinyal", "rsi", "bollinger_sinyal",
    "stokastik", "ichimoku", "genel_sinyal", "kaydedilme_zamani",
]


# ------------------------------------------------------------------ #
#  Zaman yardımcıları
# ------------------------------------------------------------------ #

def now_istanbul() -> datetime:
    """Europe/Istanbul TZ'sinde şu anki datetime."""
    if ISTANBUL_TZ is not None:
        return datetime.now(ISTANBUL_TZ)
    # Fallback: UTC+3 sabit
    from datetime import timezone, timedelta
    return datetime.now(timezone(timedelta(hours=3)))


def today_istanbul() -> date:
    return now_istanbul().date()


# ------------------------------------------------------------------ #
#  DB işlemleri
# ------------------------------------------------------------------ #

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Tabloyu oluşturur (varsa atlar)."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_log (
                tarih               TEXT NOT NULL,
                hisse               TEXT NOT NULL,
                kapanis             REAL,
                acilis              REAL,
                yuksek              REAL,
                dusuk               REAL,
                ma_sinyal           TEXT,
                macd_sinyal         TEXT,
                rsi                 REAL,
                bollinger_sinyal    TEXT,
                stokastik           TEXT,
                ichimoku            TEXT,
                genel_sinyal        TEXT,
                kaydedilme_zamani   TEXT,
                PRIMARY KEY (tarih, hisse)
            )
            """
        )
        conn.commit()


def insert_records(records: Iterable[dict], overwrite: bool = False) -> tuple[int, int]:
    """
    Kayıt listesini DB'ye ekler.

    records: her biri DB kolon adlarıyla (tarih, hisse, kapanis, ...) dict.

    overwrite=False  -> aynı (tarih, hisse) varsa atlar (eski tarihler korunur).
    overwrite=True   -> SADECE bugünün satırları için var olan satırı günceller; geçmişe dokunmaz.

    Returns: (eklenen_sayisi, atlanan_sayisi)
    """
    init_db()
    inserted = 0
    skipped = 0
    today_str = today_istanbul().isoformat()

    placeholders = ", ".join(["?"] * len(_DB_COLUMNS))
    cols_sql = ", ".join(_DB_COLUMNS)

    with _connect() as conn:
        cur = conn.cursor()
        for r in records:
            values = [r.get(c) for c in _DB_COLUMNS]

            # Güvenlik: eski tarihleri yanlışlıkla overwrite etmemek için
            # overwrite SADECE bugünün kaydı için izinli.
            if overwrite and r.get("tarih") == today_str:
                cur.execute(
                    f"INSERT OR REPLACE INTO daily_log ({cols_sql}) VALUES ({placeholders})",
                    values,
                )
                inserted += 1
            else:
                cur.execute(
                    f"INSERT OR IGNORE INTO daily_log ({cols_sql}) VALUES ({placeholders})",
                    values,
                )
                if cur.rowcount > 0:
                    inserted += 1
                else:
                    skipped += 1
        conn.commit()
    return inserted, skipped


def fetch_all() -> pd.DataFrame:
    """Tüm kayıtları, en yeni tarih önce gelecek şekilde döndürür."""
    init_db()
    with _connect() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM daily_log ORDER BY tarih DESC, hisse ASC",
            conn,
        )

    if df.empty:
        return pd.DataFrame(columns=COLUMNS)

    rename_map = dict(zip(_DB_COLUMNS, COLUMNS))
    df = df.rename(columns=rename_map)
    return df[COLUMNS]


def has_record_for_today(symbol: str | None = None) -> bool:
    """Bugün için (opsiyonel olarak belirli bir hisse için) kayıt var mı?"""
    init_db()
    today_str = today_istanbul().isoformat()
    with _connect() as conn:
        if symbol:
            cur = conn.execute(
                "SELECT 1 FROM daily_log WHERE tarih = ? AND hisse = ? LIMIT 1",
                (today_str, symbol.upper().replace(".IS", "")),
            )
        else:
            cur = conn.execute(
                "SELECT 1 FROM daily_log WHERE tarih = ? LIMIT 1",
                (today_str,),
            )
        return cur.fetchone() is not None


def distinct_dates() -> list[str]:
    init_db()
    with _connect() as conn:
        cur = conn.execute("SELECT DISTINCT tarih FROM daily_log ORDER BY tarih DESC")
        return [row[0] for row in cur.fetchall()]


def distinct_symbols() -> list[str]:
    init_db()
    with _connect() as conn:
        cur = conn.execute("SELECT DISTINCT hisse FROM daily_log ORDER BY hisse ASC")
        return [row[0] for row in cur.fetchall()]


def total_record_count() -> int:
    init_db()
    with _connect() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM daily_log")
        return int(cur.fetchone()[0])


# ------------------------------------------------------------------ #
#  Excel parsing (esnek)
# ------------------------------------------------------------------ #

_TICKER_PATTERN = re.compile(r"^[A-ZÇĞİÖŞÜ]{3,6}$")


def _normalize_token(token: str) -> str | None:
    """Bir hücre/sütun adını hisse koduna normalize eder, geçersizse None döner."""
    if token is None:
        return None
    text = str(token).strip().upper()
    text = text.replace(".IS", "").replace("BIST:", "").strip()

    # Tirelenmiş veya boşluklu ifadeleri ele
    if not text or any(ch in text for ch in [" ", "-", "/", ":", "(", ")", "%"]):
        return None
    if text in _BLACKLIST:
        return None
    if not _TICKER_PATTERN.match(text):
        return None
    return text


def parse_symbols_from_excel(path: str | os.PathLike | None = None) -> list[str]:
    """
    Excel dosyasındaki hisse kodlarını esnek şekilde çıkarır.

    Strateji:
    1. Tüm sayfaları (sheet) gez.
    2. Hem sütun başlıklarına hem ham hücrelere bak.
    3. 3-6 karakterli, sadece harf, kara listede olmayan token'ları topla.

    Dosya bulunamazsa boş liste döner.
    """
    excel_path = Path(path) if path else DEFAULT_EXCEL_PATH
    if not excel_path.exists():
        return []

    found: set[str] = set()

    try:
        xl = pd.ExcelFile(excel_path)
    except Exception:
        return []

    for sheet in xl.sheet_names:
        # Header'lı oku (bizim Excel'imizde sütun isimleri hisse kodları)
        try:
            df_h = pd.read_excel(xl, sheet_name=sheet)
            for col in df_h.columns:
                t = _normalize_token(col)
                if t:
                    found.add(t)
        except Exception:
            pass

        # Header'sız oku (kullanıcı başka bir formatta yazmış olabilir)
        try:
            df_n = pd.read_excel(xl, sheet_name=sheet, header=None)
            for col in df_n.columns:
                series = df_n[col].dropna()
                # Çok uzun sütunlarda performans için ilk 50 hücreye bak
                for v in series.head(50):
                    t = _normalize_token(v)
                    if t:
                        found.add(t)
        except Exception:
            pass

    return sorted(found)


def get_target_symbols(excel_path: str | os.PathLike | None = None) -> list[str]:
    """
    Günlük kayıt için kullanılacak hisse listesini döndürür:
    - Excel'den okunan hisseler + ALWAYS_INCLUDE (tekrarsız, sıralı)
    """
    symbols = set(parse_symbols_from_excel(excel_path))
    symbols.update(s.upper() for s in ALWAYS_INCLUDE)
    return sorted(symbols)


# ------------------------------------------------------------------ #
#  Bakım
# ------------------------------------------------------------------ #

def export_to_csv(target: str | os.PathLike) -> Path:
    """Tüm kayıtları CSV olarak dışa aktarır."""
    df = fetch_all()
    target_path = Path(target)
    df.to_csv(target_path, index=False, encoding="utf-8-sig")
    return target_path


def export_to_xlsx(target: str | os.PathLike) -> Path:
    """Tüm kayıtları XLSX olarak dışa aktarır."""
    df = fetch_all()
    target_path = Path(target)
    df.to_excel(target_path, index=False)
    return target_path
