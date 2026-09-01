"""
Grafik formasyonu tespiti (YOLOv8).

foduucom/stockmarket-pattern-detection-yolov8 modelini kullanır: mum grafiği
görüntüsü üretilir, model üzerinde formasyon kutuları aranır, bulunan kutular
tarih ve fiyat aralığına geri çevrilir.

Bağımlılıklar (ultralytics/torch) ağır olduğu için isteğe bağlıdır. Kurulu
değilse ya da model dosyası yoksa özellik sessizce kapanır, uygulamanın geri
kalanı etkilenmez.
"""

from __future__ import annotations

import base64
import io
import threading

import pandas as pd

from core.market import PROJECT_ROOT, get_stock_data, ttl_cache

# ultralytics/torch/matplotlib ağır ve isteğe bağlı olduğu için modül
# seviyesinde import EDİLMEZ; yoksa paketler kurulu olmadığında uygulama hiç
# açılmaz. Hepsi available() ve detect() içinde tembel yüklenir.

MODEL_PATH = PROJECT_ROOT / "models" / "stockmarket-yolov8.pt"
MIN_CONFIDENCE = 0.30
IMAGE_SIZE = (10.24, 7.68)                 # inç; 100 dpi ile 1024x768
DPI = 100

# Model adları teknik ama arayüzde Türkçe görünmeli
PATTERN_TR = {
    "Head and shoulders top": "Omuz-Baş-Omuz (tepe)",
    "Head and shoulders bottom": "Ters Omuz-Baş-Omuz (dip)",
    "M_Head": "M Formasyonu (çift tepe)",
    "W_Bottom": "W Formasyonu (çift dip)",
    "Triangle": "Üçgen",
    "StockLine": "Yatay seyir",
}
# Formasyonun klasik yorumu — asistana bağlam olarak veriliyor
PATTERN_BIAS = {
    "Head and shoulders top": "düşüş yönlü dönüş formasyonu",
    "Head and shoulders bottom": "yükseliş yönlü dönüş formasyonu",
    "M_Head": "düşüş yönlü dönüş formasyonu",
    "W_Bottom": "yükseliş yönlü dönüş formasyonu",
    "Triangle": "sıkışma; kırılım yönü belirleyici",
    "StockLine": "yönsüz, yatay seyir",
}

_model_lock = threading.Lock()
_model = {"obj": None}


def available() -> bool:
    """Model dosyası ve gerekli paketler yerinde mi?"""
    if not MODEL_PATH.exists():
        return False
    try:
        import mplfinance  # noqa: F401
        import ultralytics  # noqa: F401
        from PIL import Image  # noqa: F401
        return True
    except ImportError:
        return False


def _load_model():
    with _model_lock:
        if _model["obj"] is None:
            from ultralytics import YOLO
            _model["obj"] = YOLO(str(MODEL_PATH))
        return _model["obj"]


def _render_chart(df: pd.DataFrame):
    """
    Mum grafiğini PNG'ye çizer. Kutuları tarihe çevirebilmek için figürü
    kırpmadan kaydeder ve eksenin piksel yerleşimini birlikte döndürür.
    """
    import matplotlib
    matplotlib.use("Agg")                  # sunucuda ekran yok
    import matplotlib.pyplot as plt
    import mplfinance as mpf

    # Arayüzle aynı koyu palet. Tespitler çizim stiline duyarlı; koyu tema hem
    # siteyle uyumlu hem de denemede daha temiz sonuç verdi (açık temada aynı
    # grafikte hem tepe hem dip OBO işaretlenebiliyordu).
    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        marketcolors=mpf.make_marketcolors(up="#81c995", down="#f28b82",
                                           edge="inherit", wick="inherit"),
        facecolor="#1e1f20", edgecolor="#3c4043", figcolor="#1e1f20",
        gridcolor="#2c2d2f", gridstyle="-",
        rc={"axes.labelcolor": "#9aa0a6", "xtick.color": "#9aa0a6",
            "ytick.color": "#9aa0a6"},
    )
    fig, axes = mpf.plot(
        df, type="candle", style=style, volume=False,
        figsize=IMAGE_SIZE, returnfig=True, axisoff=False,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    ax = axes[0]
    height_px = fig.get_figheight() * DPI

    def px_to_index(x_px: float) -> float:
        """Görüntü pikselini mum sırasına (x veri koordinatı) çevirir."""
        x_data, _ = ax.transData.inverted().transform((x_px, 0))
        return x_data

    def px_to_price(y_px: float) -> float:
        """Görüntü y pikseli üstten sayılır, matplotlib alttan; çeviriyoruz."""
        _, y_data = ax.transData.inverted().transform((0, height_px - y_px))
        return y_data

    plt.close(fig)
    buf.seek(0)
    return buf.getvalue(), px_to_index, px_to_price


@ttl_cache(600)
def detect(symbol: str, period: str = "1y", interval: str = "1d") -> dict:
    """
    Formasyonları arar. Dönen sözlük:
      available  : özellik kullanılabilir mi
      image      : data: URI olarak mum grafiği
      detections : [{name, name_tr, bias, confidence, start, end, low, high}]
    """
    if not available():
        return {"available": False, "detections": [], "image": None,
                "error": "Formasyon modeli kurulu değil."}

    df = get_stock_data(symbol, period, interval)
    if df is None or df.empty:
        return {"available": True, "detections": [], "image": None,
                "error": "Bu sembol için veri bulunamadı."}

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if len(df) < 30:
        return {"available": True, "detections": [], "image": None,
                "error": "Formasyon araması için yeterli veri yok."}

    png, px_to_index, px_to_price = _render_chart(df)

    import numpy as np
    from PIL import Image
    image = Image.open(io.BytesIO(png)).convert("RGB")
    result = _load_model()(np.array(image), verbose=False, conf=MIN_CONFIDENCE)[0]

    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        name = result.names[int(box.cls)]
        i1 = max(0, min(len(df) - 1, int(round(px_to_index(x1)))))
        i2 = max(0, min(len(df) - 1, int(round(px_to_index(x2)))))
        if i2 < i1:
            i1, i2 = i2, i1
        prices = sorted((px_to_price(y1), px_to_price(y2)))
        detections.append({
            "name": name,
            "name_tr": PATTERN_TR.get(name, name),
            "bias": PATTERN_BIAS.get(name, ""),
            "confidence": round(float(box.conf), 3),
            "start": df.index[i1].date().isoformat(),
            "end": df.index[i2].date().isoformat(),
            "low": round(prices[0], 2),
            "high": round(prices[1], 2),
            "box": [round(x1), round(y1), round(x2), round(y2)],
        })

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return {
        "available": True,
        "error": None,
        "image": "data:image/png;base64," + base64.b64encode(png).decode(),
        "width": int(IMAGE_SIZE[0] * DPI),
        "height": int(IMAGE_SIZE[1] * DPI),
        "detections": detections,
        "period": period,
    }


def as_context(symbol: str, result: dict) -> str:
    """Tespitleri asistanın okuyabileceği düz metne çevirir."""
    if not result.get("detections"):
        return ""
    short = symbol.replace(".IS", "")
    lines = [f"{short} grafiğinde görüntü tanıma modeliyle bulunan formasyonlar "
             f"({result.get('period', '1y')} periyot):"]
    for d in result["detections"]:
        lines.append(
            f"- {d['name_tr']} ({d['bias']}), güven %{d['confidence'] * 100:.0f}, "
            f"{d['start']} – {d['end']} arası, yaklaşık {d['low']}–{d['high']} bandında."
        )
    lines.append("Model mAP@0.5 değeri 0.614; tespitler kesinlik değil, işaret olarak değerlendirilmeli.")
    return "\n".join(lines)
