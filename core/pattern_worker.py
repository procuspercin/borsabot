"""
Formasyon tespiti işçisi — AYRI SÜREÇ olarak çalışır.

torch + YOLO yüklenince süreç ~800 MB'a çıkıyor. Web sürecinin içinde
çalıştırılırsa bu bellek kalıcı olarak tutuluyor ve 2 GB'lık sunucuda yer
kalmıyor. Burada tarama yapılıp JSON stdout'a yazılır, süreç kapanır ve
bellek işletim sistemine geri döner.

Kullanım:
    python -m core.pattern_worker THYAO.IS 1y 1d
"""

from __future__ import annotations

import hashlib
import io
import json
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt          # noqa: E402
import mplfinance as mpf                 # noqa: E402
import numpy as np                       # noqa: E402
from PIL import Image                    # noqa: E402

from core.market import get_stock_data    # noqa: E402
from core.patterns import (               # noqa: E402
    DPI, IMAGE_DIR, IMAGE_SIZE, MIN_CONFIDENCE, MODEL_PATH, PATTERN_BIAS,
    PATTERN_TR, TORCH_THREADS,
)


def _render(df):
    """Mum grafiğini çizer; piksel → veri dönüşümlerini de döndürür."""
    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        marketcolors=mpf.make_marketcolors(up="#81c995", down="#f28b82",
                                           edge="inherit", wick="inherit"),
        facecolor="#1e1f20", edgecolor="#3c4043", figcolor="#1e1f20",
        gridcolor="#2c2d2f", gridstyle="-",
        rc={"axes.labelcolor": "#9aa0a6", "xtick.color": "#9aa0a6",
            "ytick.color": "#9aa0a6"},
    )
    fig, axes = mpf.plot(df, type="candle", style=style, volume=False,
                         figsize=IMAGE_SIZE, returnfig=True, axisoff=False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    ax = axes[0]
    height_px = fig.get_figheight() * DPI

    def px_to_index(x_px):
        return ax.transData.inverted().transform((x_px, 0))[0]

    def px_to_price(y_px):
        return ax.transData.inverted().transform((0, height_px - y_px))[1]

    plt.close(fig)
    buf.seek(0)
    return buf.getvalue(), px_to_index, px_to_price


def run(symbol: str, period: str, interval: str) -> dict:
    df = get_stock_data(symbol, period, interval)
    if df is None or df.empty:
        return {"error": "Bu sembol için veri bulunamadı."}

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if len(df) < 30:
        return {"error": "Formasyon araması için yeterli veri yok."}

    png, px_to_index, px_to_price = _render(df)

    import torch
    torch.set_num_threads(TORCH_THREADS)
    from ultralytics import YOLO

    model = YOLO(str(MODEL_PATH))
    image = np.array(Image.open(io.BytesIO(png)).convert("RGB"))
    result = model(image, verbose=False, conf=MIN_CONFIDENCE)[0]

    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        name = result.names[int(box.cls)]
        i1 = max(0, min(len(df) - 1, int(round(px_to_index(x1)))))
        i2 = max(0, min(len(df) - 1, int(round(px_to_index(x2)))))
        if i2 < i1:
            i1, i2 = i2, i1
        low, high = sorted((px_to_price(y1), px_to_price(y2)))
        detections.append({
            "name": name,
            "name_tr": PATTERN_TR.get(name, name),
            "bias": PATTERN_BIAS.get(name, ""),
            "confidence": round(float(box.conf), 3),
            "start": df.index[i1].date().isoformat(),
            "end": df.index[i2].date().isoformat(),
            "low": round(low, 2),
            "high": round(high, 2),
            "box": [round(x1), round(y1), round(x2), round(y2)],
        })

    # Görseli dosyaya yaz; base64 olarak JSON'dan geçirmek hem web sürecinde
    # yüz MB'larca geçici bellek tutuyor hem de sayfaya yarım MB'lık data URI
    # gömüyordu.
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{symbol}|{period}|{interval}".encode()).hexdigest()[:16]
    (IMAGE_DIR / f"{key}.png").write_bytes(png)

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return {
        "error": None,
        "image_key": key,
        "width": int(IMAGE_SIZE[0] * DPI),
        "height": int(IMAGE_SIZE[1] * DPI),
        "detections": detections,
        "period": period,
    }


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Sembol verilmedi."}))
        sys.exit(1)
    symbol = sys.argv[1]
    period = sys.argv[2] if len(sys.argv) > 2 else "1y"
    interval = sys.argv[3] if len(sys.argv) > 3 else "1d"
    try:
        print(json.dumps(run(symbol, period, interval)))
    except Exception as exc:
        print(json.dumps({"error": f"Tarama başarısız: {exc}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
