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

import importlib.util
import json
import subprocess
import sys

from core.market import PROJECT_ROOT, ttl_cache

# ultralytics/torch/matplotlib ağır ve isteğe bağlı olduğu için modül
# seviyesinde import EDİLMEZ; yoksa paketler kurulu olmadığında uygulama hiç
# açılmaz. Hepsi available() ve detect() içinde tembel yüklenir.

MODEL_PATH = PROJECT_ROOT / "models" / "stockmarket-yolov8.pt"
IMAGE_DIR = PROJECT_ROOT / "data" / "pattern-cache"
MIN_CONFIDENCE = 0.30
TORCH_THREADS = 1                          # küçük sunucuda bellek ve CPU için
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


def available() -> bool:
    """
    Model dosyası ve gerekli paketler yerinde mi?

    Paketleri gerçekten import ETMEZ: ultralytics modül seviyesinde torch'u
    yüklüyor ve bu, alt süreç izolasyonunu boşa çıkarıp ana sürece ~200 MB
    ekliyordu. find_spec modülün varlığını çalıştırmadan kontrol eder.
    """
    if not MODEL_PATH.exists():
        return False
    return all(importlib.util.find_spec(name) is not None
               for name in ("mplfinance", "ultralytics", "PIL"))


@ttl_cache(600)
def detect(symbol: str, period: str = "1y", interval: str = "1d") -> dict:
    """
    Formasyonları arar. Tarama AYRI SÜREÇTE yapılır: torch + YOLO yüklenince
    süreç ~800 MB'a çıkıyor ve web sürecinde tutulursa 2 GB'lık sunucuda yer
    kalmıyor. Alt süreç bitince bellek işletim sistemine geri dönüyor.

    Dönen sözlük:
      available  : özellik kullanılabilir mi
      image      : data: URI olarak mum grafiği
      detections : [{name, name_tr, bias, confidence, start, end, low, high}]
    """
    if not available():
        return {"available": False, "detections": [], "image": None,
                "error": "Formasyon modeli bu sunucuda etkin değil."}

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "core.pattern_worker", symbol, period, interval],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=180,
        )
    except subprocess.TimeoutExpired:
        return {"available": True, "detections": [], "image": None,
                "error": "Tarama zaman aşımına uğradı."}

    out = (proc.stdout or "").strip()
    if not out:
        detail = (proc.stderr or "").strip().splitlines()[-1:] or ["çıktı yok"]
        return {"available": True, "detections": [], "image": None,
                "error": f"Tarama başarısız: {detail[0][:160]}"}

    try:
        data = json.loads(out.splitlines()[-1])
    except json.JSONDecodeError:
        return {"available": True, "detections": [], "image": None,
                "error": "Tarama çıktısı okunamadı."}

    data["available"] = True
    data.setdefault("detections", [])
    return data


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
