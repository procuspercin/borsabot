# 📈 BorsaBot Finans

BIST hisseleri için teknik analiz, Gemini destekli yorum asistanı ve makine
öğrenmesi tabanlı fiyat beklentisi üreten bir web uygulaması.

## Özellikler

**Piyasa** — BIST/döviz/emtia/kripto/küresel/vadeli sekmeleri, mini grafikli
kartlar, izleme listesi, sektör endeksleri, en çok yükselen/düşenler ve
Bloomberg HT haber akışı.

**Teknik analiz** — MA, MACD, RSI, Bollinger, Stokastik, Ichimoku, CCI,
Fibonacci ve standart sapma; mum grafiği üzerinde indikatör katmanları ve
AL/SAT/BEKLE sinyalleri. Hisseye tıklayınca teknik özet penceresi açılır.

**Asistan** — Gemini, hesaplanan indikatör ve fiyat seviyelerini bağlam olarak
alıp somut destek/direnç yorumu yapar. Dakikada 10, günde 200 istek limiti
uygulanır.

**ML tahmin** — RandomForest yön modelleri (10/30/60/120/180 işlem günü) ve
geçmiş olasılık kalibrasyonuyla beklenen fiyat aralığı (P25–P75).

**Günlük kayıt** — Takip edilen hisselerin günlük kapanış ve sinyalleri SQLite'a
yazılır; tablo filtrelenip dışa aktarılabilir. 18:30 sonrası otomatik çalışır.

## Mimari

```
core/market.py    veri ve analiz katmanı (fiyat çekme, indikatörler, Gemini, ML köprüsü)
core/charts.py    plotly figürleri (JSON olarak tarayıcıya gider)
core/daily_log.py günlük kayıt veritabanı
web/main.py       FastAPI yönlendirmeleri
web/templates/    Jinja şablonları (htmx ile parça güncelleme)
stock_forecaster/ ML modelleri ve eğitim betikleri (kendi sanal ortamıyla çalışır)
```

Fiyatlar Yahoo'nun toplu `spark` uç noktasından çekilir: tek istekte ~20
sembol, ana sayfanın tamamı için ~0,5 sn. Sonuçlar süreç içi TTL önbelleğinde
tutulduğu için tekrar eden istekler milisaniye mertebesindedir.

## Kurulum

```bash
git clone https://github.com/kerempercin/borsabot.git
cd borsabot
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Gemini anahtarı `.streamlit/secrets.toml` içinde `GEMINI_API_KEY = "..."` olarak
ya da `GEMINI_API_KEY` ortam değişkeni olarak verilir (dosya repoda değildir).

Çalıştırma:

```bash
.venv/bin/uvicorn web.main:app --reload --port 8000
```

Sunucu kurulumu (systemd + nginx + HTTPS + şifre) için `deploy/README.md`.

## Gereksinimler

Python 3.11 · FastAPI · Jinja2 · yfinance · pandas · plotly · beautifulsoup4

ML tahmin sekmesi için `stock_forecaster/models` ve `stock_forecaster/data/raw`
gerekir; bunlar boyutları nedeniyle repoda tutulmaz.

## Uyarı

Üretilen sinyaller ve tahminler geçmiş fiyat verisinden hesaplanan istatistiksel
çıktılardır. Yatırım tavsiyesi değildir.

## Lisans

MIT
