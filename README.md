# 📈 Borsa Teknik Analiz ve Haber Platformu

Bu proje, BIST 100 hisseleri için teknik analiz yapan ve Bloomberg HT üzerinden son dakika finans haberlerini çeken modern bir Streamlit uygulamasıdır.

## 🚀 Özellikler

### 📊 Piyasa Analizi
- **Teknik İndikatörler**: MA, MACD, Bollinger Bantları, RSI, Stokastik, Fibonacci, Ichimoku, Standart Sapma.
- **İnteraktif Grafikler**: Plotly ile detaylı mum grafikleri ve indikatör çizimleri.
- **Günlük Özet**: Seçilen hissenin günlük açılış, kapanış, yüksek ve düşük değerleri.
- **Sinyal Sistemi**: İndikatörlere dayalı AL/SAT sinyalleri.

### 📰 Haberler (YENİ)
- **Canlı Akış**: Bloomberg HT RSS üzerinden anlık borsa ve finans haberleri.
- **Görsel Kartlar**: Haberler, görselleri ve özetleriyle birlikte modern kart yapısında sunulur.

## 🛠 Kurulum

1. Repoyu klonlayın:
```bash
git clone https://github.com/kullaniciadi/borsabot.git
cd borsabot
```

2. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

3. Uygulamayı çalıştırın:
```bash
streamlit run app.py
```

## 📦 Gereksinimler
- Python 3.8+
- streamlit
- yfinance
- pandas
- plotly
- beautifulsoup4
- requests

## 📝 Lisans
MIT License