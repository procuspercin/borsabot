# BorsaBot Finans — VPS kurulumu

Hedef: uygulama arka planda sürekli çalışsın, kendi alan adından HTTPS ile açılsın,
şifresiz kimse erişemesin, gizli dosyalar (API anahtarı, veritabanı) dışarı sızmasın.

Varsayım: Ubuntu 22.04/24.04 veya Debian 12, nginx, sudo yetkin var.

---

## 1. Alan adı

Herhangi bir kayıt firmasından al (Cloudflare, Namecheap, Gandi, natro vb.).
Sonra DNS panelinde tek kayıt yeterli:

| Tip | Ad | Değer |
|-----|-----|-------|
| A | `borsabot` (veya `@`) | VPS'in IPv4 adresi |

Cloudflare kullanıyorsan turuncu bulutu **kapalı** (DNS only) bırak; certbot ilk
sertifikayı böyle daha sorunsuz alır, sonra istersen açarsın.

Yayılmayı doğrula:

```bash
dig +short borsabot.ornek.com
```

## 2. Sistem hazırlığı

```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv git nginx apache2-utils ufw
```

**Python sürümü önemli:** `requirements.txt` numpy 1.23.5'e sabitli ve bu sürümün
Python 3.12 için hazır paketi yok. Ubuntu 24.04 kullanıyorsan `python3.11` depoda
olmayabilir; o zaman önce:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt update
sudo apt install -y python3.11 python3.11-venv
```

Servis kullanıcısı ve boş uygulama dizini:

```bash
sudo useradd --system --home-dir /opt/borsabot --shell /usr/sbin/nologin borsabot
sudo mkdir -p /opt/borsabot
sudo chown borsabot:borsabot /opt/borsabot
```

Güvenlik duvarı — sadece SSH ve web açık, Streamlit'in 8501 portu dışarı kapalı:

```bash
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw enable
```

## 3. Kodu yerleştir

```bash
# dizin boş olduğu için clone doğrudan içine yapılabilir
sudo -u borsabot git clone https://github.com/kerempercin/borsabot.git /opt/borsabot
cd /opt/borsabot
sudo -u borsabot python3.11 -m venv .venv
sudo -u borsabot .venv/bin/pip install -r requirements.txt
```

## 4. Gizli dosyalar

API anahtarı repoda yok, sunucuda elle oluşturulur:

```bash
sudo -u borsabot tee /opt/borsabot/.streamlit/secrets.toml >/dev/null <<'EOF'
GEMINI_API_KEY = "buraya-anahtarin"
EOF
sudo chmod 600 /opt/borsabot/.streamlit/secrets.toml
```

## 5. ML tahmin modelleri (opsiyonel)

Modeller ve ham veri repoda yok (boyut nedeniyle). ML sekmesini sunucuda da
istiyorsan kendi bilgisayarından kopyala:

```bash
# yerel makinende
rsync -avz --progress \
  stock_forecaster/models/ \
  kullanici@sunucu:/tmp/models/
rsync -avz --progress \
  stock_forecaster/data/raw/ \
  kullanici@sunucu:/tmp/raw/

# sunucuda
sudo mkdir -p /opt/borsabot/stock_forecaster/{models,data/raw}
sudo cp -r /tmp/models/* /opt/borsabot/stock_forecaster/models/
sudo cp -r /tmp/raw/*    /opt/borsabot/stock_forecaster/data/raw/
sudo chown -R borsabot:borsabot /opt/borsabot/stock_forecaster
```

Forecaster kendi sanal ortamıyla çalışır (kütüphane sürümleri ana uygulamadan farklı):

```bash
cd /opt/borsabot/stock_forecaster
sudo -u borsabot python3.11 -m venv .venv
sudo -u borsabot .venv/bin/pip install -r requirements.txt
```

Bu adımı atlarsan uygulama çalışır, sadece ML Tahmin sekmesi "model bulunamadı" der.

## 6. Servisi başlat

```bash
sudo cp /opt/borsabot/deploy/borsabot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now borsabot
systemctl status borsabot
```

Log takibi:

```bash
journalctl -u borsabot -f
```

## 7. Nginx + şifre + HTTPS

Şifre dosyası (kullanıcı adını istediğin gibi seç):

```bash
sudo htpasswd -c /etc/nginx/.htpasswd-borsabot kerem
```

Site tanımı:

```bash
sudo cp /opt/borsabot/deploy/nginx-borsabot.conf /etc/nginx/sites-available/borsabot
sudo sed -i 's/borsabot.ornek.com/borsabot.SENIN-ALANIN.com/' /etc/nginx/sites-available/borsabot
sudo ln -s /etc/nginx/sites-available/borsabot /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```

Sertifika:

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d borsabot.SENIN-ALANIN.com --redirect
```

Certbot 443 bloğunu ve otomatik yenilemeyi kendisi kurar.

## 8. Güncelleme

```bash
cd /opt/borsabot
sudo -u borsabot git pull
sudo -u borsabot .venv/bin/pip install -r requirements.txt
sudo systemctl restart borsabot
```

---

## Veriler neden public değil

- Streamlit yalnızca `127.0.0.1:8501`'i dinler; 8501 portu internete kapalı, ufw de engelliyor.
- Tek giriş nginx üzerinden ve HTTP Basic Auth ile şifreli; şifre bilmeyen uygulamayı hiç görmez.
- Tüm trafik Let's Encrypt sertifikasıyla HTTPS.
- `secrets.toml` (Gemini anahtarı), `data/daily_log.db` ve Excel dosyası sunucunun diskinde kalır,
  hiçbir HTTP yolundan servis edilmez — nginx yalnızca Streamlit'e proxy yapar, dosya sunmaz.
- `.gitignore` bu dosyaları repodan uzak tutar.

Daha sıkı isteyen için alternatifler: Cloudflare Tunnel + Access (Google hesabıyla giriş,
sunucunun IP'si hiç görünmez) ya da `--server.address 127.0.0.1` + yalnızca SSH tüneli
(`ssh -L 8501:127.0.0.1:8501 sunucu`) — bu durumda alan adına bile gerek kalmaz.

## Bilinen sınır: 18:30 otomatik kaydı

Günlük kayıt, sayfa açıkken çalışan bir kontrole bağlı (`maybe_run_auto_capture`).
Sunucuda tarayıcı açık olmadığı için tetiklenmez. Kaydın her akşam kendiliğinden
alınmasını istiyorsan bunu bir cron işine bağlamak gerekir — söyle, küçük bir
komut satırı betiği yazayım.
