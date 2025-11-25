import pandas as pd
from datetime import datetime, timedelta
import requests
import time
import random
from fake_useragent import UserAgent
import cloudscraper

class DataFetcher:
    def __init__(self, symbol, period="3mo", interval="1d"):
        self.symbol = symbol
        self.interval = interval
        self.base_url = "https://tr.investing.com"
        
        # Zaman aralığına göre period ayarla
        if interval == "15m":
            self.period = "5d"  # 15 dakikalık veriler için 5 gün
        elif interval == "1h":
            self.period = "1mo"  # Saatlik veriler için 1 ay
        elif interval == "4h":
            self.period = "3mo"  # 4 saatlik veriler için 3 ay
        else:  # 1d
            self.period = "1y"  # Günlük veriler için 1 yıl
            
        # Cloudscraper ve UserAgent oluştur
        self.scraper = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'mobile': False
            }
        )
        self.ua = UserAgent()
        
    def _get_headers(self):
        """Rastgele ve gerçekçi header'lar oluştur"""
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'tr-TR,tr;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
            'TE': 'Trailers',
        }
        
    def _convert_interval(self):
        """Interval formatını Investing.com formatına çevir"""
        interval_map = {
            "1d": "Daily",
            "4h": "4 Hours",
            "1h": "Hourly",
            "15m": "15 Minutes"
        }
        return interval_map.get(self.interval, "Daily")
        
    def _convert_period(self):
        """Period formatını Investing.com formatına çevir"""
        end_date = datetime.now() - timedelta(minutes=15)  # 15 dakika gecikme
        
        if self.period == "5d":
            start_date = end_date - timedelta(days=5)
        elif self.period == "1mo":
            start_date = end_date - timedelta(days=30)
        elif self.period == "3mo":
            start_date = end_date - timedelta(days=90)
        else:  # 1y
            start_date = end_date - timedelta(days=365)
            
        return {
            'st_date': start_date.strftime('%m/%d/%Y'),
            'end_date': end_date.strftime('%m/%d/%Y')
        }
        
    def _random_delay(self):
        """Rastgele gecikme ekle"""
        time.sleep(random.uniform(1, 3))
        
    def get_data(self):
        try:
            # Sembolü düzelt (örn: AKBNK.IS -> AKBNK)
            symbol = self.symbol.replace('.IS', '')
            
            # API endpoint ve parametreler
            endpoint = f"{self.base_url}/equities/{symbol}-historical-data"
            params = {
                'interval': self._convert_interval(),
                **self._convert_period()
            }
            
            # Rastgele gecikme ekle
            self._random_delay()
            
            # API isteği
            response = self.scraper.get(
                endpoint,
                headers=self._get_headers(),
                params=params
            )
            
            if response.status_code == 200:
                # HTML'den tablo verilerini çek
                tables = pd.read_html(response.text)
                if tables:
                    df = tables[0]  # İlk tabloyu al
                    
                    # Sütun isimlerini düzenle
                    df = df.rename(columns={
                        'Tarih': 'Date',
                        'Son': 'Close',
                        'Açılış': 'Open',
                        'Yüksek': 'High',
                        'Düşük': 'Low',
                        'Hacim': 'Volume'
                    })
                    
                    # Tarih sütununu index yap
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    
                    # Veri çekilme zamanını ekle
                    df['data_fetch_time'] = datetime.now() - timedelta(minutes=15)
                    
                    return df
                else:
                    print("Tablo verisi bulunamadı")
                    return None
            else:
                print(f"API isteği başarısız: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Veri çekme hatası: {e}")
            return None