from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QComboBox,
                            QGroupBox, QCheckBox, QScrollArea)
from PyQt6.QtCore import QTimer
from .chart_widget import ChartWidget
from core.data_fetcher import DataFetcher
from core.indicators import Indicators
from config import SYMBOL, TIMEFRAME, PERIOD

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VİOP Trading Bot")
        self.setMinimumSize(1200, 800)
        
        # Ana widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Üst panel
        top_panel = QHBoxLayout()
        # Sembol seçici
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems([
            "XU030.IS", "XU100.IS",  # Endeksler
            "AKBNK.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS", "EKGYO.IS", "EREGL.IS", "FROTO.IS", "GARAN.IS",
            "HEKTS.IS", "ISCTR.IS", "KCHOL.IS", "KOZAL.IS", "KOZAA.IS", "PETKM.IS", "PGSUS.IS", "SAHOL.IS",
            "SASA.IS", "SISE.IS", "TAVHL.IS", "TCELL.IS", "THYAO.IS", "TOASO.IS", "TSKB.IS", "TUPRS.IS",
            "VAKBN.IS", "YKBNK.IS", "YUNSA.IS", "ZOREN.IS", "AHLAT.IS", "AKSEN.IS", "ALBRK.IS", "ALCAR.IS",
            "ALGYO.IS", "ALKIM.IS", "ANACM.IS", "ASUZU.IS", "AYCES.IS", "BAGFS.IS", "BASCM.IS", "BERA.IS",
            "BIENY.IS", "BRISA.IS", "BRYAT.IS", "BUCIM.IS", "CCOLA.IS", "CEMAS.IS", "CEMTS.IS", "CIMSA.IS",
            "CUSAN.IS", "DOHOL.IS", "EGEEN.IS", "ENJSA.IS", "ENKAI.IS", "FMIZP.IS", "GESAN.IS", "GLYHO.IS",
            "GSDHO.IS", "GSDDE.IS", "HALKB.IS", "HATEK.IS", "IPEKE.IS", "ISDMR.IS", "KAREL.IS", "KARSN.IS",
            "KONTR.IS", "KONYA.IS", "KORDS.IS", "KOZAA.IS", "KRDMD.IS", "LOGO.IS", "MGROS.IS", "NETAS.IS",
            "NTHOL.IS", "ODAS.IS", "OTKAR.IS", "OYAKC.IS", "PETKM.IS", "PGSUS.IS", "POLHO.IS", "PRKAB.IS",
            "PRKME.IS", "QUAGR.IS", "SAFKN.IS", "SELEC.IS", "SELGD.IS", "SISE.IS", "SKBNK.IS", "SMRTG.IS",
            "SNGYO.IS", "SOKM.IS", "TATGD.IS", "TCELL.IS", "TKURU.IS", "TOASO.IS", "TSKB.IS", "TTKOM.IS",
            "TTRAK.IS", "TUPRS.IS", "ULKER.IS", "VAKBN.IS", "VESTL.IS", "YATAS.IS", "YKBNK.IS", "YUNSA.IS"
        ])
        top_panel.addWidget(QLabel("Sembol:"))
        top_panel.addWidget(self.symbol_combo)
        
        # Zaman aralığı seçici
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(["1d", "4h", "1h", "15m"])
        self.timeframe_combo.currentTextChanged.connect(self.refresh_data)
        top_panel.addWidget(QLabel("Zaman Dilimi:"))
        top_panel.addWidget(self.timeframe_combo)
        
        # Yenile butonu
        refresh_btn = QPushButton("Yenile")
        refresh_btn.clicked.connect(self.refresh_data)
        top_panel.addWidget(refresh_btn)
        layout.addLayout(top_panel)

        # İndikatör seçim paneli
        indicator_group = QGroupBox("İndikatörler")
        indicator_layout = QVBoxLayout()
        
        # İndikatör checkbox'ları
        self.indicator_checkboxes = {}
        indicators = {
            "MA": "Hareketli Ortalama",
            "MACD": "MACD",
            "BB": "Bollinger Bantları",
            "RSI": "RSI",
            "STOCH": "Stokastik Osilatör",
            "FIB": "Fibonacci Düzeltmesi",
            "ICHIMOKU": "Ichimoku Bulutu",
            "ATR": "ATR",
            "STD": "Standard Sapma",
            "ADX": "ADX"
        }
        
        for key, name in indicators.items():
            checkbox = QCheckBox(name)
            checkbox.stateChanged.connect(self.refresh_data)
            self.indicator_checkboxes[key] = checkbox
            indicator_layout.addWidget(checkbox)
        
        indicator_group.setLayout(indicator_layout)
        
        # Scroll area içine indikatör panelini yerleştir
        scroll = QScrollArea()
        scroll.setWidget(indicator_group)
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(150)
        layout.addWidget(scroll)
        
        # Grafik widget'ı
        self.chart = ChartWidget()
        layout.addWidget(self.chart)
        
        # Alt panel (seçili indikatörler için sinyaller)
        self.bottom_panel = QHBoxLayout()
        self.signal_labels = {}
        layout.addLayout(self.bottom_panel)
        
        # Veri çekme ve güncelleme
        self.data_fetcher = DataFetcher(SYMBOL, PERIOD, TIMEFRAME)
        self.refresh_data()
        
        # Otomatik güncelleme
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_data)
        self.timer.start(60000)
        
    def refresh_data(self):
        symbol = self.symbol_combo.currentText()
        timeframe = self.timeframe_combo.currentText()
        self.data_fetcher = DataFetcher(symbol, PERIOD, timeframe)
        df = self.data_fetcher.get_data()
        
        if df is not None and not df.empty:
            # Seçili indikatörleri hesapla
            selected_indicators = [key for key, checkbox in self.indicator_checkboxes.items() 
                                if checkbox.isChecked()]
            
            if selected_indicators:
                df = Indicators.calculate_selected(df, selected_indicators)
                self.chart.plot_chart(df, symbol, selected_indicators)
                self.update_signals(df, selected_indicators)
            else:
                print("Lütfen en az bir indikatör seçin!")
        else:
            print("Veri yok veya çekilemedi!")
    
    def update_signals(self, df, selected_indicators):
        if df is not None and not df.empty:
            last_row = df.iloc[-1]

            # Önceki sinyal etiketlerini temizle
            for label in self.signal_labels.values():
                self.bottom_panel.removeWidget(label)
                label.deleteLater()
            self.signal_labels.clear()
            
            # Seçili indikatörler için sinyalleri güncelle
            for indicator in selected_indicators:
                signal = self.calculate_signal(indicator, last_row)
                label = QLabel(f"{indicator}: {signal}")
                self.signal_labels[indicator] = label
                self.bottom_panel.addWidget(label)
    
    def calculate_signal(self, indicator, last_row):
        if indicator == "RSI":
            if 'rsi' in last_row:
                if last_row['rsi'] < 30:
                    return "GÜÇLÜ AL"
                elif last_row['rsi'] > 70:
                    return "GÜÇLÜ SAT"
            return "BEKLE"

        elif indicator == "MACD":
            if 'macd' in last_row and 'macd_signal' in last_row:
                if last_row['macd'] > last_row['macd_signal']:
                    return "AL"
                elif last_row['macd'] < last_row['macd_signal']:
                    return "SAT"
            return "BEKLE"
            
        elif indicator == "BB":
            if 'bb_low' in last_row and 'bb_high' in last_row:
                if last_row['Close'] < last_row['bb_low']:
                    return "ALT BAND - AL"
                elif last_row['Close'] > last_row['bb_high']:
                    return "ÜST BAND - SAT"
            return "BEKLE"
            
        # Diğer indikatörler için sinyal hesaplamaları buraya eklenecek
        return "BEKLE"
