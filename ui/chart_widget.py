from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel
from PyQt6.QtCore import Qt
import mplfinance as mpf
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class ChartWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Üst panel - İndikatör seçimi
        self.top_panel = QHBoxLayout()
        self.indicator_combo = QComboBox()
        self.indicator_combo.addItems([
            "Fiyat ve Bollinger",
            "RSI",
            "MACD",
            "Stokastik",
            "Ichimoku",
            "ATR",
            "ADX"
        ])
        self.indicator_combo.currentTextChanged.connect(self.on_indicator_changed)
        self.top_panel.addWidget(QLabel("İndikatör:"))
        self.top_panel.addWidget(self.indicator_combo)
        self.top_panel.addStretch()
        self.layout.addLayout(self.top_panel)
        
        # Matplotlib figure
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.layout.addWidget(self.canvas)
        
        # Veri saklama
        self.df = None
        self.symbol = None
        
    def plot_chart(self, df, symbol=None, selected_indicators=None):
        self.figure.clear()
        if df is None or df.empty:
            self.canvas.draw()
            return

        self.df = df
        self.symbol = symbol
        
        # Seçili indikatör sayısına göre subplot sayısını belirle
        num_indicators = len(selected_indicators) if selected_indicators else 0
        num_plots = num_indicators + 1  # +1 for price chart
        
        # Subplot'ları oluştur
        axes = []
        for i in range(num_plots):
            if i == 0:
                ax = self.figure.add_subplot(num_plots, 1, i+1)
            else:
                ax = self.figure.add_subplot(num_plots, 1, i+1, sharex=axes[0])
            axes.append(ax)
        
        # Fiyat grafiği
        ax1 = axes[0]
        ax1.plot(df.index, df['Close'], label='Kapanış', color='black')
        ax1.set_title(f"{symbol} - Fiyat Grafiği" if symbol else "Fiyat Grafiği")
        ax1.legend()
        
        # Seçili indikatörleri çiz
        for i, indicator in enumerate(selected_indicators, 1):
            ax = axes[i]
            
            try:
                if indicator == "MA" and all(col in df for col in ['ma20', 'ma50', 'ma200']):
                    ax.plot(df.index, df['ma20'], label='MA20', color='blue')
                    ax.plot(df.index, df['ma50'], label='MA50', color='red')
                    ax.plot(df.index, df['ma200'], label='MA200', color='green')
                    ax.set_title("Hareketli Ortalamalar")
                    
                elif indicator == "MACD" and all(col in df for col in ['macd', 'macd_signal', 'macd_diff']):
                    ax.plot(df.index, df['macd'], label='MACD', color='blue')
                    ax.plot(df.index, df['macd_signal'], label='Signal', color='red')
                    ax.bar(df.index, df['macd_diff'], label='Histogram', color='gray', alpha=0.5)
                    ax.set_title("MACD")
                    
                elif indicator == "BB" and all(col in df for col in ['bb_low', 'bb_high', 'bb_mid']):
                    ax.fill_between(df.index, df['bb_low'], df['bb_high'], color='gray', alpha=0.2, label='Bollinger')
                    ax.plot(df.index, df['bb_mid'], label='Orta Bant', color='blue')
                    ax.set_title("Bollinger Bantları")
                    
                elif indicator == "RSI" and 'rsi' in df:
                    ax.plot(df.index, df['rsi'], label='RSI', color='blue')
                    ax.axhline(70, color='red', linestyle='--')
                    ax.axhline(30, color='green', linestyle='--')
                    ax.set_title("RSI")
                    
                elif indicator == "STOCH" and all(col in df for col in ['stoch_k', 'stoch_d']):
                    ax.plot(df.index, df['stoch_k'], label='%K', color='blue')
                    ax.plot(df.index, df['stoch_d'], label='%D', color='red')
                    ax.axhline(80, color='red', linestyle='--')
                    ax.axhline(20, color='green', linestyle='--')
                    ax.set_title("Stokastik")
                    
                elif indicator == "FIB" and all(col in df for col in ['fib_0', 'fib_0.236', 'fib_0.382', 'fib_0.5', 'fib_0.618', 'fib_0.786', 'fib_1']):
                    ax.plot(df.index, df['fib_0'], label='0%', color='black')
                    ax.plot(df.index, df['fib_0.236'], label='23.6%', color='blue')
                    ax.plot(df.index, df['fib_0.382'], label='38.2%', color='green')
                    ax.plot(df.index, df['fib_0.5'], label='50%', color='yellow')
                    ax.plot(df.index, df['fib_0.618'], label='61.8%', color='orange')
                    ax.plot(df.index, df['fib_0.786'], label='78.6%', color='red')
                    ax.plot(df.index, df['fib_1'], label='100%', color='purple')
                    ax.set_title("Fibonacci Düzeltmesi")
                    
                elif indicator == "ICHIMOKU" and all(col in df for col in ['ichi_a', 'ichi_b', 'ichi_base', 'ichi_conv']):
                    ax.fill_between(
                        df.index,
                        df['ichi_base'],
                        df['ichi_conv'],
                        where=(df['ichi_base'] >= df['ichi_conv']),
                        color='g',
                        alpha=0.2,
                        label='Kumo (Yükseliş)'
                    )
                    ax.fill_between(
                        df.index,
                        df['ichi_base'],
                        df['ichi_conv'],
                        where=(df['ichi_base'] < df['ichi_conv']),
                        color='r',
                        alpha=0.2,
                        label='Kumo (Düşüş)'
                    )
                    ax.plot(df.index, df['ichi_a'], label='Tenkan-sen', color='b')
                    ax.plot(df.index, df['ichi_b'], label='Kijun-sen', color='r')
                    ax.plot(df.index, df['ichi_base'], label='Senkou Span A', color='g')
                    ax.plot(df.index, df['ichi_conv'], label='Senkou Span B', color='y')
                    ax.set_title("Ichimoku")
                    
                elif indicator == "ATR" and 'atr' in df:
                    ax.plot(df.index, df['atr'], label='ATR', color='blue')
                    ax.set_title("ATR")
                    
                elif indicator == "STD" and 'std' in df:
                    ax.plot(df.index, df['std'], label='Standart Sapma', color='blue')
                    ax.set_title("Standart Sapma")
                    
                elif indicator == "ADX" and all(col in df for col in ['adx', 'di_plus', 'di_minus']):
                    ax.plot(df.index, df['adx'], label='ADX', color='blue')
                    ax.plot(df.index, df['di_plus'], label='DI+', color='green')
                    ax.plot(df.index, df['di_minus'], label='DI-', color='red')
                    ax.set_title("ADX")
                
                ax.legend()
                
            except Exception as e:
                print(f"İndikatör çiziminde hata: {indicator} - {str(e)}")
                ax.text(0.5, 0.5, f"İndikatör verisi yok: {indicator}", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes)
        
        # Ana başlık
        self.figure.suptitle(f"{symbol} - Teknik Analiz" if symbol else "Teknik Analiz")
        
        # Grafik düzenini ayarla
        self.figure.tight_layout()
        self.canvas.draw()

    def on_indicator_changed(self, indicator_name):
        if self.df is None:
            return
            
        self.figure.clear()
        
        if indicator_name == "Fiyat ve Bollinger":
        ax1 = self.figure.add_subplot(3, 1, 1)
        ax2 = self.figure.add_subplot(3, 1, 2, sharex=ax1)
        ax3 = self.figure.add_subplot(3, 1, 3, sharex=ax1)

            # Fiyat ve Bollinger
            ax1.plot(self.df.index, self.df['Close'], label='Kapanış', color='black')
            if 'bb_low' in self.df and 'bb_high' in self.df:
                ax1.fill_between(self.df.index, self.df['bb_low'], self.df['bb_high'], color='gray', alpha=0.2, label='Bollinger')
            ax1.set_title(f"{self.symbol} - Fiyat ve Bollinger Bantları")
        ax1.legend()

        # RSI
            if 'rsi' in self.df:
                ax2.plot(self.df.index, self.df['rsi'], label='RSI', color='blue')
            ax2.axhline(70, color='red', linestyle='--')
            ax2.axhline(30, color='green', linestyle='--')
            ax2.set_title("RSI")
            ax2.legend()

        # MACD
            if all(col in self.df for col in ['macd', 'macd_signal', 'macd_diff']):
                ax3.plot(self.df.index, self.df['macd'], label='MACD', color='blue')
                ax3.plot(self.df.index, self.df['macd_signal'], label='MACD Signal', color='red')
                ax3.bar(self.df.index, self.df['macd_diff'], label='MACD Histogram', color='gray', alpha=0.5)
            ax3.set_title("MACD")
            ax3.legend()
                
        elif indicator_name == "RSI":
            ax = self.figure.add_subplot(111)
            if 'rsi' in self.df:
                ax.plot(self.df.index, self.df['rsi'], label='RSI', color='blue')
                ax.axhline(70, color='red', linestyle='--')
                ax.axhline(30, color='green', linestyle='--')
                ax.set_title("RSI")
                ax.legend()
                
        elif indicator_name == "MACD":
            ax = self.figure.add_subplot(111)
            if all(col in self.df for col in ['macd', 'macd_signal', 'macd_diff']):
                ax.plot(self.df.index, self.df['macd'], label='MACD', color='blue')
                ax.plot(self.df.index, self.df['macd_signal'], label='MACD Signal', color='red')
                ax.bar(self.df.index, self.df['macd_diff'], label='MACD Histogram', color='gray', alpha=0.5)
                ax.set_title("MACD")
                ax.legend()
                
        elif indicator_name == "Stokastik":
            ax = self.figure.add_subplot(111)
            if all(col in self.df for col in ['stoch_k', 'stoch_d']):
                ax.plot(self.df.index, self.df['stoch_k'], label='%K', color='blue')
                ax.plot(self.df.index, self.df['stoch_d'], label='%D', color='red')
                ax.axhline(80, color='red', linestyle='--')
                ax.axhline(20, color='green', linestyle='--')
                ax.set_title("Stokastik")
                ax.legend()
                
        elif indicator_name == "Ichimoku":
            ax = self.figure.add_subplot(111)
            if all(col in self.df for col in ['ichi_a', 'ichi_b', 'ichi_base', 'ichi_conv']):
                # Kumo (Bulut) alanını çiz
                ax.fill_between(
                    self.df.index,
                    self.df['ichi_base'],
                    self.df['ichi_conv'],
                    where=(self.df['ichi_base'] >= self.df['ichi_conv']),
                    color='g',
                    alpha=0.2,
                    label='Kumo (Yükseliş)'
                )
                ax.fill_between(
                    self.df.index,
                    self.df['ichi_base'],
                    self.df['ichi_conv'],
                    where=(self.df['ichi_base'] < self.df['ichi_conv']),
                    color='r',
                    alpha=0.2,
                    label='Kumo (Düşüş)'
                )
                # Çizgileri çiz
                ax.plot(self.df.index, self.df['ichi_a'], label='Tenkan-sen', color='b')
                ax.plot(self.df.index, self.df['ichi_b'], label='Kijun-sen', color='r')
                ax.plot(self.df.index, self.df['ichi_base'], label='Senkou Span A', color='g')
                ax.plot(self.df.index, self.df['ichi_conv'], label='Senkou Span B', color='y')
                ax.set_title("Ichimoku")
                ax.legend()
                
        elif indicator_name == "ATR":
            ax = self.figure.add_subplot(111)
            if 'atr' in self.df:
                ax.plot(self.df.index, self.df['atr'], label='ATR', color='blue')
                ax.set_title("ATR")
                ax.legend()
                
        elif indicator_name == "ADX":
            ax = self.figure.add_subplot(111)
            if all(col in self.df for col in ['adx', 'di_plus', 'di_minus']):
                ax.plot(self.df.index, self.df['adx'], label='ADX', color='blue')
                ax.plot(self.df.index, self.df['di_plus'], label='DI+', color='green')
                ax.plot(self.df.index, self.df['di_minus'], label='DI-', color='red')
                ax.set_title("ADX")
                ax.legend()

        self.figure.tight_layout()
        self.canvas.draw()
