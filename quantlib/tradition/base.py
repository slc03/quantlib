### 根据传统指标决策

from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.test import SMA

from stockstats import wrap, set_dft_window


class TwoSMA_Official(Strategy):
    """双均线策略（官方实现）"""
    # Define the two MA lags as *class variables*
    # for later optimization
    n1 = 10
    n2 = 20
    allow_short = False
    
    def init(self):
        # Precompute the two moving averages
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)
    
    def next(self):
        # If sma1 crosses above sma2, close any existing
        # short trades, and buy the asset
        if crossover(self.sma1, self.sma2):
            self.position.close()
            self.buy()

        # Else, if sma1 crosses below sma2, close any existing
        # long trades, and sell the asset
        elif crossover(self.sma2, self.sma1):
            self.position.close()
            if self.allow_short:
                self.sell()
                

class TwoSMA(Strategy):
    """双均线策略"""
    # Define the two MA lags as *class variables*
    # for later optimization
    n1 = 10
    n2 = 20
    allow_short = False
    
    def get_close_sma(self, windows: int):
        temp_df = wrap(self.data.df)
        arr = temp_df[f'close_{windows}_sma'].to_numpy()
        # arr[:windows-1] = np.nan      # 控制信号从完全的SMA开始
        return arr
    
    def init(self):
        # Precompute the two moving averages
        self.sma1 = self.I(self.get_close_sma, self.n1)
        self.sma2 = self.I(self.get_close_sma, self.n2)
    
    def next(self):
        # If sma1 crosses above sma2, close any existing
        # short trades, and buy the asset
        if crossover(self.sma1, self.sma2):
            self.position.close()
            self.buy()

        # Else, if sma1 crosses below sma2, close any existing
        # long trades, and sell the asset
        elif crossover(self.sma2, self.sma1):
            self.position.close()
            if self.allow_short:
                self.sell()


class MACDStrategy(Strategy):
    """MACD方法"""
    fast = 12
    slow = 26
    signal = 9
    allow_short = False

    def get_macd_diff(self):
        temp_df = wrap(self.data.df)
        return temp_df['macd'].to_numpy()

    def get_macd_signal(self):
        temp_df = wrap(self.data.df)
        return temp_df['macds'].to_numpy()

    def init(self):
        set_dft_window('macd', (self.fast, self.slow, self.signal))
        self.macd_line = self.I(self.get_macd_diff)
        self.signal_line = self.I(self.get_macd_signal)

    def next(self):
        if crossover(self.macd_line, self.signal_line):
            self.position.close()
            self.buy()
        elif crossover(self.signal_line, self.macd_line):
            self.position.close()
            if self.allow_short:
                self.sell()
                
                
class KDJRSIStrategy(Strategy):
    """KDJ结合RSI的方法"""
    rsi_period = 14
    rsi_overbought = 70
    rsi_oversold = 30
    allow_short = False

    def get_rsi(self):
        temp_df = wrap(self.data.df)
        return temp_df[f'rsi_{self.rsi_period}'].to_numpy()

    def get_k(self):
        temp_df = wrap(self.data.df)
        return temp_df['kdjk'].to_numpy()

    def get_d(self):
        temp_df = wrap(self.data.df)
        return temp_df['kdjd'].to_numpy()

    def init(self):
        self.rsi = self.I(self.get_rsi)
        self.k = self.I(self.get_k)
        self.d = self.I(self.get_d)

    def next(self):
        if self.k[-1] < 20 and self.d[-1] < 20 and self.rsi[-1] < self.rsi_oversold:
            self.position.close()
            self.buy()
        elif self.k[-1] > 80 and self.d[-1] > 80 and self.rsi[-1] > self.rsi_overbought:
            self.position.close()
            if self.allow_short:
                self.sell()
                
                
class ZeroMeanReversionStrategy(Strategy):
    """Z-Score法"""
    window = 20
    threshold = 2  # 可调参数，表示偏离标准差倍数

    def get_price_zscore(self):
        temp_df = wrap(self.data.df)
        return temp_df[f'close_{self.window}_z'].to_numpy()

    def init(self):
        self.z = self.I(self.get_price_zscore)

    def next(self):
        if self.z[-1] < -self.threshold:
            self.position.close()
            self.buy()
        elif self.z[-1] > self.threshold:
            self.position.close()
            self.sell()
