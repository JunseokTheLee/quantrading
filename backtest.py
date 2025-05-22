import backtrader as bt
import yfinance as yf
import pandas as pd
from datetime import datetime
from scipy.stats import pearsonr


# — your AAPLStrategy as given —
class AAPLStrategy(bt.Strategy):
    params = dict(
        # — ultra-short lookbacks for 1-month testing —
        ema_short     = 3,     # 3-day EMA
        ema_long      = 6,     # 6-day EMA
        sma_trend     = 10,    # 10-day SMA trend filter
        corr_period   = 5,     # only if you still use it
        rsi_period    = 7,     # faster RSI
        stoch_period  = 7,     # faster Stoch
        stoch_smoothk = 2,
        stoch_smoothd = 2,
        atr_period    = 5,     # tighter ATR

        # — tighter targets & stops —
        tp_mult       = 0.8,   # 0.8× ATR
        sl_mult       = 0.4,   # 0.4× ATR

        # — quicker pullbacks & exits —
        pullback_bars = 3,     # look for a bounce within 3 bars
        time_exit     = 5,     # exit after 5 bars if nothing else
    )

    def __init__(self):
        self.sma10   = bt.ind.SMA(self.data.close, period=self.p.sma_trend)
        self.ema6    = bt.ind.EMA(self.data.close, period=self.p.ema_long)
        self.ema3    = bt.ind.EMA(self.data.close, period=self.p.ema_short)

        self.rsi     = bt.ind.RSI(self.data.close, period=self.p.rsi_period)
        stoch        = bt.ind.Stochastic(self.data,
                                         period=self.p.stoch_period,
                                         period_dfast=self.p.stoch_smoothk,
                                         period_dslow=self.p.stoch_smoothd)
        self.stoch_k = stoch.percK

        self.atr     = bt.ind.ATR(self.data, period=self.p.atr_period)

        self.entry_bar   = None
        self.entry_price = 0.0
        self.size1       = 0
        self.size2       = 0

    def next(self):
        today = self.data.datetime.date(0)
        # only run in your 1-month window
        if not (datetime(2024,3,22).date() <= today <= datetime(2024,4,22).date()):
            return

        dt  = len(self)
        pos = self.position.size
        close = self.data.close[0]

        # 1) TIER-1 ENTRY
        if not pos:
            trend_ok = (close > self.sma10[0]) and (self.ema6[0] > self.sma10[0])
            # much looser momentum
            mom_ok   = (self.rsi[0] < 80) or (self.stoch_k[0] < 80)

            if trend_ok and mom_ok:
                cash = self.broker.getcash()
                size = int((cash * 0.5) // close)
                if size:
                    self.size1       = size
                    self.entry_bar   = dt
                    self.entry_price = close
                    self.buy(size=size)
                return

        # 2) TIER-2 PULLBACK
        elif pos == self.size1 and dt <= self.entry_bar + self.p.pullback_bars:
            # bounce off the 6-day EMA
            if (self.data.low[-1] <= self.ema6[-1] and
                close > self.ema6[0]   and
                close > self.sma10[0]):
                cash = self.broker.getcash()
                size = int((cash * 0.5) // close)
                if size:
                    self.size2 = size
                    self.buy(size=size)
                return

        # 3) EXITS
        if pos:
            atr = self.atr[0]
            sl  = self.entry_price - atr*self.p.sl_mult
            tp  = self.entry_price + atr*self.p.tp_mult

            # hit stop or target?
            if close <= sl or close >= tp:
                self.close()
                return

            # fade or time-out
            fade = self.rsi[0] > 50
            timeout = (dt >= self.entry_bar + self.p.time_exit)
            if fade or timeout:
                self.close()
                return

if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.addstrategy(AAPLStrategy)

    # 1) Download into a DataFrame
    df = yf.download('AAPL',
                     start='2015-01-01',
                     end=datetime.today().strftime('%Y-%m-%d'),multi_level_index=False)

    # 2) Wrap it—exactly the DataFrame—into PandasData
    datafeed = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(datafeed)

    # 3) Run as usual
    cerebro.broker.setcash(100_000.0)
    cerebro.broker.setcommission(commission=0.001)
    print('Starting Value:', cerebro.broker.getvalue())
    cerebro.run()
    print('Final Value:   ', cerebro.broker.getvalue())
    cerebro.plot()