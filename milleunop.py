import backtrader as bt
import yfinance as yf
import pandas as pd
from datetime import datetime
from scipy.stats import pearsonr


# — your AAPLStrategy as given —


class LongShortStrategy(bt.Strategy):
    params = dict(
        # super-responsive moving averages
        ema_short   = 3,     # 3-day EMA
        ema_long    = 6,     # 6-day EMA
        sma_trend   = 10,    # 10-day SMA trend filter

        # very sensitive oscillators
        rsi_period  = 7,     # 1-week RSI
        stoch_period    = 7,
        stoch_smoothk   = 2,
        stoch_smoothd   = 2,

        # tight TP/SL
        atr_period  = 5,     # 5-day ATR
        tp_mult     = 0.8,   # 0.8× ATR
        sl_mult     = 0.4,   # 0.4× ATR

        # quick exits & full allocation
        time_exit   = 5,     # exit after 5 bars (~1 week)
        allocation  = 1.0,
    )

    def __init__(self):
        self.sma  = bt.ind.SMA(self.data.close, period=self.p.sma_trend)
        self.rsi  = bt.ind.RSI(self.data.close, period=self.p.rsi_period)
        st      = bt.ind.Stochastic(self.data,
                                     period=self.p.stoch_period,
                                     period_dfast=self.p.stoch_smoothk,
                                     period_dslow=self.p.stoch_smoothd)
        self.stoch_k = st.percK

        self.atr = bt.ind.ATR(self.data, period=self.p.atr_period)

        self.entry_bar   = None
        self.entry_price = 0.0

    def next(self):
        today = self.data.datetime.date(0)
        if not (datetime(2024,4,22).date() <= today <= datetime(2024,5,22).date()):
            return

        dt   = len(self)
        pos  = self.position.size
        close = self.data.close[0]

        # 1) ENTRY
        if not pos:
            # much looser momentum thresholds
            long_cond = (close > self.sma[0]) and (self.rsi[0] < 80 or self.stoch_k[0] < 80)
            short_cond = (close < self.sma[0]) and (self.rsi[0] > 20 or self.stoch_k[0] > 20)

            if long_cond or short_cond:
                cash = self.broker.getcash()
                size = int((cash * self.p.allocation) // close)
                if size:
                    self.entry_bar   = dt
                    self.entry_price = close
                    self.order = self.buy(size=size) if long_cond else self.sell(size=size)
                return

        # 2) EXIT
        if pos:
            atr = self.atr[0]
            if pos > 0:
                sl, tp = self.entry_price - atr*self.p.sl_mult, self.entry_price + atr*self.p.tp_mult
                fade    = self.rsi[0] > 50
                hit_tp  = close >= tp
                hit_sl  = close <= sl
            else:
                sl, tp = self.entry_price + atr*self.p.sl_mult, self.entry_price - atr*self.p.tp_mult
                fade    = self.rsi[0] < 50
                hit_tp  = close <= tp
                hit_sl  = close >= sl

            timeout = (dt >= self.entry_bar + self.p.time_exit)

            if hit_tp or hit_sl or fade or timeout:
                self.close()
if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.addstrategy(LongShortStrategy)

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