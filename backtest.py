import backtrader as bt
import yfinance as yf
import pandas as pd
from datetime import datetime
from scipy.stats import pearsonr

start_dt = datetime(2022,1, 22).date()#2023- 2024, 2024-2025, 2022-2023
end_dt   = datetime(2025, 1, 22).date()

class SRLongShortStrategy(bt.Strategy):
    params = dict(
        ema_short     = 5,
        ema_long      = 20,
        sma_trend     = 100,
        rsi_period    = 20,
        stoch_period  = 20,
        stoch_k       = 3,
        stoch_d       = 3,
        atr_period    = 10,
        sr_period     = 20,
        sr_tol        = 0.01,
        vol_period    = 15,
        allocation    = 0.5,
        tp_mult       = 1.2,
        sl_mult       = 1.0,
        time_exit     = 30,
    )

    def __init__(self):
        # Trend indicators
        self.sma200     = bt.ind.SMA(self.data.close, period=self.p.sma_trend)
        self.ema_short  = bt.ind.EMA(self.data.close, period=self.p.ema_short)
        self.ema_long   = bt.ind.EMA(self.data.close, period=self.p.ema_long)

        # Momentum
        self.rsi        = bt.ind.RSI(self.data.close, period=self.p.rsi_period)
        stoch           = bt.ind.Stochastic(
                            self.data,
                            period=self.p.stoch_period,
                            period_dfast=self.p.stoch_k,
                            period_dslow=self.p.stoch_d)
        self.stoch_k    = stoch.percK
        self.stoch_d    = stoch.percD

        # Volatility
        self.atr        = bt.ind.ATR(self.data, period=self.p.atr_period)

        # Support & Resistance
        self.resistance = bt.ind.Highest(self.data.high, period=self.p.sr_period)
        self.support    = bt.ind.Lowest(self.data.low,  period=self.p.sr_period)

        # Volume filter
        self.vol_ma     = bt.ind.SMA(self.data.volume, period=self.p.vol_period)

        # Bookkeeping
        self.entry_bar    = None
        self.entry_price  = 0.0

    def next(self):
        # 1) DATE GATING
        today = self.data.datetime.date(0)
        if not (start_dt <= today <= end_dt):
            return

        dt_index = len(self)
        close    = self.data.close[0]
        vol_ok   = self.data.volume[0] > self.vol_ma[0]
        pos_size = self.position.size

        # ENTRY
        if not pos_size and vol_ok:
            near_sup   = close <= self.support[0] * (1 + self.p.sr_tol)
            long_cond  = (
                close > self.sma200[0] and
                (self.rsi[0] < 40 or self.stoch_k[0] < 30) and
                near_sup
            )

            near_res   = close >= self.resistance[0] * (1 - self.p.sr_tol)
            short_cond = (
                close < self.sma200[0] and
                (self.rsi[0] > 70 or self.stoch_k[0] > 80) and
                near_res
            )

            if long_cond or short_cond:
                size = int((self.broker.getcash() * self.p.allocation) // close)
                if size:
                    self.entry_bar   = dt_index
                    self.entry_price = close
                    self.order = self.buy(size=size) if long_cond else self.sell(size=size)
            return

        # EXIT
        if pos_size:
            atr     = self.atr[0]
            is_long = pos_size > 0
            sl      = (self.entry_price - atr * self.p.sl_mult) if is_long else (self.entry_price + atr * self.p.sl_mult)
            tp      = (self.entry_price + atr * self.p.tp_mult) if is_long else (self.entry_price - atr * self.p.tp_mult)

            fade    = ((self.rsi[0] > 50 or self.stoch_k[0] > self.stoch_d[0]) if is_long
                       else (self.rsi[0] < 50 or self.stoch_k[0] < self.stoch_d[0]))
            hit_tp  = (close >= tp) if is_long else (close <= tp)
            hit_sl  = (close <= sl) if is_long else (close >= sl)
            timeout = dt_index >= self.entry_bar + self.p.time_exit

            if hit_tp or hit_sl or fade or timeout:
                self.close()



def run_backtest(df):
    """Run Cerebro on df and return final portfolio value."""
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.addstrategy(SRLongShortStrategy)
    datafeed = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(datafeed)
    cerebro.broker.setcash(100_000.0)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.run()
    print('Final Value:   ', cerebro.broker.getvalue())
    cerebro.plot()
    return cerebro.broker.getvalue()


if __name__ == '__main__':
    ticker = 'XLK'

    print(f"\nDownloading {ticker} from {start_dt} to {end_dt} …")
    df = yf.download(ticker, start='2015-01-01', end=end_dt, multi_level_index=False)
    df = df[['Open','High','Low','Close','Volume']].dropna()
    obs_value = run_backtest(df)
    
