import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date


start_dt = date(2022, 1, 22)
end_dt   = date(2025, 1, 22)


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
    return cerebro.broker.getvalue()

def make_permuted_series(df):
    """
    Shuffle daily returns of df['Close'], rebuild a synthetic price path,
    and return a new OHLCV DataFrame.
    """
    rets = df['Close'].pct_change().dropna().values
    permuted = np.random.permutation(rets)
    price0 = df['Close'].iloc[0]
    new_prices = price0 * np.cumprod(1 + permuted)

    new_df = pd.DataFrame({
        'Open':   new_prices,
        'High':   new_prices,
        'Low':    new_prices,
        'Close':  new_prices,
        'Volume': df['Volume'].iloc[1:].values,
    }, index=df.index[1:])
    return new_df


if __name__ == '__main__':
    ticker = 'XLK'

    print(f"\nDownloading {ticker} from {start_dt} to {end_dt} …")
    df = yf.download(ticker, start='2015-01-01', end=end_dt, multi_level_index=False)
    df = df[['Open','High','Low','Close','Volume']].dropna()

    # 3.1) Observed backtest
    obs_value = run_backtest(df)
    print(f"\nObserved final portfolio value: {obs_value:,.2f}")

    # 3.2) Permutation testing
    n_permutations = 200
    print(f"Running {n_permutations} permuted backtests …")
    perm_results = []
    for i in range(n_permutations):
        perm_df = make_permuted_series(df)
        fv = run_backtest(perm_df)
        perm_results.append(fv)

    # 3.3) p-value
    p_val = np.mean([fv >= obs_value for fv in perm_results])
    print(f"\nPermutation p-value (≥ observed): {p_val:.4f}")

    # 3.4) Quick histogram
    try:
        import matplotlib.pyplot as plt
        plt.hist(perm_results, bins=30, alpha=0.7)
        plt.axvline(obs_value, linestyle='--', label='Observed')
        plt.title('Permutation Distribution of Final Portfolio Value')
        plt.xlabel('Final Portfolio Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"Could not plot histogram: {e}")
