import backtrader as bt
import yfinance as yf
import pandas as pd
from datetime import datetime
from scipy.stats import pearsonr


# — your AAPLStrategy as given —


class AAPLStrategy(bt.Strategy):
    params = dict(
        # — ultra-short lookbacks for 1-month testing —
        ema_short     = 3,
        ema_long      = 6,
        sma_trend     = 10,
        rsi_period    = 7,
        stoch_period  = 7,
        stoch_smoothk = 2,
        stoch_smoothd = 2,
        atr_period    = 5,

        # take-profit / stop-loss multipliers
        tp_mult       = 0.8,
        sl_mult       = 0.4,

        # pullback/time exit
        pullback_bars = 3,
        time_exit     = 5,

        # **NEW**: momentum thresholds to optimize
        rsi_thresh    = 80,   # entry if RSI < this
        stoch_thresh  = 80,   # entry if %K < this
        fade_thresh   = 50,  # exit after 5 bars if nothing else
    )

    def __init__(self):
        # indicators
        self.sma    = bt.ind.SMA(self.data.close, period=self.p.sma_trend)
        self.ema_l  = bt.ind.EMA(self.data.close, period=self.p.ema_long)
        self.ema_s  = bt.ind.EMA(self.data.close, period=self.p.ema_short)

        self.rsi    = bt.ind.RSI(self.data.close, period=self.p.rsi_period)
        stoch       = bt.ind.Stochastic(self.data,
                                        period=self.p.stoch_period,
                                        period_dfast=self.p.stoch_smoothk,
                                        period_dslow=self.p.stoch_smoothd)
        self.stochK = stoch.percK

        self.atr    = bt.ind.ATR(self.data, period=self.p.atr_period)

        # trade bookkeeping
        self.entry_bar   = None
        self.entry_price = 0.0
        self.size1       = 0

    def next(self):
        today = self.data.datetime.date(0)
        # constrain to your 1-month test window
        if not (datetime(2025,1,22).date() <= today <= datetime(2025,2,22).date()):
            return

        dt    = len(self)
        pos   = self.position.size
        close = self.data.close[0]

        # 1) TIER-1 ENTRY
        if not pos:
            trend_ok = (close > self.sma[0]) and (self.ema_l[0] > self.sma[0])
            mom_ok   = (self.rsi[0] < self.p.rsi_thresh) or (self.stochK[0] < self.p.stoch_thresh)

            if trend_ok and mom_ok:
                size = int((self.broker.getcash() * 0.5) // close)
                if size:
                    self.size1       = size
                    self.entry_bar   = dt
                    self.entry_price = close
                    self.buy(size=size)
                return

        # 2) TIER-2 PULLBACK (same logic as before)
        elif pos == self.size1 and dt <= self.entry_bar + self.p.pullback_bars:
            if (self.data.low[-1] <= self.ema_s[-1] and
                close > self.ema_s[0]    and
                close > self.sma[0]):
                size = int((self.broker.getcash() * 0.5) // close)
                if size:
                    self.buy(size=size)
                return

        # 3) EXITS
        if pos:
            atr = self.atr[0]
            sl  = self.entry_price - atr * self.p.sl_mult
            tp  = self.entry_price + atr * self.p.tp_mult

            # stop-loss or take-profit
            if close <= sl or close >= tp:
                self.close()
                return

            # fade-out or time-out
            fade    = (self.rsi[0] > self.p.fade_thresh)
            timeout = (dt >= self.entry_bar + self.p.time_exit)
            if fade or timeout:
                self.close()
                return
if __name__ == '__main__':
    cerebro = bt.Cerebro(optreturn=False)

    # Define parameter ranges for optimization
    cerebro.optstrategy(
        AAPLStrategy,
        
        ema_long     = [6, 10, 20],
        sma_trend    = [10, 20, 50],
        rsi_period   = [7, 14, 21],
        rsi_thresh   = [60, 70, 80],
        stoch_thresh = [60, 70, 80],
        
        pullback_bars= [2, 3, 5],
        
    )

    # Download data
    df = yf.download('AAPL', start='2015-01-01',
                     end=datetime.today().strftime('%Y-%m-%d'),multi_level_index=False)
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)

    # Broker settings
    cerebro.broker.setcash(100_000.0)
    cerebro.broker.setcommission(commission=0.001)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    # Run optimization
    opt_runs = cerebro.run(maxcpus=4)

    # Collect and sort results
    results = []
    
    for run in opt_runs:
        strat = run[0]
        sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', None)
        ret = strat.analyzers.returns.get_analysis().get('rperiod', None)
        params = strat.params._getkwargs()
        results.append((params, strat.broker.getvalue(), sharpe, ret))
        

    # Sort by Sharpe ratio descending
    results.sort(key=lambda x: (x[2] or 0), reverse=True)

    # Display top 5 results
    print("Top 5 parameter sets by Sharpe Ratio:")
    for params, value, sharpe, ret in results[:5]:
        print(f"Params: {params}, Final Value: {value:.2f}, Sharpe: {sharpe:}")