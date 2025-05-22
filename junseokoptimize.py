import backtrader as bt
import yfinance as yf
import pandas as pd
from datetime import datetime
from scipy.stats import pearsonr


# — your AAPLStrategy as given —


start_dt = datetime(2023, 3, 22).date()
end_dt   = datetime(2024, 3, 22).date()
# — your AAPLStrategy as given —
class SRLongShortStrategy(bt.Strategy):
    params = dict(
        # Trend filters
        ema_short     = 5,
        ema_long      = 20,
        sma_trend     = 100,

        # Momentum
        rsi_period    = 20,
        stoch_period  = 20,
        stoch_k       = 3,
        stoch_d       = 3,

        # Volatility stops
        atr_period    = 14,

        # Support/Resistance
        sr_period     = 30,
        sr_tol        = 0.01,   # 1% tolerance

        # Volume filter
        vol_period    = 5,     # new: lookback for volume MA

        # Position sizing
        allocation    = 0.5,

        # Profit target & stop-loss
        tp_mult       = 1.5,
        sl_mult       = 0.5,

        # Maximum holding duration
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

        # Support & Resistance levels
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

        dt    = len(self)
        pos   = self.position.size
        close = self.data.close[0]

        # ENTRY
        if not pos:
            # Base long/short conditions
            near_sup = close <= self.support[0] * (1 + self.p.sr_tol)
            long_cond = (
                close > self.sma200[0]
                and (self.rsi[0] < 30 or self.stoch_k[0] < 20)
                and near_sup
            )

            near_res = close >= self.resistance[0] * (1 - self.p.sr_tol)
            short_cond = (
                close < self.sma200[0]
                and (self.rsi[0] > 70 or self.stoch_k[0] > 80)
                and near_res
            )

            # NEW: volume must be above its 20-bar MA
            vol_ok = self.data.volume[0] > self.vol_ma[0]

            if vol_ok and (long_cond or short_cond):
                size = int((self.broker.getcash() * self.p.allocation) // close)
                if size:
                    self.entry_bar   = dt
                    self.entry_price = close
                    if long_cond:
                        self.order = self.buy(size=size)
                    else:
                        self.order = self.sell(size=size)
            return

        # EXIT
        if pos:
            atr = self.atr[0]
            if pos > 0:
                sl = self.entry_price - atr * self.p.sl_mult
                tp = self.entry_price + atr * self.p.tp_mult
                fade   = self.rsi[0] > 50 or self.stoch_k[0] > self.stoch_d[0]
                hit_tp = close >= tp
                hit_sl = close <= sl
            else:
                sl = self.entry_price + atr * self.p.sl_mult
                tp = self.entry_price - atr * self.p.tp_mult
                fade   = self.rsi[0] < 50 or self.stoch_k[0] < self.stoch_d[0]
                hit_tp = close <= tp
                hit_sl = close >= sl

            timeout = dt >= self.entry_bar + self.p.time_exit
            if hit_tp or hit_sl or fade or timeout:
                self.close()
if __name__ == '__main__':
    cerebro = bt.Cerebro(optreturn=False)

    # Define parameter ranges for optimization
    cerebro.optstrategy(
        SRLongShortStrategy,
        ema_short=[5,10,15],
        ema_long=[20,40,60],
        sma_trend=[100,200],
        sr_period=[10,20,30],
        
        sl_mult=[0.5,1.0],
        vol_period =[5,10,15],
        
    )

    # Download data
    df = yf.download('QQQ',
                     start='2015-01-01',
                     end=end_dt,multi_level_index=False)
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
    results.sort(key=lambda x: (x[1] or 0), reverse=True)

    # Display top 5 results
    print("Top 5 parameter sets by Value:")
    for params, value, sharpe, ret in results[:5]:
        print(f"Params: {params}, Final Value: {value:.2f}, Sharpe: {sharpe:}")
