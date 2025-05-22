import backtrader as bt
import yfinance as yf
import pandas as pd
from datetime import datetime
from scipy.stats import pearsonr


# — your AAPLStrategy as given —


class LongShortStrategy(bt.Strategy):
    params = dict(
        ema_short=10,
        ema_long=20,
        sma_trend=200,
        corr_period=20,
        rsi_period=20,
        stoch_period=20,
        stoch_smoothk=3,
        stoch_smoothd=3,
        atr_period=14,
        tp_mult=1.5,
        sl_mult=1.0,
        time_exit=30,
        allocation=0.5,  # fraction of cash per entry
    )

    def __init__(self):
        # Trend filter
        self.sma200 = bt.ind.SMA(self.data.close, period=self.p.sma_trend)

        # EMAs for correlation
        self.ema9  = bt.ind.EMA(self.data.close, period=self.p.ema_short)
        self.ema50 = bt.ind.EMA(self.data.close, period=self.p.ema_long)
       

        # Momentum indicators
        self.rsi     = bt.ind.RSI(self.data.close, period=self.p.rsi_period)
        stoch        = bt.ind.Stochastic(
                         self.data,
                         period=self.p.stoch_period,
                         period_dfast=self.p.stoch_smoothk,
                         period_dslow=self.p.stoch_smoothd)
        self.stoch_k = stoch.percK
        self.stoch_d = stoch.percD

        # Volatility for TP/SL
        self.atr = bt.ind.ATR(self.data, period=self.p.atr_period)

        # Tracking orders & entries
        self.order       = None
        self.entry_bar   = None
        self.entry_price = 0.0

    def next(self):
        trade_start = datetime(2024,4,22)
        trade_end   = datetime(2024,5,22)

        today = self.data.datetime.datetime(0)
        if not (trade_start <= today <= trade_end):
            return
        dt  = len(self)
        pos = self.position.size

        # 1) ENTRY
        if not pos:
            # LONG ENTRY?
            long_cond = (
                (self.rsi[0] < 30 or self.stoch_k[0] < 20) and
                
                (self.data.close[0] > self.sma200[0])
            )
            # SHORT ENTRY?
            short_cond = (
                (self.rsi[0] > 70 or self.stoch_k[0] > 80) and
                
                (self.data.close[0] < self.sma200[0])
            )

            if long_cond or short_cond:
                cash = self.broker.getcash()
                size = int((cash * self.p.allocation) // self.data.close[0])
                if size:
                    self.entry_bar   = dt
                    self.entry_price = self.data.close[0]
                    self.order = self.buy(size=size) if long_cond else self.sell(size=size)
                return

        # 2) EXIT
        if pos:
            # compute TP/SL levels
            atr = self.atr[0]
            if pos > 0:  # long
                sl = self.entry_price - atr * self.p.sl_mult
                tp = self.entry_price + atr * self.p.tp_mult
                fade = (self.rsi[0] > 50) or (self.stoch_k[0] > self.stoch_d[0])
                hit_tp = self.data.close[0] >= tp
                hit_sl = self.data.close[0] <= sl

            else:       # short
                sl = self.entry_price + atr * self.p.sl_mult
                tp = self.entry_price - atr * self.p.tp_mult
                fade = (self.rsi[0] < 50) or (self.stoch_k[0] < self.stoch_d[0])
                hit_tp = self.data.close[0] <= tp
                hit_sl = self.data.close[0] >= sl

            # time-based exit
            timeout = (dt >= self.entry_bar + self.p.time_exit)

            if hit_tp or hit_sl or fade or timeout:
                self.close()
                return

if __name__ == '__main__':
    cerebro = bt.Cerebro(optreturn=False)

    # Define parameter ranges for optimization
    cerebro.optstrategy(
        LongShortStrategy,
        ema_short=[5, 10, 15],
        ema_long=[20, 50, 100],
        sma_trend=[100, 200],
        rsi_period=[14, 20],
        stoch_period=[14, 20],
        atr_period=[14],
        tp_mult=[1.0, 1.5, 2.0],
        sl_mult=[0.5, 1.0],
        time_exit=[20, 30],
        allocation=[0.5, 1.0],
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