import backtrader as bt
import yfinance as yf
import pandas as pd
from datetime import datetime
from scipy.stats import pearsonr


# — your AAPLStrategy as given —


class DonchianBreakout(bt.Strategy):
    params = dict(
        dc_period = 20,   # look-back for Donchian channel
    )

    def __init__(self):
        # Upper = highest high over past dc_period bars (excluding current bar)
        self.dc_upper = bt.indicators.Highest(self.data.high(-1), period=self.p.dc_period)
        # Lower = lowest low over past dc_period bars (excluding current bar)
        self.dc_lower = bt.indicators.Lowest (self.data.low(-1),  period=self.p.dc_period)

    def next(self):
        today = self.data.datetime.date(0)
        if not (datetime(2024,1,22).date() <= today <= datetime(2024,12,22).date()):
            return
        
        close = self.data.close[0]

        # ENTRY: no position & breakout above upper band
        if not self.position and close > self.dc_upper[0]:
            size = int(self.broker.getcash() / close)
            if size:
                self.buy(size=size)
                # optional: log or store entry price

        # EXIT: in position & close below lower band
        elif self.position and close < self.dc_lower[0]:
            self.close()
if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.addstrategy(DonchianBreakout)

    # 1) Download into a DataFrame
    df = yf.download('TSLA',
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