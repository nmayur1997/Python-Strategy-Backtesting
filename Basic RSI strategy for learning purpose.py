#!/usr/bin/env python
# coding: utf-8

# In[5]:


import yfinance as yf
import backtrader as bt
import datetime

# Fetch data from Yahoo Finance
def fetch_data(symbol, start, end):
    return yf.download(symbol, start=start, end=end)

# Custom Strategy
class RSIStrategy(bt.Strategy):
    params = (
        ('rsi_period', 14),
        ('rsi_low', 30),
        ('rsi_high', 75),
        ('trade_size', 0.25),
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI_Safe(self.data.close, period=self.params.rsi_period)
        self.buy_signal = None
        self.sell_signal = None
        self.trades = []  # To store trades details
        self.lowest_after_buy = None

    def next(self):
        if self.position:
            # Update the lowest price after buying
            if self.lowest_after_buy is None or self.data.close < self.lowest_after_buy:
                self.lowest_after_buy = self.data.close[0]
        else:
            self.lowest_after_buy = None

        if self.rsi < self.params.rsi_low:
            if not self.position:
                size = int(self.broker.get_cash() * self.params.trade_size / self.data.close)
                self.buy(size=size)
                self.buy_signal = self.data.datetime.date(0)
                self.sell_signal = None
                self.buy_price = self.data.close[0]
                self.buy_quantity = size
                self.buy_cost = self.buy_price * self.buy_quantity
                self.lowest_after_buy = self.buy_price

        elif self.rsi > self.params.rsi_high:
            if self.position:
                self.sell(size=self.position.size)
                self.sell_signal = self.data.datetime.date(0)
                drawdown = ((self.lowest_after_buy - self.buy_price) / self.buy_price) * 100

                self.trades.append({
                    'Buy Date': self.buy_signal,
                    'Sell Date': self.sell_signal,
                    'Quantity': self.buy_quantity,
                    'Money Used': self.buy_cost,
                    'PnL': self.broker.getvalue() - self.buy_cost,
                    'Drawdown (%)': drawdown
                })

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, {order.executed.price:.2f}, Quantity: {order.executed.size}, Money Used: {order.executed.price * order.executed.size:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, {order.executed.price:.2f}, Quantity: {order.executed.size}')

    def log(self, txt, dt=None):
        dt = dt or self.data.datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def stop(self):
        initial_value = 100000.0
        final_value = self.broker.getvalue()
        net_return = final_value - initial_value
        cumulative_percentage_return = (net_return / initial_value) * 100
        
        print(f'Ending Value: {final_value:.2f}')
        print(f'Net Return: {net_return:.2f}')
        print(f'Cumulative Percentage Net Return: {cumulative_percentage_return:.2f}%')
        print(f'Trades: {len(self.trades)}')
        for trade in self.trades:
            print(trade)

# Run Backtest for Multiple Stocks
def run_backtests(symbols):
    start_date = '2023-01-01'
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')

    for symbol in symbols:
        print(f'Running backtest for {symbol}')
        data = fetch_data(symbol, start=start_date, end=end_date)
        if data is not None and not data.empty:
            cerebro = bt.Cerebro()
            cerebro.addstrategy(RSIStrategy)
            data_feed = bt.feeds.PandasData(dataname=data)
            cerebro.adddata(data_feed)
            cerebro.broker.setcash(100000.0)
            cerebro.broker.setcommission(commission=0.001)

            print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
            cerebro.run()
            print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())

            # Plot the results
            cerebro.plot(style='candlestick')
        else:
            print(f'No data found for {symbol}')

# List of stock symbols to backtest
symbols = ["ADANIENT.NS","ASIANPAINT.NS"]

# Run backtests for the list of symbols
run_backtests(symbols)


# In[ ]:




