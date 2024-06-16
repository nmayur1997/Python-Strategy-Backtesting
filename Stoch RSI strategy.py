#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install yfinance')


# In[1]:


pip install yfinance vectorbt matplotlib


# In[27]:


import yfinance as yf
import vectorbt as vbt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate RSI
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# Function to calculate Stoch RSI
def calculate_stoch_rsi(data, rsi_period=14, stoch_period=14, smooth_k=3, smooth_d=3):
    data['RSI'] = calculate_rsi(data, rsi_period)
    
    min_rsi = data['RSI'].rolling(window=stoch_period, min_periods=1).min()
    max_rsi = data['RSI'].rolling(window=stoch_period, min_periods=1).max()
    
    data['StochRSI'] = (data['RSI'] - min_rsi) / (max_rsi - min_rsi)
    data['%K'] = data['StochRSI'].rolling(window=smooth_k, min_periods=1).mean()
    data['%D'] = data['%K'].rolling(window=smooth_d, min_periods=1).mean()
    
    return data

# Function to backtest Stoch RSI strategy with trade information
def backtest_stoch_rsi_strategy(ticker, initial_cash=100000, start_date='2022-01-01', end_date='2023-12-30', rsi_period=14, stoch_period=14, smooth_k=3, smooth_d=3):
    # Fetch historical data
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Calculate Stoch RSI
    stoch_rsi_data = calculate_stoch_rsi(data, rsi_period, stoch_period, smooth_k, smooth_d)
    
    # Generate buy and sell signals
    buy_signals = (stoch_rsi_data['%K'] < 0.05) & (stoch_rsi_data['%K'].shift(1) >= 0.05)
    sell_signals = (stoch_rsi_data['%K'] > 0.80) & (stoch_rsi_data['%K'].shift(1) <= 0.80)
    
    # Initialize variables to track trades
    trades = []
    current_trade = None
    total_quantity = 0
    total_capital_deployed = 0
    total_profit_loss = 0
    second_entry = False  # Flag to track if the second buy entry has been made
    
    # Loop through signals to track trades
    for i in range(len(stoch_rsi_data)):
        if buy_signals[i]:
            if current_trade is None:
                # First buy entry
                current_trade = {
                    'Buy Date': stoch_rsi_data.index[i],
                    'Buy Price': stoch_rsi_data['Close'].iloc[i],
                    'Sell Date': None,
                    'Sell Price': None,
                    'Quantity': 0,
                    'Capital Deployed (%)': 0,
                    'Profit/Loss ($)': 0
                }
                # Calculate capital deployed as 25% of initial cash
                current_trade['Capital Deployed (%)'] = 25
                total_capital_deployed += current_trade['Capital Deployed (%)']
                current_trade['Quantity'] = initial_cash * (current_trade['Capital Deployed (%)'] / 100) // stoch_rsi_data['Close'].iloc[i]
                total_quantity += current_trade['Quantity']
            elif not second_entry:
                # Second buy entry, only if it hasn't been made yet
                current_trade['Buy Date 2'] = stoch_rsi_data.index[i]
                current_trade['Buy Price 2'] = stoch_rsi_data['Close'].iloc[i]
                current_trade['Quantity 2'] = initial_cash * (current_trade['Capital Deployed (%)'] / 100) // stoch_rsi_data['Close'].iloc[i]
                total_quantity += current_trade['Quantity 2']
                second_entry = True
        elif sell_signals[i] and current_trade is not None:
            # Sell signal and there is an open trade
            sell_price = stoch_rsi_data['Close'].iloc[i]
            if sell_price > current_trade['Buy Price'] and (not second_entry or sell_price > current_trade['Buy Price 2']):
                # Exit only if sell price is higher than both buy prices
                current_trade['Sell Date'] = stoch_rsi_data.index[i]
                current_trade['Sell Price'] = sell_price
                current_trade['Profit/Loss ($)'] = (current_trade['Sell Price'] - current_trade['Buy Price']) * current_trade['Quantity']
                if second_entry:
                    current_trade['Profit/Loss ($)'] += (current_trade['Sell Price'] - current_trade['Buy Price 2']) * current_trade['Quantity 2']
                trades.append(current_trade)
                total_profit_loss += current_trade['Profit/Loss ($)']
                current_trade = None  # Reset current trade after selling
                second_entry = False
    
    # Convert trades to DataFrame
    trades_df = pd.DataFrame(trades)
    
    # Print all trades
    print(f'Ticker: {ticker}')
    if not trades_df.empty:
        print(trades_df[['Buy Date', 'Buy Price', 'Buy Date 2', 'Buy Price 2', 'Sell Date', 'Sell Price', 'Quantity', 'Quantity 2', 'Capital Deployed (%)', 'Profit/Loss ($)']])
        print(f'Total Trades: {len(trades_df)}')
        print(f'Total Quantity: {total_quantity}')
        print(f'Total Capital Deployed (%): {total_capital_deployed:.2f}%')
        print(f'Total Profit/Loss ($): {total_profit_loss:.2f}')
    else:
        print('No trades executed.')
    print()
    
    # Define the entry and exit signals based on trades
    entries = buy_signals
    exits = sell_signals
    
    # Backtest the strategy using vectorbt
    portfolio = vbt.Portfolio.from_signals(
        stoch_rsi_data['Close'],
        entries,
        exits,
        init_cash=initial_cash,
        fees=0.001,  # assuming 0.1% trading fees
        cash_sharing=True,  # cash is shared between all trades
        freq='1D'
    )
    
    # Calculate final portfolio value
    final_value = portfolio.cash().iloc[-1] + portfolio.asset_value().iloc[-1]
    net_cumulative_return_percent = (final_value - initial_cash) / initial_cash * 100
    
    # Print the portfolio performance
    print(f'Final Portfolio Value: ${final_value:.2f}')
    print(f'Net Cumulative Return (%): {net_cumulative_return_percent:.2f}%')
    print(portfolio.stats())
    print()
    
    # Plot the performance
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot the price and buy/sell signals
    ax1.plot(data['Close'], label='Close Price')
    ax1.plot(stoch_rsi_data.loc[buy_signals, 'Close'], '^', markersize=10, color='g', label='Buy Signal')
    ax1.plot(stoch_rsi_data.loc[sell_signals, 'Close'], 'v', markersize=10, color='r', label='Sell Signal')
    ax1.set_title(f'{ticker} Close Price with Buy/Sell Signals')
    ax1.legend()
    
    # Plot the Stoch RSI
    ax2.plot(stoch_rsi_data['%K'], label='%K')
    ax2.axhline(0.05, linestyle='--', alpha=0.5, color='r')
    ax2.axhline(0.80, linestyle='--', alpha=0.5, color='g')
    ax2.set_title(f'{ticker} Stochastic RSI (%K)')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# List of tickers to analyze
tickers = ['BHEL.NS']

# Loop through tickers and backtest the strategy for each
for ticker in tickers:
    backtest_stoch_rsi_strategy(ticker)


# In[ ]:





# In[ ]:





# In[ ]:




