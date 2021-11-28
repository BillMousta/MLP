import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import Data
from numpy.fft import *
import math
import numpy as np
import seaborn as sns
import Visualizations


"""
Import data from yahoo finance and split it into index (sp 500) and stock (companies from SP 500) data 
and adding labels according to daily percent of Adj Closing price
Load data from yahoo finance
a Python library that gives you current and
historical stock market price data from Yahoo Finance, and so much more.
"""
def get_data(stock, start, end, interval):
    filename = stock + '.csv'
    # data = yf.download(stock, start=start, end=end, interval=interval).drop(['Open','High', 'Low', 'Close', 'Volume'], axis=1)
    # Write data
    # data.to_csv(filename, encoding='utf-8', date_format='%Y-%m-%d')
    # Read from .csv file instead of yahoo finance
    data = pd.read_csv(filename, parse_dates=True, index_col='Date')
    # data.drop(['Open','High', 'Low', 'Close', 'Volume'], axis=1, inplace= True)
    data = add_features(data, True)
    # Adding Labels
    data = add_labels(data)
    return data


"""
Add useful indicators like Bollinger Bands, Moving Average etc
to get more use useful data and try to predict the price
"""
def add_features(stock, flag):
    stock['Daily Returns'] = stock['Adj Close'].pct_change(1)
    # Compute moving average of Adj Close price 5 41 days
    stock['MA 5'] = stock['Adj Close'].rolling(5).mean().shift(-5)
    # For crash example
    if not flag:
        return stock
    stock['MA 21'] = stock['Adj Close'].rolling(21).mean().shift(-21)

    # Compute Exponential moving average
    stock['Exp Ma'] = stock['Adj Close'].ewm(com=0.5).mean()

    # Compute Bollinger Bands
    stock['std 21'] = stock['Adj Close'].rolling(21).std()
    stock['upper_band'] = stock['MA 21'] + (stock['std 21'] * 2)
    stock['lower_band'] = stock['MA 21'] - (stock['std 21'] * 2)

    # Compute MACD = long term EMA(26 periods) - short-term EMA (12 periods)
    stock['26ema'] = stock['Adj Close'].ewm(span=12, adjust=False, min_periods=12).mean()
    stock['12ema'] = stock['Adj Close'].ewm(span=26, adjust=False, min_periods=26).mean()
    stock['MACD'] = (stock['12ema'] - stock['26ema'])
    stock = stock.drop(['26ema', '12ema'], axis=1)

    # Compute fourier transform
    stock = denoising(stock)

    return stock


"""
Filter signal according to threshold
"""
def filter_signal(signal, threshold=1e4):
    fourier = rfft(signal)
    frequencies = rfftfreq(signal.size, d = 20e-3 / signal.size)
    fourier[frequencies > threshold] = 0
    return irfft(fourier, n= signal.size)


"""
Using fast fourier transform to denoise the data
accoring with closing price
"""
def denoising(data):
    data['FT 10000 comp'] = filter_signal(data['Adj Close'])
    return data


"""
Add labels according to daily percent change a Moving Average of 7 days,
if pct change > 0.5% UP Trend -> label 2 Price probably goes UP
if pct change < -0.5% Down Trend -> label 1 Price probably goes Down
else Same -> label 0 
"""
def add_labels(data):

    movements = []
    # movement = (data['Adj Close'].rolling(5).mean().shift(-5) - data['Adj Close'])/data['Adj Close']
    movement = (data['MA 5'] - data['Adj Close']) / data['Adj Close']
    # print(movement.head(30))
    for percent in movement:
        if percent > 0.007: # price goes up
                # movements.append('Up')
                movements.append(2)
        elif percent < -0.005: # price goes down
                # movements.append('Down')
                movements.append(1)
        else:
                # movements.append('Same')
                movements.append(0)

    data['Label'] = movements
    data = data.drop(['Daily Returns'],axis=1)
    data = data.dropna()
    return data


"""
Return all correlations for data index for each stock
of data set stocks
"""
def get_correlations(index, stocks, start):
    cor = []
    for stock in stocks:
        data = get_data(stock, start="2000-01-01", end='2021-01-01', interval='1d')
        correlation = index['Adj Close'][start:].pct_change(1).corr(data['Adj Close'][start:].pct_change(1))
        cor.append(correlation)

    print("Correlations")
    for i in cor:
        print(i)
    return cor


