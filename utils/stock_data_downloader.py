
from itertools import combinations
import argparse
import os,sys
import datetime as dt
import yfinance as yf
import numpy as np
import more_itertools
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from ta.momentum import WilliamsRIndicator  
import pandas_ta as ta

def calculate_ema(df,window = 10):
    ema = df.copy()
    smoothing = 2/(window+1)
    for i in range(len(df)-1):
        ema[i+1] = (1-smoothing)*ema[i] + smoothing*df[i+1]
    return ema 

def calculate_mom(df,windwow=10):
    mom = df.copy()
    mom[:] = np.nan

    for i in range(windwow-1,len(df)):
        mom[i] = df[i]-df[i-(windwow-1)]
    return mom

def calculate_rsi(df,window=10,ema=False):
    rsi = df.copy()
    rsi["diff"] = rsi["Close"].diff(1)
    rsi['gain'] = rsi['diff'].clip(lower=0)
    rsi['loss'] = rsi['diff'].clip(upper=0).abs()
    if ema == True:
	    # Use exponential moving average
        rsi['avg_gain'] = rsi['gain'].ewm(com = window - 1, adjust=True, min_periods = window).mean()  
        rsi['avg_loss'] = rsi['loss'].ewm(com = window - 1, adjust=True, min_periods = window).mean()
    else:
        # Use simple moving average
        rsi['avg_gain'] = rsi['gain'].rolling(window=window, min_periods=window).mean()
        rsi['avg_loss'] = rsi['loss'].rolling(window=window, min_periods=window).mean()

    rsi['rs'] = rsi['avg_gain'] / rsi['avg_loss']
    rsi['rsi'] = 100 - (100 / (1.0 + rsi['rs']))
    return rsi[["rsi"]]


if __name__ == "__main__":
    # Instantiate the parser
    parser = argparse.ArgumentParser(description="Download stock prices for a specified TICKER and DATE and create 15 different indicator.", epilog="""-------------------""")

    # Required positional arguments
    parser.add_argument("--ticker-name", type=str,
                        help="[string] specify the ticker name.",required=True)
    parser.add_argument("--start-interval-date", type=lambda s: dt.datetime.strptime(s, '%Y-%m-%d'),
                        help="[yyyy-MM-dd] specify start date.",required=True)
    parser.add_argument("--end-interval-date", type=lambda s: dt.datetime.strptime(s, '%Y-%m-%d'),
                        help="[yyyy-MM-dd] specify end date.",required=True)
    parser.add_argument("--output-dir", type=str,
                        help="[string] specify output directory.",required=True)

    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    TICKER_NAME = args.ticker_name
    START_DATE = args.start_interval_date
    END_DATE = args.end_interval_date
    OUTPUT_DIR = os.path.join(os.getcwd(), args.output_dir)
    if not os.path.exists(OUTPUT_DIR):
        print(f"Error: I could not find the file:\n    {OUTPUT_DIR}\nThis file is needed (launch this script with the -h flag to know more about its usage!).")
        exit(1)
    
    #search by ticker name
    tkt = yf.Ticker(TICKER_NAME)

    #get history data 
    stock_price = tkt.history(start=START_DATE,end=END_DATE)
    stock_price = stock_price.dropna()
    close_price = stock_price["Close"]

    #simple moving average 10 days
    SMA10 = close_price.rolling(10).mean()

    #weighted moving average 10 days
    weights = np.arange(1,11)
    WMA10 = close_price.rolling(10).apply(lambda x: np.sum(weights*x)) / sum(weights)

    #exponential moving average 10 days
    smoothing = 2/11
    # EMA10 = close_price.ewm(alpha=smoothing,adjust=False).mean()
    EMA10 = calculate_ema(close_price,10)

    #Momentum 10 days
    MOM = close_price - close_price.shift(9)
    # MOM = calculate_mom(close_price,10)

    #Stochastic oscillators
    k_period = 10
    k_stock = pd.DataFrame()
    k_stock["k_high"] = stock_price["High"].rolling(10).max()
    k_stock["k_low"] = stock_price["Low"].rolling(10).min()
    k_stock["Stochastic K%"] = (close_price - k_stock["k_low"]) / (k_stock["k_high"]-k_stock["k_low"]) * 100
    k_stock["Stochastic D%"] = k_stock["Stochastic K%"].rolling(k_period).mean()

    STOCHASTIC_K = k_stock["Stochastic K%"]
    STOCHASTIC_D = k_stock["Stochastic D%"]

    #Relative Strength Index (RSI)
    # RSI2 = ta.rsi(close=close_price, length=10)
    RSI = calculate_rsi(stock_price,10)

    #Moving average convergence divergence
    # m_ema12 = close_price.ewm(span=12, adjust=False, min_periods=12).mean()
    # m_ema26 = close_price.ewm(span=26, adjust=False, min_periods=26).mean()
    # macd = m_ema12 - m_ema26
    MACD = ta.macd(close=close_price, fast=12, slow=26, signal=9, append=True)["MACD_12_26_9"]

    #Larry William % range oscillator
    R = WilliamsRIndicator(high=stock_price["High"],low=stock_price["Low"],close=close_price,lbp=10).williams_r()


    # plt.plot(macd.ewm(span=9, adjust=False, min_periods=9).mean()[-500:])
    # plt.plot(macd[-500:])
    # plt.show()


    # fig = go.Figure(data=[go.Candlestick(x=stock_price.index,
    #             open=stock_price['Open'].values,
    #             high=stock_price['High'].values,
    #             low=stock_price['Low'].values,
    #             close=stock_price.values)])

    # fig.show()
    aaa="asd"


    # |0 |1 |2 |3 |4 |5 |6 |7 |8 |9 |10 |

    #example usage: python stock_data_downloader.py --ticker-name 6857.T --start-interval-date 2007-01-23 --end-interval-date 2013-12-30 --output-dir ../datasets/stock_prices/nikkei_225/
