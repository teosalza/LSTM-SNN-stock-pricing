
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
from ta.momentum import WilliamsRIndicator,ROCIndicator  
from ta.volume import AccDistIndexIndicator,OnBalanceVolumeIndicator
from ta.volatility import BollingerBands
from ta.trend import CCIIndicator
import pandas_ta as ta

def calculate_ema(df,window = 10):
    ema = df.copy()
    smoothing = 2/(window+1)
    for i in range(len(df)-1):
        ema[i+1] = (1-smoothing)*ema[i] + smoothing*df[i+1]
    return ema 

def calculate_disparity_index(close,sma_n):
    return (close-sma_n)/(sma_n*100)

def calculate_mom(df,windwow=10):
    mom = df.copy()
    mom[:] = np.nan

    for i in range(windwow-1,len(df)):
        mom[i] = df[i]-df[i-(windwow-1)]
    return mom

def calculate_psy(close_price,window=12):
    psy = close_price.copy()
    psy.iloc[:] = None
    el = close_price.rolling(window=window).apply(lambda x: np.sum(np.diff(x) > 0) + 1 if len(x) == window else np.nan)
    return el

def calculate_asy(close_price,window):
    psy = close_price.copy()
    psy.iloc[:] = None
    el = close_price.rolling(window=window).apply(lambda x: np.sum(x)/window if len(x) == window else np.nan)
    # for el in close_price.rolling(window):
    return el


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
    return rsi["rsi"]


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

    WINDOW_SIZE =10

    #simple moving average 10 days
    SMA10 = close_price.rolling(WINDOW_SIZE).mean()
    SMA10.name = "sma10"

    SMA5 = close_price.rolling(5).mean()
    SMA5.name = "sma5"

    SMA6 = close_price.rolling(6).mean()
    SMA6.name = "sma6"

    #weighted moving average 10 days
    weights = np.arange(1,11)
    WMA10 = close_price.rolling(WINDOW_SIZE).apply(lambda x: np.sum(weights*x) / sum(weights))
    WMA10.name = "wma10"

    #exponential moving average 10 days
    smoothing = 2/11
    EMA10 = close_price.ewm(alpha=smoothing,adjust=False).mean()
    # EMA10 = calculate_ema(close_price,WINDOW_SIZE)
    EMA10.name = "ema10"

    #Momentum 10 days
    MOM = close_price - close_price.shift(WINDOW_SIZE-1)
    MOM.name = "mom"
    # MOM = calculate_mom(close_price,10)

    #Stochastic oscillators
    k_period = 10
    k_stock = pd.DataFrame()
    k_stock["k_high"] = stock_price["High"].rolling(WINDOW_SIZE).max()
    k_stock["k_low"] = stock_price["Low"].rolling(WINDOW_SIZE).min()
    k_stock["Stochastic K%"] = (close_price - k_stock["k_low"]) / (k_stock["k_high"]-k_stock["k_low"]) * 100
    k_stock["Stochastic D%"] = k_stock["Stochastic K%"].rolling(k_period).mean()

    STOCHASTIC_K = k_stock["Stochastic K%"]
    STOCHASTIC_K.name = "stck%"
    STOCHASTIC_D = k_stock["Stochastic D%"]
    STOCHASTIC_D.name = "stcd%"

    #Relative Strength Index (RSI)
    RSI = ta.rsi(close=close_price, length=10)
    # RSI = calculate_rsi(stock_price,WINDOW_SIZE)

    #Moving average convergence divergence
    # m_ema12 = close_price.ewm(span=12, adjust=False, min_periods=12).mean()
    # m_ema26 = close_price.ewm(span=26, adjust=False, min_periods=26).mean()
    # macd = m_ema12 - m_ema26
    MACD = ta.macd(close=close_price, fast=12, slow=26, signal=10, append=True)["MACDs_12_26_10"]
    MACD.name = "macd"

    #Larry William % range oscillator
    R = WilliamsRIndicator(high=stock_price["High"],low=stock_price["Low"],close=close_price,lbp=10).williams_r()
    R.name = "r"

    #Accumulation Distribution oscillator
    AD = AccDistIndexIndicator(high=stock_price["High"],low=stock_price["Low"],close=close_price.shift(1),volume=stock_price["Volume"]).acc_dist_index()
    AD.name = "ad"

    #Commodity Channnel Index
    CCI = CCIIndicator(high=stock_price["High"],low=stock_price["Low"],close=close_price,window=2).cci()

    #Rate of change indicator 
    ROC = ROCIndicator(close=close_price,window=WINDOW_SIZE).roc()

    #On Balance Indicator
    OBV = OnBalanceVolumeIndicator(close=close_price,volume=stock_price["Volume"]).on_balance_volume()

    omega = close_price.diff().apply(lambda x: -1 if x < 0 else 1)
    OBV_1 = OBV.shift(1) + omega* stock_price["Volume"]

    BIAS6 = (close_price-SMA6)/SMA6 * 100
    PSY12 = calculate_psy(close_price)
    PSY12 = PSY12/12*100

    SY = np.log(close_price)/np.log(close_price.shift(1))*100
    ASY5 = calculate_asy(SY,5)
    ASY4 = calculate_asy(SY,4)
    ASY3 = calculate_asy(SY,3)
    ASY2 = calculate_asy(SY,2)
    ASY1 = calculate_asy(SY,1)

    #Disparity index momentum
    DIS = calculate_disparity_index(close=close_price,sma_n=SMA10)
    DIS.name = "dis"

    #Bollinger bands
    BB = BollingerBands(close=close_price,window=WINDOW_SIZE,window_dev=2)
    BB_LOW = BB.bollinger_lband()
    BB_HIGH = BB.bollinger_hband()

    # T_N = close_price
    # T_N.name = "t_n"
    # T_N1 = close_price.shift(-1)
    # T_N1.name = "t_n1"
    # T_N2 = close_price.shift(-2)
    # T_N2.name = "t_n2"
    
    close_trend = close_price.shift(-1) - close_price
    close_trend[close_trend >= 0] = 1
    close_trend[close_trend < 0] = 0
    T_N = close_trend
    T_N.name = "t_n"

    # T_N = close_price
    # T_N.name = "t_n"
    # T_N1 = close_price.shift(-1)
    # T_N1.name = "t_n1"
    # T_N2 = close_price.shift(-2)
    # T_N2.name = "t_n2"

    # T_N = close_price.shift(-1)
    # T_N.name = "t_n"
    # T_N1 = close_price.shift(-2)
    # T_N1.name = "t_n1"
    # T_N2 = close_price.shift(-3)
    # T_N2.name = "t_n2"

    # final_dataframe = pd.DataFrame([SMA10,WMA10,EMA10,MOM,STOCHASTIC_K,STOCHASTIC_D,RSI,MACD,R,AD,CCI,ROC,OBV,DIS,BB_LOW,BB_HIGH,T_N,T_N1,T_N2]).T
    final_dataframe = pd.DataFrame([SMA10,WMA10,EMA10,MOM,STOCHASTIC_K,STOCHASTIC_D,RSI,MACD,R,AD,CCI,ROC,OBV,DIS,BB_LOW,BB_HIGH,T_N]).T
    # final_dataframe = pd.DataFrame([SMA10,WMA10,MOM,STOCHASTIC_K,STOCHASTIC_D,RSI,R,AD,CCI,T_N]).T
    # final_dataframe = pd.DataFrame([OBV_1,SMA5,BIAS6,PSY12,ASY5,ASY4,ASY3,ASY2,ASY1,T_N]).T
    final_dataframe.dropna(inplace=True)
    final_dataframe.index = final_dataframe.index.tz_localize(None)
    final_dataframe.to_csv("{}{}.csv".format(OUTPUT_DIR,TICKER_NAME))



    # plt.plot(macd.ewm(span=9, adjust=False, min_periods=9).mean()[-500:])
    # plt.plot(macd[-500:])
    # plt.show()


    # fig = go.Figure(data=[go.Candlestick(x=stock_price.index,
    #             open=stock_price['Open'].values,
    #             high=stock_price['High'].values,
    #             low=stock_price['Low'].values,
    #             close=stock_price.values)])   

    # fig.show()


    # |0 |1 |2 |3 |4 |5 |6 |7 |8 |9 |10 |

    #example usage: python stock_data_downloader.py --ticker-name 6857.T --start-interval-date 2007-01-23 --end-interval-date 2013-12-30 --output-dir ../datasets/stock_prices/nikkei_225/
    