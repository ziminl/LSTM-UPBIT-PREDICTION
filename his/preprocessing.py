import numpy as np
import pandas_ta as ta
import pandas as pd


def window_data(data, window_size=2):
    X = []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size, :])
    return np.array(X)


def get_processed_data(data):
    data_inc = data.iloc[::-1].copy()
    data_inc.index = pd.to_datetime(data_inc['candle_date_time_kst'])
    data_inc.drop(['candle_date_time_kst', 'candle_date_time_utc', 'timestamp', 'unit', 'market'], axis=1, inplace=True)
    data_inc['rsi'] = ta.rsi(data_inc['trade_price'], length=14)
    period = 20
    ma = data_inc['trade_price'].rolling(window=period).mean()
    divergence = (data_inc['trade_price'] - ma) / ma
    data_inc['divergence'] = divergence
    data_inc.dropna(inplace=True)

    print(data_inc.tail(5))

    return data_inc
