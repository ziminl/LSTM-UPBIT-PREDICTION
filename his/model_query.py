import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
import tensorflow
import os
import pandas_ta as ta


def create_model(window_size, learning_rate, batch_size):
    model = Sequential([
        LSTM(units=128, activation='relu', input_shape=(window_size, X_train.shape[2]), return_sequences=True),
        Dropout(0.1),
        LSTM(units=64, activation='relu', return_sequences=True),
        Dropout(0.1),
        LSTM(units=32, activation='relu'),
        Dropout(0.1),
        Dense(units=16, activation='relu'),
        Dense(units=8, activation='relu'),
        Dense(units=1, activation='linear')
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mae'])

    return model

# TPU 초기화
resolver = tensorflow.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

# 데이터 불러오기 및 전처리
df = pd.read_csv('/content/drive/MyDrive/candles_30min_data2018.csv')
df = df.iloc[::-1]
df['acc_trade_price_diff'] = df['candle_acc_trade_price'] - df['candle_acc_trade_price'].shift(1)
df['acc_trade_volume_diff'] = df['candle_acc_trade_volume'] - df['candle_acc_trade_volume'].shift(1)
df.drop(0, inplace=True)
df.index = pd.to_datetime(df['candle_date_time_kst'])
df.drop(['candle_date_time_kst', 'candle_date_time_utc', 'timestamp', 'unit', 'market'], axis=1, inplace=True)
df['price_movement'] = df['high_price'] - df['low_price']
df['price_spread'] = df['trade_price'] - df['opening_price']
df['vwap'] = df['candle_acc_trade_price'] / df['candle_acc_trade_volume']
df = df.drop(df.head(20).index)
df['rsi'] = ta.rsi(df['trade_price'], length=14)
close_prices = df['trade_price']
fast_ema = close_prices.ewm(span=12, min_periods=0, adjust=False).mean()
slow_ema = close_prices.ewm(span=26, min_periods=0, adjust=False).mean()
macd = fast_ema - slow_ema
signal_line = macd.ewm(span=9, min_periods=0, adjust=False).mean()
macd_histogram = macd - signal_line
df['MACD'] = macd
df['Signal Line'] = signal_line
df['MACD Histogram'] = macd_histogram
df['future_info'] = df['trade_price'].shift(-1)
df.drop(['trade_price'], axis=1, inplace=True)
df.dropna(inplace=True)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.drop('future_info', axis=1))


def window_data(data, window_size):
    X = []
    y = []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size, :-1])
        y.append(data[i+window_size, -1])
    return np.array(X), np.array(y)

# 하이퍼파라미터 설정
window_sizes = [2, 3]
learning_rates = [0.01, 0.001]
batch_sizes = [32, 64, 128]

for window_size in window_sizes:
    for learning_rate in learning_rates:
        for batch_size in batch_sizes:
            print(f"Training model with window_size={window_size}, learning_rate={learning_rate}, batch_size={batch_size}")

            window_size = window_size
            learning_rate = learning_rate
            batch_size = batch_size

            X, y = window_data(np.concatenate((scaled_data, (df['future_info'].values.reshape(-1, 1))/10000), axis=1), window_size)
            train_size = int(0.8 * len(X))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            with strategy.scope():
                model = create_model(window_size, learning_rate, batch_size)

                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

                history = model.fit(X_train, y_train, epochs=100, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])

                prediction = model.predict(X_test)

                mae = mean_absolute_error(y_test, prediction)

                print("Mean Absolute Error:", mae)

                model.save(f'./model_ws{window_size}_lr{learning_rate}_bs{batch_size}_t3.h5')
