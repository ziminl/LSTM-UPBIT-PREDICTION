

import ccxt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

def get_candles_data():
    exchange = ccxt.binance()
    now = datetime.now()
    start_time = datetime(2017, 10, 1)
    file_path = './daily_ETH_candles_2017.csv'

    with open(file_path, 'w') as file:
        headers = "timestamp,open,high,low,close,volume\n"
        file.write(headers)

        while now > start_time:
            since = exchange.parse8601(start_time.isoformat() + 'Z')
            candles = exchange.fetch_ohlcv('ETH/USDT', timeframe='1d', since=since, limit=1000)
            if not candles:
                break

            for candle in candles:
                timestamp = datetime.fromtimestamp(candle[0] / 1000).strftime('%Y-%m-%d %H:%M:%S')
                data = ",".join([timestamp, str(candle[1]), str(candle[2]), str(candle[3]), str(candle[4]), str(candle[5])])
                file.write(data + '\n')

            start_time += timedelta(days=1000)  # Increment start_time for the next batch

    print("Data saved to:", file_path)

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df[['close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(X_train, y_train):
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
    return model

def evaluate_model(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test)
    return predictions, y_test

def predict_future(model, last_sequence, scaler, future_steps=30):
    predictions = []
    current_sequence = last_sequence

    for _ in range(future_steps):
        prediction = model.predict(current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1]))
        predictions.append(prediction[0, 0])
        current_sequence = np.append(current_sequence[1:], prediction, axis=0)
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

def main():
    file_path = './daily_ETH_candles_2017.csv'
    
    get_candles_data()  # Fetch data

    scaled_data, scaler = load_and_preprocess_data(file_path)
    
    sequence_length = 60
    X, y = create_sequences(scaled_data, sequence_length)
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = train_model(X_train, y_train)
    
    predictions, y_test = evaluate_model(model, X_test, y_test, scaler)
    
    last_sequence = X_test[-1]
    future_predictions = predict_future(model, last_sequence, scaler, future_steps=30)
    
    print("Future Predictions:", future_predictions)

if __name__ == "__main__":
    main()
