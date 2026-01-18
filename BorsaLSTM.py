
# Hamza Gunes - 18/01/2026

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import datetime
from dateutil.relativedelta import relativedelta
import warnings

warnings.filterwarnings('ignore')

class MarketAnalyzer:
    def __init__(self, ticker, look_back=60, history_years=2):
        self.ticker = ticker
        self.look_back = look_back
        self.history_years = history_years
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.df = None
        self.data = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.y_test_actual = None

    def fetch_data(self):
        end_date = datetime.date.today()
        start_date = end_date - relativedelta(years=self.history_years)
        
        print(f"Fetching data for {self.ticker}...")
        self.df = yf.download(self.ticker, start=start_date, end=end_date)
        self.data = self.df[['Close']].values
        
    def _create_dataset(self, dataset):
        X, y = [], []
        for i in range(self.look_back, len(dataset)):
            X.append(dataset[i-self.look_back:i, 0])
            y.append(dataset[i, 0])
        return np.array(X), np.array(y)

    def process_data(self, split_ratio=0.8):
        train_size = int(np.ceil(len(self.data) * split_ratio))
        
        train_data_raw = self.data[0:train_size, :]
        self.scaler.fit(train_data_raw)
        scaled_data = self.scaler.transform(self.data)

        train_data = scaled_data[0:train_size, :]
        self.x_train, self.y_train = self._create_dataset(train_data)
        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], self.x_train.shape[1], 1))

        test_data = scaled_data[train_size - self.look_back:, :]
        self.x_test, self.y_test = self._create_dataset(test_data)
        self.x_test = np.reshape(self.x_test, (self.x_test.shape[0], self.x_test.shape[1], 1))
        
        self.y_test_actual = self.data[train_size:, :]

    def build_architecture(self):
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(50, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(25))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train(self, epochs=30, batch_size=32):
        stopper = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        self.model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs, 
                       validation_data=(self.x_test, self.y_test), callbacks=[stopper], verbose=1)

    def forecast_and_visualize(self):
        last_sequence = self.data[-self.look_back:]
        last_sequence_scaled = self.scaler.transform(last_sequence)
        
        X_pred = np.array([last_sequence_scaled])
        X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1))
        
        predicted_price = self.model.predict(X_pred)
        predicted_price = self.scaler.inverse_transform(predicted_price)[0][0]
        current_price = self.data[-1][0]
        
        change = ((predicted_price - current_price) / current_price) * 100
        
        print("\n" + "="*40)
        print(f"ANALYSIS REPORT: {self.ticker}")
        print("="*40)
        print(f"Last Close:      {current_price:.2f}")
        print(f"Predicted Close: {predicted_price:.2f}")
        print(f"Expected Change: %{change:.2f}")
        
        if predicted_price > current_price:
            signal = "BUY (Bullish Signal)"
        elif predicted_price < current_price:
            signal = "SELL (Bearish Signal)"
        else:
            signal = "HOLD (Neutral)"
            
        print(f"Signal:          {signal}")
        print("="*40 + "\n")

        test_preds = self.model.predict(self.x_test)
        test_preds = self.scaler.inverse_transform(test_preds)
        
        rmse = np.sqrt(mean_squared_error(self.y_test_actual, test_preds))
        mae = mean_absolute_error(self.y_test_actual, test_preds)
        print(f"Model Metrics -> RMSE: {rmse:.2f} | MAE: {mae:.2f}\n")

        train_idx = self.df.index[:len(self.df) - len(self.y_test_actual)]
        valid_idx = self.df.index[len(self.df) - len(self.y_test_actual):]
        
        plt.figure(figsize=(14, 7))
        plt.title(f'{self.ticker} - Price Prediction Model')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.plot(train_idx, self.df['Close'][:len(train_idx)], label='History', color='grey', alpha=0.6)
        plt.plot(valid_idx, self.df['Close'][len(train_idx):], label='Actual', color='green')
        plt.plot(valid_idx, test_preds, label='Prediction', color='red', linestyle='dashed')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    analyzer = MarketAnalyzer('SASA.IS')
    analyzer.fetch_data()
    analyzer.process_data()
    analyzer.build_architecture()
    analyzer.train()
    analyzer.forecast_and_visualize()