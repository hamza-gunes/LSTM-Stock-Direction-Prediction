# LSTM Stock Trend Forecaster (LSTM ile Hisse Fiyat Tahmini)

A time-series forecasting tool designed to analyze stock market trends using Deep Learning. This project leverages Long Short-Term Memory (LSTM) networks to predict future stock prices based on historical data.

This project does not constitute investment advice !

## Overview

Financial markets are complex and volatile. This project aims to model the underlying patterns in stock price movements using a sequential neural network architecture. It fetches live data, processes it for time-series analysis, trains a predictive model, and generates actionable trading signals.

### Key Features
* **Real-time Data Integration:** Automatically fetches historical data using Yahoo Finance.
* **Sequential Modeling:** Utilizes a stacked LSTM architecture to capture temporal dependencies.
* **Robust Preprocessing:** Implements MinMax scaling and sequence generation to prevent data leakage.
* **Actionable Insights:** Provides a clear BUY/SELL/HOLD signal based on the predicted next-day close.
* **Visualization:** Plots training history, validation data, and model predictions for performance assessment.

## Technical Architecture

* **Language:** Python 3.x
* **Deep Learning Framework:** Keras (TensorFlow Backend)
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib
* **Data Source:** yfinance

## Methodology

1.  **Data Acquisition:** Fetches the last 2 years of daily closing prices.
2.  **Feature Engineering:** * Data is normalized to the [0, 1] range for optimal LSTM convergence.
    * Transforms data into a 3D structure: `[Samples, Time Steps, Features]`.
    * Uses a look-back window of 60 days.
3.  **Model Architecture:**
    * Input LSTM Layer (50 units, return sequences) + Dropout (0.2)
    * Hidden LSTM Layer (50 units) + Dropout (0.2)
    * Dense Output Layer (1 unit)
    * Optimizer: Adam | Loss Function: MSE
4.  **Evaluation:** The model is evaluated using RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error) metrics on unseen test data.

## Usage

1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/lstm-stock-forecaster.git](https://github.com/yourusername/lstm-stock-forecaster.git)
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the analyzer:
    ```bash
    python main.py
    ```

## 📉 Example Output

```text
ANALYSIS REPORT: KFEIN.IS
========================================
Last Close:      105.40
Predicted Close: 102.28
Expected Change: %-2.96
Signal:          SELL (Bearish Signal)
========================================
Model Metrics -> RMSE: 2.50 | MAE: 1.99
