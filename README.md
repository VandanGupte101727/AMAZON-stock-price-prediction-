# AMAZON-stock-price-prediction-
Amazon stock price prediction model using KNN method

Dataset from Kaggle(updated till Aug 1st 2025)

# Stock Price Prediction and Buy/Sell Classification using KNN

This project implements two machine learning models for stock market analysis using the K-Nearest Neighbors (KNN) algorithm:

1. A regression model that predicts the future closing price of a stock.
2. A classification model that provides a binary signal indicating whether to buy or sell the stock.

## Technologies Used

- Python
- pandas
- scikit-learn
- numpy
- matplotlib (optional, for visualizations)

## Dataset

The project uses a CSV file (example: `AMAZON_monthly.csv`) with the following columns:

- `Date` (format: YYYY-MM-DD)
- `Open`
- `High`
- `Low`
- `Volume`
- `Close` (target variable for regression)
- Additional columns may be included depending on available data

## Project Overview

### 1. KNN Regressor for Stock Price Prediction

- Features: All columns except `Close` and `Date`
- Target: `Close`
- The `Date` column is converted to a numerical format using `.map(pd.Timestamp.toordinal)` before model training

**Evaluation Metrics:**

- R² Score (Coefficient of Determination)
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

### 2. KNN Classifier for Buy/Sell Signal

- The label is derived by comparing the next day's closing price with the current day's:
  ```python
  data['Signal'] = (data['Close'].shift(-1) > data['Close']).map({True: 1, False: -1})
