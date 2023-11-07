# predicting-stock-prices-using-facebook-s-prophet
 

## Table of Contents
- [Introduction](#introduction)
- [Importing Libraries](#importing-libraries)
- [Importing Datasets](#importing-datasets)
- [Tesla (TSLA) Stock Analysis](#tesla-tsla-stock-analysis)
- [TCS (TCS.NS) Stock Analysis](#tcs-tcsns-stock-analysis)
- [S&P Global (GSPC) Stock Analysis](#sp-global-gspc-stock-analysis)
- [Bitcoin (BTC-USD) Stock Analysis](#bitcoin-btc-usd-stock-analysis)
- [Conclusion](#conclusion)

## Introduction
_Time series forecast can be used in a wide variety of applications such as Budget Forecasting, Stock Market Analysis, etc. But as useful it is also challenging to forecast the correct projections, Thus can’t be easily automated because of the underlying assumptions and factors. The analysts who produced accurate forecasts are also rare, and there is a big market available for them because it requires a substantial understanding of statistics and data analysis and has prior experience of producing time series forecasting.

Facebook open-sourced its time-series forecasting tool called Prophet in 2017 which produced accurate forecasts as produced by skilled analysts with a minimum amount of human efforts. The Facebook prophet is available in the form of API in Python and R/

How Prophet Works:

Facebook Prophet using Additive Regressive models using the following four components:


y(t) = g(t) + s(t) + h(t) + \epsilon_t

g(t): A piecewise linear or logistic growth curve trend. Prophet automatically detects changes in trends by selecting change points from the data.
s(t): A yearly seasonal component modeled using the Fourier series and weekly seasonal component using dummy variable
h(t): A user-provided list of important holidays.
et:  Error term used by the prophet.
Advantages of Facebook Prophet:

the prophet is optimized for business-related problems that are encountered at Facebook, it has the following characteristics:

The Facebook prophet is as accurate as a skilled analyst and can generate results in seconds
Facebook prophet requires minimal data processing and can deal with several outliers and null values.
User can add seasonality and holidays values manually, this can help easily integrate the particular domain knowledge.
In this post, we will use Facebook prophet with Python. We try to forecast the share price of Amazon Stock (from 2019-2020) using the share price data from (2015-2019).

Implementation:

For this post, we will be using Amazon Stock Price data, it can be downloaded from yahoo finance website.
First, we need to install the fbprophet tool, it can be installed with the following command in python._

Stock price forecasting is a crucial task for investors and traders. This project uses Facebook's Prophet model to predict the future stock prices of various companies. Prophet is a powerful tool for time series forecasting, which can capture daily, weekly, and yearly seasonality along with holiday effects.

In this project, we analyze stock prices of the following companies:
- Tesla (TSLA)
- Tata Consultancy Services (TCS.NS)
- S&P Global (GSPC)
- Bitcoin (BTC-USD)

We aim to provide insights into the stock price trends and make predictions using the Prophet model.

## Importing Libraries
We start by importing the necessary Python libraries such as NumPy, pandas, and Matplotlib. We also install the Facebook Prophet library using pip.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet
```

## Importing Datasets
We load historical stock price data for the selected companies using pandas. The data includes features like Date, Open, High, Low, Close, Adj Close, and Volume.

```python
# Load the dataset using pandas
data_tesla = pd.read_csv("tesla-stock.csv")
data_tcs = pd.read_csv("tcs-stock.csv")
data_sp_global = pd.read_csv("sp-global-stock.csv")
data_bitcoin = pd.read_csv("bitcoin-stock.csv")
```

## Tesla (TSLA) Stock Analysis
For each company, we perform the following steps:
1. Data preprocessing: We select and rename relevant columns.
2. Model development: We use the Prophet model with daily seasonality.
3. Forecasting: We predict future stock prices and visualize the results.
4. Visualization: We plot the forecasts and seasonal components.

```python
# Data preprocessing
data_tesla = data_tesla[["Date", "Close"]]
data_tesla = data_tesla.rename(columns={"Date": "ds", "Close": "y"})

# Model development and forecasting
m_tesla = Prophet(daily_seasonality=True)
m_tesla.fit(data_tesla)
future_tesla = m_tesla.make_future_dataframe(periods=365)
prediction_tesla = m_tesla.predict(future_tesla)

# Visualization
m_tesla.plot(prediction_tesla)
plt.title("Prediction of the Tesla Stock Price using Prophet")
plt.xlabel("Date")
plt.ylabel("Close Stock Price")
plt.show()

m_tesla.plot_components(prediction_tesla)
plt.show()
```

## TCS (TCS.NS) Stock Analysis
(Repeat the analysis steps for TCS stock)

## S&P Global (GSPC) Stock Analysis
(Repeat the analysis steps for S&P Global stock)

## Bitcoin (BTC-USD) Stock Analysis
(Repeat the analysis steps for Bitcoin stock)

## Conclusion
In this project, we used the Facebook Prophet model to analyze and forecast stock prices for different companies. We observed trends, made predictions, and visualized the results. Prophet is a powerful tool for time series forecasting that can provide valuable insights for investors and traders.

 
