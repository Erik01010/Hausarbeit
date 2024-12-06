import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
import os
from scipy.stats import norm

# declare Variables
tickers = ["AAPL", "BTC-USD", "GC=F", "^GSPC", "EURUSD=X"]
start = "2014-01-01"
end = "2024-11-04"
interval = "1d"
filename = "multi_assets.csv"

# Check if Data is already downloaded and saved in csv-file.
# Make sure, csv-file is stored in same folder as this Jupiter Notebook!

if os.path.isfile(filename):
    df = pd.read_csv(filename, header=[0, 1], index_col=0, parse_dates=[0])
    print("read from csv complete")
else:
    # Download Data from Yahoo Finance API
    df = yf.download(tickers, start, end, interval)

    # Save dataframe to csv-file.
    df.to_csv(filename)
    print("Download complete. Data saved in {}".format(filename))

# create new DataFrame with close prices only
close = df.Close.copy()

# Show important stastitic metrics
close.describe()

# Linechart close-prices for 5 instruments

close.plot(figsize=(15, 8), fontsize=13)
plt.legend(fontsize=13)
plt.ylabel("Price")
plt.title("Price Chart")

# Calculate logarithmic returns and save in new DataFrame
returns = close.copy().apply(lambda x: np.log(x.dropna()/x.dropna().shift(1))).dropna()

# Create new DataFrame containing only cumulative returns
creturns = returns.copy()
for ticker in tickers:
    creturns["{}".format(ticker)] = creturns["{}".format(ticker)].cumsum()

# Linechart cumulative returns for 5 instruments

creturns.plot(figsize=(15,8), fontsize=13, grid=True);
plt.title("Cumulative logarithmic returns for 5 istruments")
plt.ylabel("cum log returns")
plt.legend(fontsize=13)

# Linechart BTC and moving averages

# Calculate simple moving averages for BTC
btc = creturns["BTC-USD"].copy().to_frame()
btc["sma50"] = btc["BTC-USD"].rolling(50).mean()
btc["sma100"] = btc["BTC-USD"].rolling(100).mean()
btc["sma200"] = btc["BTC-USD"].rolling(200).mean()

btc.plot(figsize=(15,8), fontsize=13)
plt.legend(fontsize=13)
plt.ylabel("log returns")
plt.title("log Returns and Simple Moving Averages for BTC-USD")
plt.show()

# Histogram: compare distribution of Apple daily returns with normal distribution
data = returns["AAPL"]

# Caluclating parameters for normal distribution
mean = data.mean()
std_dev = data.std()

# Creating Data for theoretic normal distribution
x = np.linspace(data.min(), data.max(), 100)
gauss = norm.pdf(x, mean, std_dev)

plt.figure(figsize=(12, 8))

# Histogram of daily returns
plt.hist(data, bins=100, density=True, alpha=0.8, color='blue', label='Daily Returns of AAPL')

# Histogram for normal distribution
plt.hist(x, bins=100, weights=gauss, alpha=0.4, color='red', label='Normally distributed returns')


plt.title('Distribution of AAPL Daily Returns Compared with a Normal Distribution')
plt.xlabel('Daily Returns')
plt.ylabel('Density')
plt.legend()
plt.show()

# Correlation-Heatmap

# Compute correlation
correlation_matrix = returns.corr()

# Plot Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis')
plt.title('Correlation between returns of different Instruments')
plt.show()
