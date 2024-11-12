# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 

### Developed by : KULASEKARAPANDIAN K
### Register No : 212222240052
### Date: 

### AIM:
To create a project on time series analysis of cryptocurrency data (ETH-USDT) using the ARIMA model in Python and compare it with other models.

### ALGORITHM:
• Explore the Dataset of Cryptocurrency Data

• Load the cryptocurrency dataset and perform initial exploration, focusing on the timestamp and value columns. Plot the time series to visualize trends.
Check for Stationarity of the Time Series

• Plot the time series data and use the following methods to assess stationarity:
• Time Series Plot: Visualize the data for seasonality or trends.
• ACF and PACF Plots: Inspect autocorrelation and partial autocorrelation plots to understand the lag structure.
• ADF Test: Apply the Augmented Dickey-Fuller (ADF) test to check if the series is stationary.

• If the series is not stationary (as indicated by the ADF test), apply differencing to remove trends and make the series stationary.
Determine ARIMA Model Parameters (p, d, q)

• Use insights from the ACF and PACF plots to select the AR and MA terms (p and q values).
Choose d based on the differencing applied to achieve stationarity.
• Fit the ARIMA Model

• Fit an ARIMA model with the selected (p, d, q) parameters on the historical cryptocurrency data values.
Make Time Series Predictions

• Forecast future values for a specified time period (e.g., 12 time intervals) using the fitted ARIMA model.
Auto-Fit the ARIMA Model (if applicable)

• Use auto-fitting methods (such as grid search or auto_arima from pmdarima) to automatically determine the best parameters for the model if needed.
Evaluate Model Predictions

• Compare the predicted values with actual values using metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE) to assess the model's accuracy.
Plot the historical data and forecasted values to visualize the model's performance.
### PROGRAM:

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the dataset
data = pd.read_csv('/mnt/data/ETHUSDT_PERP_15m.csv')

# Convert 'timestamp' column to datetime and set it as the index
data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
data.set_index('timestamp', inplace=True)

# Extract the 'close' price column for analysis (assuming 'close' column represents the value)
series = data['close'].dropna()

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(series)
plt.title("ETH-USDT Data Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Close Price")
plt.show()

# Augmented Dickey-Fuller Test
def adf_test(series):
    result = adfuller(series)
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    return result[1] < 0.05  # Returns True if series is stationary

# Check stationarity and difference the series if needed
is_stationary = adf_test(series)
if not is_stationary:
    series_diff = series.diff().dropna()
    plt.figure(figsize=(10, 5))
    plt.plot(series_diff)
    plt.title("Differenced ETH-USDT Data")
    plt.show()
else:
    series_diff = series

# Plot ACF and PACF
plot_acf(series_diff, lags=20)
plt.title("Autocorrelation (ACF) of Differenced Series")
plt.show()

plot_pacf(series_diff, lags=20)
plt.title("Partial Autocorrelation (PACF) of Differenced Series")
plt.show()

# ARIMA model parameters (p, d, q) - adjust based on ACF/PACF plots
p, d, q = 1, 1, 1  # Modify based on ACF/PACF insights if needed

# Fit the ARIMA model
model = ARIMA(series, order=(p, d, q))
fitted_model = model.fit()
print(fitted_model.summary())

# Forecasting
forecast_steps = 12  # Number of periods to forecast (e.g., 12 15-minute intervals if data is in 15-minute frequency)
forecast = fitted_model.forecast(steps=forecast_steps)

# Set up the forecast index
last_date = series.index[-1]
forecast_index = pd.date_range(last_date, periods=forecast_steps + 1, freq='15T')[1:]  # Shift to match forecast start

# Plot the forecast
plt.figure(figsize=(10, 5))
plt.plot(series, label="Historical Data")
plt.plot(forecast_index, forecast, label="Forecast", color='orange')
plt.legend()
plt.title("ETH-USDT Data Forecast")
plt.xlabel("Timestamp")
plt.ylabel("Close Price")
plt.show()
```

### OUTPUT:
![image](https://github.com/user-attachments/assets/dd27f84e-79b8-48c1-94df-96b8b4540e3b)


### RESULT:
Thus, the project on time series analysis on cryptocurrency (ETH-USDT) data based on the ARIMA model using Python is executed successfully.
