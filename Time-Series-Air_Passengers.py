#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 17:29:58 2021

@author: richasharma
"""

# Data and package Import
#Data Source - Kaggle - https://www.kaggle.com/rakannimer/air-passengers
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima_model import ARMA
import numpy as np
from statsmodels.tsa.stattools import adfuller


Data = pd.read_csv('AirPassengers.csv')
print(Data.head(10))
#The data represents the number of passengers travelling 
#by the flight every month. 

## AIM : Predict the number of passengers flying in the near future
## What is the demand of flights in such a case?


Data.index = pd.to_datetime(Data.Month) # convert index to date
Data = Data.drop(columns = 'Month') # Drop month column
Data = Data.rename(columns = {'#Passengers':'Passengers'}) # rename column

print(Data.head(10))


####Data Exploration
# Plotting the time-series

fig = plt.figure(figsize=(10,5))
ax= fig.add_subplot(111)

plt.xlabel('Date')
plt.ylabel('Number of Air Passengers')
plt.tick_params(axis='x', which='both',bottom=True,top=True, direction="in")   
plt.tick_params(axis='y', which='both',right=True,left=True, direction="in")
plt.title('Change in Number of Air Passengers with Time')

ax.minorticks_on()

plt.plot(Data, color="blue")

## Checking for Null values

print(Data.isnull().sum())
print(Data.describe())


## Check for Stationarity in the time-series


def stationarity_test(timeseries):
    


# # Method 1: Determining Rolling Statistics

#Determing rolling statistics
   rolLmean = timeseries.rolling(12).mean()  ## window size of 12 denotes 12 months, giving rolling mean at yearly interval
   rolLstd = timeseries.rolling(12).std()



#Plot rolling statistics:

   fig = plt.figure(figsize=(10,5))
   ax= fig.add_subplot(111)

   plt.xlabel('Date')
   plt.ylabel('Number of Air Passengers')
   plt.tick_params(axis='x', which='both',bottom=True,top=True, direction="in")   
   plt.tick_params(axis='y', which='both',right=True,left=True, direction="in")
   ax.minorticks_on()


   orig = plt.plot(timeseries, color='blue',label='Original')
   mean = plt.plot(rolLmean, color='red', label='Rolling Mean')
   std = plt.plot(rolLstd, color='black', label = 'Rolling Std')
   plt.legend(loc='best')
   plt.title('Rolling Mean & Standard Deviation')
   plt.show(block=False)
 ## We observe that the Rolling mean is not constant. 
 
 
 # # Method 2:  Augmented Dickey–Fuller test
# ADF Test - null hypothesis - non-stationary - if p-value < 5% reject null hypothesis


   adfuller_result = adfuller(timeseries['Passengers'], autolag='AIC')

   print(f'ADF Statistic: {adfuller_result[0]}')

   print(f'p-value: {adfuller_result[1]}')

   for key, value in adfuller_result[4].items():
     print('Critial Values:')
     print(f'   {key}, {value}')


stationarity_test(Data)



####  we see that p-value is very large. 
###Also critical values are no where close to the Test Statistics. 
###Hence, we can safely say that our Time Series at the moment is not stationary



## To Make Series Stationary --- Using Transformation 
## I will try different methods for transformation. 

## Method 1: Log the time-series 
    
fig = plt.figure(figsize=(10,5))
ax= fig.add_subplot(111)

plt.xlabel('Date')
plt.ylabel('Number of Air Passengers (log-scale)')
plt.tick_params(axis='x', which='both',bottom=True,top=True, direction="in")   
plt.tick_params(axis='y', which='both',right=True,left=True, direction="in")
ax.minorticks_on()


ts_log = np.log(Data)
plt.plot(ts_log, color = 'blue',)    
# We see that there is not any change in the time-series.
# The plot suggests that there may be a linear trend.
# There is also seasonality, but the amplitude (height) of the cycles 
#appears to be increasing, suggesting that it is multiplicative.


## Let us try decomposition of the time-series. 



from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log,model = 'multiplicative')

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

fig = plt.figure(figsize=(10,10))
ax= fig.add_subplot(111)

plt.xlabel('Date')
plt.ylabel('Number of Air Passengers (log-scale)')
plt.tick_params(axis='x', which='both',bottom=True,top=True, direction="in")   
plt.tick_params(axis='y', which='both',right=True,left=True, direction="in")
ax.minorticks_on()

plt.subplot(411)
plt.plot(ts_log, color = 'blue', label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(trend, color = 'red', label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(seasonal,color = 'green', label='Seasonality')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(residual, color = 'purple', label='Residuals')
plt.legend(loc='upper left')

# The plots shows the different components of the time-series.
# Since it still has the trend component, it is not yet stationary.
# Therefore, we find the moving average and subtract the long-term trend

Moving_Average = ts_log.rolling(window=12).mean()

fig = plt.figure(figsize=(10,5))
ax= fig.add_subplot(111)

plt.xlabel('Date')
plt.ylabel('Number of Air Passengers (log-scale)')
plt.tick_params(axis='x', which='both',bottom=True,top=True, direction="in")   
plt.tick_params(axis='y', which='both',right=True,left=True, direction="in")
ax.minorticks_on()

plt.plot(ts_log, color='blue')
plt.plot(Moving_Average, color='red')

New_ts_LogMinusMovingAverage = ts_log - Moving_Average


fig = plt.figure(figsize=(10,5))
ax= fig.add_subplot(111)

plt.xlabel('Date')
plt.ylabel('Number of Air Passengers (log-scale)')
plt.tick_params(axis='x', which='both',bottom=True,top=True, direction="in")   
plt.tick_params(axis='y', which='both',right=True,left=True, direction="in")
ax.minorticks_on()

plt.plot(New_ts_LogMinusMovingAverage, color='orange')
print(New_ts_LogMinusMovingAverage.isnull().sum()) # There are null values which we remove
New_ts_LogMinusMovingAverage.dropna(inplace=True)



stationarity_test(New_ts_LogMinusMovingAverage)



## Method 2: Time Shift Transformation


New_ts_LogDiffShifting = ts_log - ts_log.shift()


fig = plt.figure(figsize=(10,5))
ax= fig.add_subplot(111)

plt.xlabel('Date')
plt.ylabel('Number of Air Passengers (log-scale)')
plt.tick_params(axis='x', which='both',bottom=True,top=True, direction="in")   
plt.tick_params(axis='y', which='both',right=True,left=True, direction="in")
ax.minorticks_on()

plt.plot(New_ts_LogDiffShifting, color='orange')
print(New_ts_LogDiffShifting.isnull().sum()) # There are null values which we remove
New_ts_LogDiffShifting.dropna(inplace=True)


stationarity_test(New_ts_LogDiffShifting)


## Method 3: Exponential Decay Transformation


exponential_fit_WeightedAverage = ts_log.ewm(halflife=12, min_periods=0, adjust=True).mean()

New_ts_LogScaleMinusExponentialMovingAverage = ts_log - exponential_fit_WeightedAverage



fig = plt.figure(figsize=(10,5))
ax= fig.add_subplot(111)

plt.xlabel('Date')
plt.ylabel('Number of Air Passengers (log-scale)')
plt.tick_params(axis='x', which='both',bottom=True,top=True, direction="in")   
plt.tick_params(axis='y', which='both',right=True,left=True, direction="in")
ax.minorticks_on()

plt.plot(New_ts_LogScaleMinusExponentialMovingAverage, color='orange')
print(New_ts_LogScaleMinusExponentialMovingAverage.isnull().sum()) # There are null values which we remove
New_ts_LogScaleMinusExponentialMovingAverage.dropna(inplace=True)


stationarity_test(New_ts_LogScaleMinusExponentialMovingAverage)


##Plotting ACF & PACF

from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(New_ts_LogDiffShifting, nlags=20)
lag_pacf = pacf(New_ts_LogDiffShifting, nlags=20)

import statsmodels.api as sm

fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(New_ts_LogDiffShifting.dropna(),lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(New_ts_LogDiffShifting.dropna(),lags=40,ax=ax2)



### For ARIMA MODEL ORDER  = [p,d,q]
#p = PARTIAL AUTOCORRELATION PLOT = LAG VALUE AT WHICH THE LINE TOUCHES THE CONFIDENCE INTERVAL FIRST = 1-2
#d = DIFFERENCING ORDER = 1
#q = AUTOCORRELATION PLOT = LAG VALUE AT WHICH THE LINE TOUCHES THE CONFIDENCE INTERVAL FIRST = 1-2

## Before moving to ARIMA model, we check AR and MA model individually 

from statsmodels.tsa.arima_model import ARIMA

## AR model :

model = ARIMA(ts_log, order=(2,1,0))  
results_AR = model.fit(disp=-1)

fig = plt.figure(figsize=(10,5))
ax= fig.add_subplot(111)

plt.xlabel('Date')
#plt.ylabel('Number of Air Passengers (log-scale)')
plt.tick_params(axis='x', which='both',bottom=True,top=True, direction="in")   
plt.tick_params(axis='y', which='both',right=True,left=True, direction="in")
ax.minorticks_on()

plt.plot(New_ts_LogDiffShifting)
plt.plot(results_AR.fittedvalues, color='red')

plt.title('AR model, RSS: %.4f'%sum((results_AR.fittedvalues - New_ts_LogDiffShifting['Passengers'])**2))


print(New_ts_LogDiffShifting)


## MA model :

model = ARIMA(ts_log, order=(0,1,2))  
results_MA = model.fit(disp=-1)

fig = plt.figure(figsize=(10,5))
ax= fig.add_subplot(111)

plt.xlabel('Date')
#plt.ylabel('Number of Air Passengers (log-scale)')
plt.tick_params(axis='x', which='both',bottom=True,top=True, direction="in")   
plt.tick_params(axis='y', which='both',right=True,left=True, direction="in")
ax.minorticks_on()

plt.plot(New_ts_LogDiffShifting)
plt.plot(results_MA.fittedvalues, color='red')

plt.title('MA model, RSS: %.4f'%sum((results_MA.fittedvalues - New_ts_LogDiffShifting['Passengers'])**2))


## ARIMA model :

model = ARIMA(ts_log, order=(2,1,2))  
results_ARIMA = model.fit(disp=-1) 

fig = plt.figure(figsize=(10,5))
ax= fig.add_subplot(111)

plt.xlabel('Date')
#plt.ylabel('Number of Air Passengers (log-scale)')
plt.tick_params(axis='x', which='both',bottom=True,top=True, direction="in")   
plt.tick_params(axis='y', which='both',right=True,left=True, direction="in")
ax.minorticks_on()

plt.plot(New_ts_LogDiffShifting, color='blue')
plt.plot(results_ARIMA.fittedvalues, color='red')
 
plt.title('ARIMA model, RSS: %.4f'%sum((results_ARIMA.fittedvalues - New_ts_LogDiffShifting['Passengers'])**2))


## Taking results back to original scale: Prediction & Reverse transformations



ARIMA_diff_predictions = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(ARIMA_diff_predictions.head())

#Convert to cumulative sum

ARIMA_diff_predictions_cumsum = ARIMA_diff_predictions.cumsum()
print(ARIMA_diff_predictions_cumsum.head())

ARIMA_log_prediction = pd.Series(ts_log['Passengers'].iloc[0], index=ts_log.index)
ARIMA_log_prediction = ARIMA_log_prediction.add(ARIMA_diff_predictions_cumsum,fill_value=0)
ARIMA_log_prediction.head()



# Inverse of log is exp.

plt.figure(figsize=(10,5))
predictions_ARIMA = np.exp(ARIMA_log_prediction)
plt.plot(Data, color='blue')
plt.plot(predictions_ARIMA, color='red')
#plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA - Data)**2)/len(Data)))



# Forecast into future


results_ARIMA.plot_predict(1,204)
x=results_ARIMA.forecast(steps=120)


## SARIMA [(p,d,q)x(P,D,Q,s)]¶


data_diff_seas = New_ts_LogDiffShifting.diff(12)
data_diff_seas = data_diff_seas.dropna()
dec = sm.tsa.seasonal_decompose(data_diff_seas,period = 12)
dec.plot()
plt.show()


model = sm.tsa.statespace.SARIMAX(Data['Passengers'],order = (2,1,2),seasonal_order = (1,1,2,12))
results = model.fit()
print(results.summary())


Data['FORECAST'] = results.predict(start = 120,end = 144,dynamic = True)
Data[['Passengers','FORECAST']].plot(figsize = (10,5))


exp = [Data.iloc[i,0] for i in range(120,len(Data))]
pred = [Data.iloc[i,1] for i in range(120,len(Data))]
Data = Data.drop(columns = 'FORECAST')
#print(mean_absolute_error(exp,pred))

from pandas.tseries.offsets import DateOffset

future_dates = [Data.index[-1] + DateOffset(months = x)for x in range(0,25)]
df = pd.DataFrame(index = future_dates[1:],columns = Data.columns)



forecast = pd.concat([Data,df])
forecast['FORECAST'] = results.predict(start = 144,end = 168,dynamic = True)
forecast[['Passengers','FORECAST']].plot(figsize = (10,5))

