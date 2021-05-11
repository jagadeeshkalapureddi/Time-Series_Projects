#!/usr/bin/env python
# coding: utf-8

# In[42]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from pandas import Series
from statsmodels.tsa.stattools import adfuller
from math import sqrt
from pandas import DataFrame
from pandas import concat
from numpy import mean
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt


# In[43]:


dff = pd.read_csv("sovenir.csv",index_col=0, parse_dates=True, skipinitialspace=True)
dff.head()


# In[44]:


print("Shape:",dff.shape)


# In[45]:


dff.isnull().sum()


# In[46]:


dff.describe()


# In[47]:


plt.figure(figsize=(15,6))
plt.plot(dff)
plt.title('souvenir sales')
plt.xlabel("Time")
plt.ylabel("sales")
plt.show()


# Lets apply log to the data

# In[48]:


from numpy import log
df = log(dff)


# In[49]:


df.head()


# In[50]:


plt.figure(figsize=(15,6))
plt.plot(df)
plt.title('souvenir sales')
plt.xlabel("Time")
plt.ylabel("logofsales")
plt.show()


# Adfuller Test

# In[51]:


timeseries = adfuller(df)
print('ADF Statistic: %f' % timeseries[0])
print('p-value: %f' % timeseries[1])
print('Critical Values:')
for key, value in timeseries[4].items():
    print('\t%s: %.3f' % (key, value))
if timeseries[0] > timeseries[4]["5%"]:
    print ("Failed to Reject Ho - Time Series is Non-Stationary")
else:
    print("Reject Ho - Time Series is Stationary")


# In[52]:


X = df.values
X = np.sqrt(X)
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
if result[0] > result[4]["5%"]:
    print ("Failed to Reject Ho - Time Series is Non-Stationary")
else:
    print("Reject Ho - Time Series is Stationary")


# In[53]:


data_stationery = df.diff().dropna()
result = adfuller(data_stationery)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
if result[0] > result[4]["5%"]:
    print ("Failed to Reject Ho - Time Series is Non-Stationary")
else:
    print("Reject Ho - Time Series is Stationary")


# In[54]:


train=df[0:59] 
test=df[59:85]


# In[55]:


#Plotting data
train.plot(figsize=(15,8), title= 'Sales', fontsize=14)
test.plot(figsize=(15,8), title= 'Sales', fontsize=14)
plt.show()


# Moving Average

# In[56]:


width = 3
lag1 = df.shift(1)
lag3 = df.shift(width - 1)
window = lag3.rolling(window=width)
means = window.mean()
dataframe = concat([means, lag1, df], axis=1)
dataframe.columns = ['mean', 't-1', 't+1']
print(dataframe.head(10))


# In[57]:


df = pd.DataFrame(df)


# In[58]:


# prepare situation
X = df.Sales
window = 10
history = [X[i] for i in range(window)]
test = [X[i] for i in range(window, len(X))]
predictions = list()
# walk forward over time steps in test
for t in range(len(test)):
    length = len(history)
    yhat = mean([history[i] for i in range(length-window,length)])
    obs = test[t]
    predictions.append(yhat)
    history.append(obs)

error = mean_squared_error(test, predictions)
print('Test RMSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
# zoom plot
pyplot.plot(test[0:30])
pyplot.plot(predictions[0:30], color='red')
pyplot.show()


# In[59]:


predictions, history


# ARIMA

# In[60]:


x = df['Sales']
# fit model
model = ARIMA(x, order=(5,1,1))
model_fit = model.fit()
# summary of fit model
print(model_fit.summary())
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# line plot of residuals
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
# density plot of residuals
residuals.plot(kind='kde')
pyplot.show()

residuals.plot(kind = 'hist', color = 'red')
pyplot.show()
# summary stats of residuals
print(residuals.describe())


# In[61]:


df['forecast']=model_fit.predict(start=59,end=85,dynamic=True)
df[['Sales','forecast']].plot(figsize=(12,8))


# In[62]:


model=sm.tsa.statespace.SARIMAX(df['Sales'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
results=model.fit()
df['forecast']=results.predict(start=59,end=85,dynamic=True)
df[['Sales','forecast']].plot(figsize=(12,8))


# 
# Holt winter method
# 

# In[63]:


train=df[0:59] 
test=df[59:85]


# In[64]:


y_hat_avg = test.copy()
fit1 = ExponentialSmoothing(np.asarray(train['Sales']) ,seasonal_periods=7 ,trend='add', seasonal='add',).fit()
y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
plt.figure(figsize=(16,8))
plt.plot( train['Sales'], label='Train')
plt.plot(test['Sales'], label='Test')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.show()


# In[65]:


rms = sqrt(mean_squared_error(test.Sales, y_hat_avg.Holt_Winter))
print('RMSE:', rms)


# In[66]:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Sales'],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Sales'],lags=40,ax=ax2)


# In[67]:


autocorrelation_plot(df['Sales'])
plt.show()


# LJUNG BOX TEST
# 

# In[68]:


import statsmodels.api as sm


# In[69]:


res = sm.tsa.ARMA(df['Sales'], (1,1)).fit(disp=-1)


# In[70]:


sm.stats.acorr_ljungbox(res.resid, lags=[5], return_df=True)


# In[71]:


sm.stats.acorr_ljungbox(res.resid, lags=[12], return_df=True)

