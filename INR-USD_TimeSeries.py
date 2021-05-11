#!/usr/bin/env python
# coding: utf-8

# ## `Exchange Rate of Time Series Analysis`

# In[44]:


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
import warnings
warnings.filterwarnings("ignore")


# In[45]:


dff = pd.read_csv("USD-INR.csv",index_col=0, parse_dates=True, skipinitialspace=True)
dff.head()


# In[46]:


print("Shape:",dff.shape)


# In[47]:


dff.isnull().sum()


# In[48]:


dff.describe()


# In[49]:


plt.figure(figsize=(15,6))
plt.plot(dff)
plt.title('IND-USD Echange Rate')
plt.xlabel("Time")
plt.ylabel("Exchange - Rate")
plt.show()


# In[50]:


dff.boxplot()


# In[51]:


df = dff['Rate'].resample('M').mean()
df.head()


# In[52]:


# Originial Data Representation
df.plot(figsize=(15, 6))
plt.title("Originial Data after Imputation via reampling")
plt.xlabel("Time")
plt.ylabel("Rate")
plt.style.use('seaborn-whitegrid')
plt.show()


# ### `Adfuller Test `

# In[53]:


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


# In[54]:


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


# In[55]:


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


# ### `Creating train and test set `

# In[56]:


train=df[0:90] 
test=df[90:120]


# #### `Let’s visualize the data (train and test together) to know how it varies over a time period.`

# In[57]:


#Plotting data
train.plot(figsize=(15,8), title= 'Rate', fontsize=14)
test.plot(figsize=(15,8), title= 'Rate', fontsize=14)
plt.show()


# ### `Moving Average`

# In[58]:


width = 3
lag1 = df.shift(1)
lag3 = df.shift(width - 1)
window = lag3.rolling(window=width)
means = window.mean()
dataframe = concat([means, lag1, df], axis=1)
dataframe.columns = ['mean', 't-1', 't+1']
print(dataframe.head(10))


# In[59]:


df = pd.DataFrame(df)


# In[60]:


# prepare situation
X = df.Rate
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

predictions, history
# ### `ARIMA`

# In[61]:


x = df['Rate']
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


# In[62]:


df['forecast']=model_fit.predict(start=90,end=103,dynamic=True)
df[['Rate','forecast']].plot(figsize=(12,8))


# In[63]:


model=sm.tsa.statespace.SARIMAX(df['Rate'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
results=model.fit()
df['forecast']=results.predict(start=90,end=120,dynamic=True)
df[['Rate','forecast']].plot(figsize=(12,8))


# ### `LJUNG BOX TEST`

# In[64]:


res = sm.tsa.ARMA(results.resid, (1,1)).fit(disp=-1)


# In[65]:


sm.stats.acorr_ljungbox(results.resid, lags=[12], return_df=True, boxpierce = True)


# If The p_value is less than 0.05 so, we can reject the Ho. (The residuals are independently distributed.)
# 
# H0: The residuals are independently distributed.
# 
# HA: The residuals are not independently distributed; they exhibit serial correlation.

# ### `Weighted Movig Average`

# ### `Creating train and test set `

# In[66]:


train=df[0:90] 
test=df[90:120]


# In[67]:


weights = np.arange(1,11) #this creates an array with integers 1 to 10 included
weights


# In[68]:


wma10 = train['Rate'].rolling(10).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
wma10.head(20)


# In[69]:


plt.figure(figsize = (12,6))
plt.plot(train['Rate'], label="Price")
plt.plot(wma10, label=" WMA")

plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()


# ### `simple Exponential Smoothening`

# ### `Creating train and test set `

# In[70]:


train=df[0:90] 
test=df[90:120]


# In[91]:


y_hat_avg = test.copy()
fit2 = SimpleExpSmoothing(np.asarray(train['Rate'])).fit(smoothing_level=0.1,optimized=False)
y_hat_avg['SES'] = fit2.forecast(len(test))
plt.figure(figsize=(16,8))
plt.plot(train['Rate'], label='Train')
plt.plot(test['Rate'], label='Test')
plt.plot(y_hat_avg['SES'], label='SES')
plt.legend(loc='best')
plt.show()


# In[93]:


rms = sqrt(mean_squared_error(test.Rate, y_hat_avg.SES))
print('RMSE:', rms)


# ### `Holt's Winter Method`

# In[104]:


y_hat_avg = test.copy()
fit = ExponentialSmoothing(np.asarray(train['Rate']) ,seasonal_periods=2,trend='add', seasonal='add',).fit()
y_hat_avg['Holt_Winter'] = fit.forecast(len(test))
plt.figure(figsize=(16,8))
plt.plot( train['Rate'], label='Train')
plt.plot(test['Rate'], label='Test')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.show()


# In[105]:


residuals = DataFrame(fit.resid)
residuals.plot()
pyplot.show()
# density plot of residuals
residuals.plot(kind='kde')
pyplot.show()

residuals.plot(kind = 'hist', color = 'red')
pyplot.show()
# summary stats of residuals
print(residuals.describe())


# In[101]:


rms = sqrt(mean_squared_error(test.Rate, y_hat_avg.Holt_Winter))
print('RMSE:', rms)


# In[76]:


y_hat_avg


# In[77]:


res = sm.tsa.ARMA(results.resid, (1,1)).fit(disp=-1)


# In[78]:


sm.stats.acorr_ljungbox(results.resid, lags=[12], return_df=True, boxpierce = True)


# If The p_value is less than 0.05 so, we can reject the Ho. (The residuals are independently distributed.)
# 
# H0: The residuals are independently distributed.
# 
# HA: The residuals are not independently distributed; they exhibit serial correlation.

# ### `AUTO CORRELATION AND PARTIAL AUTO CORRELATION PLOT` 

# In[81]:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Rate'],lags=12,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Rate'],lags=12,ax=ax2)


# ### `AUTO CORREALTION PLOT`

# In[80]:


autocorrelation_plot(df['Rate'])
plt.show()


# ### `     --  ---- ---- ----- ----- --- ----- ------ ------ ------ ----- -Thank You`

# In[ ]:




