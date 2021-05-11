#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


df = pd.read_csv('births.csv',index_col=0, skipinitialspace=False)

df.head()df.isnull().sum()df.shapedf.describe()df.skew()df.info()df.boxplot()df.plot(figsize=(15, 6),color='green',linestyle='--', alpha=0.9, linewidth = 1.5)
plt.xlabel("Time_Period")
plt.ylabel("Birth_Rate")
plt.title('Birth Rate line plot')
plt.style.use('seaborn-whitegrid')
plt.show()
# ### `Adfuller Test `
timeseries = adfuller(df)
print('ADF Statistic: %f' % timeseries[0])
print('p-value: %f' % timeseries[1])
print('Critical Values:')
for key, value in timeseries[4].items():
    print('\t%s: %.3f' % (key, value))
if timeseries[0] > timeseries[4]["5%"]:
    print ("Failed to Reject Ho - Time Series is Non-Stationary")
else:
    print("Reject Ho - Time Series is Stationary")X = df.values
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
# In[3]:


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

# In[4]:


train=data_stationery[0:100] 
test=data_stationery[100:168]


# In[5]:


plt.figure(figsize=(16,8))
plt.plot( train['Counts'], label='Train')
plt.plot(test['Counts'], label='Test')


# ## Moving Average

# In[6]:


width = 3
lag1 = df.shift(1)
lag3 = df.shift(width - 1)
window = lag3.rolling(window=width)
means = window.mean()
dataframe = pd.concat([means, lag1, df], axis=1)
dataframe.columns = ['mean', 't-1', 't+1']
print(dataframe.head(10))


# In[7]:


type(df)


# In[8]:


df = pd.DataFrame(df)


# In[9]:


# prepare situation
X = df.Counts
window = 10
history = [X[i] for i in range(window)]
test = [X[i] for i in range(window, len(X))]
predictions = list()
# walk forward over time steps in test
for t in range(len(test)):
    length = len(history)
    yhat = np.mean([history[i] for i in range(length-window,length)])
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


# ## ARIMA

# In[10]:


x = df['Counts']
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


# In[11]:


df['forecast']=model_fit.predict(start=100,end=168,dynamic=True)
df[['Counts','forecast']].plot(figsize=(12,8))


# In[12]:


model=sm.tsa.statespace.SARIMAX(df['Counts'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
results=model.fit()
df['forecast']=results.predict(start=100,end=168,dynamic=True)
df[['Counts','forecast']].plot(figsize=(12,8))


# In[15]:


res = sm.tsa.ARMA(df['Counts'], (1,1)).fit(disp=-1)
sm.stats.acorr_ljungbox(res.resid, lags=[2], return_df=True)


# In[16]:


# After 11 th lag the p_value is less than 0.05 so, we can reject the Ho. (The residuals are independently distributed.)

# H0: The residuals are independently distributed.

# HA: The residuals are not independently distributed; they exhibit serial correlation.


# # Weighted Moving Average

# In[17]:


train=df[0:100] 
test=df[100:168]


# In[18]:


weights = np.arange(1,11) #this creates an array with integers 1 to 10 included
weights


# In[19]:


wma10 = train['Counts'].rolling(10).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
wma10.head(20)


# In[20]:


plt.figure(figsize = (12,6))
plt.plot(train['Counts'], label="Price")
plt.plot(wma10, label=" WMA")

plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()


# ## Exponential Smoothening

# In[21]:


train=df[0:100] 
test=df[100:168]


# In[22]:


y_hat_avg = test.copy()
fit2 = SimpleExpSmoothing(np.asarray(train['Counts'])).fit(smoothing_level=0.6,optimized=False)
y_hat_avg['SES'] = fit2.forecast(len(test))
plt.figure(figsize=(16,8))
plt.plot(train['Counts'], label='Train')
plt.plot(test['Counts'], label='Test')
plt.plot(y_hat_avg['SES'], label='SES')
plt.legend(loc='best')
plt.show()


# In[23]:


rms = sqrt(mean_squared_error(test.Counts, y_hat_avg.SES))
print('RMSE:', rms)


# ### `Holt's Winter Method`

# In[24]:


y_hat_avg = test.copy()
fit = ExponentialSmoothing(np.asarray(train['Counts']) ,seasonal_periods=7 ,trend='add', seasonal='add',).fit()
y_hat_avg['Holt_Winter'] = fit.forecast(len(test))
plt.figure(figsize=(16,8))
plt.plot( train['Counts'], label='Train')
plt.plot(test['Counts'], label='Test')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.show()


# In[25]:


residuals = pd.DataFrame(fit.resid)
residuals.plot()
plt.show()
# density plot of residuals
residuals.plot(kind='kde')
plt.show()

residuals.plot(kind = 'hist', color = 'red')
plt.show()
# summary stats of residuals
print(residuals.describe())


# In[29]:


# ljungbox test

sm.stats.acorr_ljungbox(residuals, lags=[2], return_df=True)


# In[30]:


rms = sqrt(mean_squared_error(test.Counts, y_hat_avg.Holt_Winter))
print('RMSE:', rms)


# In[31]:


y_hat_avg


# ### `AUTO CORRELATION AND PARTIAL AUTO CORRELATION PLOT` 

# In[32]:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(data_stationery['Counts'],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(data_stationery['Counts'],lags=40,ax=ax2)


# ### `AUTO CORREALTION PLOT`

# In[33]:


autocorrelation_plot(data_stationery['Counts'])
plt.show()


# In[ ]:




