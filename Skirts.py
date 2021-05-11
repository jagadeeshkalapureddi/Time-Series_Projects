#!/usr/bin/env python
# coding: utf-8

# In[104]:


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


# In[105]:


df = pd.read_csv('skirts.csv',index_col=0, skipinitialspace=False)


# In[106]:


df.head()


# In[107]:


df.isnull().sum()


# In[108]:


df.shape


# In[109]:


df.describe()


# In[110]:


df.skew()


# In[111]:


df.info()


# In[112]:


df.boxplot()


# In[113]:


df.plot(figsize=(15, 6),color='green',linestyle='--', alpha=0.9, linewidth = 1.5)
plt.xlabel("Time_Period")
plt.ylabel("Birth_Rate")
plt.title('Birth Rate line plot')
plt.style.use('seaborn-whitegrid')
plt.show()


# ### `Adfuller Test `

# In[114]:


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


# In[115]:


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


# In[116]:


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

# In[117]:


train=df[0:35] 
test=df[35:46]


# In[118]:


plt.figure(figsize=(16,8))
plt.plot( train['Diameter'], label='Train')
plt.plot(test['Diameter'], label='Test')


# ## Moving Average

# In[119]:


width = 3
lag1 = df.shift(1)
lag3 = df.shift(width - 1)
window = lag3.rolling(window=width)
means = window.mean()
dataframe = pd.concat([means, lag1, df], axis=1)
dataframe.columns = ['mean', 't-1', 't+1']
print(dataframe.head(10))


# In[120]:


type(df)


# In[121]:


df = pd.DataFrame(df)


# In[122]:


df.head()


# ## ARIMA

# In[123]:


x = df['Diameter']
# fit model
model = ARIMA(x, order=(5,1,1))
model_fit = model.fit()
# summary of fit model
print(model_fit.summary())
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


# In[124]:


df = pd.DataFrame(df)
df.shape


# In[125]:


df['forecast']=model_fit.predict(start=35,end=46,dynamic=True)
df[['Diameter','forecast']].plot(figsize=(12,8))


# In[126]:


model=sm.tsa.statespace.SARIMAX(df['Diameter'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
results=model.fit()
df['forecast']=results.predict(start=35,end=46,dynamic=True)
df[['Diameter','forecast']].plot(figsize=(12,8))


# In[127]:


res = sm.tsa.ARMA(df['Diameter'], (1,1)).fit(disp=-1)
sm.stats.acorr_ljungbox(res.resid, lags=[1], return_df=True)


# # Weighted Moving Average

# In[128]:


train=df[0:35] 
test=df[35:46]


# In[129]:


weights = np.arange(1,11) #this creates an array with integers 1 to 10 included
weights


# In[130]:


wma10 = train['Diameter'].rolling(10).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
wma10.head(20)


# In[131]:


plt.figure(figsize = (12,6))
plt.plot(train['Diameter'], label="Diameter")
plt.plot(wma10, label=" WMA")

plt.xlabel("Date")
plt.ylabel("Diameter")
plt.legend()
plt.show()


# ## Exponential Smoothening

# In[132]:


train=df[0:35] 
test=df[35:46]


# In[133]:


y_hat_avg = test.copy()
fit2 = SimpleExpSmoothing(np.asarray(train['Diameter'])).fit(smoothing_level=0.3,optimized=False)
y_hat_avg['SES'] = fit2.forecast(len(test))
plt.figure(figsize=(16,8))
plt.plot(train['Diameter'], label='Train')
plt.plot(test['Diameter'], label='Test')
plt.plot(y_hat_avg['SES'], label='SES')
plt.legend(loc='best')
plt.show()


# In[134]:


rms = sqrt(mean_squared_error(test.Diameter, y_hat_avg.SES))
print('RMSE:', rms)


# ### `Holt's Winter Method`

# In[135]:


y_hat_avg = test.copy()
fit = ExponentialSmoothing(np.asarray(train['Diameter']) ,seasonal_periods=7 ,trend='add', seasonal='add',).fit()
y_hat_avg['Holt_Winter'] = fit.forecast(len(test))
plt.figure(figsize=(16,8))
plt.plot( train['Diameter'], label='Train')
plt.plot(test['Diameter'], label='Test')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.show()


# In[136]:


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


# In[137]:


rms = sqrt(mean_squared_error(test.Diameter, y_hat_avg.Holt_Winter))
print('RMSE:', rms)


# In[138]:


y_hat_avg[['Diameter', 'Holt_Winter']]


# In[139]:


# ljungbox test

sm.stats.acorr_ljungbox(residuals, lags=[1], return_df=True)


# ### `AUTO CORRELATION AND PARTIAL AUTO CORRELATION PLOT` 

# In[140]:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Diameter'],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Diameter'],lags=40,ax=ax2)


# ### `AUTO CORREALTION PLOT`

# In[141]:


autocorrelation_plot(df['Diameter'])
plt.show()

