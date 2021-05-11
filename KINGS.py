#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[4]:


df = pd.read_csv('kings.csv',index_col=0, skipinitialspace=False)


# In[5]:


df.head()


# In[6]:


df.isnull().sum()


# In[7]:


df.shape


# In[8]:


df.describe()


# In[9]:


df.skew()


# In[10]:


df.info()


# In[11]:


df.boxplot()


# In[12]:


df.plot(figsize=(15, 6),color='green',linestyle='--', alpha=0.9, linewidth = 1.5)
plt.xlabel("Time_Period")
plt.ylabel("Birth_Rate")
plt.title('Birth Rate line plot')
plt.style.use('seaborn-whitegrid')
plt.show()


# ### `Adfuller Test `

# In[15]:


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


# ### `Creating train and test set `

# In[17]:


train=df[0:28]
test=df[28:41]


# In[19]:


test.shape


# In[20]:


train.shape


# In[22]:


plt.figure(figsize=(16,8))
plt.plot( train['Lifespan'], label='Train')
plt.plot(test['Lifespan'], label='Test')


# ## Moving Average

# In[24]:


from pandas import concat

width = 3
lag1 = df.shift(1)
lag3 = df.shift(width - 1)
window = lag3.rolling(window=width)
means = window.mean()
dataframe = concat([means, lag1, df], axis=1)
dataframe.columns = ['mean', 't-1', 't+1']
print(dataframe.head(10))


# In[25]:


df = pd.DataFrame(df)


# In[26]:


df.head()


# ## ARIMA

# In[27]:


x = df['Lifespan']
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


# In[30]:


df.head()


# In[31]:


df['forecast']=model_fit.predict(start=28,end=41,dynamic=True)
df[['Lifespan','forecast']].plot(figsize=(12,8))


# In[32]:


model=sm.tsa.statespace.SARIMAX(df['Lifespan'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
results=model.fit()
df['forecast']=results.predict(start=100,end=168,dynamic=True)
df[['Lifespan','forecast']].plot(figsize=(12,8))


# In[34]:


res = sm.tsa.ARMA(df['Lifespan'], (1,1)).fit(disp=-1)
sm.stats.acorr_ljungbox(res.resid, lags=[12], return_df=True)
#sm.stats.acorr_ljungbox(res.resid, lags=[12], return_df=True)


# # Weighted Moving Average

# In[35]:


train=df[0:28] 
test=df[28:41]


# In[36]:


weights = np.arange(1,11) #this creates an array with integers 1 to 10 included
weights


# In[37]:


wma10 = train['Lifespan'].rolling(10).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
wma10.head(20)


# In[38]:


plt.figure(figsize = (12,6))
plt.plot(train['Lifespan'], label="kingslifespan")
plt.plot(wma10, label=" WMA")

plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()


# ## Exponential Smoothening

# In[39]:


train=df[0:28] 
test=df[28:41]


# In[40]:


y_hat_avg = test.copy()
fit2 = SimpleExpSmoothing(np.asarray(train['Lifespan'])).fit(smoothing_level=0.6,optimized=False)
y_hat_avg['SES'] = fit2.forecast(len(test))
plt.figure(figsize=(16,8))
plt.plot(train['Lifespan'], label='Train')
plt.plot(test['Lifespan'], label='Test')
plt.plot(y_hat_avg['SES'], label='SES')
plt.legend(loc='best')
plt.show()


# In[41]:


rms = sqrt(mean_squared_error(test.Lifespan, y_hat_avg.SES))
print('RMSE:', rms)


# ### `Holt's Winter Method`

# In[42]:


y_hat_avg = test.copy()
fit = ExponentialSmoothing(np.asarray(train['Lifespan']) ,seasonal_periods=7 ,trend='add', seasonal='add',).fit()
y_hat_avg['Holt_Winter'] = fit.forecast(len(test))
plt.figure(figsize=(16,8))
plt.plot( train['Lifespan'], label='Train')
plt.plot(test['Lifespan'], label='Test')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.show()


# In[43]:


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


# In[56]:


# ljungbox test

sm.stats.acorr_ljungbox(residuals, lags=[16], return_df=True)


# In[57]:


rms = sqrt(mean_squared_error(test.Lifespan, y_hat_avg.Holt_Winter))
print('RMSE:', rms)


# In[59]:


y_hat_avg[['Lifespan', 'Holt_Winter']]


# ### `AUTO CORRELATION AND PARTIAL AUTO CORRELATION PLOT` 

# In[60]:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Lifespan'],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Lifespan'],lags=40,ax=ax2)


# ### `AUTO CORREALTION PLOT`

# In[61]:


autocorrelation_plot(df['Lifespan'])
plt.show()

