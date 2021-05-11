#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


df = pd.read_csv("rain.csv",index_col=0, parse_dates=True, skipinitialspace=True)
df.head()


# In[4]:


print("Shape:",df.shape)


# In[5]:


import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'


# In[6]:


df.isnull().sum()


# In[7]:


df.info()


# In[8]:


# checking null values 

df.isna().sum()


# In[9]:


# statistical analysis

df.describe()


# In[10]:


plt.figure(figsize=(15,6))
plt.plot(df)
plt.title('Rainfall')
plt.xlabel("Time")
plt.ylabel("Rain in mm")
plt.show()


# In[11]:


df.boxplot()


# In[12]:


df.resample('Y').mean().plot.bar(y=['mm'], figsize=[25,10])


# In[ ]:





# In[13]:


y = df['mm'].resample('y').mean()
y.head()


# In[14]:


# Originial Data Representation
y.plot(figsize=(15, 6))
plt.title("Originial Data after Imputation via reampling")
plt.xlabel("Time")
plt.ylabel("Rainfall measure in mm")
plt.style.use('seaborn-whitegrid')
plt.show()


# In[15]:


# We are using adfuller test for Checking Stationarity
from pandas import Series
from statsmodels.tsa.stattools import adfuller
result = adfuller(y)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


# ## adfuller Test for stationarity

# ### Null Hypothesis:  It has some time dependent structure(series is non-stationary).
# 
# ### Alternative Hypothesis: It has no time dependent structure(series is stationary).
# 
# 
# 

# In[ ]:





# In[16]:


from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(y)

plt.plot(y, label = 'Original')
plt.legend(loc = 'best')

trend = decomposition.trend
plt.show()
plt.plot(trend, label = 'Trend')
plt.legend(loc = 'best')

seasonal = decomposition.seasonal
plt.show()
plt.plot(seasonal, label = 'Seasonal')
plt.legend(loc = 'best')

residual = decomposition.resid
plt.show()
plt.tight_layout()
plt.plot(residual, label = 'Residual')
plt.legend(loc='best')


# In[17]:


#Creating train and test set 

train=df[0:70]
test=df[70:100]


# In[18]:


type(df)


# #### Let’s visualize the data (train and test together) to know how it varies over a time period.

# In[19]:


#Plotting data
train.plot(figsize=(15,8), title= 'Rainfall', fontsize=14)
test.plot(figsize=(15,8), title= 'Rainfall', fontsize=14)
plt.show()


# ### Method : – Simple Average

# In[20]:


y_hat_avg = test.copy()
y_hat_avg['avg_forecast'] = train.mm.mean()
plt.figure(figsize=(12,8))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(y_hat_avg['avg_forecast'], label='Average Forecast')
plt.legend(loc='best')
plt.show()


# #####  We will now calculate RMSE to check to accuracy of our model.

# In[21]:


# import math
from math import sqrt
from sklearn.metrics import mean_squared_error

rms = sqrt(mean_squared_error(test.mm, y_hat_avg.avg_forecast))
print(rms)


# # --------------------------------

# ## Moving Average

# In[22]:


from pandas import concat

width = 3
lag1 = df.shift(1)
lag3 = df.shift(width - 1)
window = lag3.rolling(window=width)
means = window.mean()
dataframe = concat([means, lag1, df], axis=1)
dataframe.columns = ['mean', 't-1', 't+1']
print(dataframe.head(10))


# In[23]:


df = pd.DataFrame(df)


# In[24]:


# prepare situation
from numpy import mean
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error

X = df.mm
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


# ### `ARIMA`

# In[25]:


from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame
from math import sqrt


x = df['mm']
# fit model
model = ARIMA(x, order=(1,1,0))
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


# In[33]:


res = sm.tsa.ARMA(df['mm'], (1,1)).fit(disp=-1)
print(sm.stats.acorr_ljungbox(res.resid, lags=[1], return_df=True))
print(sm.stats.acorr_ljungbox(res.resid, lags=[20], return_df=True))


# In[27]:


df['forecast']=model_fit.predict(start=90,end=120,dynamic=True)
df[['mm','forecast']].plot(figsize=(12,8))


# In[27]:


model=sm.tsa.statespace.SARIMAX(df['mm'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
results=model.fit()
df['forecast']=results.predict(start=70,end=100,dynamic=True)
df[['mm','forecast']].plot(figsize=(12,8))


# ### `Weighted Moving Average`

# In[28]:


weights = np.arange(1,11) #this creates an array with integers 1 to 10 included
weights


# In[29]:


wma10 = train['mm'].rolling(10).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
wma10.head(20)


# In[30]:


plt.figure(figsize = (12,6))
plt.plot(train['mm'], label="Price")
plt.plot(wma10, label=" WMA")

plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()


# ### simple Exponential Smoothening

# In[41]:


train=df[0:70]
test=df[70:100]


# In[32]:


y_hat_avg = test.copy()
fit2 = SimpleExpSmoothing(np.asarray(train['mm'])).fit(smoothing_level=0.6,optimized=False)
y_hat_avg['SES'] = fit2.forecast(len(test))
plt.figure(figsize=(16,8))
plt.plot(train['mm'], label='Train')
plt.plot(test['mm'], label='Test')
plt.plot(y_hat_avg['SES'], label='SES')
plt.legend(loc='best')
plt.show()


# In[33]:


rms = sqrt(mean_squared_error(test.mm, y_hat_avg.SES))
print('RMSE:', rms)


# ### `Holt's Winter Method`

# In[43]:


y_hat_avg = test.copy()
fit = ExponentialSmoothing(np.asarray(train['mm']) ,seasonal_periods=7 ,trend='add', seasonal='add',).fit()
y_hat_avg['Holt_Winter'] = fit.forecast(len(test))
plt.figure(figsize=(16,8))
plt.plot( train['mm'], label='Train')
plt.plot(test['mm'], label='Test')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.show()


# In[44]:


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


# In[64]:


# ljungbox test

sm.stats.acorr_ljungbox(residuals, lags=[46], return_df=True)


# In[46]:


rms = sqrt(mean_squared_error(test.mm, y_hat_avg.Holt_Winter))
print('RMSE:', rms)


# In[37]:


y_hat_avg


# ### `AUTO CORRELATION AND PARTIAL AUTO CORRELATION PLOT` 

# In[63]:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['mm'],lags=46,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['mm'],lags=46,ax=ax2)


# ### `AUTO CORREALTION PLOT`

# In[39]:


autocorrelation_plot(df['mm'])
plt.show()


# In[ ]:




