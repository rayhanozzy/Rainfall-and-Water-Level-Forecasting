#!/usr/bin/env python
# coding: utf-8

# # [Facebook Prophet](https://github.com/facebook/prophet) and [NeuralProphet](https://github.com/ourownstory/neural_prophet) Comparison
# By: Rayhan Ozzy Ertarto
# 
# The goal of this notebook is to compare the *expected values* forecasted by these two models and compare them against the actuals in order to calculate the performance metrics and define which model performs better using this time series dataset (Water Level in Karet Floodgate, Central Jakarta)

# Importing basic libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


np.random.seed(1234)


# In[3]:


plt.style.use('ggplot')


# Reading the time series

# In[4]:


gsheetkey = "1d0g-fOczYG3WGt3CpmAQzTfT4S0q4MI5BCtHR2HvQoE"

url=f'https://docs.google.com/spreadsheet/ccc?key={gsheetkey}&output=csv'
df_tma = pd.read_csv(url)
df_tma.head(10)


# In[5]:


df_tma.info()


# In[6]:


df_tma = df_tma[['pintu_air_id','nama_pintu_air','ketinggian','jam_laporan','updated_at']]
df_tma.head(10)


# In[7]:


df_tma['updated_at'] = pd.to_datetime(df_tma['updated_at']).dt.date
df_tma.head(10)


# In[8]:


df_tma = df_tma.rename(columns={'updated_at': 'tanggal'})
df_tma.head(10)


# In[9]:


df_tma['jam_laporan'] = pd.to_datetime(df_tma.jam_laporan).dt.strftime('%H:%M')
df_tma.head(10)


# In[10]:


df_tma_jakpus = df_tma.loc[(df_tma['pintu_air_id'] == 4) | (df_tma['pintu_air_id'] == 5) |
                           (df_tma['pintu_air_id'] == 6) | (df_tma['pintu_air_id'] == 7) |
                           (df_tma['pintu_air_id'] == 15) | (df_tma['pintu_air_id'] == 16) |
                           (df_tma['pintu_air_id'] == 19)]
df_tma_jakpus.head(20)


# In[11]:


df_tma_kar = df_tma_jakpus.loc[(df_tma['pintu_air_id'] == 5)]
df_tma_kar.head(10)


# In[12]:


df_tma_kar['tanggal'] = df_tma_kar['tanggal'].astype(str)


# In[13]:


df_tma_kar['ketinggian'] = df_tma_kar['ketinggian'].astype(int)


# In[14]:


df_tma_kar['waktu'] = df_tma_kar[['tanggal','jam_laporan']].agg(' '.join,axis=1)
df_tma_kar.head(10)


# In[15]:


df_tma_kar = df_tma_kar[['waktu','ketinggian']]
df_tma_kar.head(10)


# In[16]:


df_tma_kar.drop_duplicates(subset='waktu',keep='first',inplace=True)
df_tma_kar.head()


# In[17]:


# Renaming columns
df_tma_kar.rename(columns = {'waktu':'ds', 'ketinggian':'y'}, inplace = True)
df_tma_kar.head()


# In[18]:


df_tma_kar_time = df_tma_kar.set_index('ds')
df_tma_kar_time.head()


# In[19]:


#Plot of decompotition
import statsmodels.api as sm
from pylab import rcParams
rcParams['figure.figsize'] = 11, 9
decomposition = sm.tsa.seasonal_decompose(df_tma_kar_time, 
                                         model = 'additive',
                                         period=60) 
fig = decomposition.plot()
plt.show()


# In[20]:


df_tma_kar['ds'] = pd.DatetimeIndex(df_tma_kar['ds'])
df_tma_kar.info()


# In[21]:


df_tma_kar.head(10)


# In[22]:


df_tma_kar.tail(10)


# In[23]:


df_tma_kar.set_index('ds').plot(figsize=(12,6))
plt.title('Time Series Plot')


# ## Prophet Model

# In[24]:


get_ipython().system('pip install prophet -q')


# In[25]:


from prophet import Prophet


# In[26]:


m = Prophet(seasonality_mode='additive')


# Using default settings, only the seasonality mode is set to *Additive*
# 
# 

# In[27]:


m.fit(df_tma_kar)


# In[28]:


future = m.make_future_dataframe(periods=1440, freq='H')


# In[29]:


future.tail(5)


# In[30]:


forecast = m.predict(future)


# In[31]:


forecast.tail()


# In[32]:


m.plot(forecast);
plt.title("Forecast of the Time Series in the next 60 days (1440 hours)")
plt.xlabel("Dates")
plt.ylabel("Water Level (cm)")


# In[33]:


m.plot_components(forecast);
print("Components of the time series:")


# In[34]:


#p_forecast = forecast[forecast['ds']>'2022-02-18 16:00:00'][['ds','yhat_lower','yhat','yhat_upper']]
p_forecast = forecast[['ds','yhat_lower','yhat','yhat_upper']]
p_forecast.info()


# In[35]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=45)
plt.title("Detail of Forecast using Prophet")
plt.plot(p_forecast['ds'], p_forecast['yhat'], marker='.', c='navy')
plt.fill_between(p_forecast['ds'],p_forecast['yhat_lower'], p_forecast['yhat_upper'], alpha=0.1, color='cyan')
plt.xlabel("Dates")
plt.ylabel("Water Level (cm)")


# ### Performance Metrics

# In[36]:


from sklearn.metrics import mean_squared_error


# In[37]:


df_tma_kar.info()


# In[38]:


df_tma_kar_merge = pd.merge(df_tma_kar, forecast[['ds','yhat_lower','yhat_upper','yhat']],on='ds')
df_tma_kar_merge = df_tma_kar_merge[['ds','yhat_lower','yhat_upper','yhat','y']]
df_tma_kar_merge.head()


# In[39]:


df_tma_kar_merge.tail()


# In[40]:


prophet_mse = mean_squared_error(df_tma_kar_merge['y'], df_tma_kar_merge['yhat'])
prophet_rmse = np.sqrt(mean_squared_error(df_tma_kar_merge['y'], df_tma_kar_merge['yhat']))


# In[41]:


print("Prophet MSE: {:.4f}".format(prophet_mse))
print("Prophet RMSE: {:.4f}".format(prophet_rmse))


# ## NeuralProphet

# In[42]:


get_ipython().system('pip install neuralprophet -q')


# In[43]:


from neuralprophet import NeuralProphet, set_random_seed


# In[44]:


set_random_seed(42)


# In[45]:


nm = NeuralProphet(seasonality_mode='additive')


# In[46]:


nm.fit(df_tma_kar, freq='H')


# In[47]:


n_future = nm.make_future_dataframe(df_tma_kar, periods=1440, n_historic_predictions=len(df_tma_kar))
n_future


# In[48]:


n_future.tail()


# In[49]:


n_forecast = nm.predict(n_future)


# In[50]:


n_forecast.info()


# In[51]:


n_forecast.tail()


# In[52]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=45)
plt.title("Detail of Forecast using NeuralProphet")
plt.plot(n_forecast['ds'], n_forecast['yhat1'], marker='.', c='red')
plt.legend()
plt.xlabel("Dates")
plt.ylabel("Water Level (cm)")


# In[53]:


nm.plot(pd.concat([df_tma_kar, n_forecast], ignore_index=True));
plt.title("Forecast of the Time Series in the next 60 days (1440 hours)")
plt.xlabel("Dates")
plt.ylabel("Water Level (cm)")


# In[54]:


nm.plot_components(pd.concat([df_tma_kar, n_forecast], ignore_index=True));


# ### Performance Metrics

# In[55]:


n_forecast


# In[56]:


n_forecast_merge = pd.merge(df_tma_kar, n_forecast[['ds','yhat1','residual1']],on='ds')
n_forecast_merge = n_forecast_merge[['ds','yhat1','residual1','y']]
n_forecast_merge.head()


# In[57]:


n_prophet_mse = mean_squared_error(n_forecast_merge['y'], n_forecast_merge['yhat1'])
n_prophet_rmse = np.sqrt(mean_squared_error(n_forecast_merge['y'], n_forecast_merge['yhat1']))


# In[58]:


print("Neural Prophet MSE: {:.4f}".format(n_prophet_mse))
print("Neural Prophet RMSE: {:.4f}".format(n_prophet_rmse))


# In[59]:


print("Prophet MSE: {:.4f}".format(prophet_mse))
print("Prophet RMSE: {:.4f}".format(prophet_rmse))


# In[60]:


n_prophet_mse - prophet_mse


# In[61]:


n_prophet_rmse - prophet_rmse


# In[62]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=45)
plt.title("Models Comparison")
plt.plot(p_forecast['ds'], p_forecast['yhat'], marker='.', c='navy', label='Prophet')
plt.plot(n_forecast['ds'], n_forecast['yhat1'], marker='.', c='red', label='NeuralProphet')
plt.legend()
plt.xlabel("Dates")
plt.ylabel("Water Level (cm)")


# In[63]:


pd.DataFrame({'metrics':['MSE','RMSE'],
              'Prophet ':[prophet_mse, prophet_rmse],
              'Neural Prophet':[n_prophet_mse, n_prophet_rmse]
             })


# ## Final Comments

# *   At least for this particular dataset and using the default arguments,  the **NeuralProphet** model scored a **MSE** of **672.783893** and **RMSE** of **25.938078** whereas the **Prophet** model scored a **MSE** of **683.233403** and **RMSE** of **26.138734**, a **10.449510050300546 and 0.20065571766378199 difference of MSE and RMSE respectively** compared against the first model.
