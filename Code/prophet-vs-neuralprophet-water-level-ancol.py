#!/usr/bin/env python
# coding: utf-8

# # [Facebook Prophet](https://github.com/facebook/prophet) and [NeuralProphet](https://github.com/ourownstory/neural_prophet) Comparison
# By: Rayhan Ozzy Ertarto
# 
# The goal of this notebook is to compare the *expected values* forecasted by these two models and compare them against the actuals in order to calculate the performance metrics and define which model performs better using this time series dataset (Water Level in Ancol Flushing Floodgate, Central Jakarta)

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


# In[ ]:


df_tma['updated_at'] = pd.to_datetime(df_tma['updated_at']).dt.date
df_tma.head(10)


# In[ ]:


df_tma = df_tma.rename(columns={'updated_at': 'tanggal'})
df_tma.head(10)


# In[ ]:


df_tma['jam_laporan'] = pd.to_datetime(df_tma.jam_laporan).dt.strftime('%H:%M')
df_tma.head(10)


# In[ ]:


df_tma_jakut = df_tma.loc[(df_tma['pintu_air_id'] == 12) | (df_tma['pintu_air_id'] == 13) |
                           (df_tma['pintu_air_id'] == 17)]
df_tma_jakut.head(20)


# In[ ]:


df_tma_anc = df_tma_jakut.loc[(df_tma['pintu_air_id'] == 13)]
df_tma_anc.head(10)


# In[ ]:


df_tma_anc['tanggal'] = df_tma_anc['tanggal'].astype(str)


# In[ ]:


df_tma_anc['ketinggian'] = df_tma_anc['ketinggian'].astype(int)


# In[ ]:


df_tma_anc['waktu'] = df_tma_anc[['tanggal','jam_laporan']].agg(' '.join,axis=1)
df_tma_anc.head(10)


# In[ ]:


df_tma_anc = df_tma_anc[['waktu','ketinggian']]
df_tma_anc.head(10)


# In[ ]:


df_tma_anc.drop_duplicates(subset='waktu',keep='first',inplace=True)
df_tma_anc.head()


# In[ ]:


# Renaming columns
df_tma_anc.rename(columns = {'waktu':'ds', 'ketinggian':'y'}, inplace = True)
df_tma_anc.head()


# In[ ]:


df_tma_anc_time = df_tma_anc.set_index('ds')
df_tma_anc_time.head()


# In[ ]:


#Plot of decompotition
import statsmodels.api as sm
from pylab import rcParams
rcParams['figure.figsize'] = 11, 9
decomposition = sm.tsa.seasonal_decompose(df_tma_anc_time, 
                                         model = 'additive',
                                         period=60) 
fig = decomposition.plot()
plt.show()


# In[ ]:


df_tma_anc['ds'] = pd.DatetimeIndex(df_tma_anc['ds'])
df_tma_anc.info()


# In[ ]:


df_tma_anc.head(10)


# In[ ]:


df_tma_anc.tail(10)


# In[ ]:


#df_tma_anc.set_index('ds').plot(figsize=(12,6))
#plt.title('Time Series Plot')


# ## Prophet Model

# In[ ]:


get_ipython().system('pip install prophet -q')


# In[ ]:


from prophet import Prophet


# In[ ]:


m = Prophet(seasonality_mode='additive')


# Using default settings, only the seasonality mode is set to *Additive*
# 
# 

# In[ ]:


m.fit(df_tma_anc)


# In[ ]:


future = m.make_future_dataframe(periods=1440, freq='H')


# In[ ]:


future.tail(5)


# In[ ]:


forecast = m.predict(future)


# In[ ]:


forecast.tail()


# In[ ]:


m.plot(forecast);
plt.title("Forecast of the Time Series in the next 60 days (1440 hours)")
plt.xlabel("Dates")
plt.ylabel("Water Level (cm)")


# In[ ]:


m.plot_components(forecast);
print("Components of the time series:")


# In[ ]:


#p_forecast = forecast[forecast['ds']>'2022-02-18 16:00:00'][['ds','yhat_lower','yhat','yhat_upper']]
p_forecast = forecast[['ds','yhat_lower','yhat','yhat_upper']]
p_forecast.info()


# In[ ]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=45)
plt.title("Detail of Forecast using Prophet")
plt.plot(p_forecast['ds'], p_forecast['yhat'], marker='.', c='navy')
plt.fill_between(p_forecast['ds'],p_forecast['yhat_lower'], p_forecast['yhat_upper'], alpha=0.1, color='cyan')
plt.xlabel("Dates")
plt.ylabel("Water Level (cm)")


# ### Performance Metrics

# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


df_tma_anc.info()


# In[ ]:


df_tma_anc_merge = pd.merge(df_tma_anc, forecast[['ds','yhat_lower','yhat_upper','yhat']],on='ds')
df_tma_anc_merge = df_tma_anc_merge[['ds','yhat_lower','yhat_upper','yhat','y']]
df_tma_anc_merge.head()


# In[ ]:


df_tma_anc_merge.tail()


# In[ ]:


prophet_mse = mean_squared_error(df_tma_anc_merge['y'], df_tma_anc_merge['yhat'])
prophet_rmse = np.sqrt(mean_squared_error(df_tma_anc_merge['y'], df_tma_anc_merge['yhat']))


# In[ ]:


print("Prophet MSE: {:.4f}".format(prophet_mse))
print("Prophet RMSE: {:.4f}".format(prophet_rmse))


# ## NeuralProphet

# In[ ]:


get_ipython().system('pip install neuralprophet -q')


# In[ ]:


from neuralprophet import NeuralProphet, set_random_seed


# In[ ]:


set_random_seed(42)


# In[ ]:


nm = NeuralProphet(seasonality_mode='additive')


# In[ ]:


nm.fit(df_tma_anc, freq='H')


# In[ ]:


n_future = nm.make_future_dataframe(df_tma_anc, periods=1440, n_historic_predictions=len(df_tma_anc))
n_future


# In[ ]:


n_future.tail()


# In[ ]:


n_forecast = nm.predict(n_future)


# In[ ]:


n_forecast.info()


# In[ ]:


n_forecast.tail()


# In[ ]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=45)
plt.title("Detail of Forecast using NeuralProphet")
plt.plot(n_forecast['ds'], n_forecast['yhat1'], marker='.', c='red')
plt.legend()
plt.xlabel("Dates")
plt.ylabel("Water Level (cm)")


# In[ ]:


nm.plot(pd.concat([df_tma_anc, n_forecast], ignore_index=True));
plt.title("Forecast of the Time Series in the next 60 days (1440 hours)")
plt.xlabel("Dates")
plt.ylabel("Water Level (cm)")


# In[ ]:


nm.plot_components(pd.concat([df_tma_anc, n_forecast], ignore_index=True));


# ### Performance Metrics

# In[ ]:


n_forecast


# In[ ]:


n_forecast_merge = pd.merge(df_tma_anc, n_forecast[['ds','yhat1','residual1']],on='ds')
n_forecast_merge = n_forecast_merge[['ds','yhat1','residual1','y']]
n_forecast_merge.head()


# In[ ]:


n_prophet_mse = mean_squared_error(n_forecast_merge['y'], n_forecast_merge['yhat1'])
n_prophet_rmse = np.sqrt(mean_squared_error(n_forecast_merge['y'], n_forecast_merge['yhat1']))


# In[ ]:


print("Neural Prophet MSE: {:.4f}".format(n_prophet_mse))
print("Neural Prophet RMSE: {:.4f}".format(n_prophet_rmse))


# In[ ]:


print("Prophet MSE: {:.4f}".format(prophet_mse))
print("Prophet RMSE: {:.4f}".format(prophet_rmse))


# In[ ]:


n_prophet_mse - prophet_mse


# In[ ]:


n_prophet_rmse - prophet_rmse


# In[ ]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=45)
plt.title("Models Comparison")
plt.plot(p_forecast['ds'], p_forecast['yhat'], marker='.', c='navy', label='Prophet')
plt.plot(n_forecast['ds'], n_forecast['yhat1'], marker='.', c='red', label='NeuralProphet')
plt.legend()
plt.xlabel("Dates")
plt.ylabel("Water Level (cm)")


# In[ ]:


pd.DataFrame({'metrics':['MSE','RMSE'],
              'Prophet ':[prophet_mse, prophet_rmse],
              'Neural Prophet':[n_prophet_mse, n_prophet_rmse]
             })


# ## Final Comments

# *   At least for this particular dataset and using the default arguments,  the **NeuralProphet** model scored a **MSE** of **264.679741** and **RMSE** of **16.268981** whereas the **Prophet** model scored a **MSE** of **269.443288** and **RMSE** of **16.414728**, a **4.763546237988635 and 0.14574680858210698 difference of MSE and RMSE respectively** compared against the first model.
