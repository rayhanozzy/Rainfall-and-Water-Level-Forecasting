#!/usr/bin/env python
# coding: utf-8

# # [Facebook Prophet](https://github.com/facebook/prophet) and [NeuralProphet](https://github.com/ourownstory/neural_prophet) Comparison
# By: Rayhan Ozzy Ertarto
# 
# The goal of this notebook is to compare the *expected values* forecasted by these two models and compare them against the actuals in order to calculate the performance metrics and define which model performs better using this time series dataset (Rainfall in Kemayoran BMKG Station, Central Jakarta)

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

# In[62]:


gsheetkey = "1QUan4wVK8yNaIDvLvQEXocTWch8xkjy8kvS53PAKt_U"

url=f'https://docs.google.com/spreadsheet/ccc?key={gsheetkey}&output=csv'
df_km = pd.read_csv(url)
df_km.head(10)


# In[63]:


df_km.info()


# In[64]:


df_km['RR'] = df_km['RR'].replace([8888.0],['NaN'])


# In[65]:


df_km['RR'] = df_km['RR'].replace([9999.0],['NaN'])


# In[66]:


df_km.head(10)


# In[67]:


df_km.info()


# In[68]:


df_km['RR'] = df_km['RR'].astype(float)
df_km.info()


# In[69]:


df_km.head()


# Check for Missing Values

# In[70]:


df_km.isna().sum()


# Fill Missing Values by Interpolation

# In[71]:


df_km = df_km.interpolate()


# In[72]:


df_km.isna().sum()


# In[73]:


df_km.head(10)


# In[74]:


df_km_time = df_km.set_index('Tanggal')
df_km_time.head()


# In[75]:


#Plot of decompotition
import statsmodels.api as sm
from pylab import rcParams
rcParams['figure.figsize'] = 11, 9
decomposition = sm.tsa.seasonal_decompose(df_km_time, 
                                         model = 'additive',
                                         period=60) 
fig = decomposition.plot()
plt.show()


# In[16]:


df_km['Tanggal'] = pd.to_datetime(df_km['Tanggal'])
df_km.info()


# In[17]:


# Renaming columns
df_km.rename(columns = {'Tanggal':'ds', 'RR':'y'}, inplace = True)
df_km.head()


# In[18]:


df_km.tail()


# In[21]:


df_km.set_index('ds').plot(figsize=(12,6))
plt.title('Time Series Plot')


# ## Prophet Model

# In[22]:


get_ipython().system('pip install prophet -q')


# In[23]:


from prophet import Prophet


# In[24]:


m = Prophet(seasonality_mode='additive')


# Using default settings, only the seasonality mode is set to *Additive*
# 
# 

# In[25]:


m.fit(df_km)


# In[26]:


future = m.make_future_dataframe(periods=60, freq='D')


# In[27]:


future.tail(5)


# In[28]:


forecast = m.predict(future)


# In[29]:


forecast.tail()


# In[30]:


m.plot(forecast);
plt.title("Forecast of the Time Series in the next 60 days")
plt.xlabel("Dates")
plt.ylabel("Rainfall (mm)")


# In[31]:


m.plot_components(forecast);
print("Components of the time series:")


# In[32]:


#p_forecast = forecast[forecast['ds']>'2022-02-18'][['ds','yhat_lower','yhat','yhat_upper']]
p_forecast = forecast[['ds','yhat_lower','yhat','yhat_upper']]
p_forecast.info()


# In[33]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=45)
plt.title("Detail of Forecast using Prophet")
plt.plot(p_forecast['ds'], p_forecast['yhat'], marker='.', c='navy')
plt.fill_between(p_forecast['ds'],p_forecast['yhat_lower'], p_forecast['yhat_upper'], alpha=0.1, color='cyan')
plt.xlabel("Dates")
plt.ylabel("Rainfall (mm)")


# ### Performance Metrics

# In[34]:


from sklearn.metrics import mean_squared_error


# In[35]:


df_km.info()


# In[36]:


df_km_merge = pd.merge(df_km, forecast[['ds','yhat_lower','yhat_upper','yhat']],on='ds')
df_km_merge = df_km_merge[['ds','yhat_lower','yhat_upper','yhat','y']]
df_km_merge.head()


# In[37]:


df_km_merge.tail()


# In[38]:


prophet_mse = mean_squared_error(df_km_merge['y'], df_km_merge['yhat'])
prophet_rmse = np.sqrt(mean_squared_error(df_km_merge['y'], df_km_merge['yhat']))


# In[39]:


print("Prophet MSE: {:.4f}".format(prophet_mse))
print("Prophet RMSE: {:.4f}".format(prophet_rmse))


# ## NeuralProphet

# In[40]:


get_ipython().system('pip install neuralprophet -q')


# In[41]:


from neuralprophet import NeuralProphet, set_random_seed


# In[42]:


set_random_seed(42)


# In[43]:


nm = NeuralProphet(seasonality_mode='additive')


# In[44]:


nm.fit(df_km, freq='D')


# In[45]:


n_future = nm.make_future_dataframe(df_km, periods=60, n_historic_predictions=len(df_km))
n_future


# In[46]:


n_future.tail()


# In[47]:


n_forecast = nm.predict(n_future)


# In[48]:


n_forecast.info()


# In[49]:


n_forecast.tail()


# In[50]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=45)
plt.title("Detail of Forecast using NeuralProphet")
plt.plot(n_forecast['ds'], n_forecast['yhat1'], marker='.', c='red')
plt.legend()
plt.xlabel("Dates")
plt.ylabel("Rainfall (mm)")


# In[51]:


nm.plot(pd.concat([df_km, n_forecast], ignore_index=True));
plt.title("Forecast of the Time Series in the next 60 days")
plt.xlabel("Dates")
plt.ylabel("Rainfall (mm)")


# In[52]:


nm.plot_components(pd.concat([df_km, n_forecast], ignore_index=True));


# ### Performance Metrics

# In[53]:


n_forecast


# In[54]:


n_forecast_merge = pd.merge(df_km, n_forecast[['ds','yhat1','residual1']],on='ds')
n_forecast_merge = n_forecast_merge[['ds','yhat1','residual1','y']]
n_forecast_merge.head()


# In[55]:


n_prophet_mse = mean_squared_error(n_forecast_merge['y'], n_forecast_merge['yhat1'])
n_prophet_rmse = np.sqrt(mean_squared_error(n_forecast_merge['y'], n_forecast_merge['yhat1']))


# In[56]:


print("Neural Prophet MSE: {:.4f}".format(n_prophet_mse))
print("Neural Prophet RMSE: {:.4f}".format(n_prophet_rmse))


# In[57]:


print("Prophet MSE: {:.4f}".format(prophet_mse))
print("Prophet RMSE: {:.4f}".format(prophet_rmse))


# In[58]:


n_prophet_mse - prophet_mse


# In[59]:


n_prophet_rmse - prophet_rmse


# In[60]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=45)
plt.title("Models Comparison")
plt.plot(p_forecast['ds'], p_forecast['yhat'], marker='.', c='navy', label='Prophet')
plt.plot(n_forecast['ds'], n_forecast['yhat1'], marker='.', c='red', label='NeuralProphet')
plt.legend()
plt.xlabel("Dates")
plt.ylabel("Rainfall (mm)")


# In[61]:


pd.DataFrame({'metrics':['MSE','RMSE'],
              'Prophet ':[prophet_mse, prophet_rmse],
              'Neural Prophet':[n_prophet_mse, n_prophet_rmse]
             })


# ## Final Comments

# *   At least for this particular dataset and using the default arguments,  the **NeuralProphet** model scored a **MSE** of **326.223130** and **RMSE** of **18.061648** whereas the **Prophet** model scored a **MSE** of **332.051434** and **RMSE** of **18.222279**, a **5.828303990666825 and 0.16063046496710953 difference of MSE and RMSE respectively** compared against the first model.
