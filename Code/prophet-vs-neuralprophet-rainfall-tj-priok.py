#!/usr/bin/env python
# coding: utf-8

# # [Facebook Prophet](https://github.com/facebook/prophet) and [NeuralProphet](https://github.com/ourownstory/neural_prophet) Comparison
# By: Rayhan Ozzy Ertarto
# 
# The goal of this notebook is to compare the *expected values* forecasted by these two models and compare them against the actuals in order to calculate the performance metrics and define which model performs better using this time series dataset (Rainfall in Tanjung Priok BMKG Station, North Jakarta)

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


gsheetkey = "1qAe0nBzswVA1vekBH97Sj2uBQQWdN8RQ3g95pjc7CT4"

url=f'https://docs.google.com/spreadsheet/ccc?key={gsheetkey}&output=csv'
df_tp = pd.read_csv(url)
df_tp.head(10)


# In[5]:


df_tp.info()


# In[6]:


df_tp['RR'] = df_tp['RR'].replace([8888.0],['NaN'])


# In[7]:


df_tp['RR'] = df_tp['RR'].replace([9999.0],['NaN'])


# In[8]:


df_tp.head(10)


# In[9]:


df_tp.info()


# In[10]:


df_tp['RR'] = df_tp['RR'].astype(float)
df_tp.info()


# In[11]:


df_tp.head()


# Check for Missing Values

# In[12]:


df_tp.isna().sum()


# Fill Missing Values by Interpolation

# In[13]:


df_tp = df_tp.interpolate()


# In[14]:


df_tp.isna().sum()


# In[15]:


df_tp.head(10)


# In[16]:


df_tp_time = df_tp.set_index('Tanggal')
df_tp_time.head()


# In[17]:


#Plot of decompotition
import statsmodels.api as sm
from pylab import rcParams
rcParams['figure.figsize'] = 11, 9
decomposition = sm.tsa.seasonal_decompose(df_tp_time, 
                                         model = 'additive',
                                         period=60) 
fig = decomposition.plot()
plt.show()


# In[18]:


df_tp['Tanggal'] = pd.to_datetime(df_tp['Tanggal'])
df_tp.info()


# In[19]:


# Renaming columns
df_tp.rename(columns = {'Tanggal':'ds', 'RR':'y'}, inplace = True)
df_tp.head()


# In[20]:


df_tp.tail()


# In[21]:


#df_tp.set_index('ds').plot(figsize=(12,6))
#plt.title('Time Series Plot')


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


m.fit(df_tp)


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


#p_forecast = forecast[forecast['ds']>'2022-02-28'][['ds','yhat_lower','yhat','yhat_upper']]
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


df_tp.info()


# In[36]:


df_tp_merge = pd.merge(df_tp, forecast[['ds','yhat_lower','yhat_upper','yhat']],on='ds')
df_tp_merge = df_tp_merge[['ds','yhat_lower','yhat_upper','yhat','y']]
df_tp_merge.head()


# In[37]:


df_tp_merge.tail()


# In[38]:


prophet_mse = mean_squared_error(df_tp_merge['y'], df_tp_merge['yhat'])
prophet_rmse = np.sqrt(mean_squared_error(df_tp_merge['y'], df_tp_merge['yhat']))


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


nm.fit(df_tp, freq='D')


# In[45]:


n_future = nm.make_future_dataframe(df_tp, periods=60, n_historic_predictions=len(df_tp))
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


nm.plot(pd.concat([df_tp, n_forecast], ignore_index=True));
plt.title("Forecast of the Time Series in the next 60 days")
plt.xlabel("Dates")
plt.ylabel("Rainfall (mm)")


# In[52]:


nm.plot_components(pd.concat([df_tp, n_forecast], ignore_index=True));


# ### Performance Metrics

# In[53]:


n_forecast


# In[54]:


n_forecast_merge = pd.merge(df_tp, n_forecast[['ds','yhat1','residual1']],on='ds')
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

# *   At least for this particular dataset and using the default arguments,  the **NeuralProphet** model scored a **MSE** of **228.275052** and **RMSE** of **15.108774** whereas the **Prophet** model scored a **MSE** of **242.997213** and **RMSE** of **15.588368**, a **14.722161218709658 and 0.47959387483461136 difference of MSE and RMSE respectively** compared against the first model.
