#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[2]:


os.getcwd()


# In[3]:


os.chdir(r'C:\Users\harap\Downloads\city_temperature.csv')


# In[4]:


df = pd.read_csv('city_temperature.csv')


# In[5]:


df.head()


# In[7]:


df.shapepe


# In[11]:


df.duplicated()


# In[12]:


df.isna().sum()


# In[13]:


#Taking out only Delhi data
delhi=df[df["City"]=="Delhi"]
delhi.reset_index(inplace=True)
delhi.drop('index',axis=1,inplace=True)
delhi.describe()   


# # Plotting the temperatures

# In[14]:


plt.figure(figsize=(15,6))
plt.plot(delhi["AvgTemperature"])
plt.ylabel("Temperature",fontsize=20)


# In[15]:


from sklearn.impute import SimpleImputer
imputer=SimpleImputer()
delhi["AvgTemperature"].replace(-99,np.nan,inplace=True)#Replacing wrong entries with nan 
delhi["AvgTemperature"]=pd.DataFrame(imputer.fit_transform(delhi.loc[:,"AvgTemperature":]))


# In[16]:


#Let's see how many years of data we have in our data.
print(min(delhi["AvgTemperature"]))
years=delhi["Year"].unique()
years


# In[17]:


#Defining training and testing data
training_set=delhi[delhi["Year"]<=2015]
test_set=delhi[delhi["Year"]>2015]


# In[18]:


#Min of the temperatures
delhi.iloc[:,-1].min()


# ##how our data looks after dealing with all the wrong values.

# In[19]:


plt.figure(figsize=(15,7))
plt.plot(delhi.iloc[:,-1])
plt.xlabel("Time Series",fontsize=20)
plt.ylabel("Temperature",fontsize=20)
#making a list of values to be plotted on y axis
y_values=[x for x in range(50,101,10)]
y_values.extend([delhi.iloc[:,-1].min(),delhi.iloc[:,-1].max(),delhi.iloc[:,-1].mean()])
plt.yticks(y_values)
plt.axhline(y=delhi.iloc[:,-1].mean(), color='r', linestyle='--',label="Mean")
plt.legend(loc=1)
plt.axhline(y=delhi.iloc[:,-1].max(), color='g', linestyle=':')
plt.axhline(y=delhi.iloc[:,-1].min(), color='g', linestyle=':')


# In[20]:


from statsmodels.graphics.tsaplots import plot_acf
plot_acf(delhi["AvgTemperature"],lags=365)
#plt.show()


# In[21]:


from statsmodels.tsa.ar_model import AutoReg
model_AR=AutoReg(training_set["AvgTemperature"],lags=365)
model_fit_AR=model_AR.fit()
predictions_AR = model_fit_AR.predict(training_set.shape[0],training_set.shape[0]+test_set.shape[0]-1)


# In[22]:


import seaborn as sns
plt.figure(figsize=(15,5))
plt.ylabel("Temperature",fontsize=20)
plt.plot(test_set["AvgTemperature"],label="Original Data")
plt.plot(predictions_AR,label="Predicted Data")
plt.legend()


# In[23]:


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(predictions_AR,test_set["AvgTemperature"])
mse


# ##Now with only lag=365 taking into consideration

# In[24]:


from statsmodels.tsa.ar_model import AutoReg
model_AR2=AutoReg(training_set["AvgTemperature"],lags=[365])
model_fit_AR2=model_AR2.fit()
predictions_AR2= model_fit_AR2.predict(training_set.shape[0],training_set.shape[0]+test_set.shape[0]-1)


# In[25]:


plt.figure(figsize=(15,5))
plt.ylabel("Temperature",fontsize=20)
plt.plot(test_set["AvgTemperature"],label="Original Data")
plt.plot(predictions_AR2,label="Predicted Data")
plt.legend()


# In[26]:


#This graph looks accurate than the previous one but let's check if it actually works better or not
mse=mean_squared_error(predictions_AR2,test_set["AvgTemperature"])
mse


# # AVERAGE

# In[27]:


from statsmodels.tsa.arima_model import ARMA
model_MA=ARMA(training_set["AvgTemperature"],order=(0,10))
model_fit_MA=model_MA.fit()
predictions_MA=model_fit_MA.predict(test_set.index[0],test_set.index[-1])


# In[28]:


plt.figure(figsize=(15,5))
plt.ylabel("Temperature",fontsize=20)
plt.plot(test_set["AvgTemperature"],label="Original Data")
plt.plot(predictions_MA,label="Predictions")
plt.legend()


# In[29]:


mse=mean_squared_error(predictions_MA,test_set["AvgTemperature"])
mse


# # Autoregressive Moving Average (ARMA)

# In[ ]:


model_ARMA=ARMA(training_set["AvgTemperature"],order=(5,10))
model_fit_ARMA=model_ARMA.fit()
predictions_ARMA=model_fit_ARMA.predict(test_set.index[0],test_set.index[-1])


# In[ ]:


plt.figure(figsize=(15,5))
plt.ylabel("Temperature",fontsize=20)
plt.plot(test_set["AvgTemperature"],label="Original Data")
plt.plot(predictions_ARMA,label="Predictions")
plt.legend()


# In[ ]:


mse=mean_squared_error(predictions_ARMA,test_set["AvgTemperature"])
mse


# In[ ]:





# In[ ]:





# In[ ]:




