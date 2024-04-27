#!/usr/bin/env python
# coding: utf-8

# ### Predicting canada's per capita income of different year

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# ### Import CSV file

# In[50]:


df=pd.read_csv('canada_per_capita_income.csv' )
df


# In[4]:


df.head(5)


# In[47]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('year')
plt.ylabel('per capita income (US$)')
plt.scatter(df['year'], df['per capita income (US$)'], color='red', marker='+')
plt.plot(df['year'], reg.predict(df[['year']]), color='blue', label='Regression Line')
plt.show()


# ### Training Model

# In[38]:


reg = linear_model.LinearRegression()
reg.fit(df[['year']], df['per capita income (US$)'])


# #### PREDICT PER CAPITA INCOME OF YEAR 2020
# 

# In[42]:


reg.predict([[2020]])


# #### PREDICT PER CAPITA INCOME OF YEAR 1992

# In[24]:


reg.predict([[1992]])


# #### OUTPUT

# In[25]:


predicted_price_of_year_2020 = reg.predict([[2020]])
print("Predicted price of year 2020 is ",predicted_price_of_year_2020)


# #### CALCULATING INTERCEPT

# In[26]:


reg.intercept_


# #### CALCULATING coefficient

# In[27]:


reg.coef_


# #### VERIFICATION WITH LINEAR EQUATION  Y=M*X+B WHERE Y=PREDICTION(PER CAPITA INCOME), M= SLOPE, X=YEAR AND B= INTERCEPT

# In[28]:


828.46507522*2020+(-1632210.7578554575)

