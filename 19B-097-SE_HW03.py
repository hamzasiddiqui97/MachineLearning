#!/usr/bin/env python
# coding: utf-8

# # TASK 1

# In[1]:


import pandas as pd
from pandas import read_csv
data = pd.read_csv("Property_with_Feature_Engineering.csv")
df = pd.DataFrame(data)
df.head()


# # TASK 2:

# In[2]:


types = data.dtypes
print(types)


# In[3]:


data.isna().sum()


# In[4]:


data.isnull().sum()


# In[5]:


data


# # task 3:

# In[6]:


data.fillna(2,inplace=True)
data.isna().sum()


# # TASK 4:

# In[7]:


data_frame= pd.DataFrame(data[['price_bin','city','location','province_name','baths','bedrooms','area']])
data_frame.head()


# # Task 5

# Target Feature:
#     Price
#     
# Descriptive Feature:
#     City
#     Location
#     Province Name
#     Baths
#     Bedrooms
#     Area
#     Latitude
#     Longitude
#     Property Type
#     Purpose
#     Date Added
#     Agency
#     Agent

# # TASK 6:

# In[8]:


data_frame.describe()


# # TASK 7:

# In[9]:


#COVARIANCE MATRIX
data_frame.cov()


# In[10]:


#CORREALTION MATRIX
data_frame.corr()


# # TASK 8:

# In[11]:


data = data.groupby("city")
data.head(1)


# In[12]:


data2 = data_frame.groupby('location')
data2.head(1)


# In[13]:


data3 = data_frame.groupby('area')
data3.head(1)


# # TASK 9:

# In[14]:


cleanup_nums = {"property_type": {"House": 1 ,"Flat": 2 ,"Upper Portion": 3 ,"Lower Portion": 4 ,"Room": 5 ,"Farm House": 6 , " Penthouse": 7} ,"province_name": {"Punjab": 1 , "Sindh": 2 , "KPK": 3 , "Balochistan": 4 ,"GB": 5 , "Islamabad Capital": 6}}

encoded_df = df.replace(cleanup_nums)
encoded_df.head()


# # Task 10

# In[16]:


nums = {" property_type ": {" House ": 1 , " Flat ": 2 ," Upper Portion": 3 ," Lower Portion ": 4 ," Room ": 5 ," Farm House ": 6 , " Penthouse ": 7} ,
"province_name": {" Punjab ": 1 , " Sindh ": 2, " Islamabad Capital ": 3}}
encoded_df = df.replace(nums)
encoded_df.head()

