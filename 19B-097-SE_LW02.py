#!/usr/bin/env python
# coding: utf-8

# In[2]:


## if library in not installed in the directory
# !pip install -U --user seaborn


# # Example1

# In[10]:


import seaborn as sns
import pandas as pd
df = sns.load_dataset ('tips')

# df.head()   # if we dont mention value of n then it will show beginning rows

df.head(n=10)  # if n=10 then it will show 10 records

# dataset provided by seaborn library for practice


# In[11]:


df.describe()


# # Example2

# In[12]:


import seaborn as sns

sns.set()
# Load an example dataset
tips = sns.load_dataset("tips")
sns.relplot(
data = tips ,
x ="total_bill", y ="tip", col ="time",
hue ="smoker", style ="smoker", size ="size",
)


# In[13]:


# import seaborn as sns
# tips = sns.load_dataset("tips")
sns.displot(data=tips,x="total_bill", col = "time", kde= True)


# In[14]:


df.describe(include='category').T


# # Example3

# In[15]:


from numpy import isnan
from pandas import read_csv
from sklearn . impute import SimpleImputer
# load dataset
url = 'https://raw.githubusercontent.com//jbrownlee//Datasets//master//horse-colic.csv'
dataframe = read_csv(url,header=None,na_values='?')
# split into input and output elements
data = dataframe.values
ix = [ i for i in range( data.shape[1]) if i != 23]
X , y = data [: , ix ] , data [: , 23]
# print total missing
print (f'Missing : {sum(isnan(X ).flatten())}' )
# define imputer
imputer = SimpleImputer( strategy ='mean')
# fit on the dataset
imputer.fit (X)
# transform the dataset
Xtrans = imputer.transform (X)
# print total missing
zzz = sum(isnan(Xtrans).flatten())
print (f'Missing : {zzz}')


# In[31]:


pip install seaborn==0.11.0 --user


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




