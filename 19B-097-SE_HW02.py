#!/usr/bin/env python
# coding: utf-8

# # Task1

# In[1]:


import seaborn as sns
import pandas as pd
df = sns.load_dataset ('tips')
df.head (n= 10)


# In[2]:


df.describe()


# # Task2

# In[3]:


import seaborn as sns
import pandas as pd
df=pd.read_csv('pima-indians-diabetes.csv')
df.head(n=10)


# In[4]:


import seaborn as sns
import pandas as pd
df=pd.read_csv('diabetes.csv')
df.head(n=10)


# # Task3

# In[5]:


import seaborn as sns
import pandas as pd
sns.set_style("dark")
data= pd.read_csv('heart.csv')
sns.relplot(data = data, x ='Cholesterol', y ='Age', col ='Sex')


# In[6]:


sns.relplot(data = data, x ='RestingECG', y ='Age', col ='Sex')


# In[7]:


sns.relplot(data = data, x ='MaxHR', y ='Age', col ='Sex')


# # Task4

# In[8]:


from numpy import isnan
from pandas import read_csv
from sklearn.impute import SimpleImputer
# load dataset
url='https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.csv'
dataframe = read_csv ( url , header = None , na_values ='?')
# split into input and output elements
data = dataframe.values
ix=[i for i in range(data.shape[1]) if i != 23]
X,y=data[:,ix ],data[:,23]
# print total missing
print (f'Missing : {sum(isnan(X).flatten())}')
# define imputer
imputer=SimpleImputer(strategy='mean')
# fit on the dataset
imputer.fit(X)
# transform the dataset
Xtrans = imputer.transform(X)
# print total missing
print(f'Missing : {sum(isnan(Xtrans).flatten())}')


# # Task5 

# In[10]:


dataframe = read_csv ('heart.csv', na_values ='NA',)
print("The Total Null Values of each column Before Imputation are: ", dataframe.isnull().sum())
dataframe.fillna('0', inplace = True)
print("The Total Null Values After", dataframe.isna().sum())


# In[ ]:




