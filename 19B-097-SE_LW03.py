#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install pandas


# In[3]:


pip install matplotlib


# In[2]:


import pandas as pd
data = pd.read_csv("MIFClaimABTFull.csv")
df = pd.DataFrame(data)
df.head()


# In[4]:


from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

filename = 'MIFClaimABTFull.csv'
name = ['ID','InsuranceType','IncomeofPolicyHolder','MaritalStatus','NumClaimants','InjuryType','OvernightHospitalStay','ClaimAmount','TotalClaimed','NumClaims','NumSoftTissue','SoftTissue','ClaimAmountReceived','FraudFlag']
data = read_csv(filename)
peek = data.head()
print(peek)


# In[7]:


df.dtypes


# In[5]:


df.describe()


# In[17]:


data2 = pd.read_csv("diabetes.csv")
df2 = pd.DataFrame(data2, columns =[ 'Glucose',"BMI"])
df2.head()

df2.loc[df2['BMI'] >= 50]


# In[41]:


from pandas import set_option
set_option('display.width', 100)
set_option('display.precision', 2)
correlations = data.corr(method='pearson')
print(correlations)


# In[38]:


from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

filename = 'MIFClaimABTFull.csv'
name = ['ID','InsuranceType','IncomeofPolicyHolder','MaritalStatus','NumClaimants','InjuryType','OvernightHospitalStay','ClaimAmount','TotalClaimed','NumClaims','NumSoftTissue','%SoftTissue','ClaimAmountReceived','FraudFlag']
data=read_csv(filename)
peek = data.head()
print(peek)


# In[42]:


skew = data.skew()
print(skew)


# In[44]:


#histograms
data.hist()
pyplot.show()


# In[45]:


#scatter plot matrix
scatter_matrix(data)
pyplot.show()


# In[58]:


#scatter plot matrix
data1 = pd.read_csv('Iris.csv')
data2 = pd.DataFrame(data1)
df = data1.iloc[:,2:4]
scatter_matrix(df)
pyplot.show()


# In[62]:


#import seaborn
get_ipython().system('pip install seaborn')


# In[63]:


import seaborn as sns
import pandas as pd

MIFC = pd.read_csv('MIFClaimABTFull.csv')
sns.set_theme()

sns.scatterplot(data=MIFC,x= 'IncomeofPolicyHolder', y = 'ClaimAmount')


# In[68]:


get_ipython().system('pip install pandas-profiling')


# In[71]:


from pandas_profiling import ProfileReport
import pandas as pd
names = []
data = pd.read_csv('Iris.cvs',names = names)
df = pd.DataFrame(data)
prof = ProfileReport(df)
prof.to_file(output_file ='output.html')


# In[76]:


import pandas as pd
data = pd.read_csv('Property.csv', delimiter = ';')    #names removed from the parameter

df = pd.DataFrame(data)
df.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[39]:


data = pd.read_csv ("diabetes.csv")
df = pd.DataFrame( data)
df.drop(columns=['Pregnancies','SkinThickness','DiabetesPedigreeFunction'])


# In[40]:


df = pd.DataFrame({
'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie'] ,
'style': ['cup', 'cup', 'cup', 'pack', 'pack'] ,
'rating': [4 , 4 , 3.5 , 15 , 5]
})
df.drop_duplicates()


# In[20]:


import numpy as np
df = pd . DataFrame ({" name ": ['Alfred', 'Batman','Catwoman'],
" toy": [ np.nan , 'Batmobile', 'Bullwhip'] ,
" born ": [ pd . NaT , pd . Timestamp ("1940-04-25") ,
pd.NaT]})
df.dropna()


# In[21]:


df = pd . DataFrame ([[ np . nan , 2 , np . nan , 0] ,
[3 , 4 , np . nan , 1] ,
[ np . nan , np . nan , np . nan , np . nan ] ,
[ np . nan , 3 , np . nan , 4]] ,
columns = list("ABCD"))
df . fillna (0)


# In[22]:


values = {"A": 0 , "B": 1 , "C": 2 , "D": 3}
df . fillna ( value = values )


# In[30]:


pip install sklearn


# In[37]:


from numpy import array
import pandas as pd
from sklearn.impute import SimpleImputer
data = array ([[ 1. , 2. , 3. , 4.] ,[ 5. , 6. , np.nan , 8.] ,[ 10. , 11. , 12. , np.nan ]])
df = pd.DataFrame( data )
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit( df )
imputed_data = imr.transform(df.values )
print( imputed_data )


# In[32]:


df = pd . DataFrame ({ 'Animal': ['Falcon', 'Falcon',
'Parrot', 'Parrot'] ,
'Max Speed': [380. , 370. , 24. , 26.]})
df.groupby (['Animal']).mean ()


# In[34]:


data2 = pd . read_csv ("diabetes.csv ")
df2 = pd . DataFrame ( data2 )
df2 . cov ()
df2 . corr ()


# In[35]:


cleanup_nums = {" property_type ": {" House ": 1 , " Flat ": 2 ," Upper Portion": 3 ," Lower Portion ": 4 ," Room ": 5 ," Farm House ": 6 , " Penthouse ": 7} ," province_name ": {" Punjab ": 1 , " Sindh ": 2 , " KPK ": 3 , " Balochistan ": 4 ,"GB": 5 , " Islamabad Capital ": 6}}
encoded_df = df . replace ( cleanup_nums )
encoded_df . head ( n =100)


# In[36]:


import numpy as np
class_mapping = { label : idx for idx , label in enumerate ( np.unique ( df ['Animal']))}
print (class_mapping)

