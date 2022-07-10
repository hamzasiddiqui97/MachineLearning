#!/usr/bin/env python
# coding: utf-8

# # TASK 01:

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif


# In[2]:


dataset = pd.read_csv('MIFClaimABTFull.csv')
dataset = dataset.replace('CI', 0)
dataset = dataset.replace('Married', 0)
dataset = dataset.replace('Single', 1)
dataset = dataset.replace('Divorced', 2)
dataset = dataset.replace('Soft Tissue', 0)
dataset =  dataset.replace('Back', 1)
dataset = dataset.replace('Broken Limb', 2)
dataset = dataset.replace('Serious', 3)
dataset = dataset.replace('No', 0)
dataset = dataset.replace('Yes', 1)
dataset.fillna(0, inplace = True)
X = dataset[['MaritalStatus','InjuryType','OvernightHospitalStay','NumClaimants','TotalClaimed','NumClaims','NumSoftTissue','%SoftTissue','ClaimAmountReceived']]
y = dataset['FraudFlag']
importance = mutual_info_classif(X , y)
feat_importance = pd.Series(importance, ['MaritalStatus','InjuryType','OvernightHospitalStay','NumClaimants','TotalClaimed','NumClaims','NumSoftTissue','%SoftTissue','ClaimAmountReceived'])
feat_importance.plot ( kind ='barh', color ='teal')


# # TASK 2:

# In[3]:


import pandas as pd
churn_df = pd.read_csv('diabetes.csv')
churn_df.head()
churn_df_sel = churn_df.loc[:,['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']]
churn_df_sel.head()


# In[4]:


from sklearn . feature_selection import chi2
X = churn_df_sel.drop ('Outcome', axis =1)
y = churn_df_sel ['Outcome']

chi_scores = chi2 (X , y )
chi_scores


# In[5]:


p_values = pd.Series(chi_scores[1], index= X.columns)
print(p_values [:])
p_values .sort_values(ascending=True, inplace=True)


# In[6]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
X_new = SelectPercentile(chi2, percentile=50) 
X_new.fit_transform(X,y)


# In[7]:


from sklearn . feature_selection import SelectKBest
from sklearn . feature_selection import SelectPercentile
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
X_new = SelectKBest (chi2 , k = 3)
X_new.fit_transform (X , y )


# In[8]:


new_features = []
mask = X_new.get_support ()
for bool, feature in zip(mask, feature_names):
    if bool:
        new_features.append(feature)
new_features


# # TASK 03:

# In[9]:


df = pd.read_csv('MIFClaimABTFull.csv')
df.head()


# In[10]:


from skfeature.function.similarity_based import fisher_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
data= pd.read_csv('MIFClaimABTFull.csv')
df = pd.DataFrame (data)
churn_df = df.loc [: ,['ID','IncomeofPolicyHolder','NumClaimants','InjuryType',
      'OvernightHospitalStay','ClaimAmount','TotalClaimed','NumClaims','NumSoftTissue','%SoftTissue',
                       'ClaimAmountReceived','FraudFlag']]
label_encoder = LabelEncoder()
churn_df ['NumSoftTissue'] = label_encoder.fit_transform (df['NumSoftTissue'])
churn_df ['MaritalStatus'] = label_encoder.fit_transform (df['MaritalStatus'])
churn_df ['InjuryType'] = label_encoder.fit_transform (df['InjuryType'])
churn_df ['OvernightHospitalStay'] = label_encoder.fit_transform (df['OvernightHospitalStay'])
df ['NumSoftTissue'] = label_encoder.fit_transform (df['NumSoftTissue'])
X_train = churn_df.drop ('FraudFlag', axis =1)
y_train = churn_df['FraudFlag']
# calculate score
rank = fisher_score.fisher_score(X_train.to_numpy(), y_train.to_numpy(),mode ='rank')
# print histogram graph
feat_importances = pd . Series(rank)
feat_importances.plot( kind ='barh', color ='teal')
plt.show()


# # TASK 4

# In[11]:


df = pd.read_csv('Property.csv',delimiter=';')
df.head()


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# corelation matrix
df = pd.read_csv('Property.csv',delimiter=';')
corr = df.corr ()
plt.figure (figsize=(10 ,6) )
sns.heatmap(corr,annot = True )


# In[12]:


from sklearn.feature_selection import VarianceThreshold
x=df[['baths','bedrooms','latitude','longitude','price']]
v_threshold = VarianceThreshold ( threshold =0)
v_threshold.fit(x)
v_threshold.get_support ()


# # TASK 5:

# In[13]:


import pandas as pd
import numpy as np
data = 'diabetes.csv'
df = pd.read_csv(data)
df.head ()
X = df.iloc[:,1:8]
Y = df.iloc[:,8]
mad = X.mad(axis=0)
print (" Mean absolute deviation of columns :", mad)


# # TASK 06:

# In[14]:


data = 'Churn_Modelling.csv'
churn_df = pd.read_csv(data)
churn_df = churn_df[['Geography', 'Gender', 'NumOfProducts', 'HasCrCard', 'IsActiveMember','Exited']]
churn_df ['Geography'] = label_encoder.fit_transform ( churn_df ['Geography'])
churn_df ['Gender'] = label_encoder.fit_transform ( churn_df ['Gender'])
# X = churn_df.drop('Exited', axis=1)
# Y = churn_df['Exited']
X = churn_df.iloc[:,1:5]
Y = churn_df.iloc[:,5]
X = X + 1
am = np.mean (X , axis =0)
gm = np.power (np.product (X , axis =0) ,1/X.shape [0])
# ratio of arithmatic mean and geometric mean
disp_ratio = am / gm
print(disp_ratio)
plt.bar ( np.arange(X.shape[1]) , disp_ratio , color ='teal')


# In[ ]:




