#!/usr/bin/env python
# coding: utf-8

# # Filter Methods for Features Selection

# # Information Gain

# In[7]:


import pandas as pd
dataframe=pd.read_csv('diabetes.csv')
dataframe.columns


# In[8]:


dataframe.head()


# In[18]:


import pandas as pd
from sklearn.feature_selection import mutual_info_classif

dataframe=pd.read_csv('diabetes.csv')
X= dataframe.iloc[:,0:8]
Y = dataframe['Outcome']


importance = mutual_info_classif (X,Y)
feat_importance = pd.Series (importance , dataframe.columns [0:len(dataframe.columns)-1])
ax = feat_importance.plot ( kind ='barh', color ='teal')
fig = ax.get_figure()
fig.savefig('figure_IGFS.pdf')


# # x2 test:

# In[25]:


import pandas as pd
data = pd.read_csv('Churn_Modelling.csv')
churn_df = pd.DataFrame(data)
churn_df.head()


# In[30]:


from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns


# In[26]:


import pandas as pd
from sklearn.feature_selection import mutual_info_classif
churn_df_sel = pd.read_csv('Churn_Modelling.csv')
churn_df_sel.head()
churn_df_sel = churn_df.loc [: ,['Geography','Gender','HasCrCard', 'IsActiveMember','Exited']]
churn_df_sel.head ()


# In[41]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
churn_df_sel['Geography'] = label_encoder.fit_transform(churn_df['Geography'])
churn_df_sel['Gender'] = label_encoder.fit_transform(churn_df['Gender'])
churn_df_sel.head(2)


# In[46]:


from sklearn.feature_selection import chi2
X = churn_df_sel.drop ('Exited',axis =1)
y = churn_df_sel['Exited']
# chi square test with 95% confidence interval
chi_scores = chi2(X,y)
print(churn_df_sel.shape)
chi_scores[0]


# In[37]:


p_values = pd.Series(chi_scores[1] , index=X.columns)
print (p_values[:])
p_values.sort_values(ascending=True , inplace=True)


# In[70]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
X_new = SelectPercentile(chi2 , percentile =50) 
#X_new.fit_transform(X , y)
feature_names = ['HasCrCard', 'IsActiveMember','Exited']

# use percentile method
X_new = SelectKBest (chi2 , k = 3)
# k best method
X_new.fit_transform(X , Y)



# In[57]:


new_features = [] 
# The list of your K best features
mask = X_new.get_support()
# Print selected fatures
 # The list of your K best features
new_features = []
for bool , feature in zip(mask , feature_names) :
    if bool :
        new_features.append(feature)
new_features


# # EXAMPLES:

# In[79]:


get_ipython().system('pip install scipy-stats')


# In[75]:


from scipy.stats import chi_contingency
from scipy.stats import chi1


table = [[250,200],[50,1000]]
print(table)
stat,p , dof, expected = chi2_contingency(table)
print('dof=%d')


# # FISHER SCORE

# In[22]:


get_ipython().system('pip install skfeature_chappers')


# In[81]:


from skfeature.function.similarity_based import fisher_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
data = pd.read_csv ('Churn_Modelling.csv')
df = pd.DataFrame(data)
churn_df = df.loc[: ,[ 'Geography','Gender','HasCrCard', 'IsActiveMember','Exited']]

label_encoder = LabelEncoder()
churn_df['Geography'] = label_encoder.fit_transform(churn_df['Geography'])
churn_df['Gender'] = label_encoder.fit_transform (churn_df['Gender' ])
X_train = churn_df.drop('Exited', axis =1)
y_train = churn_df['Exited']
# calculate score
rank = fisher_score.fisher_score(X_train.to_numpy(), y_train.to_numpy() ,mode ='rank')
print(rank)


# In[82]:


# print histogram graph
feat_importances = pd.Series(rank , churn_df.columns[0: len (churn_df.columns)-1])
feat_importances.plot ( kind ='barh', color ='teal')
plt.show()


# # Correlation Coefficient & Variance method:
# 

# In[14]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# corelation matrix
corr = churn_df.corr ()
plt.figure (figsize =(10 ,6))
sns.heatmap (corr,annot = True


# # VARIANCE THRESHOLD:

# In[59]:


from sklearn.feature_selection import VarianceThreshold
v_threshold = VarianceThreshold (threshold=0)
v_threshold.fit (churn_df)
v_threshold.get_support()


# # Mean Absolute Difference (MAD):

# In[48]:


# load data
import pandas as pd
import numpy as np
data = 'Iris.csv'
names = ['ID','sepalLength','sepalWidth','petalLength','petalWidth','species']
df = pd.read_csv (data,names=names)
df.head ()
X = df.iloc [: ,1:5]
Y = df.iloc [: ,5]
X.head()


# In[83]:


mad = X.mad (axis =0);
print (" Mean absolute deviation of columns :");
print(mad);


# # DISPERSION RATIO:

# In[84]:


import matplotlib.pyplot as plt
X = X +1
am=0
gm=0
am = np.mean(X , axis =0)
gm = np.power(np.product(X , axis =0),1/ X.shape [0])
# ratio of arithmatic mean and geometric mean
disp_ratio = am/gm
plt.bar(np.arange(X.shape [1]),disp_ratio,color ='teal')


# In[ ]:




