#!/usr/bin/env python
# coding: utf-8

# # HOMEWORK TASKS:

# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif


# In[16]:


get_ipython().system('pip install sklearn')


# # TASK 1:

# In[ ]:


import pandas as pd 
from sklearn.preprocessing import LabelEncoder
col_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
data = pd.read_csv('housing.csv',names=col_names,header=None, delimiter=r"\s+")
df = pd.DataFrame(data)
# 2. split data into descriptive and target features in X and y variables respectively .
X = df .iloc [: ,1:12]
le = LabelEncoder()
df.iloc[:,13] = le.fit_transform(df.iloc[:,13])
Y = df . iloc [: ,13]
# 3. load important libraries
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
# 4. Apply Model
lr = LogisticRegression ( class_weight ="balanced", solver ="lbfgs", random_state=42 , n_jobs = -1 , max_iter =500)
lr.fit(X,Y)
# 5. Select best features
bfs = SFS (lr ,
           k_features ='best',
           forward = True ,
           floating = False ,
           verbose =2 ,
           scoring ='accuracy',
           cv =0)
bfs.fit (X , Y )
# 6. print feature list
features = list ( bfs.k_feature_names_ )
print (features)


# # TASK 2:

# In[ ]:


import pandas as pd 
from sklearn.preprocessing import LabelEncoder
col_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
data = pd.read_csv('housing.csv',names=col_names,header=None, delimiter=r"\s+")
df = pd.DataFrame(data)
# 2. split data into descriptive and target features in X and y variables respectively .
X = df .iloc [: ,1:12]
le = LabelEncoder()
df.iloc[:,13] = le.fit_transform(df.iloc[:,13])
Y = df . iloc [: ,13]
# 3. load important libraries
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
# 4. Apply Model
lr = LogisticRegression ( class_weight ="balanced", solver ="lbfgs", random_state=42 , n_jobs = -1 , max_iter =500)
lr.fit(X,Y)
# 5. Select best features
bfs = SFS (lr ,
           k_features ='best',
           forward = True ,
           floating = False ,
           verbose =2 ,
           scoring ='accuracy',
           cv =0)
bfs.fit (X , Y )
# 6. print feature list
features = list ( bfs.k_feature_names_ )
print (features)


# # TASK 3: EXHAUSTIVE FEATURE SELECTION

# In[18]:


import pandas as pd 
from sklearn.preprocessing.label import LabelEncoder
df=pd.read_csv('MIFClaimABTFull.csv')
X = df.iloc [: ,1:12]
le = LabelEncoder()
df.iloc[:,13] = le.fit_transform(df.iloc[:,13])
Y = df . iloc [: ,13]
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

knn = KNeighborsClassifier(n_neighbors=3)
efs1 = EFS (knn,min_features =1 ,max_features =4,scoring ='accuracy',print_progress=True,cv=5)
efs1 = efs1.fit(X,Y)
print ('Best accuracy score: %.2f'% efs1.best_score_)
print ('Best subset (indices):',efs1.best_idx_ )
print ('Best subset (corresponding names ):',efs1.best_feature_names_)


# # TASK 4:

# In[19]:


# Recursive Feature Elimination:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('BreastCancerData.csv')
df = pd.DataFrame(data)
X=df.drop(['id','diagnosis'],axis=1)
X.head()
Y=df['diagnosis']
features = X.columns
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.30,random_state =42)
from sklearn . pipeline import Pipeline
from sklearn . model_selection import RepeatedStratifiedKFold
from sklearn . model_selection import cross_val_score
from sklearn . feature_selection import RFE
import numpy as np
from sklearn . ensemble import GradientBoostingClassifier
# initialize recursive feature elimination class
rfe = RFE ( estimator = GradientBoostingClassifier () , n_features_to_select =6)
model = GradientBoostingClassifier ()
# make a piline for execution
pipe = Pipeline ([('Feature Selection',rfe),('Model',model)])
cv = RepeatedStratifiedKFold ( n_splits =10 , n_repeats =5 , random_state=36851234)
# compute and print score of each feature
n_scores = cross_val_score(pipe,X,Y, scoring ='accuracy', cv=cv , n_jobs = -1)
print (np.mean (n_scores))
# Execute RFE model
pipe.fit (X, Y)
rfe.support_
rfe . support_


# # TASK 5:LASSO REGRESSION
#     

# In[9]:


col_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
data = pd.read_csv('housing.csv',names=col_names,header=None, delimiter=r"\s+")
df = pd.DataFrame(data)
df.head()

import numpy as np
from sklearn . pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso,Ridge

X = df .iloc [: ,1:12]
Y = df . iloc [: ,13]
features = X.columns
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size =0.33,random_state =42)
pipeline = Pipeline ([('scaler', StandardScaler()),('model',Lasso())])
search = GridSearchCV(pipeline,{'model__alpha':np.arange(0.1,10 ,0.1)},cv=5,scoring ="neg_mean_squared_error",verbose=3)
search.fit(X_train,y_train)
coefficients = search.best_estimator_.named_steps['model']. coef_
importance = np.abs(coefficients)
print(importance)
print(np.array(features)[importance > 0])


# # TASK 6:RANDOM FOREST FEATURE SELECTION

# In[5]:


import pandas as pd
df = pd.read_csv('diabetes.csv')
df.head()


# In[6]:


import pandas as pd
#Random forest features selection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
features = df.columns
X = df.iloc[: ,1:7]
y = df.iloc[: ,8]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.33,random_state=42)
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train,y_train)
rf.feature_importances_


# In[7]:


import matplotlib.pyplot as plt
f_i = list(zip(features,rf.feature_importances_))
f_i.sort(key=lambda x:x[1])
plt.barh([x[0] for x in f_i],[x[1] for x in f_i])
plt.show()


# In[ ]:




