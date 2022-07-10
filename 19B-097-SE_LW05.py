#!/usr/bin/env python
# coding: utf-8

# # Forward Feature Selection:
# 



# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif


# In[2]:

import pandas as pd
df=pd.read_csv('Iris.csv')
X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
Y = df[['Species']]
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
# 4. Apply Model
lr = LogisticRegression (class_weight ='balanced', solver ='lbfgs', random_state=42 , n_jobs = -1 , max_iter =500)
lr.fit (X , Y)
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


# In[37]:


import pandas as pd
import numpy as np
data = 'Iris.csv'
df1=pd.read_csv(data)
df = pd.DataFrame(df1)
df = df.dropna()
# 2. split data into descriptive and target features in X and y variables respectively .
X=df.iloc[:,1:5]
Y=df.iloc[:,5]
# 3. load important libraries
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
# 4. Apply Model
lr = LogisticRegression(class_weight='balanced', solver ='lbfgs', random_state=42 , n_jobs = -1 , max_iter =500)
lr.fit(X,Y)
# 5. Select best features
bfs=SFS(lr,
k_features ='best',
forward = True ,
floating = False ,
verbose =2 ,
scoring ='accuracy',
cv =0)
bfs.fit(X,Y)
# 6. print feature list
features = list (bfs.k_feature_names_)
print(features)


# # BACKWARD FEATURE SELECTION:

# In[28]:


import pandas as pd
df=pd.read_csv('Iris.csv')
X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
Y = df[['Species']]
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
# 4. Apply Model
lr = LogisticRegression (class_weight ='balanced', solver ='lbfgs', random_state=42 , n_jobs = -1 , max_iter =500)
lr.fit (X , Y)
# 5. Select best features
bfs = SFS (lr ,
           k_features ='best',
           forward = False,
           floating = False ,
           verbose =2 ,
           scoring ='accuracy',
           cv =0)
bfs.fit (X , Y )
# 6. print feature list
features = list ( bfs.k_feature_names_ )
print (features)


# In[38]:


import pandas as pd
import numpy as np
data = 'Iris.csv'
df1=pd.read_csv(data)
df = pd.DataFrame(df1)
df = df.dropna()
# 2. split data into descriptive and target features in X and y variables respectively .
X=df.iloc[:,1:5]
Y=df.iloc[:,5]
# 3. load important libraries
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
# 4. Apply Model
lr = LogisticRegression(class_weight='balanced', solver ='lbfgs', random_state=42 , n_jobs = -1 , max_iter =500)
lr.fit(X,Y)
# 5. Select best features
bfs=SFS(lr,
k_features ='best',
forward = False ,
floating = False ,
verbose =2 ,
scoring ='accuracy',
cv =0)
bfs.fit(X,Y)
# 6. print feature list
features = list (bfs.k_feature_names_)
print(features)


# In[5]:


get_ipython().system('pip install mlxtend')


# # Exhaustive Feature Selection:
# 

# In[44]:


data = 'Iris.csv'
df1=pd.read_csv(data)
df = pd.DataFrame(df1)
df = df.dropna()
# 2. split data into descriptive and target features in X and y variables respectively .
X=df.iloc[:,1:5]
Y=df.iloc[:,5]
# Exhaustive Feature Selection:
from sklearn.neighbors import KNeighborsClassifier
from mlxtend .feature_selection import ExhaustiveFeatureSelector as EFS
knn=KNeighborsClassifier(n_neighbors =3)
efs1 = EFS(knn,min_features =1,max_features =4,scoring ='accuracy',print_progress=True,cv =5)
efs1 = efs1.fit(X,Y)
print('Best accuracy score : %.2f'%efs1.best_score_)
print('Best subset (indices ):', efs1.best_idx_)
print('Best subset (corresponding names):', efs1.best_feature_names_)


# # Recursive Feature Elimination:

# In[6]:


import pandas as pd
data = pd.read_csv ("diabetes.csv")
df = pd.DataFrame (data)
df.head ()


# In[15]:


x = df.drop(['Outcome'],axis=1)
y = df['Outcome']


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test, x_train, x_test = train_test_split(x, y,random_state=0)


# In[9]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
import numpy as np
from sklearn . ensemble import GradientBoostingClassifier

rfe = RFE ( estimator = GradientBoostingClassifier () , n_features_to_select =6)
model = GradientBoostingClassifier ()


# In[10]:


pipe = Pipeline ([('Feature Selection', rfe ) , ('Model', model ) ])
cv = RepeatedStratifiedKFold ( n_splits =10 , n_repeats =5 , random_state=36851234)

n_scores = cross_val_score ( pipe , X , Y , scoring ='accuracy', cv =cv , n_jobs = -1)
print(np.mean(n_scores))

pipe.fit ( X , Y )
rfe.support_


# # C. Embedded Methods:
# 

# # LASSO Regularization (L1):

# In[16]:


import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso,Ridge
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv('Iris.csv')
label_encoder=LabelEncoder()
df['Species']= label_encoder.fit_transform(df['Species'])
X=df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
Y=df['Species']
features = X.columns
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size =0.33,random_state =42)
pipeline = Pipeline ([('scaler', StandardScaler()),('model',Lasso())])
search = GridSearchCV(pipeline,{'model__alpha':np.arange(0.1,10 ,0.1)},cv=5,scoring ="neg_mean_squared_error",verbose=3)
search.fit(X_train,y_train)
coefficients = search.best_estimator_.named_steps['model'].coef_
importance = np.abs(coefficients)
print(importance)
print(np.array(features)[importance > 0])


# # Random forest features selection

# In[46]:


from sklearn.ensemble import RandomForestRegressor

col_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
data = pd.read_csv('housing.csv',names=col_names,header=None, delimiter=r"\s+")
df = pd.Datarame(data)
print(df.head())
features =df.columns


# In[47]:


import matplotlib.pyplot as plt

X = df.iloc [: ,1:12]
y = df.iloc [: ,13]

X_train , X_test , y_train , y_test = train_test_split ( X , y , test_size =0.33 ,random_state =42)

rf = RandomForestRegressor ( random_state =0)
rf.fit ( X_train , y_train )

rf = RandomForestRegressor ( random_state =0)
rf . fit ( X_train , y_train )

f_i = list(zip(features,rf.feature_importances_))
f_i.sort(key=lambda x:x[1])
plt.barh([x[0] for x in f_i],[x[1] for x in f_i])
plt.show()


# # Hybrid methods

# # Genetic Algorithm

# In[24]:


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data=pd.read_csv("BreastCancerData.csv")
df=pd.DataFrame(data)
X=df.drop(['id','diagnosis'],axis=1)

X.head()
y = df['diagnosis']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state =42)


# In[27]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
print("Accuracy = "+str(accuracy_score(y_test,predictions)))


# In[42]:


import random
import warnings
warnings.filterwarnings('ignore')
logmodel = LogisticRegression()
def initilization_of_population ( size , n_feat ) :
    population = []
    for i in range ( size ) :
        chromosome = np.ones ( n_feat , dtype = np . bool )
        chromosome [: int (0.3* n_feat ) ]= False
        np . random.shuffle ( chromosome )
        population.append ( chromosome )
    return population
def fitness_score ( population ) :
    scores = []
    for chromosome in population :
        logmodel . fit ( X_train . iloc [: , chromosome ] , y_train )
        predictions = logmodel . predict ( X_test . iloc [: , chromosome ])
        scores . append ( accuracy_score ( y_test , predictions ) )
    scores , population = np . array ( scores ) , np . array ( population )
    inds = np . argsort ( scores )
    return list ( scores [ inds ][:: -1]) , list ( population [ inds ,:][:: -1])

def selection ( pop_after_fit , n_parents ) :
    population_nextgen = []
    for i in range(n_parents) :
        population_nextgen.append ( pop_after_fit [ i ])
    return population_nextgen

def crossover (pop_after_sel) :
    population_nextgen = pop_after_sel
    for i in range (len(pop_after_sel ) ) :
        child = pop_after_sel [ i ]
        child [3:7]= pop_after_sel [( i +1) %len ( pop_after_sel ) ][3:7]
        population_nextgen.append ( child )
    return population_nextgen

def mutation ( pop_after_cross,mutation_rate ) :
    population_nextgen = []
    for i in range (0,len(pop_after_cross ) ) :
        chromosome = pop_after_cross [ i ]
        for j in range(len( chromosome ) ) :
            if random . random () < mutation_rate :
                chromosome [ j ]= not chromosome [ j ]
        population_nextgen . append ( chromosome )
# print ( population_nextgen )
    return population_nextgen

def generations ( size , n_feat , n_parents , mutation_rate , n_gen , X_train ,X_test , y_train , y_test ) :
        best_chromo = []
        best_score = []
        population_nextgen = initilization_of_population ( size , n_feat )
        for i in range ( n_gen ) :
            scores , pop_after_fit = fitness_score ( population_nextgen )
            print ( scores [:2])
            pop_after_sel = selection ( pop_after_fit , n_parents )
            pop_after_cross = crossover ( pop_after_sel )
            population_nextgen = mutation ( pop_after_cross , mutation_rate )
            best_chromo . append ( pop_after_fit [0])
            best_score . append ( scores [0])
        return best_chromo , best_score
# Implementing GA
chromo,score = generations ( size =200,n_feat =30,n_parents =100,mutation_rate=0.10,n_gen=5,X_train=X_train,X_test=X_test,y_train = y_train , y_test =y_test )
logmodel.fit(X_train.iloc[:,chromo[-1]],y_train)
predictions = logmodel . predict(X_test.iloc[:,chromo[-1]])
print (" Accuracy score after genetic algorithm is="+str(accuracy_score (
y_test , predictions ) ) )
print ('list of important features',X_train.iloc[:,chromo[-1]].columns )


