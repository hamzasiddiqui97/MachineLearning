# In[12]:

import pandas as pd
import numpy as np
import matplotlib as plt


# In[2]:

from math import e
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
n_samples , n_features = 10,5
rng = np.random.RandomState (0)
y = rng.randn (n_samples)
X = rng.randn (n_samples,n_features)
# Always scale the input . The most convenient way is to use a pipeline .
reg = make_pipeline (StandardScaler(),SGDRegressor(max_iter =1000 , tol =e**-3))
reg.fit(X,y)


# In[13]:


dataset = pd.read_csv('startup.csv')
X = dataset.iloc [: , : -1]
y = dataset.iloc [: , 4]
# Convert the column into categorical columns
states = pd.get_dummies (X ['State'] , drop_first = True )
# Drop the state coulmn
X = X.drop ('State', axis =1)
# concat the dummy variables
X = pd.concat ([ X , states ] , axis =1)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split (X , y , test_size = 0.2 ,random_state = 0)
# Fitting Multiple Linear Regression to the Training set
from sklearn . linear_model import LinearRegression
regressor = LinearRegression ()
regressor.fit ( X_train, y_train)
# Predicting the Test set results
y_pred = regressor.predict ( X_test )
from sklearn.metrics import r2_score
score = r2_score( y_test , y_pred )
print (score)


# # Handling Categorical target variable - Logistic Regression

# In[45]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv ('RPMVibration2.csv')
df = pd.DataFrame ( data )
le = LabelEncoder ()
X = np.array ( df.loc [: ,[ 'RPM','Vibration']])
df ['Status'] = le.fit_transform ( df['Status'])
y = np . array ( df['Status'])


# In[5]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression ( solver ='liblinear', random_state =0)
model.fit(X, y)
print ('Model Information :')
print ('Model classes :', model . classes_ , '\n Model Intercept w [0]:',model . intercept_ ,'\n Model coefficients w [1] and w [2]:', model . coef_ )


# In[6]:


model.predict_proba ( X )
model.predict ( X )
model.score (X , y )


# In[5]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
confusion_matrix (y , model.predict (X))
classification_report (y,model.predict(X))


# In[8]:


import pandas as pd
import numpy as np
df = pd.read_csv('RainGrowth.csv')
df.head()


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns
X = df['Rain']
y = df['Growth']
plt.scatter(X, y)
plt.show()


# In[11]:


def objective(X,a,b,c):
    return(a*(X**2)) + (b*X) + c

# define the true Sigmoid function
def sigmoid(X,a,b,c):
    y = 1/(1 + np.exp((a*(X**2)) + (b*X) + c))
    return y

objective(X, -1.717, 8.475, 3.707)
Y = sigmoid(X, -1.717, 8.475, 3.707)


# In[22]:


from scipy.optimize import curve_fit
xdata = X/max(X)
ydata = Y/max(Y)

popt, pcov = curve_fit(sigmoid, xdata, ydata)
# Now we plot our resulting regression model .
x = np.linspace(1 ,7 ,50)
x = x/max(x)
plt.figure(figsize =(8 ,5))
y = sigmoid(x, *popt)
plt.plot(xdata,ydata,'ro ',label ='data')
plt.plot(x,y,linewidth =3.0,label ='fit')
plt.legend(loc ='best')
plt.ylabel('Grass Growth')
plt.xlabel('Rain Fall')
plt.show()


# # TASK 1:

# In[58]:


import pandas as pd
import numpy as np
df = pd.read_csv('food.csv')
df.head()


# In[59]:


y = df['incomebc']
X = df.drop('incomebc',axis = 1)
reg = make_pipeline ( StandardScaler () ,SGDRegressor ( max_iter =1000 , tol =e**-3) )
reg . fit (X , y)


# In[61]:


import matplotlib.pyplot as plt
plt.scatter(X['food_exp'],y)
plt.show()


# In[15]:


plt.scatter(X['income'],y)
plt.show()


# # TASK 2:

# In[62]:


dataset = pd.read_csv ('flights.csv')
dataset.head()


# In[63]:


dataset = dataset.drop[('tailnum', axis = 1)


# In[64]:


dataset.isna().sum()


# In[65]:


dataset.fillna(0,inplace=True)


# In[66]:


dataset.isna().sum()


# In[67]:


final_dataset = pd.get_dummies(dataset,drop_first=True)
final_dataset.head()


# In[68]:


X = final_dataset.drop(['dep_delay'], axis = 1)
y = final_dataset['dep_delay']
from sklearn . model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 ,random_state = 0)
from sklearn . linear_model import LinearRegression
regressor = LinearRegression()
regressor . fit ( X_train , y_train )
# Predicting the Test set results
y_pred = regressor . predict ( X_test )
from sklearn . metrics import r2_score
score = r2_score ( y_test , y_pred )
print ( score )


# In[77]:


flight_delay =  y_test - y_pred
print(flight_delay)


# # TASK 3:

# In[28]:


import pandas as pd
import numpy as np
data = pd.read_csv ('ALCustomers.csv')
df = pd.DataFrame ( data )
df.head()


# In[29]:


df.shape


# In[30]:


df.isna().sum()


# In[31]:


df.fillna(0,inplace=True)


# In[32]:


df['satisfaction'] = le.fit_transform(df['satisfaction'])
df['Gender'] = le.fit_transform(df['Gender'])
df['Customer Type'] = le.fit_transform(df['Customer Type'])
df['Type of Travel'] = le.fit_transform(df['Type of Travel'])
df['Class'] = le.fit_transform(df['Class'])


# In[33]:


X = df.drop('satisfaction',axis=1)
df ['satisfaction'] = le.fit_transform ( df['satisfaction'])
y = np . array ( df['satisfaction'])


# In[34]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression ( solver ='liblinear', random_state =0)
model.fit(X, y)
print ('Model Information :')
print ('Model classes :', model . classes_ , '\n Model Intercept w [0]:',model . intercept_ ,'\n Model coefficients w [1] and w [2]:', model . coef_ )


# In[35]:


model . predict_proba ( X )
model . predict ( X )
model . score (X , y )


# In[36]:


confusion_matrix(y, model.predict(X))
classification_report(y, model.predict(X))


# In[37]:


import pandas as pd
import numpy as np
df = pd.read_csv('EEG_responses.csv')
df.head()


# In[38]:


X = df['P20']
y = df['P45']
plt.scatter(X, y)
plt.show()


# In[69]:


def objective(X,y):
    return ((0.0683*(X*y))+ (0.7155*(y**3)) + (0.1937*(X**3))+ (0.5404*(y**2)) + (-0.6319*(X**2)) + (-0.8479*y) + (0.0929*X) + -0.6319)

# define the true Sigmoid function
def sigmoid(X,y):
    y = 1/(1 + np.exp((0.0683*(X*y))+ (0.7155*(y**3)) + (0.1937*(X**3))+ (0.5404*(y**2)) + (-0.6319*(X**2)) + (-0.8479*y) + (0.0929*X) + -0.6319))
    return y
           
Y =sigmoid(X, y)


# In[74]:
xdata = X/max(X)
ydata = Y/max(Y)
# In[75]:
from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid,xdata,ydata)
# Now we plot our resulting regression model .
x = np.linspace(1 ,7 ,50)
x = x/max(x)
plt.figure(figsize =(8 ,5))
y = sigmoid(x, *popt)
plt.plot(xdata,ydata,'ro ',label ='data')
plt.plot(x,y,linewidth =3.0,label ='fit')
plt.legend(loc ='best')
plt.ylabel('Grass Growth')
plt.xlabel('Rain Fall')
plt.show()
