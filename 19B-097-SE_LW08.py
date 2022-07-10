# # LAB TASKS:
# In[21]:
import pandas as pd
df = pd.read_csv('diabetes.csv')
X = df.drop(['Outcome'],axis=1)
y = df['Outcome']
# # TASK 1-2
# In[22]:

from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score,accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np

X_train , X_test , y_train , y_test = train_test_split (X , y , random_state =0)
max_features = 3
kfold = model_selection . KFold ( n_splits =10 , shuffle = True , random_state =2020)
rf = DecisionTreeClassifier ( max_features = max_features )
num_trees = 100
model = BaggingClassifier ( base_estimator = rf , n_estimators = num_trees ,
random_state =2020)
results = model_selection . cross_val_score ( model , X_train ,y_train,cv = kfold )
print (" Accuracy : %0.2f (+/ - %0.2f)" % (results . mean () , results . std () ) )


# # Random Forest Classification
# In[23]:
kfold = model_selection . KFold ( n_splits =10 , shuffle = True , random_state =2020)
rf = DecisionTreeClassifier ()
num_trees = 100
max_features = 3
kfold = model_selection . KFold ( n_splits =10 , shuffle = True , random_state =2020)
model = RandomForestClassifier ( n_estimators = num_trees,max_features =max_features )
results = model_selection.cross_val_score ( model,X_train,y_train,cv =kfold)
print (" Accuracy : %0.2f (+/ - %0.2f)" % (results.mean(), results.std()))

# In[24]:
clf_boosting = AdaBoostClassifier (
DecisionTreeClassifier ( max_depth =1) ,
n_estimators =200)
clf_boosting.fit(X_train ,y_train )
predictions =clf_boosting .predict ( X_test )
print(" For Boosting : F1 Score {} , Accuracy {}".format(round(f1_score(y_test, predictions),2),round(accuracy_score(y_test,predictions),2)))


# # Random Forest as a Bagging classifier
# 

# In[29]:


clf_bagging = RandomForestClassifier ( n_estimators =200 , max_depth =1)
clf_bagging . fit ( X_train , y_train )
predictions = clf_bagging . predict ( X_test )
print (" For Bagging : F1 Score {} , Accuracy {}". format ( round ( f1_score ( y_test ,
predictions ) ,2) ,round ( accuracy_score ( y_test , predictions ) ,2) ) )


# # Comparison Bagging, Boosting and Stacking

# In[38]:

boosting_clf_ada_boost = AdaBoostClassifier (
DecisionTreeClassifier ( max_depth =1) ,
n_estimators =3)
bagging_clf_rf = RandomForestClassifier ( n_estimators =200 , max_depth =1,random_state =2020)
clf_rf = RandomForestClassifier ( n_estimators =200 , max_depth =1 , random_state=2020)

# In[39]:
clf_ada_boost = AdaBoostClassifier (
DecisionTreeClassifier ( max_depth =1 , random_state =2020) ,
n_estimators =3)
clf_logistic_reg = LogisticRegression ( solver ='liblinear', random_state =2020)
# Customizing and Exception message

class NumberOfClassifierException ( Exception ) :
    pass
# Creating a stacking class
class Stacking () :

    def __init__ ( self , classifiers ) :
        if(len( classifiers ) < 2) :
            raise numberOfClassifierException (" You must fit your classifier with 2 classifiers at least")
        else :
            self . _classifiers = classifiers
    def fit ( self , data_x , data_y ) :
        stacked_data_x = data_x . copy ()
        for classfier in self . _classifiers [: -1]:
            classfier.fit ( data_x , data_y )
            stacked_data_x = np . column_stack (( stacked_data_x , classfier .
            predict_proba ( data_x ) ) )
        last_classifier = self . _classifiers [ -1]
        last_classifier . fit ( stacked_data_x , data_y )
    def predict ( self , data_x ) :
        stacked_data_x = data_x.copy ()
        for classfier in self._classifiers [: -1]:
            prob_predictions = classfier . predict_proba ( data_x )
            stacked_data_x = np.column_stack (( stacked_data_x ,
            prob_predictions))
        last_classifier = self. _classifiers [ -1]
        return last_classifier. predict(stacked_data_x )
bagging_clf_rf.fit(X_train , y_train)
boosting_clf_ada_boost.fit(X_train,y_train)
classifers_list = [ clf_rf ,clf_ada_boost , clf_logistic_reg ]
clf_stacking = Stacking ( classifers_list )

clf_stacking.fit(X_train,y_train )
predictions_bagging = bagging_clf_rf . predict ( X_test )
predictions_boosting = boosting_clf_ada_boost . predict ( X_test )
predictions_stacking = clf_stacking . predict ( X_test )
print (" For Bagging : F1 Score {} , Accuracy {}". format ( round ( f1_score ( y_test ,
predictions_bagging ) ,2) ,round ( accuracy_score ( y_test , predictions_bagging )
,2) ) )
print (" For Boosting : F1 Score {} , Accuracy {}". format ( round ( f1_score ( y_test
, predictions_boosting ) ,2) ,round ( accuracy_score ( y_test ,
predictions_boosting ) ,2) ) )

# In[40]:

print (" For Stacking : F1 Score {} , Accuracy {}". format ( round ( f1_score ( y_test
, predictions_stacking ) ,2) ,round ( accuracy_score ( y_test ,
predictions_stacking ) ,2) ) )


# # TASK 3-4:

# In[41]:


from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score,accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target
X_train , X_test , y_train , y_test = train_test_split (X , y , random_state =0)
max_features = 3
kfold = model_selection . KFold ( n_splits =10 , shuffle = True , random_state =2020)
rf = DecisionTreeClassifier ( max_features = max_features )
num_trees = 100
model = BaggingClassifier ( base_estimator = rf , n_estimators = num_trees ,
random_state =2020)
results = model_selection . cross_val_score ( model , X_train ,y_train,cv = kfold )
print (" Accuracy : %0.2f (+/ - %0.2f)" % (results . mean () , results . std () ) )


# In[42]:


clf_boosting = AdaBoostClassifier (
DecisionTreeClassifier ( max_depth =1) ,
n_estimators =200)
clf_boosting.fit(X_train ,y_train )
predictions =clf_boosting .predict ( X_test )
print(" For Boosting : F1 Score {} , Accuracy {}".format(round(f1_score(y_test, predictions),2),round(accuracy_score(y_test,predictions),2)))


# In[43]:


clf_bagging = RandomForestClassifier ( n_estimators =200 , max_depth =1)
clf_bagging . fit ( X_train , y_train )
predictions = clf_bagging . predict ( X_test )
print (" For Bagging : F1 Score {} , Accuracy {}". format ( round ( f1_score ( y_test ,
predictions ) ,2) ,round ( accuracy_score ( y_test , predictions ) ,2) ) )


# # Comparison Bagging, Boosting and Stacking
# In[44]:
boosting_clf_ada_boost = AdaBoostClassifier (
DecisionTreeClassifier ( max_depth =1) ,
n_estimators =3)
bagging_clf_rf = RandomForestClassifier ( n_estimators =200 , max_depth =1,random_state =2020)
clf_rf = RandomForestClassifier ( n_estimators =200 , max_depth =1 , random_state=2020)
clf_ada_boost = AdaBoostClassifier (
DecisionTreeClassifier ( max_depth =1 , random_state =2020) ,
n_estimators =3)
clf_logistic_reg = LogisticRegression ( solver ='liblinear', random_state =2020)


# In[45]:


class NumberOfClassifierException ( Exception ) :
    pass
# Creating a stacking class
class Stacking () :

    def __init__ ( self , classifiers ) :
        if(len( classifiers ) < 2) :
            raise numberOfClassifierException (" You must fit your classifier with 2 classifiers at least")
        else :
            self . _classifiers = classifiers
    def fit ( self , data_x , data_y ) :
        stacked_data_x = data_x . copy ()
        for classfier in self . _classifiers [: -1]:
            classfier.fit ( data_x , data_y )
            stacked_data_x = np . column_stack (( stacked_data_x , classfier .
            predict_proba ( data_x ) ) )
        last_classifier = self . _classifiers [ -1]
        last_classifier . fit ( stacked_data_x , data_y )
    def predict ( self , data_x ) :
        stacked_data_x = data_x.copy ()
        for classfier in self._classifiers [: -1]:
            prob_predictions = classfier . predict_proba ( data_x )
            stacked_data_x = np.column_stack (( stacked_data_x ,
            prob_predictions))
        last_classifier = self. _classifiers [ -1]
        return last_classifier. predict(stacked_data_x )
    
bagging_clf_rf.fit(X_train , y_train)
boosting_clf_ada_boost.fit(X_train,y_train)
classifers_list = [ clf_rf ,clf_ada_boost , clf_logistic_reg ]
clf_stacking = Stacking ( classifers_list )
clf_stacking.fit(X_train,y_train )
predictions_bagging = bagging_clf_rf . predict ( X_test )
predictions_boosting = boosting_clf_ada_boost . predict ( X_test )
predictions_stacking = clf_stacking . predict ( X_test )
print (" For Bagging : F1 Score {} , Accuracy {}". format ( round ( f1_score ( y_test ,
predictions_bagging ) ,2) ,round ( accuracy_score ( y_test , predictions_bagging )
,2) ) )
print (" For Boosting : F1 Score {} , Accuracy {}". format ( round ( f1_score ( y_test
, predictions_boosting ) ,2) ,round ( accuracy_score ( y_test ,
predictions_boosting ) ,2) ) )

# In[46]:
print (" For Stacking : F1 Score {} , Accuracy {}". format ( round ( f1_score ( y_test
, predictions_stacking ) ,2) ,round ( accuracy_score ( y_test ,
predictions_stacking ) ,2) ) )


# # TASK 5-6

# In[48]:


from sklearn import preprocessing
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
train_data.head(10)
test_ids  = test_data["PassengerId"]

#Clean data
def clean(data):
    data = data.drop(["Ticket","Cabin","PassengerId","Name"], axis = 1)
    col = ["SibSp","Parch","Fare","Age"]
    for i in col:
        data[i].fillna(data[i].median(), inplace = True)
    data.Embarked.fillna("U", inplace = True) #Fill those empty data points with an unknown token
    return data

train_data = clean(train_data)
test_data = clean(test_data)

label_encod = preprocessing.LabelEncoder()

cols = ["Sex","Embarked"]
for col in cols:
    train_data[col] = label_encod.fit_transform(train_data[col])
    test_data[col] = label_encod.transform(test_data[col])
y_pred = train_data["Survived"]
x_pred = train_data.drop("Survived", axis=1)

X_train, X_test, y_train, y_test = train_test_split(x_pred, y_pred, test_size  = 0.2, random_state = 42)
kfold = model_selection . KFold ( n_splits =10 , shuffle = True , random_state =2020)
rf = DecisionTreeClassifier ( max_features = max_features )
num_trees = 100
model = BaggingClassifier ( base_estimator = rf , n_estimators = num_trees ,
random_state =2020)
results = model_selection . cross_val_score ( model , X_train ,y_train,cv = kfold )
print (" Accuracy : %0.2f (+/ - %0.2f)" % (results . mean () , results . std () ) )


# In[49]:


clf_boosting = AdaBoostClassifier (
DecisionTreeClassifier ( max_depth =1) ,
n_estimators =200)
clf_boosting.fit(X_train ,y_train )
predictions =clf_boosting .predict ( X_test )
print(" For Boosting : F1 Score {} , Accuracy {}".format(round(f1_score(y_test, predictions),2),round(accuracy_score(y_test,predictions),2)))


# In[50]:


clf_bagging = RandomForestClassifier ( n_estimators =200 , max_depth =1)
clf_bagging . fit ( X_train , y_train )
predictions = clf_bagging . predict ( X_test )
print (" For Bagging : F1 Score {} , Accuracy {}". format ( round ( f1_score ( y_test ,
predictions ) ,2) ,round ( accuracy_score ( y_test , predictions ) ,2) ) )


# # Comparison Bagging, Boosting and Stacking
# In[51]:
boosting_clf_ada_boost = AdaBoostClassifier (
DecisionTreeClassifier ( max_depth =1) ,
n_estimators =3)
bagging_clf_rf = RandomForestClassifier ( n_estimators =200 , max_depth =1,random_state =2020)
clf_rf = RandomForestClassifier ( n_estimators =200 , max_depth =1 , random_state=2020)
clf_ada_boost = AdaBoostClassifier (
DecisionTreeClassifier ( max_depth =1 , random_state =2020) ,
n_estimators =3)
clf_logistic_reg = LogisticRegression ( solver ='liblinear', random_state =2020)


# In[52]:


class NumberOfClassifierException ( Exception ) :
    pass
# Creating a stacking class
class Stacking () :

    def __init__ ( self , classifiers ) :
        if(len( classifiers ) < 2) :
            raise numberOfClassifierException (" You must fit your classifier with 2 classifiers at least")
        else :
            self . _classifiers = classifiers
    def fit ( self , data_x , data_y ) :
        stacked_data_x = data_x . copy ()
        for classfier in self . _classifiers [: -1]:
            classfier.fit ( data_x , data_y )
            stacked_data_x = np . column_stack (( stacked_data_x , classfier .
            predict_proba ( data_x ) ) )
        last_classifier = self . _classifiers [ -1]
        last_classifier . fit ( stacked_data_x , data_y )
    def predict ( self , data_x ) :
        stacked_data_x = data_x.copy ()
        for classfier in self._classifiers [: -1]:
            prob_predictions = classfier . predict_proba ( data_x )
            stacked_data_x = np.column_stack (( stacked_data_x ,
            prob_predictions))
        last_classifier = self. _classifiers [ -1]
        return last_classifier. predict(stacked_data_x )
    
bagging_clf_rf.fit(X_train , y_train)
boosting_clf_ada_boost.fit(X_train,y_train)
classifers_list = [ clf_rf ,clf_ada_boost , clf_logistic_reg ]
clf_stacking = Stacking ( classifers_list )
clf_stacking.fit(X_train,y_train )
predictions_bagging = bagging_clf_rf . predict ( X_test )
predictions_boosting = boosting_clf_ada_boost . predict ( X_test )
predictions_stacking = clf_stacking . predict ( X_test )
print (" For Bagging : F1 Score {} , Accuracy {}". format ( round ( f1_score ( y_test ,
predictions_bagging ) ,2) ,round ( accuracy_score ( y_test , predictions_bagging )
,2) ) )
print (" For Boosting : F1 Score {} , Accuracy {}". format ( round ( f1_score ( y_test
, predictions_boosting ) ,2) ,round ( accuracy_score ( y_test ,
predictions_boosting ) ,2) ) )

# In[53]:
print (" For Stacking : F1 Score {} , Accuracy {}". format ( round ( f1_score ( y_test
, predictions_stacking ) ,2) ,round ( accuracy_score ( y_test ,
predictions_stacking ) ,2) ) )
