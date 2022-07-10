#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn . neighbors import KNeighborsClassifier
X , Y = load_iris(return_X_y=True)
X_train , X_test , y_train , y_test = train_test_split (X , Y , test_size = 0.2,random_state =32)
sc = StandardScaler ()
sc . fit ( X_train )
X_train = sc . transform ( X_train )
sc . fit ( X_test )
X_test = sc . transform ( X_test )
X . shape
# Step 2
error1 = []
error2 = []
for k in range (1 ,15) :
    knn = KNeighborsClassifier ( n_neighbors = k )
    knn . fit ( X_train , y_train )
    y_pred1 = knn . predict ( X_train )
    error1 . append ( np . mean ( y_train != y_pred1 ) )
    y_pred2 = knn . predict ( X_test )
    error2 . append ( np . mean ( y_test != y_pred2 ) )
# plt . figure ( figsize (10 ,5))
plt . plot ( range (1 ,15) , error1 , label ="train")
plt . plot ( range (1 ,15) , error2 , label ="test")
plt . xlabel ('k Value ')
plt . ylabel ('Error')
plt . legend ()


# In[3]:


from sklearn import metrics
knn = KNeighborsClassifier ( n_neighbors =7)
knn . fit ( X_train , y_train )
y_pred = knn . predict ( X_test )
metrics . accuracy_score ( y_test , y_pred )


# In[4]:


df = pd.read_csv('Surgical-deepnet.csv')
y_pred=np.array([0,0,1,1,0,0,0,1,0,0])
y_true = df[['baseline_cvd','baseline_dementia','baseline_diabetes','baseline_digestive','baseline_osteoart','baseline_psych','baseline_pulmonary','gender','mort30','complication']]


# In[5]:


from sklearn . metrics import jaccard_score , confusion_matrix
for i in range (10) :
    print ( jaccard_score(y_true.iloc[i,:],y_pred))
    print ( confusion_matrix(y_true.iloc[10,:],y_pred,labels =[0 ,1]) )


# # Cosine

# In[12]:


doc_trump = "Mr. Trump became president after winning the political election.Though he lost the support of some republican friends , Trump isfriends with President Putin"
doc_election = " President Trump says Putin had no political interference isthe election outcome.He says it was a witchhunt by political parties.He claimed President Putin is a friend who had nothing to do with theelection"
doc_putin = " Post elections , Vladimir Putin became President of Russia.President Putin had served as the Prime Minister earlier in hispolitical career "
documents = [ doc_trump , doc_election , doc_putin ]


# In[13]:


from sklearn . feature_extraction . text import CountVectorizer
from sklearn . metrics . pairwise import cosine_similarity
import pandas as pd
# Create the Document Term Matrix
count_vectorizer = CountVectorizer(stop_words='english')
count_vectorizer = CountVectorizer ()
sparse_matrix = count_vectorizer . fit_transform ( documents )
# OPTIONAL : Convert Sparse Matrix to Pandas Dataframe if you want to see the word frequencies .
doc_term_matrix = sparse_matrix.todense ()
count_vectorizer.get_feature_names
df = pd.DataFrame ( doc_term_matrix ,
columns = count_vectorizer.get_feature_names(),index =['doc_trump','doc_election','doc_putin'])
print(cosine_similarity ( df , df ) )


# # Mahalanobis Distance

# In[8]:


import pandas as pd
from scipy import linalg
import numpy as np
data = pd . read_csv ('diamonds.csv')
df = pd . DataFrame ( data ).iloc [: , [0 ,4 ,6]]
def mahalanobis( x = None,data = None , cov = None ) :
    x_minus_mu= x-np.mean ( data , axis =0)
    if not cov:
        cov = np . cov ( data .values . T )
        inv_covmat = linalg.inv ( cov )
        left_term = np . dot ( x_minus_mu , inv_covmat )
        mahal = np . dot ( left_term , x_minus_mu . T )
    return mahal . diagonal ()
df_x = df [[ 'carat', 'depth', 'price']]. head (500)
df_x ['mahala'] = mahalanobis ( x = df_x , data = df [['carat', 'depth', 'price']])
df_x . head ()


# # Multivariate outlier detection using Mahalanobis distance

# In[9]:


from scipy . stats import chi2
chi2 . ppf ((1 -0.01) , df =2)
# Compute the P- Values
df_x ['p_value'] = 1 - chi2 . cdf ( df_x ['mahala'] , 2)
# Extreme values with a significance level of 0.01
df_x .loc [df_x . p_value < 0.01]. head (10)


# # KD Tree

# In[11]:


import numpy as np
from sklearn.neighbors import KDTree
data = pd .read_csv ('Agility_Speed.csv')
df =pd.DataFrame ( data )
x =df['Speed']
y =df['Agility']
X = df[['Speed','Agility']]
tree= KDTree(X,leaf_size=2)
dist,ind = tree . query ([[6 ,3.5]] , k =2)
print(ind)
print(dist)
plt. scatter(x,y , color ='red', marker ='o')
plt.scatter(6,3.5 , color ='blue',marker ='v')
plt.show()


# # LAB TASK

# In[ ]:


#TASK NO 1:


# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn . neighbors import KNeighborsClassifier
label_encoder = LabelEncoder()
df = pd.read_csv('heart.csv')
df['Sex']= label_encoder.fit_transform(df['Sex'])
df['ChestPainType']= label_encoder.fit_transform(df['ChestPainType'])
df['RestingECG']= label_encoder.fit_transform(df['RestingECG'])
df['ExerciseAngina']= label_encoder.fit_transform(df['ExerciseAngina'])
df['ST_Slope']= label_encoder.fit_transform(df['ST_Slope'])
X = df.drop('HeartDisease',axis=1)
Y = df['HeartDisease']
X_train , X_test , y_train , y_test = train_test_split (X , Y , test_size = 0.2,random_state =32)
sc = StandardScaler ()
sc . fit ( X_train )
X_train = sc . transform ( X_train )
sc . fit ( X_test )
X_test = sc . transform ( X_test )
X . shape
# Step 2
error1 = []
error2 = []
for k in range (1 ,15) :
    knn = KNeighborsClassifier ( n_neighbors = k )
    knn . fit ( X_train , y_train )
    y_pred1 = knn . predict ( X_train )
    error1 . append ( np . mean ( y_train != y_pred1 ) )
    y_pred2 = knn . predict ( X_test )
    error2 . append ( np . mean ( y_test != y_pred2 ) )
# plt . figure ( figsize (10 ,5))
plt . plot ( range (1 ,15) , error1 , label ="train")
plt . plot ( range (1 ,15) , error2 , label ="test")
plt . xlabel ('k Value ')
plt . ylabel ('Error')
plt . legend ()


# In[ ]:


#TASK NO 2:


# In[3]:


import numpy as np
from sklearn . feature_extraction . text import CountVectorizer
from sklearn . metrics . pairwise import cosine_similarity
import pandas as pd
df = pd.read_csv('IMDBdata_MainData.csv')

features = ['Genre', 'Plot', 'Language']
for feature in features:
    df[feature] = df[feature].fillna('')

def combined_features(row):
    return row['Genre']+" "+row['Plot']+" "+row['Language']
df["combined_features"] = df.apply(combined_features, axis =1)

cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])
count_matrix.toarray()
cosine_sim = cosine_similarity(count_matrix)
print(cosine_sim)


# In[1]:


#TASK NO 3:


# In[1]:


import pandas as pd
from scipy import linalg
import numpy as np
data = pd . read_csv ('diabetes.csv')
df = pd . DataFrame ( data )
def mahalanobis( x = None,data = None , cov = None ) :
    x_minus_mu= x-np.mean ( data , axis =0)
    if not cov:
        cov = np . cov ( data .values . T )
        inv_covmat = linalg.inv ( cov )
        left_term = np . dot ( x_minus_mu , inv_covmat )
        mahal = np . dot ( left_term , x_minus_mu . T )
    return mahal . diagonal ()
df_x = df [['Pregnancies','Glucose', 'BloodPressure','SkinThickness','Insulin', 'BMI','DiabetesPedigreeFunction','Age','Outcome']]. head (500)
df_x ['mahala'] = mahalanobis ( x = df_x , data = df [['Pregnancies','Glucose', 'BloodPressure','SkinThickness','Insulin', 'BMI','DiabetesPedigreeFunction','Age','Outcome']])
df_x.head()


# In[2]:


from scipy . stats import chi2
chi2 . ppf ((1 -0.05) , df =2)
df_x ['p_value'] = 1 - chi2 . cdf ( df_x ['mahala'] , 2)
df_x .loc [df_x . p_value < 0.05]. head (10)


# In[ ]:


#TASK NO 4:


# In[14]:


from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
data = pd .read_csv ('Agility_Speed.csv')
df =pd.DataFrame ( data )
x =df['Speed']
y =df['Agility']
X = list (zip(x , y ) )
tree= KDTree(X,leaf_size=2)
dist,ind = tree . query ([[6 ,3.5]] , k =2)
print(ind)
print(dist)
plt. scatter(x,y , color ='red', marker ='o')
plt.scatter(6,3.5 , color ='blue',marker ='v')
plt.show()


# In[ ]:


#TASK NO 5:


# In[2]:


import numpy as np
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler


class KDTree:

    # class initialization function
    def __init__(self, data, mins, maxs):
        self.data = np.asarray(data)

        # data should be two-dimensional
        assert self.data.shape[1] == 2

        if mins is None:
            mins = data.min(0)
        if maxs is None:
            maxs = data.max(0)

        self.mins = np.asarray(mins)
        self.maxs = np.asarray(maxs)
        self.sizes = self.maxs - self.mins

        self.child1 = None
        self.child2 = None

        if len(data) > 1:
            # sort on the dimension with the largest spread
            largest_dim = np.argmax(self.sizes)
            i_sort = np.argsort(self.data[:, largest_dim])
            self.data[:] = self.data[i_sort, :]

            # find split point
            N = self.data.shape[0]
            half_N = int(N / 2)
            split_point = 0.5 * (self.data[half_N, largest_dim]
                                 + self.data[half_N - 1, largest_dim])

            # create subnodes
            mins1 = self.mins.copy()
            mins1[largest_dim] = split_point
            maxs2 = self.maxs.copy()
            maxs2[largest_dim] = split_point

            # Recursively build a KD-tree on each sub-node
            self.child1 = KDTree(self.data[half_N:], mins1, self.maxs)
            self.child2 = KDTree(self.data[:half_N], self.mins, maxs2)

    def draw_rectangle(self, ax, depth=None):
       
        if depth == 0:
            rect = plt.Rectangle(self.mins, *self.sizes, ec='k', fc='none')
            ax.add_patch(rect)

        if self.child1 is not None:
            if depth is None:
                self.child1.draw_rectangle(ax)
                self.child2.draw_rectangle(ax)
            elif depth > 0:
                self.child1.draw_rectangle(ax, depth - 1)
                self.child2.draw_rectangle(ax, depth - 1)


np.random.seed(0)

X = pd.DataFrame(df[['Speed','Agility']]).to_numpy()
scaler = StandardScaler()
X = scaler.fit_transform(X)
X[:, 1] *= 0.1
X[:, 1] += X[:, 0] ** 2


KDT = KDTree(X, [-1.1, -0.1], [1.1, 1.1])

fig = plt.figure(figsize=(10, 10))
fig.subplots_adjust(wspace=0.1, hspace=0.15,
                    left=0.1, right=0.9,
                    bottom=0.05, top=0.9)

for level in range(1, 5):
    ax = fig.add_subplot(2, 2, level, xticks=[], yticks=[])
    ax.scatter(X[:, 0], X[:, 1], s=9)
    KDT.draw_rectangle(ax, depth=level - 1)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.15, 1.15)
    ax.set_title('level %i' % level)

# suptitle() adds a title to the entire figure
fig.suptitle('KD-tree')
plt.show()


# In[ ]:




