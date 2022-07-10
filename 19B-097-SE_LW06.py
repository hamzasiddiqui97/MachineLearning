import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


# In[1]:


#names = ["ID","sepalLength","sepalWidth","petalLength","petalWidth","species"]
data = load_iris()

from sklearn.model_selection import train_test_split 
X = data.data
y = data.target
X_train , X_test , y_train , y_test = train_test_split (X , y , random_state =0)


# In[77]:


from sklearn . tree import DecisionTreeClassifier
clf = DecisionTreeClassifier ( criterion ="gini", max_depth =4 , random_state =1)
clf . fit ( X_train , y_train )


# # TREE STRUCTURE

# In[19]:
n_nodes = clf . tree_ . node_count
children_left = clf . tree_ . children_left
children_right = clf . tree_ . children_right
feature = clf . tree_ . feature
threshold = clf . tree_ . threshold

node_depth = np . zeros ( shape = n_nodes , dtype = np . int64 )
is_leaves = np . zeros ( shape = n_nodes , dtype = bool )
stack = [(0 , 0) ] # start with the root node id (0) and its depth (0)
while len( stack ) > 0:
    # ‘pop ‘ ensures each node is only visited once
    node_id , depth = stack . pop ()
    node_depth [ node_id ] = depth
    # If the left and right child of a node is not the same we have a split
    # node
    is_split_node = children_left [ node_id ] != children_right [ node_id ]
    # If a split node , append left and right children and depth to ‘stack ‘
    # so we can loop through them
    if is_split_node :
        stack . append (( children_left [ node_id ] , depth + 1) )
        stack . append (( children_right [ node_id ] , depth + 1) )
    else :
        is_leaves [ node_id ] = True
print (
    " The binary tree structure has {n} nodes and has "
    " the following tree structure :\n". format ( n = n_nodes )
)




# In[18]:


for i in range ( n_nodes ) :
    if is_leaves [ i ]:
        print(
            "{space} node ={node} is a leaf node .". format (
                space = node_depth [ i ] * "\t", node = i))
    else :
        print(
            "{space} node ={node} is a split node : "
            "go to node {left} if X[: , {feature}] <= {threshold} "
            " else to node {right}.". format (
                space = node_depth [i] * "\t",
                node =i,
                left = children_left [i],
                feature = feature [i],
                threshold = threshold [i],
                right = children_right [i]))


# In[12]:


from sklearn import tree
plt.figure(figsize=(10,10))
tree . plot_tree ( clf )
plt . show ()


# # DECISION PATH

# In[13]:


node_indicator = clf . decision_path ( X_test )
leaf_id = clf . apply ( X_test )
sample_id = 0
# obtain ids of the nodes ‘sample_id ‘ goes through , i.e. , row ‘sample_id ‘
node_index = node_indicator . indices [
    node_indicator . indptr [ sample_id ] : node_indicator . indptr [ sample_id + 1]
]
print (" Rules used to predict sample {id}:\n". format (id= sample_id ) )
for node_id in node_index :
    # continue to the next node if it is a leaf node
    if leaf_id [ sample_id ] == node_id :
        continue
    # check if value of the split feature for sample 0 is below threshold
    if X_test [ sample_id , feature [ node_id ]] <= threshold [ node_id ]:
        threshold_sign = "<="
    else :
        threshold_sign = ">"
    print (
        " decision node {node} : ( X_test [{sample} , {feature}] = {value}) "
        "{inequality} {threshold})". format (
        node = node_id ,
        sample = sample_id ,
        feature = feature [ node_id ] ,
        value = X_test [ sample_id , feature [ node_id ]] ,
        inequality = threshold_sign ,
        threshold = threshold [ node_id ] ,))

# # ID3 ALGORITHM:

# In[52]:
class ID3 :
    def __init__ (self , data , t_label ) :
        self . data = data
        self . t_label = t_label
    def getFeatureEntropy ( self , data = None , t_label = None ) :
        if data == None:
            data = self.data
        if t_label == None :
            t_label = self . t_label
        target = data [ t_label ]
        class_list = target.unique()
        total_row = data.shape [0] # the total size of the dataset
        total_entr = 0
        # print (" Total rows " , total_row )
        for c in class_list : # for each class in the label
            total_class_count = data [ data [ t_label ] == c ]. shape [0] # number of the class
        # print ( ’ Row with "{}" class {} ’. format (c, total_class_count ))
        # print ( ’ Entropy of class "{0}" is {1}/{2} * log2 ({1}/{2}) ’.format (c, total_class_count , total_row ))
            total_class_entr = - ( total_class_count / total_row ) * np . log2 (total_class_count / total_row ) # entropy of the class
            total_entr += total_class_entr # adding the class entropy to the total entropy of the dataset
        return total_entr
    
    def get_rem_by_entropy(self) :
        desc_features = pd . DataFrame ()
        target_feature = pd . DataFrame ()
        feature = pd . DataFrame ()
        desc_features = self . data . drop ([ self . t_label ] , axis =1)
        target_feature = self . data [ self . t_label ]
        target_list = list ()
        target_list = target_feature . unique ()
        class_count = desc_features . shape [0]
        rem_list = list ()
        entropy = 0
        class_list = list ()
        feature_list = list ()
        feature_list = desc_features . columns
        for item in feature_list :
            # print ( ’ fetaure : ’ , item )
            rem_feature_entropy =0
            class_list = desc_features [ item ]. unique ()
            new_feature = desc_features [ item ]
            for level in class_list :
                label_class_count = desc_features [desc_features [item] == level].shape [0]
                entropy_class = 0
                feature_level_entropy =0
                sum_feature_entropy =0
                # print ( ’ level "{0}" of feature "{1}" total count {2} ’. format(level ,item , label_class_count ))
                if label_class_count != 0:
                    probability_class = label_class_count / class_count #probability of the class
                    # print ( ’ Probability value of {0}/{1} is {2:.4 f} ’. format (label_class_count , class_count , probability_class ))
                    for tvalue in target_list :
                        count_level_frequency =0
                        for i in range ( class_count ):
                            if ( new_feature [ i ] == level ) and ( target_feature[ i ] == tvalue ):
                                count_level_frequency +=1
                        if count_level_frequency !=0:
                            feature_prob = count_level_frequency / label_class_count
                            feature_level_entropy = - ( feature_prob * np .log2 ( feature_prob ) )
                            # feature level entropy
                            # print ( ’ pribability {0}/{1} is {2:.4 f} of target
                            #value {3} ’. format ( count_level_frequency , label_class_count ,
                            #feature_level_entropy , tvalue ))
                            sum_feature_entropy += feature_level_entropy
                    ProbXfeature_entropy = probability_class * sum_feature_entropy
                rem_feature_entropy += ProbXfeature_entropy
                # print ( ’ Feature {0} entropy is: {1} ’. format (item ,rem_feature_entropy ))
            rem_list . append (rem_feature_entropy)
        return rem_list
    
    def getInfoGain_by_entropy ( self ) :
        IG_list = list ()
        target_entropy = self . getFeatureEntropy ()
        rem = self . get_rem_by_entropy ()
        for i in range (len( rem ) ) :
            IG_list . append ( target_entropy - rem [ i ])
        return IG_list
    
    def getGR_by_entropy ( self ) :
        data = self . data
        target_label = self . t_label
        feature_list = list ()
        GR_list = list ()
        count =0
        desc_features = data . drop ([ target_label ] , axis =1)
        feature_list = desc_features . columns
        IG = self .getInfoGain_by_entropy()
        for item in feature_list :
            feat_entropy = self . getFeatureEntropy ( None , item )
            # print ( feat_entropy )
            tt = IG [ count ]/ feat_entropy
            GR_list . append ( tt )
            count +=1
        return GR_list


# In[73]:


dataset = pd.read_csv('vegetation.csv')
id3 = ID3(dataset, 'Vegetation')
print('Entropy: ', id3.getFeatureEntropy(None,None))
print('Remainder Values by entropy method:',id3.get_rem_by_entropy())
print('Information Gain (IG) of each feature:',id3.getInfoGain_by_entropy())
print('Gain Ratio (GR) of rach feature:',id3.getGR_by_entropy())


# # LAB TASKS:

# # GINI METHOD:

# In[28]:


class GINI:
    def __init__ (self , data , t_label ) :
        self . data = data
        self . t_label = t_label
    def getFeatureEntropy_by_gini ( self,data = None ,t_label = None):
        if data == None:
            data = self.data
        if t_label == None :
            t_label = self . t_label
        target = data [ t_label ]
        class_list = target.unique()
        total_row = data.shape [0] # the total size of the dataset
        total_entr = 0
        # print (" Total rows " , total_row )
        for c in class_list : # for each class in the label
            total_class_count = data [ data [ t_label ] == c ]. shape [0] # number of the class
        # print ( ’ Row with "{}" class {} ’. format (c, total_class_count ))
        # print ( ’ Entropy of class "{0}" is {1}/{2} * log2 ({1}/{2}) ’.format (c, total_class_count , total_row ))
            total_class_entr =  ( total_class_count / total_row )**2 # entropy of the class
            total_entr += total_class_entr # adding the class entropy to the total entropy of the dataset
        return 1-total_entr
        
    def get_rem_by_entropy_gini(self) :
        desc_features = pd . DataFrame ()
        target_feature = pd . DataFrame ()
        feature = pd . DataFrame ()
        desc_features = self . data . drop ([ self . t_label ] , axis =1)
        target_feature = self . data [ self . t_label ]
        target_list = list ()
        target_list = target_feature . unique ()
        class_count = desc_features . shape [0]
        rem_list = list ()
        entropy = 0
        class_list = list ()
        feature_list = list ()
        feature_list = desc_features . columns
        for item in feature_list :
            # print ( ’ fetaure : ’ , item )
            rem_feature_entropy =0
            class_list = desc_features [ item ]. unique ()
            new_feature = desc_features [ item ]
            for level in class_list :
                label_class_count = desc_features [desc_features [item] == level].shape [0]
                entropy_class = 0
                feature_level_entropy =0
                sum_feature_entropy =0
                if label_class_count != 0:
                    probability_class = label_class_count / class_count #probability of the class
                    for tvalue in target_list :
                        count_level_frequency =0
                        for i in range ( class_count ):
                            if ( new_feature [ i ] == level ) and ( target_feature[ i ] == tvalue ):
                                count_level_frequency +=1
                        if count_level_frequency !=0:
                            feature_prob = count_level_frequency / label_class_count
                            feature_level_entropy = ( feature_prob )**2
                            #feature_level_entropy =( total_class_count / total_row )**2
                            sum_feature_entropy += feature_level_entropy
                    ProbXfeature_entropy = probability_class * (1-sum_feature_entropy)
                rem_feature_entropy += ProbXfeature_entropy
            rem_list . append (rem_feature_entropy)
        return rem_list
    
    def getInfoGain_by_gini ( self ) :
        IG_list = list ()
        target_entropy = self .getFeatureEntropy_by_gini ()
        rem = self.get_rem_by_entropy_gini()
        for i in range (len( rem ) ) :
            IG_list . append ( target_entropy - rem [ i ])
        return IG_list
    
    def getGR_by_gini ( self ) :
        data = self . data
        target_label = self . t_label
        feature_list = list ()
        GR_list = list ()
        count =0
        desc_features = data . drop ([ target_label ] , axis =1)
        feature_list = desc_features . columns
        IG = self .getInfoGain_by_gini()
        for item in feature_list :
            feat_entropy = self .getFeatureEntropy_by_gini ( None , item )
            # print ( feat_entropy )
            tt = IG [ count ]/ feat_entropy
            GR_list . append ( tt )
            count +=1
        return GR_list


dataset = pd.read_csv('vegetation.csv')
Gini = GINI(dataset, 'Vegetation')
print("Gini Index:",Gini.getFeatureEntropy_by_gini())
print("------------------------------------------------------------------")
print('Remainder Values by gini index method:',Gini.get_rem_by_entropy_gini())
print('Information Gain (IG) of each feature by gini:',Gini.getInfoGain_by_gini())
print('Gain Ratio (GR) of rach feature by gini:',Gini.getGR_by_gini())

# In[84]:


