import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split
dataset = pd.read_csv('california_housing_train.csv')
X = dataset.drop('median_house_value', axis=1)
Y = dataset['median_house_value']

decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
X_train , X_test , y_train , y_test = train_test_split (X , Y , random_state =0)
decision_tree.fit(X_train , y_train)
decision_tree = decision_tree.fit(X_train, y_train)
from sklearn import tree
plt.figure(figsize=(20,20))
tree.plot_tree(decision_tree)
plt.show ()


# In[18]:


n_nodes = decision_tree.tree_.node_count
children_left = decision_tree.tree_.children_left
children_right = decision_tree.tree_.children_right
feature = decision_tree.tree_.feature
threshold = decision_tree.tree_.threshold
node_depth = np.zeros ( shape = n_nodes , dtype = np . int64 )
is_leaves = np.zeros ( shape = n_nodes , dtype = bool )
stack = [(0 , 0)] # start with the root node id (0) and its depth (0)
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
print (" The binary tree structure has {n} nodes and has the following tree structure".format ( n = n_nodes ))
for i in range ( n_nodes):
    if is_leaves [ i ]:
        print ("{} node ={} is a leaf node .". format (node_depth [ i ] * "\t",i))
    else :
        print ("{} node ={} is a split node : go to node {} if X[: , {}] <= {} else to node {}.".format(node_depth [ i ] * "\t",i ,children_left [ i ] ,feature [ i ] ,threshold [ i ] ,children_right [ i ]))
node_indicator = decision_tree . decision_path ( X_test )
leaf_id = decision_tree . apply ( X_test )
sample_id = 0
# obtain ids of the nodes ‘sample_id ‘ goes through , i.e. , row ‘sample_id ‘
node_index = node_indicator . indices[node_indicator . indptr [ sample_id ] : node_indicator . indptr [ sample_id + 1]]


# In[88]:


sample_ids = [0 , 1]
# boolean array indicating the nodes both samples go through
common_nodes = node_indicator . toarray () [ sample_ids ]. sum (axis =0) == len(sample_ids )
# obtain node ids using position in array
common_node_id = np . arange ( n_nodes ) [ common_nodes ]
print ("\nThe following samples {samples} share the node (s) {nodes} in the tree.".format (samples = sample_ids , nodes = common_node_id))
print ("This is {prop}% of all nodes .". format ( prop =100 * len ( common_node_id )/n_nodes ) )



# In[87]:


