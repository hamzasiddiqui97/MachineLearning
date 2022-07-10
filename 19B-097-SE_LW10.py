#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes Categorical data

# In[1]:


class NBClassifier :
    def __init__ ( self , data , target ) :
        self . data = data
        self . target = target
        self.prob_all= list()
        self.tprob_dict = dict()
        
        
    def target_prob ( self ) :
        class_list = self.data.loc[:,self.target].unique()
        total_rows = self.data.shape[0]
        
        for c in class_list : # for each class in the label
            total_class_count = self . data [ self . data [ self . target ] == c ].shape [0] # number of the class
            self . tprob_dict [ c ]= total_class_count / total_rows
        return self . tprob_dict


    def cond_prob(self ,feature ) :
        feature_class_list=self.data.loc[:,feature].unique()
        target_class_list= self.data.loc[:,self.target].unique()
        total_rows=self.data.shape[0]
        cond_prob_list=list()
        counter =0
        for t_level in target_class_list :
            t_level_count = self . data [ self . data [ self . target ] == t_level ].shape [0]

            for f_level in feature_class_list :
                counter =0
                for i in range ( total_rows ) :
                    if ( self . data . loc [i , feature ] == f_level ) and ( self . data .loc [i , self . target ] == t_level ):
                        counter +=1
                cond_prob_list . append (( feature , f_level , t_level , round ( counter/ t_level_count ,3) ) )
        return cond_prob_list
    

    def cond_prob_LPSmoother ( self , feature , k ) :
        feature_class_list = self.data.loc[:,feature].unique()
        target_class_list = self.data.loc[:,self.target].unique()
        total_rows = self.data.shape[0]
        counter = 0
        cond_prob_list = list()
        f_level_count =len (feature_class_list)
        for t_level in target_class_list :
            t_level_count = self . data [ self . data [ self . target ] == t_level ].shape [0]
            for f_level in feature_class_list :
                counter =0
                for i in range ( total_rows ) :
                    if ( self . data . loc [i , feature ] == f_level ) and ( self . data .loc [i , self . target ] == t_level ) :
                        counter +=1
                cp_smoother = ( counter + k ) /( t_level_count +( k * f_level_count ))
                cond_prob_list . append (( feature , f_level , t_level , round (cp_smoother ,3) ) )
        return cond_prob_list
    

    def count_all_prob ( self , k = None ) :
        X=self.data.drop([self.target],axis=1)
        if k!=None:
            for feature in X.columns :
                cond_list = self . cond_prob_LPSmoother(feature,k)
                self.prob_all = self . prob_all + cond_list
        else:
            for feature in X.columns :
                cond_list = self . cond_prob(feature)
                self.prob_all = self . prob_all + cond_list
            
        return self.prob_all
    def fit (self,smoothing='LPS',k=None) :
        self.target_prob()
        if smoothing == 'LPS' and k!=None:
            self . count_all_prob (k)
        else:
            self . count_all_prob ()
        return None

    def pred_score ( self , query ) :
        num_of_arg=self.data.shape[1]-1
        query_length=len(query)
        prob_dict=dict()
        prob_prod=1
        feature_list=self.data.columns
        if query_length != num_of_arg :
            print ('Query required ', num_of_arg ,' number of arguments')
            print ('Data set column ', self.data.columns )
        else :
            i =0
            for item in query :
                if item not in self . data . iloc [: , i ]. unique () :
                    print ( item ,'is not in coulmn ', self . data . columns [ i ])
                i +=1
            for t_level in self . data [ self . target ]. unique () :
                for cnt in range ( query_length ) :
                    for item in self . prob_all :
                        if item [0]== feature_list [cnt] and item [1]== query [cnt] and item [2]== t_level :
                            prob_prod = prob_prod * item [3]
                prob_dict [ t_level ]= round ( prob_prod * self . tprob_dict [ t_level] ,4)
                prob_prod = 1
        return prob_dict
    
    def pred ( self , query ) :
        pr_dict = self . pred_score ( query )
        return max( pr_dict , key = pr_dict . get )
    
    def get_prob_dict ( self ) :
        print ('Target probabilities : ', self . tprob_dict )
        print ('Conditional features Probabilities : ', self . prob_all )
        return None
    
    
    


# In[2]:


import pandas as pd
data = pd . read_csv ('LoanApplication_cat.csv', sep =',')
df = pd . DataFrame ( data )
df = df.drop(['ID'],axis=1)
nbc = NBClassifier ( df ,'Fraud')
nbc . fit ()
query = ['paid','none','rent']
nbc . pred_score ( query )


# In[3]:


nbc . get_prob_dict ()


# In[4]:


nbc = NBClassifier ( df ,'Fraud')
nbc . fit ( smoothing ='LPS',k =3)
query = ['paid','guarantor','free']
nbc . pred_score ( query )


# In[5]:


nbc.pred(query)


# In[6]:


nbc.get_prob_dict()


# # CONTINUOUS DATA

# In[7]:


class NBClassifier :
    def __init__ ( self , data , target,bins ) :
        self . data = data
        self . target = target
        self.prob_all= list()
        self.tprob_dict = dict()
        self.bins = bins
        
        
    def target_prob ( self ) :
        class_list = self.data.loc[:,self.target].unique()
        total_rows = self.data.shape[0]
        
        for c in class_list : # for each class in the label
            total_class_count = self . data [ self . data [ self . target ] == c ].shape [0] # number of the class
            self . tprob_dict [ c ]= total_class_count / total_rows
        return self . tprob_dict


    def cond_prob(self ,feature ) :
        feature_class_list=self.data.loc[:,feature].unique()
        target_class_list= self.data.loc[:,self.target].unique()
        total_rows=self.data.shape[0]
        cond_prob_list=list()
        counter =0
        for t_level in target_class_list :
            t_level_count = self . data [ self . data [ self . target ] == t_level ].shape [0]

            for f_level in feature_class_list :
                counter =0
                for i in range ( total_rows ) :
                    if ( self . data . loc [i , feature ] == f_level ) and ( self . data .loc [i , self . target ] == t_level ):
                        counter +=1
                cond_prob_list . append (( feature , f_level , t_level , round ( counter/ t_level_count ,3) ) )
        return cond_prob_list
    

    def cond_prob_LPSmoother ( self , feature , k ) :
        feature_class_list = self.data.loc[:,feature].unique()
        target_class_list = self.data.loc[:,self.target].unique()
        total_rows = self.data.shape[0]
        counter = 0
        cond_prob_list = list()
        f_level_count =len (feature_class_list)
        for t_level in target_class_list :
            t_level_count = self . data [ self . data [ self . target ] == t_level ].shape [0]
            for f_level in feature_class_list :
                counter =0
                for i in range ( total_rows ) :
                    if ( self . data . loc [i , feature ] == f_level ) and ( self . data .loc [i , self . target ] == t_level ) :
                        counter +=1
                cp_smoother = ( counter + k ) /( t_level_count +( k * f_level_count ))
                cond_prob_list . append (( feature , f_level , t_level , round (cp_smoother ,3) ) )
        return cond_prob_list
    

    def count_all_prob ( self , k = None ) :
        X=self.data.drop([self.target],axis=1)
        if k!=None:
            for feature in X.columns :
                cond_list = self . cond_prob_LPSmoother(feature,k)
                self.prob_all = self . prob_all + cond_list
        else:
            for feature in X.columns :
                cond_list = self . cond_prob(feature)
                self.prob_all = self . prob_all + cond_list
            
        return self.prob_all
    def fit (self,smoothing='LPS',k=None) :
        self.target_prob()
        if smoothing == 'LPS' and k!=None:
            self . count_all_prob (k)
        else:
            self . count_all_prob ()
        return None

    def pred_score ( self , query ) :
        num_of_arg=self.data.shape[1]-1
        query_length=len(query)
        prob_dict=dict()
        prob_prod=1
        feature_list=self.data.columns
        if query_length != num_of_arg :
            print ('Query required ', num_of_arg ,' number of arguments')
            print ('Data set column ', self.data.columns )
        else :
            i =0
            ci =0
            decided_bin= ''
            query_copy = [i for i in query] 
            for item in query :
                if type(item) != int and type(item) !=float:
                    if item not in self . data . iloc [: , i ]. unique () :
                        print ( item ,'is not in coulmn ', self . data . columns [ i ])
                    i +=1
                else:
                    if self.bins[ci][0] <= item <= self.bins[ci][1]:
                        decided_bin ='bin1'
                    elif self.bins[ci][1] <= item <= self.bins[ci][2]:
                        decided_bin ='bin2'
                    elif self.bins[ci][2] <= item <= self.bins[ci][3]:
                        decided_bin ='bin3'
                    else:
                        decided_bin ='bin4'
                        
                    query_copy[i] = decided_bin
                    i+=1
                    ci+=1   
    
            for t_level in self . data [ self . target ]. unique () :
                for cnt in range ( query_length ) :
                    for item in self . prob_all :
                        if item [0]== feature_list [cnt] and item [1]== query_copy [cnt] and item [2]== t_level :
                            prob_prod = prob_prod * item [3]
                prob_dict [ t_level ]= round ( prob_prod * self . tprob_dict [ t_level] ,4)
                prob_prod = 1
        return prob_dict
    
    def pred ( self , query ) :
        pr_dict = self . pred_score ( query )
        return max( pr_dict , key = pr_dict . get )
    
    def get_prob_dict ( self ) :
        print ('Target probabilities : ', self . tprob_dict )
        print ('Conditional features Probabilities : ', self . prob_all )
        return None
    
    


# In[8]:


class Binning:
    def __init__ ( self , df , features ) :
        self . df = df
        self . features = features
        self.bins = list()
    
    
    def binning_ranges(self,feature):
        bin_ranges= []
        col = df[feature]
        col= col.sort_values(ascending=True)
        col_values= [i for i in col]
        bin_freq = col.shape[0]/4
        n = 1
        for i in range(5):
            if i == 0:
                bin_ranges.append(0)

            elif i !=4:
                avg = (col_values[int(round(n*bin_freq-1))] + col_values[int(round(n*bin_freq))])/2
                bin_ranges.append(avg)
                n+=1

            else:
                avg = col_values[int(round(n*bin_freq-1))]
                bin_ranges.append(avg)
        return bin_ranges
    
    def make_bins(self,df,feature):
        label_names = ["bin1", "bin2","bin3","bin4" ]
        cut_points = self.binning_ranges(feature)
        self.bins.append(cut_points)
        self.df[feature+" Binnig"] = pd.cut(df[feature], cut_points, labels=label_names)
        
    def get_bins(self):
        return self.bins
       
    def get_new_dataset(self):
        for i in self.features:
            self.make_bins(self.df,i)
        self.df =self.df.drop(self.features,axis=1)
    
        return self.df
    


# In[11]:


import pandas as pd
data = pd . read_csv ('LoanApplication_cont.csv', sep =',')
df = pd . DataFrame ( data )
binning = Binning(df,['AccountBalance','LoanAmount'])
df = binning.get_new_dataset()


# # without smoothing

# In[12]:


df = df.drop(['ID'],axis=1)
nbc = NBClassifier ( df ,'Fraud',binning.get_bins())
nbc . fit ()
query = ['paid','guarantor','free',759.07,8000]
nbc . pred_score ( query )


# In[13]:


nbc.pred(query)


# In[14]:


nbc.get_prob_dict()


# # With Smoothing

# In[15]:


nbc = NBClassifier ( df ,'Fraud',binning.get_bins())
nbc . fit (smoothing ='LPS',k =3)
query = ['paid','guarantor','free',759.07,8000]
nbc . pred_score ( query )


# In[16]:


nbc.pred(query)


# In[17]:


nbc.get_prob_dict()


# In[ ]:




