# # Regression Model Using Gradient Descent Algorithm

# In[1]:


class GradientDescent :
    def __init__ ( self,df ,target):
        self.df = df
        self.target = target
        
    def setAlpha ( self , value):
        self.alpha = value
    
    def setWeightVector ( self , WVector =[] , random = False) :
        if rnadom == False and len ( WVector ) !=0:
            W.self = WVector
        else:
            n = df.shape [1] -1
            self.W = genRandom (n)
        
    def genRandom (self , n ) :
        import numpy as np
        W = np.random.uniform ( low = -1 , high =1 , size = n )
        return W

    def L2c (self , df , target , pred_target ) :
        y = df.loc [: ,[ target ]].values.flatten().tolist()
        y_pred = pred_target
        nrows = df.shape [0]
        loss =0
        for j in range ( nrows ) :
            loss += ( y [ j ] - y_pred [ j ])
        return loss
    
    def L2f (self , df , target , pred_target , feature ) :
        y = df.loc [: ,[ target ]].values.flatten().tolist ()
        y_pred = pred_target
        d = df.iloc [: ,feature].values.flatten().tolist ()
        nrows = df.shape [0]
        loss =0
        j =0
        for j in range ( nrows ) :
            loss += ( y [ j ] - y_pred [ j ]) * d [ j ]
        return loss
    
    def errorDelta (self , loss , alpha ) :
        return alpha * loss
    
    def pred ( self ,dFrame , target , WVector ) :
        if dFrame . shape [1]!= len( WVector ) :
            print ('feature length is not equal to weight vector')
            return []
        pred = list ()
        predict =0
        df = dFrame.drop (['RentalPrice'] , axis =1)
        W = WVector
        nrows = df.shape [0]
        ncolumns = df.shape [1]
        for i in range (nrows):
            j =0
            predict =0
            A = df.iloc [ i ]
            row = A.values.flatten().tolist()
            for j in range ( ncolumns ) :
                if j ==0:
                    predict += W [0]
                else :
                    predict += W [ j ]* row [j -1]
            pred.append ( predict )
        return pred

    def weights_update (self, df , target , WVector ) :
        loss_vec = list ()
        new_WVector = list ()
        W = WVector
        print ('Current Weights :', W )
        print ('Current target values : ', df . loc [: ,[ target ]].values.flatten().tolist () )
        pred_target = self.pred ( df , target , W )
        print ('Predicted target values :', pred_target )
        nrows = df . shape [0]
        ncolumns = df . shape [1]
        # print ( ’ Shape : ’ ,nrows , ’x ’, ncolumns )
        j =0
        loss = self.L2c ( df , target , pred_target )
        loss_vec.append ( loss )
        # print ( ’W [0]: ’, new_WVector [0])
        for j in range ( ncolumns -1) :
            loss =0
            # feature = df. iloc [j]
            loss = self.L2f ( df , target , pred_target , j )
            loss_vec . append ( loss )
        for i in range (len( loss_vec ) ) :
            new_WVector . append ( WVector [ i ]+( loss_vec [ i ]* alpha ) )
        print ('Updated Weights :', new_WVector )
        return new_WVector
    
    def epoch (self,n):
        self.epochs = n
        
    
    def fit(self, W) :
        for i in range ( self . epochs ) :
            print ('Iteration - ',i )
            print ('--------------------------------------')
            newW = self.weights_update( df , self.target , W )
            print ('--------------------------------------')
            W = newW
            
            
#driver code
import pandas as pd
alpha = 0.00000002
W = [-0.146,0.185,0.044,0.119]
n = 60 #no of iteration
df = pd.read_csv('RentalPrice1.csv')
df = df.drop(['ID','EnergyRating'],axis=1)
LR = LinearRegression(df,'RentalPrice')
LR.setAlpha(alpha)
LR.epoch(n)
LR.fit(W)

