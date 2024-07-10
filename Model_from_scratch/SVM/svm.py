import numpy as np

class SVM_Classifier():
    def __init__(self,lr,epochs,lamb):
        self.lr = lr
        self.epochs = epochs
        self.lamb = lamb
        
    def fit(self,x,y):
        self.m,self.n = x.shape
        
        self.w = np.zeros(self.n)
        self.b = 0
        self.x = x
        self.y = y
        
        for i in range(int(self.epochs)):
            self.update_weights()
            
        
    def update_weights(self):
        y_label = np.where(self.y<=0,-1,1)
        for index,x_i in enumerate(self.x):
            condition = y_label[index]*(np.dot(x_i,self.w)-self.b)
            if condition >=1:
                derv_w = 2*self.w*self.lamb
                derv_b = 0
            else:
                derv_w = 2*self.w*self.lamb - np.dot(x_i,y_label[index])
                derv_b = y_label[index]
            self.w = self.w - self.lr * derv_w
            self.b = self.b - self.lr * derv_b
                
        
    def predict(self,x):
        output = np.dot(x,self.w) - self.b
        pred_labels = np.sign(output)
        y_hat = np.where(pred_labels<=-1,0,1)
        return y_hat
        