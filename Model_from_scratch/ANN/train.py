import numpy as np
from functions import initialise,gradientDescent,computeGradient,forwardProp
from tqdm import tqdm

class Trainer():
    def __init__(self,input_neuron,hidden_neuron,output_neuron,epochs,lr,x_input,y_output):
        self.epochs = epochs
        self.lr = lr
        self.x_input = x_input
        self.y_output = y_output
        self.w1,self.b1,self.w2,self.b2 = initialise(input_neuron,hidden_neuron,output_neuron)
        self.cost = []
        
    def train(self):
        for i in tqdm(range(self.epochs)):
            cost = 0
            for j in range(len(self.x_input)-1):
                input1 = self.x_input[j,:].T.reshape(2,1)
                output1 = self.y_output[j,:].T.reshape(2,1)
                z1,a1,z2,a2 = forwardProp(input1,self.w1,self.b1,self.w2,self.b2)
                loss,db2,dw2,db1,dw1 = computeGradient(a2,output1,z2,a1,self.w2,z1,input1)
                cost += loss
                self.cost.append(cost)
                self.b2,self.w2,self.b1,self.w1 = gradientDescent(self.b2,db2,self.w2,dw2,self.b1,db1,self.w1,dw1,self.lr)
                
    def value_checkpoints(self):
        return self.cost,self.b1,self.w1,self.b2,self.w2

    