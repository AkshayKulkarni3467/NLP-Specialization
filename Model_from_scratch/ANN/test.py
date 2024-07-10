import numpy as np
from functions import forwardProp
from train import Trainer
import matplotlib.pyplot as plt

x_input = np.random.uniform(-1,1,(100,2))
y_output = (np.dot(np.array([-1,0,0,1]).reshape(2,2),x_input.T)).T

trainer = Trainer(input_neuron=2,hidden_neuron=3,output_neuron=2,epochs=1000,lr=0.1,x_input=x_input,y_output=y_output)

trainer.train()

cost,b1,w1,b2,w2 = trainer.value_checkpoints()

test_x = np.arange(0,1,.01)
test_y = test_x**2

plt.scatter(test_x, test_y)
plt.savefig('charts/test_data.png')

test = np.column_stack([test_x,test_y])

n = test.shape[0]
output_x = []
output_y = []

for i in range(n):
    a0 = test[i,:].reshape(2,1)
    _,_,_,a2 = forwardProp(a0,w1,b1,w2,b2)
    output_x.append(a2[0,0])
    output_y.append(a2[1,0])
    
plt.scatter(output_x, output_y,c='blue')
plt.scatter(test[:,0],test[:,1],c='red')
plt.savefig('charts/test_output.png')