import numpy as np
from train import Trainer
from functions import forwardProp
import matplotlib.pyplot as plt

x_input = np.random.uniform(-1,1,(100,2))
y_output = (np.dot(np.array([-1,0,0,1]).reshape(2,2),x_input.T)).T

trainer = Trainer(input_neuron=2,hidden_neuron=3,output_neuron=2,epochs=200,lr=0.005,x_input=x_input,y_output=y_output)

trainer.train()

cost,b1,w1,b2,w2 = trainer.value_checkpoints()

length = len(cost)


plt.plot(range(length),cost)
plt.savefig('charts/cost.png')