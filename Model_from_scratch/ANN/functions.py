import numpy as np

def initialise(input_neuron,hidden_neuron,output_neuron):
    w1 = np.random.rand(hidden_neuron,input_neuron)-0.5
    w2 = np.random.rand(output_neuron,hidden_neuron)-0.5
    b1 = np.zeros((hidden_neuron,1))
    b2 = np.zeros((output_neuron,1))
    return w1,b1,w2,b2

def relu(x):
    a = 1/(1+np.exp(-x))
    return a
def linear(x):
    return x

def derivative_linear(x):
    return 1

def derivative_relu(x):
    x = relu(x)
    a = 1-x
    return x*a

def forwardProp(a0,w1,b1,w2,b2):
    z1 = np.dot(w1,a0)+b1
    a1 = relu(z1)
    z2 = np.dot(w2,a1)+b2
    a2 = linear(z2)
    return z1,a1,z2,a2

def computeGradient(y_hat,y,z2,a1,w2,z1,a0):
    compute_loss = np.sum((y_hat - y)**2)
    derivative_b2 = (y_hat - y)*derivative_linear(z2)
    derivative_w2 = np.dot(derivative_b2,a1.T)
    derivative_b1 = np.dot(w2.T,derivative_b2)*derivative_relu(z1)
    derivative_w1 = np.dot(derivative_b1,a0.T)
    return compute_loss,derivative_b2,derivative_w2,derivative_b1,derivative_w1

def gradientDescent(b2,db2,w2,dw2,b1,db1,w1,dw1,lr):
    b2 = b2 - lr*db2
    w2 = w2 - lr*dw2
    b1 = b1 - lr*db1
    w1 = w1 - lr*dw1
    return b2,w2,b1,w1

