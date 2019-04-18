# -*- coding: utf-8 -*-
"""
Spyder Editor
Author: Abhijeet Paul
##this is an implementation of the 2 input XOR gate using neural network
idea for the XOR gate is from here --> https://towardsdatascience.com/implementing-the-xor-gate-using-backpropagation-in-neural-networks-c1f255b4f20d
"""

#########################################
## import the relevant libraries
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


##############################################
# fnctions needed for the calculations ##
#############################################


def activation (x,acttype="sigmoid"):
    """ this is the activation function sigmoid function"""
    if acttype.lower()=="sigmoid":
        return 1/(1 + np.exp(-x))
    else:
        return x*(x>0) ##this is relu --> 0 if x<0, x=x, for X>=0
        



def activation_derivative(x,acttype="sigmoid"):
    """ this is sigmoid function derivative used for back propogation"""
    if acttype.lower()=="sigmoid":
        return x * (1 - x)
    else: ##relu
        return 1*(x>0) ## this is the derivative of relu
    





#### choose gate type (input and output)

def get_2inputlogicgate_in_out(gate_type="nor"):
    print("chosen gate type = ",gate_type,"\n");
    if gate_type.lower() == "xor" :
        inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
        expected_output = np.array([[0],[1],[1],[0]])
    elif gate_type.lower() == "and" :
        inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
        expected_output = np.array([[0],[0],[0],[1]])
        ##implement the AND 
    else: ##this is for NOR
        inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
        expected_output = np.array([[1],[0],[0],[0]])

    return inputs,expected_output

######################################################################################
### user inputs.
epochs = 11000 ##no. of iterations
##hyper parameter.
lr = 0.1 ##learning rate.

gate_type = "and"
activation_type = 'relu' ##other one = relu

###################################################################
##newral network structure
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2,2,1
inputs,expected_output = get_2inputlogicgate_in_out(gate_type)

#Random weights and bias initialization
hidden_weights = np.random.uniform(size=(inputLayerNeurons,hiddenLayerNeurons))
hidden_bias =np.random.uniform(size=(1,hiddenLayerNeurons))
output_weights = np.random.uniform(size=(hiddenLayerNeurons,outputLayerNeurons))
output_bias = np.random.uniform(size=(1,outputLayerNeurons))



print("Initial hidden weights: ",end='')
print(*hidden_weights)

print("Initial hidden biases: ",end='')
print(*hidden_bias)

print("Initial output weights: ",end='')
print(*output_weights)

print("Initial output biases: ",end='')
print(*output_bias)





#Training algorithm
train_iter = 0
error_val = np.zeros((epochs,1))
for _ in range(epochs):

    
    	#Forward Propagation
    
    ##1 hidden layer with 2 nurons from the input layer to 2 nurons in the hidden layer.
    hidden_layer_activation = np.dot(inputs,hidden_weights) ##W dot X
    hidden_layer_activation += hidden_bias # this is WX + B
    hidden_layer_output = activation(hidden_layer_activation,activation_type) ##activation using sigmoid
    
    
    ## this is the output layer (with single neuron created using hidden layer)
    output_layer_activation = np.dot(hidden_layer_output,output_weights)
    output_layer_activation += output_bias
    predicted_output = activation(output_layer_activation,activation_type)
    
    
    ### this is the update of the waights and bias part of the network
    	#Backpropagation
    
    
    error = expected_output - predicted_output
    error_val[train_iter-1]= np.linalg.norm(error,axis=0,keepdims=True) ##l2 norm for tracking.
    train_iter +=1
    print("learning iterations no. {a} \t Error = {err:4.3e} \n".format(a=train_iter,err=error_val[train_iter-1,0]))
    
    
    d_predicted_output = error * activation_derivative(predicted_output,activation_type)
    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer     = error_hidden_layer * activation_derivative(hidden_layer_output,activation_type,)
    
    
    
    	#Updating Weights and Biases
    
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
    output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * lr
    
    hidden_weights += inputs.T.dot(d_hidden_layer) * lr
    hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * lr



print("Final hidden weights: ",end='')
print(*hidden_weights)

print("Final hidden bias: ",end='')
print(*hidden_bias)

print("Final output weights: ",end='')
print(*output_weights)

print("Final output bias: ",end='')
print(*output_bias)



print("\nOutput from neural network after 10,000 epochs: ",end='')

print(*predicted_output)

plt.close('all')
plt.figure(1)
sns.lineplot(np.arange(error_val.shape[0]),error_val[:,0])
plt.show()
