# -*- coding: utf-8 -*-
import numpy as np
from scipy import special

class neuralNetwork:
    
    #initialize neural net
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        #weights
        self.wih = np.random.normal(0.0,pow(self.inodes, -0.5),(self.hnodes,self.inodes))
        self.who = np.random.normal(0.0,pow(self.hnodes, -0.5),(self.onodes,self.hnodes))
        
        #learning rate
        self.lr = learningrate
        
        #activation funcion - sigmoid function
        self.activation_function = lambda x: special.expit(x);
        
        pass
    
    def train(self,inputs_list,targets_list):
        #convert inputs list to 2d array
        inputs = np.array(inputs_list,ndmin=2).T
        targets = np.array(targets_list,ndmin=2).T
        
        #calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        #calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #calculate signals into final output layer
        final_inputs = np.dot(self.who,hidden_outputs)
        #calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        #output layer error calculation
        output_errors = targets - final_outputs;
        #hidden layer 2 error is the output_errors, split by weights,
        #recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)
        
        #update weights for the links between the hidden and output
        #layers
        self.who += self.lr * np.dot((output_errors * final_outputs *(1.0 - final_outputs)),np.transpose(hidden_outputs))
        
        #update weights for links between input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        
        pass
    
    def query(self, inputs_list):
        #convert inputs list to 2d array
        inputs = np.array(inputs_list,ndmin=2).T
        
        #calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        #calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #calculate signals into final output layer
        final_inputs = np.dot(self.who,hidden_outputs)
        #calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
    

#number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

#learning rate
learning_rate = 0.3

#Instantiate neural net
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

#load the mnist data
training_data_file = open("mnist_data/mnist_train.csv")
training_data_list = training_data_file.readlines()
training_data_file.close()

#train the neural net

#for loop to go through all the data points
for record in training_data_list:
    all_values = record.split(',')
    
    #scale and shift inputs
    inputs = (np.asfarray(all_values[1:])/255.0 * 0.99) + 0.01
    
    #create the target output values
    targets = np.zeros(output_nodes) + 0.01
    
    #all_values[0] is the target label for each record
    targets[int(all_values[0])] = 0.99
    n.train(inputs,targets)
    pass


#test the neural net

#load the mnist data
test_data_file = open("mnist_data/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()

#scorecard to track performance
scorecard = []

#go through all the recs in the test data
for record in test_data_list:
    #split the rec by the commas
    all_values = record.split(',')
    
    #correct ans is first value
    correct_label = int(all_values[0])
    print(correct_label, "correct label")
    
    #scale and shift the inputs
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    
    #query the network
    outputs = n.query(inputs)
    
    #index of the highest value corresponds to the label
    label = np.argmax(outputs)
    print(label, "network's answer")
    
    #append correct or incorrect to list
    if(label == correct_label):
        #network's ans matches correct ans
        scorecard.append(1)
    else:
        #network's ans doesn't match correct ans
        scorecard.append(0)
        pass
    pass

#calculate the performance percentage
scorecard_array = np.asarray(scorecard)
print("performance = ", scorecard_array.sum()/scorecard_array.size)

