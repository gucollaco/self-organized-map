# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:43:08 2018

@author: gustavo.collaco
"""

import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

# dataset preparation function
def dataset():
    # read csv file
    data = pd.read_csv("iris_dataset.csv", header=None)

    # inputs
    values = data.iloc[:, :-1]
    values = values.values

    # weights matrix
    dimension_x = dimension_y = int(np.sqrt(round(np.sqrt(2) * np.size(values, 0))))
    n_inputs = len(values[0])
    random.seed(30)
    neuron_weights = np.random.uniform(low=0.05, high=0.2, size=(dimension_x, dimension_y, n_inputs))

    # returning values and answers
    return values, neuron_weights

# competition
def kohonen(values, neuron_weights, learning_rate=0.3, acceptable_error=0.1, n_epochs=300):
    #epochs = []
    n_inputs = len(values[0])
    total_inputs = len(values)
    dimension = len(neuron_weights[0])
    
    #for epoch in n_epochs:
    for i in range(total_inputs):
        distances = [[] for i in range(dimension)]
            
        for j in range(dimension):
            for k in range(dimension):
                    
                distance = 0
                for l in range(n_inputs):
                    distance += (neuron_weights[j][k][l] - values[i][l]) ** 2
                
                distances[j].append(distance)
                
        #epochs.append(epoch+1)
        
    print('distances', distances)

# main function
if __name__ == "__main__":
    # returning values and answers from the dataset function
    values, neuron_weights = dataset()
    
    #normalized_values = normalize(values)
    kohonen(values, neuron_weights)

    #print('VAL', values)
    print('WEIGHTS', neuron_weights)
    #print('KOHONEN', len(neuron_weights[0]))