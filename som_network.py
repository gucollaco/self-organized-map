# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:43:08 2018

@author: gustavo.collaco
"""

import pandas as pd
import math
import random
import numpy as np
import matplotlib.pyplot as plt

# normalize the values array
def normalize(values):
    return (values - values.min()) / (values.max() - values.min())

# plot the values
def plot_data(dimension):
    img = np.zeros((dimension, dimension))
    for i in range(dimension):
        for j in range(dimension):
            img[i][j] = 1
            
    plt.imshow(img)

# dataset preparation function
def dataset():
    # read csv file
    data = pd.read_csv("iris_dataset.csv", header=None)

    # inputs
    values = data.iloc[:, :-1]
    values = normalize(values).values

    # weights matrix
    dimension_x = dimension_y = int(np.sqrt(round(np.sqrt(2) * np.size(values, 0))))
    n_inputs = len(values[0])
    random.seed(30)
    neuron_weights = np.random.uniform(low=0.05, high=0.2, size=(dimension_x, dimension_y, n_inputs))

    # returning values and answers
    return values, neuron_weights

# competition
def kohonen(values, neuron_weights, learning_rate=0.3, n_epochs=50):
    #epochs = []
    n_inputs = len(values[0])
    total_inputs = len(values)
    dimension = len(neuron_weights[0])
    
    # new 
    sigma0 = 0.9
    initial_learning_rate = learning_rate
    
    # iterate through all epochs
    for epoch in range(n_epochs):
        for i in range(total_inputs):
            
            # calculate all the distances for this input
            distances = []
            for j in range(dimension):
                for k in range(dimension):
                    
                    distance = 0
                    for l in range(n_inputs):
                        distance += (neuron_weights[j][k][l] - values[i][l]) ** 2
                    
                    distances.append(np.sqrt(distance))
            
            # printing the distances
            print('distances', distances)
            
            # minimum distance value (winner)
            index_winner = np.argmin(distances)
            x_winner = index_winner % dimension
            y_winner = int(index_winner / dimension)
            
            # printing the winner information
            print('winner axis on the distances array', index_winner)
            print('winner x axis', x_winner)
            print('winner y axis', y_winner)
            
            # calculate the distances related to the winner      
            distances_winner = []
            for j in range(dimension):
                for k in range(dimension):
                    
                    distance = 0
                    x_neuron = k
                    y_neuron = j
                    distance = (x_winner - x_neuron) ** 2 + (y_winner - y_neuron) ** 2
                    distances_winner.append(np.sqrt(distance))
            
            # printing the distances related to the winner
            print('distances to the winner', distances_winner)
            
            #sigma0 = dimendion / 2
            #tau = n_epochs / math.log(sigma0)
            #sigma = sigma0 * np.exp(((-1) * epoch) / tau)
            
            # modify sigma and learning rate according to the epoch
            sigma = sigma0 * (1 - epoch / n_epochs)
            learning_rate = initial_learning_rate * (1 - epoch / n_epochs)
            
            # update the weights
            for j in range(dimension):
                for k in range(dimension):
                    pos = j * dimension + k
                    h = math.exp((-1) * distances_winner[pos]**2 / (2 * sigma**2))
                    
                    for l in range(n_inputs):
                        neuron_weights[j][k][l] = neuron_weights[j][k][l] + learning_rate * h * distances[pos]
            
        # plotting the data
        plot_data(dimension)
            
        # prints the actual epoch
        print('Actual epoch', epoch)
        
# main function
if __name__ == "__main__":
    # returning values and neuron_weights from the dataset function
    # the neuron_weights is a 'square', with same length and width
    # its depth is based on the amount of parameters on the input
    values, neuron_weights = dataset()
    
    # calling the network processing
    kohonen(values, neuron_weights)

    print('VAL', values)
    print('WEIGHTS', neuron_weights)
    #print('KOHONEN', len(neuron_weights[0]))