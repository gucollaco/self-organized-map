# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:43:08 2018

@author: gustavo.collaco
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# normalize the values array
def normalize(values):
    return (values - values.min()) / (values.max() - values.min())

# find representation (slow)
def find_representation(dimension, neuron_weights, n_inputs, values, answers):
    most_similar = np.zeros((dimension, dimension))
    total_inputs = len(values)
    distances = []
    for j in range(dimension):
        for k in range(dimension):
            
            for i in range(total_inputs):
                distance = 0
                for l in range(n_inputs):
                    distance += (values[i][l] - neuron_weights[j][k][l]) ** 2
                
                distances.append(np.sqrt(distance))
            
            index_most_similar = np.argmin(distances)
            
            distances = []
            most_similar[j][k] = answers[index_most_similar]
    
    # array containing the most similar values (results from the expected output array)
    # for each 'pixel'
    return most_similar
    
# plot the values
def plot_data_final(dimension, neuron_weights, n_inputs):
    img = np.zeros((dimension, dimension))
    for i in range(dimension):
        for j in range(dimension):
            aux = 0
            for k in range(n_inputs):
                aux += neuron_weights[i][j][k]**2
            img[i][j] = np.sqrt(aux)
            
    plt.figure()
    plt.title("Weights updated")
    plt.imshow(img)

# plot the best matching units
def plot_bmu(dimension, best_matching_units, best_matching_units_result, n_inputs):
    img = np.zeros((dimension, dimension))
    fig, ax = plt.subplots()
    
    for i in range(dimension):
        for j in range(dimension):
            value = len(set(best_matching_units_result)) + 1
            value_all = []
            
            if [i, j] in best_matching_units:
                all_found = [k for k, e in enumerate(best_matching_units) if e == [i, j]]
                value = best_matching_units_result[all_found[0]]
                
                if(len(all_found) > 0):
                    for z in range(len(all_found)):
                        value_all.append(best_matching_units_result[all_found[z]])
                        
                    value_all = list(set(value_all))
                    value = round(sum(value_all) / len(value_all), 2)
                    
            img[j][i] = value
    
    img_plt = ax.imshow(img)
    
    # annotations
    for i in range(dimension):
        for j in range(dimension):
            text = ax.text(i, j, img[j, i], ha="center", va="center", color="w")

    #plt.figure()
    plt.title("Last epochs' best matching units")
    plt.show()


# plot the umatrix
def plot_umatrix(dimension, values, neuron_weights, n_inputs, total_inputs, answers):
    img = np.zeros((dimension, dimension))
    
    fig, ax = plt.subplots()
    
    for i in range(dimension):
        for j in range(dimension):
            value = 0
            # checking all possible positions on the grid
            # top left
            if(i == j == 0):
                distance = 0                
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i][j+1][l]) ** 2)                
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i+1][j+1][l]) ** 2)                    
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i+1][j][l]) ** 2)                
                value = distance / 3
                
            # top right
            elif(i == 0 and j == dimension-1):
                distance = 0                
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i][j-1][l]) ** 2)                
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i+1][j-1][l]) ** 2)                    
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i+1][j][l]) ** 2)                
                value = distance / 3
                
            # bottom left
            elif(i == dimension-1 and j == 0):
                distance = 0                
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i-1][j][l]) ** 2)
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i-1][j+1][l]) ** 2)
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i][j+1][l]) ** 2)
                value = distance / 3
                
            # bottom right
            elif(i == dimension-1 and j == dimension-1):
                distance = 0                
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i-1][j][l]) ** 2)
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i-1][j-1][l]) ** 2)
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i][j-1][l]) ** 2)
                value = distance / 3
                
            # left middle
            elif(j == 0 and i != 0 and i != dimension-1):
                distance = 0                
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i-1][j][l]) ** 2)
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i+1][j][l]) ** 2)
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i-1][j+1][l]) ** 2)
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i+1][j+1][l]) ** 2)
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i][j+1][l]) ** 2)
                value = distance / 5
            
            # top middle
            elif(i == 0 and j != 0 and j != dimension-1):
                distance = 0                
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i][j-1][l]) ** 2)
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i+1][j-1][l]) ** 2)
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i+1][j][l]) ** 2)
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i+1][j+1][l]) ** 2)
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i][j+1][l]) ** 2)
                value = distance / 5
            
            # bottom middle
            elif(i == dimension-1 and j != 0 and j != dimension-1):
                distance = 0                
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i][j-1][l]) ** 2)
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i-1][j-1][l]) ** 2)
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i-1][j][l]) ** 2)
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i-1][j+1][l]) ** 2)
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i][j+1][l]) ** 2)
                value = distance / 5
                
            # right middle
            elif(j == dimension-1 and i != 0 and i != dimension-1):
                distance = 0                
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i-1][j][l]) ** 2)
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i-1][j-1][l]) ** 2)
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i][j-1][l]) ** 2)
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i+1][j-1][l]) ** 2)
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i+1][j][l]) ** 2)
                value = distance / 5
                
            # filling the middle
            else:
                distance = 0
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i-1][j][l]) ** 2)
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i-1][j-1][l]) ** 2)
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i][j-1][l]) ** 2)
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i+1][j-1][l]) ** 2)
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i+1][j][l]) ** 2)
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i+1][j+1][l]) ** 2)
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i][j+1][l]) ** 2)
                for l in range(n_inputs): distance += np.sqrt((neuron_weights[i][j][l] - neuron_weights[i-1][j+1][l]) ** 2)
                value = distance / 8
                
            img[i][j] = value

    img_plt = ax.imshow(img)
    
    most_similar = find_representation(dimension, neuron_weights, n_inputs, total_inputs, answers)
    
    # annotations
    for i in range(dimension):
        for j in range(dimension):
            text = ax.text(j, i, most_similar[i, j], ha="center", va="center", color="w")

    plt.title("Umatrix")
    plt.show()

# dataset preparation function
def dataset():
    # read csv file
    data = pd.read_csv("dataset_iris.csv", header=None, sep=",")
    # shuffles the data
    data = data.sample(frac=1).reset_index(drop=True)
    # inputs
    values = data.iloc[:, :-1]
    # answers
    answers = data.iloc[:, -1]
    
    # rearranging the values and answers
    values = normalize(values).values
    answers = pd.factorize(answers)[0]
    
    # weights matrix
    dimension_x = dimension_y = int(np.sqrt(round(np.sqrt(2) * np.size(values, 0))))
    n_inputs = len(values[0])
    neuron_weights = np.random.uniform(low=-0.1, high=0.1, size=(dimension_x, dimension_y, n_inputs))

    # returning values, answers and weights
    return values, answers, neuron_weights

# competition
def kohonen(values, answers, neuron_weights, learning_rate=0.3, n_epochs=100):
    n_inputs = len(values[0])
    total_inputs = len(values)
    dimension = len(neuron_weights[0])
    
    # new 
    sigma0 = None
    sigma = None
    initial_learning_rate = learning_rate
    new_learning_rate = learning_rate
    
    # iterate through all epochs
    for epoch in range(n_epochs):
        best_matching_units = []
        best_matching_units_result = []
        
        for i in range(total_inputs):
            
            # calculate all the distances for this input
            distances = []
            for j in range(dimension):
                for k in range(dimension):
                    
                    distance = 0
                    for l in range(n_inputs):
                        distance += (values[i][l] - neuron_weights[j][k][l]) ** 2
                    
                    distances.append(np.sqrt(distance))
                    #distances.append(distance)
            
            # minimum distance value (winner)
            index_winner = np.argmin(distances)
            x_winner = index_winner % dimension
            y_winner = int(index_winner / dimension)
            
            # calculate the distances related to the winner      
            distances_winner = []
            for y_neuron in range(dimension):
                for x_neuron in range(dimension):
                    
                    distance = 0
                    distance = (x_winner - x_neuron) ** 2 + (y_winner - y_neuron) ** 2
                    distances_winner.append(np.sqrt(distance))
                    #distances_winner.append(distance)

            # adding the winner to the best_matching_units array
            best_matching_units.append([x_winner, y_winner])
            best_matching_units_result.append(answers[i])
            
            if sigma is None:
                sigma = sigma0 = np.sqrt(-(dimension**2) / (2*np.log(0.1)))
                tau = n_epochs / np.log(sigma0/0.1)
                #sigma = sigma0 = np.sqrt(dimension ** 2 + dimension ** 2)
                #tau = (-1) * n_epochs / np.log((5*10**(-5)) / sigma0)#sigma0
            
            # update the weights
            for j in range(dimension):
                for k in range(dimension):
                    pos = j * dimension + k
                    
                    h = np.exp((-1) * distances_winner[pos]**2 / (2 * sigma**2))
                        
                    for l in range(n_inputs):
                        neuron_weights[j][k][l] = neuron_weights[j][k][l] + new_learning_rate * h * (values[i][l] - neuron_weights[j][k][l]) #distances[pos]

        sigma = sigma0 * np.exp((-1) * epoch / tau)
        new_learning_rate = initial_learning_rate * np.exp((-1) * epoch / tau)
        
        # prints the actual epoch
        print('Actual epoch: ', epoch)
        print('Learning Rate: ', new_learning_rate)
        print('Sigma: ', sigma)
        print('____________________')

    # plotting the final data
    plot_data_final(dimension, neuron_weights, n_inputs)
    plot_bmu(dimension, best_matching_units, best_matching_units_result, n_inputs)
    plot_umatrix(dimension, values, neuron_weights, n_inputs, values, answers)

# main function
if __name__ == "__main__":
    # returning values and neuron_weights from the dataset function
    # the neuron_weights is a 'square', with same length and width
    # its depth is based on the amount of parameters on the input
    values, answers, neuron_weights = dataset()
    
    # calling the network processing
    kohonen(values, answers, neuron_weights)