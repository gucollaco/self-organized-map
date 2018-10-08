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

    # answers
    answers_factorized = pd.factorize(data[np.size(values,1)])[0]
    n_answers = len(set(answers_factorized))
    answers = np.zeros(shape=[np.size(values,0), n_answers])
    count = 0
    for i in answers_factorized:
        for j  in np.unique(answers_factorized): answers[count][j] = 1 if (i==j) else 0
        count += 1

    # weights matrix
    n_neuron = int(round(np.sqrt(2) * np.size(values, 0)))
    weights = []
    n_inputs = len(values[0])
    random.seed(30)
    for i in range(n_neuron):
        aux = [random.uniform(0, 0.1) for i in range(n_inputs)]
        weights.append(aux)
    weights = np.array(weights)

    # returning values and answers
    return values, weights, answers

# competition
#def kohonen(learning_rate=0.3, acceptable_error=0.1`):
#    while(e > acceptable_error):
    

# main function
if __name__ == "__main__":

    # returning values and answers from the dataset function
    values, weights, answers = dataset()

    print('values', np.size(values, 0))
    print('VAL', values)
    print('answers', answers)
    print('WEIGHTS', weights)