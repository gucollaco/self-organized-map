# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:43:08 2018

@author: gustavo.collaco
"""

import pandas as pd
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
    
    # returning values and answers
    return values, answers

# main function
if __name__ == "__main__":
    # returning values and answers from the dataset function
    values, answers = dataset()
    
    n_neuron = np.sqrt(2) * np.size(values, 0)

    print('values', np.size(values, 0))
    print('n_neuron', n_neuron)
    print('answers', answers)