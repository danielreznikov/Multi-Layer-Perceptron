import numpy as np
import matplotlib.pyplot as plt
import math
from pprint import pprint
import utilities
from multi_layer_preceptron import *
from time import time
import sys


def main():
    data = utilities.read_data(train_size=None, test_size=None)

    # Initialize MLP Object
    mlp = MLP(input_units=785, hidden_layers=1, hidden_units=65, output_units=10)

    mlp.set_mlp_data(data)

    # Train the Model on the Training Set
    num_epochs = 50
    mlp.train(max_epochs=num_epochs, learning_rate_init=0.007, annealing=num_epochs * .90, batch_size=128, shuffle=False)

    # Display Accuracy and Loss over Epochs to Show Convergence of Model
    mlp.train_diagnostics('4a') # Problem (3E)

main()
