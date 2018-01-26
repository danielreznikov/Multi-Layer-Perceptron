import numpy as np
import matplotlib.pyplot as plt
import math
from pprint import pprint
import utilities
from multi_layer_preceptron import *
from time import time
import sys


def main():
    data = utilities.read_data(train_size=20000, test_size=2000)
    xTrain, yTrain, xValid, yValid, xTest, yTest = utilities.split_data(data)

    # Initialize MLP Object
    mlp = MLP(input_units=785, hidden_layers=2, hidden_units=65, output_units=10)

    # Train the Model on the Training Set
    num_epochs = 10
    mlp.train(xTrain, yTrain, max_epochs=num_epochs, learning_rate_init=0.005, annealing=num_epochs * .90, batch_size=200, shuffle=True, gradient_checking=False)

    # Evaluate on a Test Set to Measure Model Performance
    predictions = mlp.get_model_predictions(xTest)
    accuracy = utilities.accuracy(yTest, predictions, softmax=True)
    print('Test Accuracy: ', accuracy)

    # Display Accuracy and Loss over Epochs to Show Convergence of Model
    mlp.train_diagnostics()

main()
