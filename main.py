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

    # Initialize MLP Object
    mlp = MLP(input_units=785, hidden_layers=1, hidden_units=64, output_units=10)

    # Split Data into Train, Test, and Validation
    xTrain = np.copy(data['xTrain'])
    yTrain = np.copy(data['yTrain'])
    xTest = np.copy(data['xTest'])
    yTest = np.copy(data['yTest'])


    # Randomly split data into train and validation
    np.random.seed(2018)
    indices = np.random.randint(0, xTrain.shape[0], xTrain.shape[0] // 10)
    xValid = xTrain[indices]
    yValid = yTrain[indices]
    np.delete(xTrain, indices, axis=0)
    np.delete(yTrain, indices, axis=0)

    # Train the Model on the Training Set
    num_epochs = 100


    mlp.train_non_modular(xTrain, yTrain, max_epochs=num_epochs, learning_rate_init=0.0001, annealing=num_epochs*.90, batch_size=200, shuffle=True)

    # weights, accuracies = mlp.train(data, max_epochs=num_epochs, learning_rate_init=0.00005, lam=0, reg=None, annealing=10, batch_size=50, shuffle=False)



    # Evaluate on a Test Set to Measure Model Performance
    # predictions = mlp.evaluate(xTest)
    # accuracy = utilities.accuracy(yTest, predictions, softmax=True)
    # print('Test Accuracy: ', accuracy)

    # Display Accuracy and Loss over Epochs to Show Convergence of Model
    mlp.train_diagnostics()




    # x = np.arange(num_epochs)
    # plt.plot(x, accuracies['train_acc'])
    # plt.xlabel('epoch')
    # plt.ylabel('Percent Accuracy')
    # # plt.legend(['train_acc', 'valid_acc', 'test_acc'], loc='upper right')
    # plt.show()


main()
