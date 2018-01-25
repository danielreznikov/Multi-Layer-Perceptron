import numpy as np
import matplotlib.pyplot as plt
import math
from pprint import pprint
import utilities
from multi_layer_preceptron import *

def main():
    data = utilities.read_data()

    mlp = MLP(input_units=785, hidden_layers=2, hidden_units=64, output_units=10)

    weights, accuracies = mlp.train(max_epochs=10, learning_rate_init=0.0001, lam=0, reg=None, annealing=1000)

    x = np.arange(10)

    plt.plot(x, accuracies['train_acc'])
    plt.xlabel('epoch')
    plt.ylabel('Percent Accuracy')
    # plt.legend(['train_acc', 'valid_acc', 'test_acc'], loc='upper right')
    plt.show()


main()
