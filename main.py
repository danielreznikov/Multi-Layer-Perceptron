import numpy as np
import matplotlib.pyplot as plt
import math
from pprint import pprint
import utilities
from multi_layer_preceptron import *
from time import time


def main():
    data = utilities.read_data(train_size=5000, test_size=1)

    num_epochs = 100

    mlp = MLP(input_units=785, hidden_layers=2, hidden_units=128, output_units=10)

    strt = time()
    print("Training...")
    weights, accuracies = mlp.train(data, max_epochs=num_epochs, learning_rate_init=0.5, lam=0, reg=None, annealing=50, batch_size=None, shuffle=False)
    print("Done. Took", time()-strt, " secs.")

    x = np.arange(num_epochs)

    plt.plot(x, accuracies['train_acc'])
    plt.xlabel('epoch')
    plt.ylabel('Percent Accuracy')
    # plt.legend(['train_acc', 'valid_acc', 'test_acc'], loc='upper right')
    plt.show()


main()
