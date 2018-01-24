import numpy as np
import matplotlib.pyplot as plt
import math
from pprint import pprint
import utilities
from multi_layer_preceptron import *



def main():
    print("hello world?")

    data = utilities.read_data()

    print(data['xTrain'].shape)

    mlp = MLP()


main()
