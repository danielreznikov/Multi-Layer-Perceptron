import os
from mnist import MNIST
import numpy as np

def read_data(train_size=None, test_size=None):
    '''Load data'''
    dir = os.getcwd()
    mndata = MNIST(dir + '/data')
    mndata.gz = True
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    # Use the first 20,000 training images and the last 2,000 test images
    xTrain = np.array(train_images)[:20000][:train_size]
    xTrain = np.divide(xTrain, 127.5)
    xTrain = np.subtract(xTrain, 1)
    xTrain = np.insert(xTrain, 0, 1, axis=1)
    yTrain = np.array(train_labels)[:20000][:train_size]

    xTest = np.array(test_images)[-2000:][:test_size]
    xTest = np.divide(xTest, 127.5)
    xTest = np.subtract(xTrain, 1)
    xTest = np.insert(xTest, 0, 1, axis=1)
    yTest = np.array(test_labels)[-2000:][:test_size]

    yTrain1Hot = np.zeros((yTrain.shape[0], 10))
    yTest1Hot = np.zeros((yTest.shape[0], 10))

    for idx in range(yTrain.shape[0]):
        yTrain1Hot[idx][yTrain[idx]] = 1

    for idx in range(yTest.shape[0]):
        yTest1Hot[idx][yTest[idx]] = 1

    data = {'xTrain': xTrain,
            'yTrain': yTrain1Hot,
            'xTest': xTest,
            'yTest': yTest1Hot,
            }

    return data

def sigmoid_activation(x):
    return 1.0/(1.0 + np.exp(-x))

def softmax_activation(net_input):
    '''Input n x 10 numpy array and Outputs a nx10 numpy array where the rows will sum to 1'''
    expon = np.exp(net_input - np.max(net_input, axis=1).reshape(-1, 1))

    retval = expon / expon.sum(axis=1).reshape(-1, 1)
    assert (retval.shape == net_input.shape)

    return retval

