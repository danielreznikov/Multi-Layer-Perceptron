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
    xTrain = np.array(train_images)[:train_size]
    xTrain = np.divide(xTrain, 127.5)
    xTrain = np.subtract(xTrain, 1)
    xTrain = np.insert(xTrain, 0, 1, axis=1)
    yTrain = np.array(train_labels)[:train_size]

    xTest = np.array(test_images)[:test_size]
    xTest = np.divide(xTest, 127.5)
    xTest = np.subtract(xTest, 1)
    xTest = np.insert(xTest, 0, 1, axis=1)
    yTest = np.array(test_labels)[:test_size]

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

def split_data(data):
    # Split Data into Train, Test, and Validation
    xTrain = np.copy(data['xTrain'])
    yTrain = np.copy(data['yTrain'])
    xTest = np.copy(data['xTest'])
    yTest = np.copy(data['yTest'])

    # Randomly split data into train and validation
    np.random.seed(2018)
    indices = np.random.randint(0, xTrain.shape[0], xTrain.shape[0] // 5)
    xValid = xTrain[indices]
    yValid = yTrain[indices]
    np.delete(xTrain, indices, axis=0)
    np.delete(yTrain, indices, axis=0)

    return xTrain, yTrain, xValid, yValid, xTest, yTest

def sigmoid_activation(net_input):
    return 1.0/(1.0 + np.exp(-net_input))

def softmax_activation(net_input):
    '''Input n x 10 numpy array and Outputs a nx10 numpy array where the rows will sum to 1'''
    expon = np.exp(net_input - np.max(net_input, axis=1).reshape(-1, 1))

    retval = expon / expon.sum(axis=1).reshape(-1, 1)
    assert (retval.shape == net_input.shape)

    return retval

def cross_entropy_loss(actuals, predicted, softmax=True):
    '''
    Actuals   - nX10, 1-Hot encoded
    Predicted - nX10, softmax values
    Return    - scalar value for loss
    '''
    assert (actuals.shape == predicted.shape)

    dims = 785

    if softmax:
        loss = np.sum(actuals * np.log(predicted), axis=(1, 0))
        return -loss / actuals.shape[0]

    else:
        loss = np.multiply(actuals, np.log(predicted)) + np.multiply((np.ones((dims, 1)) - actuals), np.log(np.ones((dims, 1)) - predicted))
        loss = -np.sum(loss) / actuals.shape[0]
        return loss

def accuracy(actuals, predictions, softmax=True):
    '''Computes the percent accuracy of a model.'''

    if not softmax:
        raise Exception("ERROR cannot compute accuracy for non-softmax 1-hot models.")

    # From 1-Hot encoding to [0-9]
    func = lambda lis: np.argmax(lis, axis=1)
    actuals = func(actuals)
    predictions = func(predictions)

    correct = np.sum([actual == pred for actual, pred in zip(actuals, predictions)])
    accuracy = correct / actuals.shape[0]

    return accuracy

def early_stopping(arr):
    '''True if last 3 epochs have higher loss than 4 epochs ago.'''
    return arr[-3] > arr[-4] and arr[-2] > arr[-3] and arr[-1] > arr[-4]