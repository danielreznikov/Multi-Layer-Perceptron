import numpy as np
from sklearn.datasets import load_digits
import pandas as pd
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import sys


def sigmoid(x, derivative=False):


    g = 1/(1 + np.exp(-x))


    if derivative == True:
        return g * (1 - g)
    else:
        return g


def softmax(x, derivative=False):
    expon = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
    return expon / expon.sum(axis=1).reshape(-1, 1)

    # if i == j:
    #     self.gradient[i, j] = self.value[i] * (1 - self.value[i))
    #     else:
    #     self.gradient[i, j] = -self.value[i] * self.value[j]


def show_img(image):
    plt.imshow(image)
    plt.show()
    return None




digits = load_digits()

data = digits.data
data = np.hstack((data, np.ones((data.shape[0], 1))))

target = digits.target
weights = np.zeros((65, 10))
t = np.asarray(pd.get_dummies(target))

# print(target)
# sys.exit()


epochs = 1000
alpha = .0001


input_units = 65
hidden_units1 = 150
hidden_units2 = 150
output_units = 10

W_hidden1 = np.random.normal(loc=0, scale=1, size=(input_units, hidden_units1))
W_hidden2 = np.random.normal(loc=0, scale=1, size=(hidden_units1, hidden_units2))
W_output = np.random.normal(loc=0, scale=1, size=(hidden_units2, output_units))




accuracy_over_epoch = []

for i in range(epochs):

    # Forward Prop

    # Layer 1
    net_input_h1 = np.dot(data, W_hidden1)
    hidden_layer_out1 = sigmoid(net_input_h1)

    # Layer 2
    net_input_h2 = np.dot(hidden_layer_out1, W_hidden2)
    hidden_layer_out2 = sigmoid(net_input_h2)

    # Output Layer
    net_input_o = np.dot(hidden_layer_out2, W_output)
    y = softmax(net_input_o)


    # Back Prop
    delta_output = (t - y)
    delta_hidden2 = sigmoid(net_input_h2) * (1-sigmoid(net_input_h2)) * np.dot(delta_output, W_output.T)
    delta_hidden1 = sigmoid(net_input_h1) * (1 - sigmoid(net_input_h1)) * np.dot(delta_hidden2, W_hidden2.T)

    W_output = W_output + alpha * np.dot(hidden_layer_out2.T, delta_output)
    W_hidden2 = W_hidden2 + alpha * np.dot(hidden_layer_out1.T, delta_hidden2)
    W_hidden1 = W_hidden1 + alpha * np.dot(data.T, delta_hidden1)


    predictions = np.argmax(y, axis=1)
    accuracy = accuracy_score(target, predictions)
    accuracy_over_epoch.append(accuracy)





print('training done')

plt.plot(np.arange(epochs), accuracy_over_epoch)
plt.show()

# for col in weights.T:
#     vec = col[:-1]
#     img = vec.reshape((8, 8))
#     show_img(img)


print('done')