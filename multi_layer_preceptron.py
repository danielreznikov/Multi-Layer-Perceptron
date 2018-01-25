import numpy as np
import utilities
import sys

class MLP(object):
    """This is our neural network object."""

    def __init__(self, input_units, hidden_layers, hidden_units, output_units,
                 hidden_activation='sigmoid', output_activation='softmax'):

        self.input_units = input_units
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.dims = 785
        self.num_layers = hidden_layers + 1

        if hidden_activation == 'sigmoid':
            self.hidden_activation = utilities.sigmoid_activation
        else:
            raise Exception("ERROR: unsuported activation function," + hidden_activation)

        if output_activation == 'softmax':
            self.output_activation = utilities.softmax_activation
        else:
            raise Exception("ERROR: unsuported activation function," + output_activation)


    def train(self, data, max_epochs=100, learning_rate_init=0.0001, lam=0, reg=None, annealing=100):

        xTrain = np.copy(data['xTrain'])
        yTrain = np.copy(data['yTrain'])
        xTest = np.copy(data['xTest'])
        yTest = np.copy(data['yTest'])

        # train_loss_record = []
        # valid_loss_record = []
        # test_loss_record = []
        weights_record = []

        train_accuracy = []
        # valid_accuracy = []
        # test_accuracy = []

        # Split data into train and validation
        indices = np.random.randint(0, xTrain.shape[0], xTrain.shape[0] // 10)
        xValid = xTrain[indices]
        yValid = yTrain[indices]
        np.delete(xTrain, indices, axis=0)
        np.delete(yTrain, indices, axis=0)


        datasets = {'train': xTrain, 'valid': xValid, 'test': xTest}

        weights = []
        net_inputs = []
        net_outputs = []
        deltas = []

        np.random.seed(2018)

        # Initialize weights list indexing into each layer (looks good)
        for layer in range(self.num_layers):
            if layer == 0: # layer0
                weights.append(np.random.normal(loc=0, scale=1, size=(self.dims, self.hidden_units)))
            elif layer < self.num_layers - 1: # hidden layers
                weights.append(np.random.normal(loc=0, scale=1, size=(self.hidden_units, self.hidden_units)))
            else: # last layer
                weights.append(np.random.normal(loc=0, scale=1, size=(self.hidden_units, self.output_units)))


        # Iterate over epochs!
        for epoch in range(max_epochs):
            learning_rate = learning_rate_init / (1 + epoch / annealing)

            # Forward propogation to get model weights
            for layer in range(self.num_layers):
                if layer == 0:
                    net_inputs.append(np.dot(xTrain, weights[layer]))
                    net_outputs.append(self.hidden_activation(net_inputs[layer]))
                elif layer < self.num_layers - 1:
                    net_inputs.append(np.dot(net_outputs[layer-1], weights[layer]))
                    net_outputs.append(self.hidden_activation(net_inputs[-1]))
                else:
                    net_inputs.append(np.dot(net_outputs[layer-1], weights[layer]))
                    net_outputs.append(self.output_activation(net_inputs[-1]))

            # Backprop of partial gradients via deltas
            for layer in range(self.num_layers-1, -1, -1):
                # print(layer)


                if layer == self.num_layers-1:
                    deltas.append(yTrain - net_outputs[-1])
                else:
                    delta = net_outputs[layer]*(1 - net_outputs[layer]) * np.dot(deltas[-1], weights[layer+1].T)
                    deltas.append(delta)

            deltas = list(reversed(deltas))


            # Update weights via gradient descent
            for layer in range(self.num_layers):

                if layer == 0:
                    grad = np.dot(xTrain.T, deltas[layer])
                    weights[layer] += learning_rate * np.dot(xTrain.T, deltas[layer])

                else:
                    grad = np.dot(net_outputs[layer-1].T, deltas[layer])
                    weights[layer] += learning_rate * np.dot(net_outputs[layer-1].T, deltas[layer])

            # Regularization
            if reg == 'L2':
                weights -= lam * 2 * weights
            elif reg == 'L1':
                weights -= lam * np.sign(weights)

            weights_record.append(weights)

            # Get Model Predictions
            # predictions =  np.argmax(net_outputs[-1], axis=1) #TODO commented out hard model predictions

            # for key, dataset in datasets:
            #     # TODO update to predict for valid/test datasets
            #     predictions[key] = np.argmax(net_outputs[-1], axis=1)

            # Compute Accuracy

            # print(np.argmax(yTrain, axis=1)[:10], np.argmax(net_outputs[-1], axis=1)[:10])
            # sys.exit()

            train_accuracy.append(utilities.accuracy(yTrain, net_outputs[-1])) #TODO Changed predictions to netout bc 1hot
            # valid_accuracy.append(accuracy_softmax(yValid, predictions_valid))
            # test_accuracy.append(accuracy_softmax(yTest, predictions_test))

            # Compute Loss
            # train_loss_record.append(cross_entropy_loss_softmax(yTrain, predictions_train))
            # valid_loss_record.append(cross_entropy_loss_softmax(yValid, predictions_valid))
            # test_loss_record.append(cross_entropy_loss_softmax(yTest, predictions_test))

        # Format output
        # losses = {'train_loss': train_loss_record,
        #           'valid_loss': valid_loss_record,
        #           'test_loss': test_loss_record,
        #           }

        accuracies = {'train_acc': train_accuracy,
                      # 'valid_acc': valid_accuracy,
                      # 'test_acc': test_accuracy
                      }

        # return weights, losses, weights_record, accuracies
        return weights, accuracies

