import numpy as np
import utilities
from matplotlib import pyplot as plt
from time import time
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

        self.sigmoid_weights = np.random.normal(loc=0, scale=1/np.sqrt(self.input_units), size=(self.input_units, self.hidden_units))
        self.softmax_weights = np.random.normal(loc=0, scale=1/np.sqrt(self.hidden_units), size=(self.hidden_units, self.output_units))

        self.accuracy_over_epoch = []
        self.loss_over_epoch = []



        if hidden_activation == 'sigmoid':
            self.hidden_activation = utilities.sigmoid_activation
        else:
            raise Exception("ERROR: unsuported activation function," + hidden_activation)

        if output_activation == 'softmax':
            self.output_activation = utilities.softmax_activation
        else:
            raise Exception("ERROR: unsuported activation function," + output_activation)


    def train(self, data, max_epochs=100, learning_rate_init=0.0001, lam=0, reg=None, annealing=100, batch_size=None, shuffle=False):

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

        weights = []
        net_inputs = []
        net_outputs = []
        deltas = []

        np.random.seed(2018)

        # Split data into train and validation
        indices = np.random.randint(0, xTrain.shape[0], xTrain.shape[0] // 10)
        xValid = xTrain[indices]
        yValid = yTrain[indices]
        np.delete(xTrain, indices, axis=0)
        np.delete(yTrain, indices, axis=0)

        datasets = {'train': xTrain, 'valid': xValid, 'test': xTest}

        # Mini-batching
        batches = []
        if shuffle:
            indices = np.random.permutation(xTrain.shape[0])
        else:
            indices = np.arange(xTrain.shape[0])

        if batch_size==None:
            batch_size = xTrain.shape[0]

        for batch_num in range(xTrain.shape[0] // batch_size):
            indxs = indices[batch_num*batch_size:(batch_num+1)*batch_size]
            batches.append((xTrain[indxs], yTrain[indxs]))

        # Initialize weights list indexing into each layer (looks good)
        for layer in range(self.num_layers):
            if layer == 0: # layer0
                weights.append(np.random.normal(loc=0, scale=1, size=(self.dims, self.hidden_units)))
            elif layer < self.num_layers - 1: # hidden layers
                weights.append(np.random.normal(loc=0, scale=1, size=(self.hidden_units, self.hidden_units)))
            else: # last layer
                weights.append(np.random.normal(loc=0, scale=1, size=(self.hidden_units, self.output_units)))

        # Iterate over epochs
        for epoch in range(max_epochs):
            for train, labels in batches:
                # learning_rate = learning_rate_init / (1 + epoch / annealing)
                learning_rate = learning_rate_init

                # Forward propogation to get model weights
                for layer in range(self.num_layers):
                    if layer == 0:
                        net_inputs.append(np.dot(train, weights[layer]))
                        net_outputs.append(self.hidden_activation(net_inputs[layer]))
                    elif layer < self.num_layers - 1:
                        net_inputs.append(np.dot(net_outputs[layer-1], weights[layer]))
                        net_outputs.append(self.hidden_activation(net_inputs[-1]))
                    else:
                        net_inputs.append(np.dot(net_outputs[layer-1], weights[layer]))
                        net_outputs.append(self.output_activation(net_inputs[-1]))

                # Backprop of partial gradients via deltas
                for layer in range(self.num_layers-1, -1, -1):
                    if layer == self.num_layers-1:
                        deltas.append(labels - net_outputs[-1])
                    else:
                        delta = net_outputs[layer]*(1 - net_outputs[layer]) * np.dot(deltas[-1], weights[layer+1].T)
                        deltas.append(delta)

                deltas = list(reversed(deltas))

                # Update weights via gradient descent
                for layer in range(self.num_layers):
                    if layer == 0:
                        grad = np.dot(train.T, deltas[layer])
                        weights[layer] += learning_rate * np.dot(train.T, deltas[layer])
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

    def train_non_modular(self, x, t, max_epochs=100, learning_rate_init=0.0001, annealing=100, batch_size=None, shuffle=False, gradient_checking=False):

        # Mini-batching
        batches = []
        num_samples = x.shape[0]

        if shuffle:
            indices = np.random.permutation(num_samples)
        else:
            indices = np.arange(num_samples)

        if batch_size == None:
            batch_size = num_samples

        for batch_num in range(num_samples // batch_size):
            indxs = indices[batch_num * batch_size:(batch_num + 1) * batch_size]
            batches.append((x[indxs], t[indxs]))

        # Initialize Weights as Random
        # W_hidden1 = np.random.normal(loc=0, scale=1/np.sqrt(self.input_units), size=(self.input_units, self.hidden_units))
        # W_hidden2 = np.random.normal(loc=0, scale=1/np.sqrt(hidden_units1), size=(hidden_units1, hidden_units2))
        # W_output = np.random.normal(loc=0, scale=1/np.sqrt(self.hidden_units), size=(self.hidden_units, self.output_units))

        # Initialize Weights as the Default Random Weight Specified in the Constructor
        W_hidden1 = self.sigmoid_weights
        W_output = self.softmax_weights

        # Start a Timer for Training
        strt = time()
        print("Training...")

        # Iterate
        for epoch in range(max_epochs):
            # Decay Learning Rate
            alpha = learning_rate_init / (1 + epoch / annealing)

            # Iterate Over Mini-Batches
            for x_batch, t_batch in batches:
                # Forward Prop

                # Layer 1
                net_input_h1 = np.dot(x_batch, W_hidden1)
                hidden_layer_out1 = utilities.sigmoid_activation(net_input_h1)

                # Layer 2
                # net_input_h2 = np.dot(hidden_layer_out1, W_hidden2)
                # hidden_layer_out2 = utilities.sigmoid_activation(net_input_h2)

                # Output Layer
                net_input_o = np.dot(hidden_layer_out1, W_output)
                y = utilities.softmax_activation(net_input_o)

                # Back Prop (deltas)
                delta_output = (t_batch - y)
                # delta_hidden2 = utilities.sigmoid_activation(net_input_h2) * (1 - utilities.sigmoid_activation(net_input_h2)) * np.dot(delta_output, W_output.T)
                delta_hidden1 = utilities.sigmoid_activation(net_input_h1) * (1 - utilities.sigmoid_activation(net_input_h1)) * np.dot(delta_output, W_output.T)

                if gradient_checking == True:

                    # Tune which weight and which layer for gradient checking here!
                    weight_indices = (3, 4)
                    layer_tag = 'output'

                    if layer_tag == 'output':

                        numerical_grad = self.get_numerical_gradient(x_batch, t_batch, layer_tag, weight_indices)
                        backprop_grad = - np.dot(hidden_layer_out1.T, delta_output)[weight_indices]

                        print('Numerical Gradient:', numerical_grad)
                        print('Backprop Gradient:', backprop_grad)
                        print('Difference between Gradient:', numerical_grad - backprop_grad)

                    elif layer_tag == 'hidden':

                        numerical_grad = self.get_numerical_gradient(x_batch, t_batch, layer_tag, weight_indices)
                        backprop_grad = - np.dot(x_batch.T, delta_hidden1)[weight_indices]

                        print('Numerical Gradient:', numerical_grad)
                        print('Backprop Gradient:', backprop_grad)
                        print('Difference between Gradient:', numerical_grad - backprop_grad)

                    else:
                        print('Invalid Tag')
                        sys.exit()


                # Gradient Descent
                W_output = W_output + alpha * np.dot(hidden_layer_out1.T, delta_output)
                # W_hidden2 = W_hidden2 + alpha * np.dot(hidden_layer_out1.T, delta_hidden2)
                W_hidden1 = W_hidden1 + alpha * np.dot(x_batch.T, delta_hidden1)

                # Store the Model
                self.sigmoid_weights = W_hidden1
                self.softmax_weights = W_output


            predictions = self.evaluate(x)

            # Store Accuracies over Epochs
            accuracy = utilities.accuracy(t, predictions)
            self.accuracy_over_epoch.append(accuracy)

            # Store Cross-Entropy Loss over Epochs
            loss = utilities.cross_entropy_loss(t, predictions)
            self.loss_over_epoch.append(loss)

            if epoch % 10 == 0:
                print("\nEpoch:", epoch)
                print("\tAccuracy:", accuracy)
                print("\tLoss:", loss)


        print('\n\nTraining Done! Took', time() - strt, " secs.")
        print('Final Training Accuracy: ', self.accuracy_over_epoch[-1])

        # Store the Model
        # self.sigmoid_weights = W_hidden1
        # self.softmax_weights = W_output


        return None

    def get_numerical_gradient(self, x, t, weights, weight_indices):
        """ Computes the numerical gradient for a given input with respect to a single weight specified by the tuple in
            weight_indices """

        epsilon = 0.00001

        if weights == 'hidden':

            sigmoid_weights_plus = np.copy(self.sigmoid_weights)
            sigmoid_weights_plus[weight_indices] += epsilon

            sigmoid_weights_minus = np.copy(self.sigmoid_weights)
            sigmoid_weights_minus[weight_indices] -= epsilon

            pred_plus = self.evaluate(x, sigmoid_weights=sigmoid_weights_plus)
            pred_minus = self.evaluate(x, sigmoid_weights=sigmoid_weights_minus)

            loss_plus = utilities.cross_entropy_loss(t, pred_plus)
            loss_minus = utilities.cross_entropy_loss(t, pred_minus)

            gradient = (loss_plus - loss_minus)/ (2 * epsilon)

            return gradient

        elif weights == 'output':
            softmax_weights_plus = np.copy(self.softmax_weights)
            softmax_weights_plus[weight_indices] += epsilon

            softmax_weights_minus = np.copy(self.softmax_weights)
            softmax_weights_minus[weight_indices] -= epsilon

            pred_plus  = self.evaluate(x, sigmoid_weights=self.sigmoid_weights, softmax_weights=softmax_weights_plus)
            pred_minus = self.evaluate(x, sigmoid_weights=self.sigmoid_weights, softmax_weights=softmax_weights_minus)

            loss_plus = utilities.cross_entropy_loss(t, pred_plus)
            loss_minus = utilities.cross_entropy_loss(t, pred_minus)

            gradient = (loss_plus - loss_minus) / (2 * epsilon)

            return gradient

        else:
            print('Invalid Weights.  Input hidden or output.')

    def evaluate(self, inputs, sigmoid_weights=None, softmax_weights=None):
        """ Perform Forward Prop with Curren Saved Weights
            Outputs Model Predictions for Data """

        if type(sigmoid_weights) == type(np.array([0])):
            # Validates Inputs
            assert sigmoid_weights.shape[0] == inputs.shape[1], 'Invalid Dimensions with given Weights'

            # Layer 1
            net_input_h1 = np.dot(inputs, sigmoid_weights)
            hidden_layer_out1 = utilities.sigmoid_activation(net_input_h1)

        else:
            # Layer 1
            net_input_h1 = np.dot(inputs, self.sigmoid_weights)
            hidden_layer_out1 = utilities.sigmoid_activation(net_input_h1)

        if type(softmax_weights) == type(np.array([0])):

            # Validates Inputs
            assert softmax_weights.shape[0] == hidden_layer_out1.shape[1], 'Invalid Dimensions with given Weights'

            # Output Layer
            net_input_o = np.dot(hidden_layer_out1, softmax_weights)
            y = utilities.softmax_activation(net_input_o)

        else:
            # Output Layer
            net_input_o = np.dot(hidden_layer_out1, self.softmax_weights)
            y = utilities.softmax_activation(net_input_o)

        return y

    def disp_train_accuracy(self):
        num_epochs = len(self.accuracy_over_epoch)

        plt.plot(np.arange(num_epochs), self.accuracy_over_epoch)
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy Over Epochs')
        plt.show()

        return None

    def disp_train_loss(self):
        num_epochs = len(self.loss_over_epoch)

        plt.plot(np.arange(num_epochs), self.loss_over_epoch)
        plt.xlabel('Epochs')
        plt.ylabel('Training Loss')
        plt.title('Training Cross-Entropy Loss Over Epochs')
        plt.show()

        return None

    def train_diagnostics(self):

        plt.subplot(2, 1, 1)
        num_epochs = len(self.accuracy_over_epoch)

        plt.plot(np.arange(num_epochs), self.accuracy_over_epoch)
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Accuracy Vs. Epochs')

        plt.subplot(2, 1, 2)
        plt.plot(np.arange(num_epochs), self.loss_over_epoch)
        plt.xlabel('Epochs')
        plt.ylabel('Training Loss')
        plt.title('Cross-Entropy Loss Vs. Epochs')

        plt.show()

        return None





















