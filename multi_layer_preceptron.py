import numpy as np
import utilities
from matplotlib import pyplot as plt
from time import time
import sys

class MLP(object):
    """This is our neural network object."""

    def __init__(self, input_units, hidden_layers, hidden_units, output_units, hidden_activation='sigmoid', output_activation='softmax'):
        assert(hidden_layers == 1 or hidden_layers ==2)

        self.input_units = input_units
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.dims = 785
        self.num_layers = hidden_layers + 1
        self.train_stats = None

        self.losses = {'train_loss': [], 'valid_loss': [], 'test_loss': []}
        self.accuracies = {'train_acc': [],'valid_acc': [],'test_acc': []}

        if hidden_layers == 1:
            self.weights = [
                # Experiment 3E, 4A, 4B
                # np.random.uniform(low=-0.5, high=0.5, size=(self.input_units, self.hidden_units-1)),
                # np.random.uniform(low=-0.5, high=0.5, size=(self.hidden_units, self.output_units))

                np.random.normal(loc=0, scale=1/np.sqrt(self.input_units),  size=(self.input_units, self.hidden_units-1)),
                np.random.normal(loc=0, scale=1/np.sqrt(self.hidden_units), size=(self.hidden_units, self.output_units))
            ]
        elif hidden_layers == 2:
            self.weights = [
                np.random.normal(loc=0, scale=1 / np.sqrt(self.input_units),  size=(self.input_units, self.hidden_units  - 1)),
                np.random.normal(loc=0, scale=1 / np.sqrt(self.hidden_units), size=(self.hidden_units, self.hidden_units - 1)),
                np.random.normal(loc=0, scale=1 / np.sqrt(self.hidden_units), size=(self.hidden_units, self.output_units))
            ]

        self.best_model = [sys.maxsize, self.weights]

        if hidden_activation == 'sigmoid':
            self.hidden_activation = utilities.sigmoid_activation
        elif hidden_activation == 'tanh':
            self.hidden_activation = utilities.tanh_activation
        elif hidden_activation == 'relu':
            self.hidden_activation = utilities.relu_activation
        else:
            raise Exception("ERROR: unsuported activation function," + hidden_activation)

        if output_activation == 'softmax':
            self.output_activation = utilities.softmax_activation
        else:
            raise Exception("ERROR: unsuported activation function," + output_activation)


    def train_hidden1(self, max_epochs=100, learning_rate_init=0.0001, annealing=100, batch_size=None, shuffle=False, gradient_checking=False, momentum=False):
        assert (self.hidden_layers == 1)

        # Start a Timer for Training
        strt = time()
        print("Training...")


        W_hidden1 = self.weights[0]
        W_output = self.weights[1]

        momentum1 = 0
        momentum2 = 0

        # Iterate
        for epoch in range(max_epochs):
            # Decay Learning Rate
            alpha = learning_rate_init / (1 + epoch / annealing)

            # Mini-batching
            batches = []
            num_samples = self.xTrain.shape[0]

            if shuffle:
                indices = np.random.permutation(num_samples)
            else:
                indices = np.arange(num_samples)

            if batch_size == None:
                batch_size = num_samples

            for batch_num in range(num_samples // batch_size):
                indxs = indices[batch_num * batch_size:(batch_num + 1) * batch_size]
                batches.append((self.xTrain[indxs], self.yTrain[indxs]))

            # Iterate Over Mini-Batches
            for x_batch, t_batch in batches:
                # Forward Prop
                net_input_h1 = np.dot(x_batch, W_hidden1)
                hidden_layer_out1 = self.hidden_activation(net_input_h1)
                hidden_layer_out1 = np.insert(hidden_layer_out1, 0, 1, axis=1)

                # Output Layer
                net_input_o = np.dot(hidden_layer_out1, W_output)
                y = utilities.softmax_activation(net_input_o)

                # Backprop (deltas)
                delta_output = (t_batch - y)
                if self.hidden_activation == utilities.sigmoid_activation:
                    delta_hidden1 = utilities.sigmoid_activation(net_input_h1) * (1 - utilities.sigmoid_activation(net_input_h1)) * np.dot(delta_output, W_output[1:,:].T)
                elif self.hidden_activation == utilities.tanh_activation:
                    delta_hidden1 = (2/3) * (1.7159 - 1.0/1.7159*np.power(utilities.tanh_activation(net_input_h1), 2)) * np.dot(delta_output, W_output[1:,:].T)
                elif self.hidden_activation == utilities.relu_activation:
                    delta_hidden1 =  (net_input_h1 >= 0) * np.dot(delta_output, W_output[1:,:].T) # fix relu here
                else:
                    raise Exception("ERROR: Not supported hidden activation function!")

                if gradient_checking == True:
                    # Tune which weight and which layer for gradient checking here!
                    weight_indices = (0, 3)
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
                if momentum == True:

                    current_grad1 = alpha * np.dot(hidden_layer_out1.T, delta_output)
                    current_grad2 = alpha * np.dot(x_batch.T, delta_hidden1)

                    W_output = W_output + current_grad1 + (0.9 * momentum1)
                    W_hidden1 = W_hidden1 + current_grad2 + (0.9 * momentum2)

                    momentum1 = current_grad1 + (0.9 * momentum1)
                    momentum2 = current_grad2 + (0.9 * momentum2)

                else:
                    W_output = W_output + alpha * np.dot(hidden_layer_out1.T, delta_output)
                    W_hidden1 = W_hidden1 + alpha * np.dot(x_batch.T, delta_hidden1)

                # Store the Model
                self.weights[0] = W_hidden1
                self.weights[1] = W_output

            # Get model predictions
            predictions_train = self.get_model_predictions(self.xTrain)
            predictions_valid = self.get_model_predictions(self.xValid)
            predictions_test = self.get_model_predictions(self.xTest)

            # Compute accuracies over epochs
            self.accuracies['train_acc'].append(utilities.accuracy(self.yTrain, predictions_train))
            self.accuracies['valid_acc'].append(utilities.accuracy(self.yValid, predictions_valid))
            self.accuracies['test_acc'].append(utilities.accuracy(self.yTest, predictions_test))

            # Code Profiling
            if not self.train_stats and self.accuracies['valid_acc'][-1] >= 0.97:
                self.train_stats = (time() - strt, epoch)

            # Cross-Entropy Loss over epochs
            self.losses['train_loss'].append(utilities.cross_entropy_loss(self.yTrain, predictions_train))
            self.losses['valid_loss'].append(utilities.cross_entropy_loss(self.yValid, predictions_valid))
            self.losses['test_loss'].append(utilities.cross_entropy_loss(self.yTest, predictions_test))

            # Update best model so far
            if self.losses['valid_loss'][-1] < self.best_model[0]:
                self.best_model[0] = self.losses['valid_loss'][-1]
                self.best_model[1] = self.weights

            # Early Stopping
            if epoch > 4 and utilities.early_stopping(self.losses['valid_loss']):
                print("\tEarly Stopping (3 consecutive increases) at epoch =", epoch)
                break
            elif epoch > 2 and np.abs(self.losses['valid_loss'][-1] - self.losses['valid_loss'][-2]) < 0.00001:
                print("\tEarly Stopping, error below epsilon.", self.losses['valid_loss'][-1])
                break

            # Debug statements
            if epoch % 10 == 0:
                print("Epoch:", epoch)
                # print("\tTraining Accuracy:", self.accuracies['train_acc'][-1])
                # print("\tValidation Accuracy:", self.accuracies['valid_acc'][-1])
                # print("\tTest Accuracy:", self.accuracies['train_acc'][-1])
                # print("\tLoss:", self.losses['train_loss'][-1])

        if not self.train_stats:
            self.train_stats = (time() - strt, epoch)

        print('\n\nTraining Done! Took', round(time() - strt, 3), " secs.")
        # print('Final Training Accuracy: ', self.accuracies['train_acc'][-1], " in ", epoch, " epochs.")
        # print('Final Validation Accuracy: ', self.accuracies['valid_acc'][-1], " in ", epoch, " epochs.")
        # print('Final Test Accuracy: ', self.accuracies['test_acc'][-1], " in ", epoch, " epochs.\n")


        return 1

    def train_hidden2(self, max_epochs=100, learning_rate_init=0.0001, annealing=100, batch_size=None, shuffle=False, momentum=False):
        assert (self.hidden_layers == 2)

        # Start a Timer for Training
        strt = time()
        print("Training...")

        W_hidden1 = self.weights[0]
        W_hidden2 = self.weights[1]
        W_output = self.weights[2]

        momentum1 = 0
        momentum2 = 0
        momentum3 = 0

        # Iterate
        for epoch in range(max_epochs):
            # Decay Learning Rate
            alpha = learning_rate_init / (1 + epoch / annealing)

            # Mini-batching
            batches = []
            num_samples = self.xTrain.shape[0]

            if shuffle:
                indices = np.random.permutation(num_samples)
            else:
                indices = np.arange(num_samples)

            if batch_size == None:
                batch_size = num_samples

            for batch_num in range(num_samples // batch_size):
                indxs = indices[batch_num * batch_size:(batch_num + 1) * batch_size]
                batches.append((self.xTrain[indxs], self.yTrain[indxs]))

            # Iterate Over Mini-Batches
            for x_batch, t_batch in batches:
                # Forward Prop
                net_input_h1 = np.dot(x_batch, W_hidden1)
                hidden_layer_out1 = utilities.sigmoid_activation(net_input_h1)
                hidden_layer_out1 = np.insert(hidden_layer_out1, 0, 1, axis=1)

                net_input_h2 = np.dot(hidden_layer_out1, W_hidden2)
                hidden_layer_out2 = utilities.sigmoid_activation(net_input_h2)
                hidden_layer_out2 = np.insert(hidden_layer_out2, 0, 1, axis=1)

                net_input_o = np.dot(hidden_layer_out2, W_output)
                y = utilities.softmax_activation(net_input_o)

                # Back Prop (deltas)
                delta_output = (t_batch - y)
                if self.hidden_activation == utilities.sigmoid_activation:
                    delta_hidden2 = utilities.sigmoid_activation(net_input_h2) * (1 - utilities.sigmoid_activation(net_input_h2)) * np.dot(delta_output, W_output[1:,:].T)
                    delta_hidden1 = utilities.sigmoid_activation(net_input_h1) * (1 - utilities.sigmoid_activation(net_input_h1)) * np.dot(delta_hidden2, W_hidden2[1:,:].T)
                elif self.hidden_activation == utilities.tanh_activation:
                    delta_hidden2 = (2/3) * (1.7159 - 1.0/1.7159*np.power(utilities.tanh_activation(net_input_h2), 2)) * np.dot(delta_output, W_output[1:,:].T)
                    delta_hidden1 = (2/3) * (1.7159 - 1.0/1.7159*np.power(utilities.tanh_activation(net_input_h1), 2)) * np.dot(delta_hidden2, W_hidden2[1:,:].T)
                else:
                    raise Exception("ERROR: Not supported hidden activation function!")

                # Gradient Descent
                if momentum == True:
                    current_grad1 = alpha * np.dot(hidden_layer_out2.T, delta_output)
                    current_grad2 = alpha * np.dot(hidden_layer_out1.T, delta_hidden2)
                    current_grad3 = alpha * np.dot(x_batch.T, delta_hidden1)


                    W_output = W_output + current_grad1 + (0.9 * momentum1)
                    W_hidden2 = W_hidden2 + current_grad2 + (0.9 * momentum2)
                    W_hidden1 = W_hidden1 + current_grad3 + (0.9 * momentum3)

                    momentum1 = current_grad1 + (0.9 * momentum1)
                    momentum2 = current_grad2 + (0.9 * momentum2)
                    momentum3 = current_grad3 + (0.9 * momentum3)

                else:
                    W_output = W_output + alpha * np.dot(hidden_layer_out2.T, delta_output)
                    W_hidden2 = W_hidden2 + alpha * np.dot(hidden_layer_out1.T, delta_hidden2)
                    W_hidden1 = W_hidden1 + alpha * np.dot(x_batch.T, delta_hidden1)


                # Store the Model
                self.weights[0] = W_hidden1
                self.weights[1] = W_hidden2
                self.weights[2] = W_output

            # Get model predictions
            predictions_train = self.get_model_predictions(self.xTrain)
            predictions_valid = self.get_model_predictions(self.xValid)
            predictions_test = self.get_model_predictions(self.xTest)

            # Compute accuracies over epochs
            self.accuracies['train_acc'].append(utilities.accuracy(self.yTrain, predictions_train))
            self.accuracies['valid_acc'].append(utilities.accuracy(self.yValid, predictions_valid))
            self.accuracies['test_acc'].append(utilities.accuracy(self.yTest, predictions_test))

            # Code Profiling
            if not self.train_stats and self.accuracies['valid_acc'][-1] >= 0.97:
                self.train_stats = (time() - strt, epoch)

            # Cross-Entropy Loss over epochs
            self.losses['train_loss'].append(utilities.cross_entropy_loss(self.yTrain, predictions_train))
            self.losses['valid_loss'].append(utilities.cross_entropy_loss(self.yValid, predictions_valid))
            self.losses['test_loss'].append(utilities.cross_entropy_loss(self.yTest, predictions_test))

            # Update best model so far
            if self.losses['valid_loss'][-1] < self.best_model[0]:
                self.best_model[0] = self.losses['valid_loss'][-1]
                self.best_model[1] = self.weights

            # Early Stopping
            # if epoch > 4 and utilities.early_stopping(self.losses['valid_loss']):
            #     print("\tEarly Stopping at epoch =", epoch)
            #     break
            # elif epoch > 2 and np.abs(self.losses['valid_loss'][-1] - self.losses['valid_loss'][-2]) < 0.00001:
            #     print("\tEarly Stopping, error below epsilon.", self.losses['valid_loss'][-1])
                # break

            # Debug statements
            if epoch % 10 == 0:
                print("Epoch:", epoch)
                # print("\tAccuracy:", self.accuracies['train_acc'][-1])
                # print("\tLoss:", self.losses['train_loss'][-1])

        if not self.train_stats:
            self.train_stats = (time() - strt, epoch)

        print('\n\nTraining Done! Took', round(time() - strt, 3), " secs.")
        # print('Final Training Accuracy: ', self.accuracies['train_acc'][-1], " in ", epoch, " epochs.")

        return 1

    def train(self, max_epochs=100, learning_rate_init=0.0001, annealing=100, batch_size=None, shuffle=False, gradient_checking=False, momentum=False):
        if self.hidden_layers == 1:
            self.train_hidden1(max_epochs=max_epochs, learning_rate_init=learning_rate_init, annealing=annealing,
                               batch_size=batch_size, shuffle=shuffle, gradient_checking=gradient_checking, momentum=momentum)

        elif self.hidden_layers == 2:
            self.train_hidden2(max_epochs=max_epochs, learning_rate_init=learning_rate_init, annealing=annealing,
                               batch_size=batch_size, shuffle=shuffle, momentum=momentum)

        return

    def get_numerical_gradient(self, x, t, layer_tag, weight_indices):
        """ Computes the numerical gradient for a given input with respect to a single weight specified by the tuple in
            weight_indices """
        assert (self.hidden_layers == 1)

        def evaluate(inputs, sigmoid_weights=None, softmax_weights=None):
            """ Perform Forward Prop with Curren Saved Weights
                Outputs Model Predictions for Data """
            assert (self.hidden_layers == 1)

            if type(sigmoid_weights) == type(np.array([0])):
                # Validates Inputs
                assert sigmoid_weights.shape[0] == inputs.shape[1], 'Invalid Dimensions with given Weights'

                # Layer 1
                net_input_h1 = np.dot(inputs, sigmoid_weights)
                hidden_layer_out1 = utilities.sigmoid_activation(net_input_h1)

            else:
                # Layer 1
                net_input_h1 = np.dot(inputs, self.weights[0])
                hidden_layer_out1 = utilities.sigmoid_activation(net_input_h1)

            hidden_layer_out1 = np.insert(hidden_layer_out1, 0, 1, axis=1)  # prepend bias term

            if type(softmax_weights) == type(np.array([0])):

                # Validates Inputs
                assert softmax_weights.shape[0] == hidden_layer_out1.shape[1], 'Invalid Dimensions with given Weights'

                # Output Layer
                net_input_o = np.dot(hidden_layer_out1, softmax_weights)
                y = utilities.softmax_activation(net_input_o)

            else:
                # Output Layer
                net_input_o = np.dot(hidden_layer_out1, self.weights[1])
                y = utilities.softmax_activation(net_input_o)

            return y

        epsilon = 0.01

        if layer_tag == 'hidden':

            sigmoid_weights_plus = np.copy(self.weights[0])
            sigmoid_weights_plus[weight_indices] += epsilon

            sigmoid_weights_minus = np.copy(self.weights[0])
            sigmoid_weights_minus[weight_indices] -= epsilon

            pred_plus = evaluate(x, sigmoid_weights=sigmoid_weights_plus)
            pred_minus = evaluate(x, sigmoid_weights=sigmoid_weights_minus)

            loss_plus = utilities.cross_entropy_loss(t, pred_plus)
            loss_minus = utilities.cross_entropy_loss(t, pred_minus)

            gradient = (loss_plus - loss_minus)/ (2 * epsilon)

            return gradient

        elif layer_tag == 'output':
            softmax_weights_plus = np.copy(self.weights[1])
            softmax_weights_plus[weight_indices] += epsilon

            softmax_weights_minus = np.copy(self.weights[1])
            softmax_weights_minus[weight_indices] -= epsilon

            pred_plus  = evaluate(x, sigmoid_weights=self.weights[0], softmax_weights=softmax_weights_plus)
            pred_minus = evaluate(x, sigmoid_weights=self.weights[0], softmax_weights=softmax_weights_minus)

            loss_plus = utilities.cross_entropy_loss(t, pred_plus)
            loss_minus = utilities.cross_entropy_loss(t, pred_minus)

            gradient = (loss_plus - loss_minus) / (2 * epsilon)

            return gradient

        else:
            print('Invalid Weights.  Input hidden or output.')

    def get_model_predictions(self, x):
        """ Perform Forward Prop with Curren Saved Weights
            Outputs Model Predictions for Data """

        if self.hidden_layers == 1:
            weights_hidden = self.best_model[1][0]
            weights_output = self.best_model[1][1]

            net_input_hidden = np.dot(x, weights_hidden)
            net_output_hidden = self.hidden_activation((net_input_hidden))
            net_output_hidden = np.insert(net_output_hidden, 0, 1, axis=1)

            predictions = self.output_activation(np.dot(net_output_hidden, weights_output))

        elif self.hidden_layers == 2:
            weights_hidden1 = self.best_model[1][0]
            weights_hidden2 = self.best_model[1][1]
            weights_output = self.best_model[1][2]

            net_input_hidden1 = np.dot(x, weights_hidden1)
            net_output_hidden1 = self.hidden_activation((net_input_hidden1))
            net_output_hidden1 = np.insert(net_output_hidden1, 0, 1, axis=1)

            net_input_hidden2 = np.dot(net_output_hidden1, weights_hidden2)
            net_output_hidden2 = self.hidden_activation((net_input_hidden2))
            net_output_hidden2 = np.insert(net_output_hidden2, 0, 1, axis=1)

            predictions = self.output_activation(np.dot(net_output_hidden2, weights_output))

        return predictions


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

    def train_diagnostics(self, prob):

        plt.subplot(2, 1, 1)
        num_epochs = len(self.accuracies['train_acc'])

        plt.plot(np.arange(num_epochs), self.accuracies['train_acc'])
        plt.plot(np.arange(num_epochs), self.accuracies['valid_acc'])
        plt.plot(np.arange(num_epochs), self.accuracies['test_acc'])
        plt.xlabel('Epochs')
        plt.ylabel('Percent Accuracy')
        plt.title('Accuracy Vs. Epochs')
        plt.legend(['Training', 'Validation', 'Test'], loc='lower right')

        plt.subplot(2, 1, 2)
        plt.plot(np.arange(num_epochs), self.losses['train_loss'])
        plt.plot(np.arange(num_epochs), self.losses['valid_loss'])
        plt.plot(np.arange(num_epochs), self.losses['test_loss'])
        plt.xlabel('Epochs')
        plt.ylabel('Normalized Loss')
        plt.title('Cross-Entropy Loss Vs. Epochs')
        plt.legend(['Training', 'Validation', 'Test'], loc='upper right')

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)

        plt.savefig('report_images/' + prob + '.png')
        plt.show()

    def set_mlp_data(self, data):

        xTrain, yTrain, xValid, yValid, xTest, yTest = utilities.split_data(data)

        self.xTrain = xTrain
        self.yTrain = yTrain

        self.xValid = xValid
        self.yValid = yValid

        self.xTest = xTest
        self.yTest = yTest

    def set_mlp_data_pca(self, data):

        xTrain, yTrain, xValid, yValid, xTest, yTest = utilities.split_data(data)


        self.xTrain = self.pca(xTrain)
        self.yTrain = yTrain

        self.xValid = self.pca(xValid)
        self.yValid = yValid

        self.xTest = self.pca(xTest)
        self.yTest = yTest

    def pca(self, data):
        sample_mean = np.mean(data, axis=0)
        mean_centered = data - sample_mean
        cov = np.dot(mean_centered.T, mean_centered)/mean_centered.shape[0]
        eig_vals, principal_comp = np.linalg.eigh(cov) # ascending order

        # plt.plot(np.flipud(eig_vals))
        # plt.title('Principal Component Number Vs. Eigenvalue')
        # plt.xlabel('Principal Component Number')
        # plt.ylabel('Eigenvalue')
        # plt.show()

        return principal_comp[:, :100]









