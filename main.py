from multi_layer_preceptron import *


def main():
    data = utilities.read_data(train_size=None, test_size=None)

    # Initialize MLP Object
    mlp = MLP(input_units=785, hidden_layers=1, hidden_units=65, output_units=10, hidden_activation='tanh')

    mlp.set_mlp_data(data)

    # Train the Model on the Training Set
    num_epochs = 15
    mlp.train(max_epochs=num_epochs, learning_rate_init=0.0001, annealing=num_epochs * 10.7, batch_size=128, shuffle=True)

    msg = "++++++++++++++++++++++++++++++++++++++++"
    try:
        print(msg, "\nExperiment Stats: Time to train(", round(mlp.train_stats[0],4), ' secs) Epochs to train (', mlp.train_stats[1], ')\n', msg)
    except TypeError:
        print(msg, "\nTarget validation accuracy of 0.97 was never reached.\n", msg)

    # Display Accuracy and Loss over Epochs to Show Convergence of Model
    mlp.train_diagnostics('delete') # Problem (3E)

main()
