from multi_layer_preceptron import *


def main():
    data = utilities.read_data(train_size=None, test_size=None)

    # Initialize MLP Object
    mlp = MLP(input_units=785, hidden_layers=2, hidden_units=65, output_units=10, hidden_activation='tanh')

    mlp.set_mlp_data(data)

    # Train the Model on the Training Set
    num_epochs = 50
    mlp.train(max_epochs=num_epochs, learning_rate_init=0.0005, annealing=750.0, batch_size=128, shuffle=True, momentum=True)

    msg = "++++++++++++++++++++++++++++++++++++++++"
    print("Final test accuracy (", mlp.accuracies['test_acc'][-1], ")")
    if mlp.accuracies['valid_acc'][-1] >= 0.97:
        print(msg, "\nExperiment Stats: Time to train(", round(mlp.train_stats[0],4), ' secs) Epochs to train (', mlp.train_stats[1], ')\n', msg)
    else:
        print(msg, "\nTarget validation accuracy of 0.97 was never reached.")
        print("Experiment Stats: Time to train(", round(mlp.train_stats[0], 4), ' secs) Epochs to train (',
              mlp.train_stats[1], ') acc(', round(mlp.accuracies['valid_acc'][-1],4), ')\n', msg)

    # Display Accuracy and Loss over Epochs to Show Convergence of Model
    mlp.train_diagnostics('delete') # Problem (3E)

main()
