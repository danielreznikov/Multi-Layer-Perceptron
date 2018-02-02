from multi_layer_preceptron import *

def main():
    data = utilities.read_data(train_size=None, test_size=None)

    # Train the Model on the Training Set
    num_epochs = 50

    learning_rates = [0.001, 0.003, 0.007, 0.0005, 0.0003, 0.00075, 0.00001, 0.00003, 0.00005]
    annealing = [100, 200, 300, 450, 600, 750]

    msg = "++++++++++++++++++++++++++++++++++++++++"

    for eta in learning_rates:
        for anneal in annealing:

            mlp = MLP(input_units=785, hidden_layers=1, hidden_units=65, output_units=10, hidden_activation='relu')
            mlp.set_mlp_data(data)

            print("\n", msg, "Learning for eta = ", eta, " and annealing = ", anneal, '\n')

            mlp.train(max_epochs=num_epochs, learning_rate_init=eta, annealing=anneal, batch_size=128, shuffle=True, momentum=True)

            # print("Final test accuracy (", mlp.accuracies['test_acc'][-1], ")")
            if mlp.accuracies['valid_acc'][-1] >= 0.97:
                print(msg, "\nExperiment Stats: Time to train(", round(mlp.train_stats[0],4), ' secs) Epochs to train (', mlp.train_stats[1], ')\n', msg)
            else:
                print(msg, "\nTarget validation accuracy of 0.97 was never reached.")
                print("Experiment Stats: Time to train(", round(mlp.train_stats[0], 4), ' secs) Epochs to train (',
                      mlp.train_stats[1], ') Valid acc(', round(mlp.accuracies['valid_acc'][-1],4), ')\n', msg)
                mlp.train_stats = None

    # Display Accuracy and Loss over Epochs to Show Convergence of Model
    # mlp.train_diagnostics('delete') # Problem (3E)

main()
