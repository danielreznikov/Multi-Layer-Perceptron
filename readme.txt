This file serves to instruct users on how to run our Mulit-Layer Preceptron code. 

Authors: Daniel Reznikov, Christian Koguchi.

There are 3 files:
	main.py : 
		This the entry point to the code. There are global variables at the top that serve configure hyper paramters.
				train_size / test_size 	- allow the user to subsample the input data for faster iteration. Default None uses full datasets.
				num_epochs 					- maximum training epochs to run the network on.
				learning_rates				- a list of candidate learning rates to try during grid search for model selection.
				annealing_rates			- a list of candidate T values to anneal the learning rate with, used in grid search.
				mlp.init()					- this line initializes the network topology.

	multi_layer_preceptron.py : 	
		This file is the main class definition for the network. 
		The class init method sets the layers, hidden nodes, and activation functions.
		mlp.train() delegates to train subroutines and this is where most of the time is spent during learning.
		This file also contains methods for numerical gradient checking and progress logging.

	utilities.py :
		This file contians utility methods for reading data, splitting datasets into train and validation, 
		and the functional forms of the various activation functions that we support.

To run the program:
	1. Download these files and the data/ directory.
	2. Set hyper paramaters in main.py (or leave them as is and what the network run on the last configuration).
	3. $python3 main.py
	4. Console output shows all relevant outputs.
