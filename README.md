# Code for Image Recognition with Python
This repo contains an image recognition program which uses machine learning to train for a given dataset and can proceed to identify new images.

Currently it uses data from the MNIST database (http://yann.lecun.com/exdb/mnist/), it trains using 60,000 unique images and then proceeds to test itself using 10,000 image test set.

## Method:
- The imported images will first have their pixel grayscale value normalized (brought to a number between 0 & 1) and then the new matrix of normalized grayscale value for the pixels are flaten to 1 vector.
- The "correct" labels for the training data, which is also imported, are converted to binary index labels.
- The neural network is initialized (the depth and length of the neurons can be set by changing the neuron vector in `Main.py`) with random values for the weights and biases of every connection.
- A sigmoidal activation function is used to normalize the weight sum for each neuron. The final output neurons use a cost soft max function.
- Finally the training is initialized (the batch size, amount of iterations, and rate of learning can all be adjusted very easily in `Main.py`). The training uses stochastic gradient descent to find a local minima, this results in an accuracy of 92-94%.

### Notes:
The mathematical description for backprogations and my notes learning the principles of machine learning be found in `Notes.pdf`

## Here's a general description of each of the files:
`Main.py` - Main files from which data processing, initialization of neural net, and training is called.

`NeuralNetwork.py` - Code for the neural network and backpropagation.

`ProcessData.py` - Code to normalize, flatten, and convert labels.
