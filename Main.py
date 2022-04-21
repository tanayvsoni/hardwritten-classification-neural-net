import NerualNetwork as nn
import ProcessData as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import mnist

# Storing mnist image data
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# For Visual only, showing grid of images for the test values
columns = 10    
rows = 10
fig=plt.figure(figsize=(8, 8))

for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(test_X[i-1], cmap=plt.get_cmap('gray'))
plt.show()

# Processing Data to make it usable
training_data = pd.processData().normFlat(train_X)
test_data = pd.processData().normFlat(test_X)

# Designating Neurons in each layer
neurons = [training_data.shape[1],30,30,30,10]

# Converting labels into binary labels
training_labels = pd.processData().oneHot_labels(train_X,neurons,train_y)
test_labels = pd.processData().oneHot_labels(test_X,neurons,test_y)

# Initializing Neural Network
net = nn.NeuralNetwork(neurons)

# Initial Accuracy and Cost display, should be around 10%
net.accuracy(test_data,test_labels,rows,columns)
net.average_Cost(test_data,test_labels)

# # Training the network
net.trainNetwork(training_data,training_labels,10,3,3)

# # # Final display of accuracy and cost for the test values
net.accuracy(test_data,test_labels,rows,columns)
net.average_Cost(test_data,test_labels)