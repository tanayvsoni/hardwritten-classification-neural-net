import numpy as np
from keras.datasets import mnist

class NeuralNetwork:

    def __init__(self,neurons):
        """
        Initializes weights and biases, using random values
        Creates activation array to store values for neurons
        """
        # Weights
        weightShape = [(a,b) for a,b in zip(neurons[1:],neurons[:-1])]
        self.weights = [np.random.standard_normal(s)/np.sqrt(s[1]) for s in weightShape]
        #Biases
        self.biases = [np.zeros((s,1)) for s in neurons[1:]]

        #Storing size of output layer
        self.outputLayer = neurons[-1]
        #Stores # of neuron layers
        self.numNeurons_layers = len(neurons)
        
        #Creates array to store values neurons
        #Example: neurons = [784,16,16,10]
        #         activations = [[784,1],[16,784],[16,16].[10,16]]
        self.activations = np.array([np.zeros(s) for s in neurons],dtype='object')

    def neuronVal(self,inputNeurons):
        """
        Will calculate and stores values of the activation of each neuron
        For single images
        """
        #Stores initial values for inputNeurons of the Image
        self.activations[0] = inputNeurons

        #Runs through a for loop, for all layer expcept last output layer
        for i in range(self.numNeurons_layers - 1):
            #Calculates the value for each neuron, [W][a] + [b]
            z = np.matmul(self.weights[i],self.activations[i]) + self.biases[i]
            #Stores values of the neurons into our array 
            self.activations[i+1] = self.sigmoid(z)
        #Returns the output layer only
        return self.activations[-1]

    def accuracy(self,inputData,labels,r,c):
        # Calculates the predictions values
        predictions = self.neuronVal(inputData)

        # Setting up temp varibles to be used to display a portion of the predictions for the data
        # Dependent on r,c 
        temp = np.zeros((r,c))
        iteration = 0

        for i in range(0,r):
            for j in range(0,c):
                temp[i][j] = np.argmax(predictions[iteration])
                iteration += 1
        print(temp)

        # Calculating number of predictions which are correct and than printing them
        num_correct = sum([np.argmax(p) == np.argmax(l) for p, l in zip(predictions, labels)])
        print("{0}/{1} accuracy: {2}%".format(num_correct, inputData.shape[0], (num_correct / inputData.shape[0]) * 100))

    def average_Cost(self,inputData,labels):
        predictions = self.neuronVal(inputData)

        average_cost = sum([self.cost(p, l) for p, l in zip(predictions, labels)]) / len(inputData)
        print("average cost: {0}".format(average_cost))

    def backProg(self,inputNeurons,label):
        """
        Calculating changes in weights & biases to minimize the 
        cost function for a single image
        """
        #Creating delta arrays in the sample shape as weights & biases
        biasDeltas = [np.zeros(b.shape) for b in self.biases]
        weightDeltas = [np.zeros(w.shape) for w in self.weights]
        #run image through the network once, so we have values for each neuron
        self.neuronVal(inputNeurons)
        """
        The delta of the cost function can be taken w/r either weights or biases
        Below we are calculating both partial derivatives of cost
        """
        #From 3b1b's theory video on neural networks~
        L = -1

        #This is the (derivative of activation w/r to z) * (derivative of cost w/r activation)
        partialDeltas = self.costSoftMax_derivative(self.activations[L],label)*self.sigmoid_derivative(self.activations[L])
        #b(L) = 1
        biasDeltas[L] = partialDeltas # *1 but this is unnessary
        #w(L) = a(L-1)
        weightDeltas[L] = np.dot(partialDeltas, self.activations[L-1].T)
        
        #Count backwards from the 2nd last layer to the 2nd layer (Avoid input & output layers)
        while L > -self.numNeurons_layers + 1:
            #We need to update the partialDeltas for the a previous layer's activation
            #To do this, take the partialDeltas and multiple them by the derivative of z w/r a(L-1)
            previousDeltas = np.dot(self.weights[L].T, partialDeltas)
            #Same process as above, but now we are doing this for all the hidden layers
            partialDeltas = previousDeltas * self.sigmoid_derivative(self.activations[L-1])
            biasDeltas[L-1] = partialDeltas
            weightDeltas[L-1] = np.dot(partialDeltas, self.activations[L-2].T)
            
            L -= 1

        return biasDeltas, weightDeltas

    def updateBatch(self,batch,rateOfLearning):
        """
        Calculate the change in direction given a batch
        of results and the delta values for weights and biases
        """
        #Initialize the gradient arrays
        biasGradient = [np.zeros(b.shape) for b in self.biases]
        weightGradient = [np.zeros(w.shape) for w in self.weights]

        for inputNeurons,label in batch:
            #Calculating the changes in bias and weights for a single img within the batch
            biasDeltas,weightDeltas = self.backProg(inputNeurons,label)

            #Adding the results from previous iteration, within the batch, to newly calculated results
            biasGradient = [delta + gradient for gradient,delta in zip(biasGradient,biasDeltas)]
            weightGradient = [delta + gradient for gradient,delta in zip (weightGradient,weightDeltas)]
    
        #applying the gradient found with previous batch's values. Using the learning rate as the variable to control
        #the sizes of steps taken to find a minima. New Value = Previous Value + the negative gradient * size of step
        self.biases = [b - (rateOfLearning/len(batch)) * gradient for b,gradient in zip(self.biases,biasGradient)]
        self.weights = [w - (rateOfLearning/len(batch)) * gradient for w,gradient in zip(self.weights,weightGradient)]

    def trainNetwork(self,inputData,labels,batchSize,iterationOfTraining,rateOfLearning):
        """
        Using Stochastic Gradient Decent
        """
        #Storing the correct pixel values of the imgs with the right label
        #Forms a array of shape (#ofImages,([pixel data,1],label))
        trainingData = [(x,y) for x,y in zip(inputData,labels)]

        print("Starting Training")
        for i in range(iterationOfTraining):
            #Creating a array full of subarrays containing batches
            #Example would be: batches.shape = (Amount of batches,(batchSize,[pixels,label]))
            batches = [trainingData[k:k + batchSize] for k in range(0,len(trainingData),batchSize)]

            for batch in batches:
                #Runs through all the imgs within a batch
                self.updateBatch(batch,rateOfLearning)
            print("Training Cycle {0} Compeleted".format(i+1))
        print("Training Done")

    @staticmethod
    #Activation Function
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    @staticmethod
    #Calculation of Cost function output-trueValue
    def cost(prediction,label):
        # return np.sum(label*np.log(prediction))
        return np.sum((prediction - label)**2)
        
    @staticmethod
    #Find the derivative of cost w.r.t to a, where a = sigmoid(z)
    def costSoftMax_derivative(prediction,label):
        return 2*(prediction - label)
        
    @staticmethod
    #This is the derivative of activation w/r z, where a = sigmoid(z)
    def sigmoid_derivative(x):
        """
        We take sigmoid(z) as a input so we don't have to store the values of z
        and it simipfies the math. Derivative of sigmoid = sigmoid(z) * (1-sigmoid(z))
        """
        return x * (1-x)

        