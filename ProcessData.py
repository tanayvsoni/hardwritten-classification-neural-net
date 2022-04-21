import NerualNetwork as nn
import numpy as np

class processData:
    #Flatten out data, such that the output array has the shape (60000,784,1)
    def normFlat(self,data):
        dataSet = data/(np.max(data)-np.min(data))
        output = []

        for i in range(0,dataSet.shape[0]):
            output.append(dataSet[i].flatten())
        output = np.array(output)
        output.shape += (1,)
        return output

    #Creates a label array filled with zeros expect for the index value of the orginal label array.
    def oneHot_labels(self,data,neurons,labels):

        oneHot_labels = np.zeros((data.shape[0],neurons[-1],1))

        for i in range(oneHot_labels.shape[0]):
            for j in range(oneHot_labels.shape[1]):
                        if j == labels[i]:
                            oneHot_labels[i][j] = 1
        return oneHot_labels