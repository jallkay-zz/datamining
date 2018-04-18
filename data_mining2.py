import math
import operator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set()

# Import data
mydata = pd.read_csv("dataset.csv")

class CreateNeuron():
    def __init__(self, numOfNeurons, numPerNeuron):
        self.synapticWeights = 2 * np.random.random((numPerNeuron, numOfNeurons)) - 1
        self.bias            = np.random.uniform(size=(1,numOfNeurons))

class ANN():
    def __init__(self, hiddenLayer, outputLayer):
        self.hiddenLayer = hiddenLayer
        self.outputLayer = outputLayer
        
    def sigmoid (self, x): 
        return 1/(1 + np.exp(-x))      # activation function

    def sigmoid_(self, x): 
        return x * (1 - x)

    def trainNetwork(self, trainX, trainY, iterations):
        for i in range(0, iterations):
            hiddenLayerOutput, outputLayerOutput = self.evaluate(trainX)

            # Calculate error to groud truth
            outputLayerError = trainY - outputLayerOutput
            # How far it is off
            outputLayerDelta = outputLayerError * self.sigmoid_(outputLayerOutput)
            
            # Calculate the error to the groud truth of the hidden layer
            hiddenLayerError = outputLayerDelta.dot(self.outputLayer.synapticWeights.T)
            hiddenLayerDelta = hiddenLayerError * self.sigmoid_(hiddenLayerOutput)

            # Work out adjustment for the weights
            hiddenLayerAdjustment = trainX.T.dot(hiddenLayerDelta)
            outputLayerAdjustment = hiddenLayerOutput.T.dot(outputLayerDelta)

            # Change the weightings
            self.hiddenLayer.synapticWeights += hiddenLayerAdjustment * 0.1
            self.hiddenLayer.bias += np.sum(hiddenLayerDelta, axis=0,keepdims=True) * 0.1
            self.outputLayer.synapticWeights += outputLayerAdjustment * 0.1
            self.outputLayer.bias += np.sum(outputLayerDelta, axis=0,keepdims=True) * 0.1


    def evaluate(self, inputs):
        hiddenLayerOutput = self.sigmoid(np.dot(inputs, self.hiddenLayer.synapticWeights) + self.hiddenLayer.bias)
        outputLayerOutput = self.sigmoid(np.dot(hiddenLayerOutput, self.outputLayer.synapticWeights) + self.outputLayer.bias)

        return hiddenLayerOutput, outputLayerOutput
    
    def showWeights(self):
        print "Hidden Layer Weights:"
        print self.hiddenLayer.synapticWeights
        print "Output Layer Weights:"
        print self.outputLayer.synapticWeights

def cleandata(mydata, KNN=False, CV=False, CVNumber=0):
    if not CV:
        trainX = mydata.sample(frac=0.9)
        testX = mydata.drop(trainX.index)
    else:
        size = len(mydata) / 10
        testX = []
        idxStart = int(math.floor(CVNumber * size))
        idxEnd = int(math.floor((CVNumber + 1) * size) - (size / 2))
        split = len(mydata) / 2
        
        sideA = mydata[idxStart:idxEnd]
        sideB = mydata[idxStart + split:idxEnd + split]
        testX = pd.concat([sideA, sideB])
        trainX = mydata.drop(testX.index)
    
    trainY = []
    testY = []

    for type in trainX.Class:
        if type == "Diabetes":
            trainY.append([1])
        elif type == "DR":
            trainY.append([0])

    for type in testX.Class:
        if type == "Diabetes":
            testY.append([1])
        elif type == "DR":
            testY.append([0])

    trainY = np.array(trainY)
    testY  = np.array(testY)
    trainX = trainX.drop("Class", axis=1) # Remove the identifying class
    testX  = testX.drop("Class", axis=1) # Remove the identifying class

    if KNN:
        trainX.insert(10, 'Condition', trainY)
        testX.insert(10, 'Condition', testY)
        
        
    trainX = np.array(trainX)
    testX  = np.array(testX)
    
    if not KNN:
        trainX = np.array((trainX-trainX.min())/(trainX.max()-trainX.min())) # Normalize to 0 
        testX  = np.array((testX-testX.min())/(testX.max()-testX.min())) # Normalize to 0 

    return trainX, trainY, testX, testY


def visualiseData(mydata):
    # Visualise data columns
    print mydata.describe()
    # Check for empty values
    print ("Empty values: {}.".format(mydata.isnull().values.any()))
    # Split data into diabetic and non dieabetic 
    diabetes   = mydata[mydata.Class == "Diabetes"]
    nodiabetes = mydata[mydata.Class == "DR"] 
    # Get Arterial Blood Pressure from both 
    data = [diabetes.PressureA, nodiabetes.PressureA]
    # Subplot the data to a boxplot for both
    plt.figure()
    ax1 = plt.subplot(121)
    plt.boxplot(data, labels = ["Diabetes", "No Diabetes"])
    # Subplot the density plot of the diabetes and non diabetes
    ax2 = plt.subplot(122)
    pt = sns.kdeplot(diabetes.Tortuosity, shade=True, label="Diabetes")
    pt = sns.kdeplot(nodiabetes.Tortuosity, shade=True, label="No Diabetes")
    # Show plots
    plt.show()


def getDistance(x, y, length):
    ''' Function for getting the euclidean distance from one point to another (by squaring and then rooting) '''
    distance = 0
    for i in range(length):
        distance += pow((x[i] - y[i]), 2)
    return math.sqrt(distance)

def getNeighbours(trainX, testInstance, k):
    distances = []
    testLength = len(testInstance)-1
    for i in range(len(trainX)):
        dist = getDistance(testInstance, trainX[i], testLength)
        distances.append((trainX[i], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbours = []
    for j in range(k):
        neighbours.append(distances[j][0])
    return neighbours

def getVotes(neighbours):
    classVotes = {}
    for i in range(len(neighbours)):
        response = neighbours[i][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getKNN(trainX, trainY, testX, testY, k):
    predictions = []
    for i in range(len(testX)):
        neighbours = getNeighbours(trainX, testX[i], k)
        result = getVotes(neighbours)
        predictions.append(result)
    return predictions


def getAccuracy(groundTruth, predicted, KNN=False, debug=True):
    percents = []
    for i, j in zip(groundTruth, predicted):
        distance = abs(i[0] - j[0]) if not KNN else abs(i[0] - j)
        percent = (1 - distance) * 100 
        percents.append(percent)
        if debug:
            print ("Ground Truth: %i -- Predicted -- %.2f -- Accuracy %.2f%%" % (i, j, percent))
    print("Overall accuracy %f" % np.mean(percents))
    return np.mean(percents)

def evaluateANN():
    trainX, trainY, testX, testY = cleandata(mydata)
    plt.figure(figsize=(16, 32))
    # create a mesh to plot in
    h = .02
    x_min, x_max = trainX[:, 0].min() - 1, trainX[:, 0].max() + 1
    y_min, y_max = trainX[:, 1].min() - 1, trainX[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    neurons = [2, 3, 4, 5, 10, 20, 50]
    for i, nn_hdim in enumerate(neurons):
        plt.subplot(5, 2, i+1)
        plt.title('Hidden Layer size %d' % nn_hdim)
        print "Initialising"
        hiddenLayer = CreateNeuron(nn_hdim, 10)
        outputLayer = CreateNeuron(1, nn_hdim)
        neuralNetwork = ANN(hiddenLayer, outputLayer)
        print "Training"
        neuralNetwork.trainNetwork(trainX, trainY, 60000)

        Z = outputLayer.synapticWeights.reshape(xx.shape)
        plt.contourf(xx, yy, outputLayer.synapticWeights)
        yesX = []
        yesY = []
        noX = []
        noY = []
        for i, k in enumerate(trainX):
            if trainY[i][0] == 0:
                noX.append(k[6])
                noY.append(k[1])
            elif trainY[i][0] == 1:
                yesX.append(k[6])
                yesY.append(k[1])
        plt.scatter(noX, noY, color='red')
        plt.scatter(yesX, yesY, color='green')
        
    plt.show()


def cvANN():
    neurons = [2, 10, 50]
    overallPercents = []
    for i, nn_hdim in enumerate(neurons):
        percents = []
        for fold in range(0, 10):
            trainX, trainY, testX, testY = cleandata(mydata, CV=True, CVNumber=fold)
            hiddenLayer = CreateNeuron(nn_hdim, 10)
            outputLayer = CreateNeuron(1, nn_hdim)
            neuralNetwork = ANN(hiddenLayer, outputLayer)
            neuralNetwork.trainNetwork(trainX, trainY, 60000)
            hiddenData, outputData = neuralNetwork.evaluate(testX)
            print "Fold %i:" % fold
            ANNPercent = getAccuracy(testY, outputData, KNN=False, debug=False)
           
            percents.append(ANNPercent)
            
        print("Neuron: %i after 10 Fold CV percentage %.2f" % (nn_hdim, np.mean(percents)))
        overallPercents.append(np.mean(percents))
    plt.subplot(121)
    plt.plot(neurons, overallPercents)


def cvKNN():

    k = [1, 5, 10]
    overallPercents = []
    for i, iterk in enumerate(k):
        percents = []
        for fold in range(0, 10):
            trainX, trainY, testX, testY = cleandata(mydata, KNN=True, CV=True, CVNumber=fold)
            predictions = getKNN(trainX, trainY, testX, testY, iterk)
            print "KNN Predictions"
            KNNPercent = getAccuracy(testY, predictions, KNN=True)
            percents.append(KNNPercent)
            
        print("K: %i after 10 Fold CV percentage %.2f" % (iterk, np.mean(percents)))
        overallPercents.append(np.mean(percents))

    plt.subplot(122)
    plt.plot(k, overallPercents)
    

if __name__ == "__main__":


    visualiseData(mydata)
    #Init
    np.random.seed(1)

    plt.figure()
    cvANN()
    cvKNN()
    plt.show()
