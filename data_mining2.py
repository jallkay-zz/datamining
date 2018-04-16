import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import operator

sns.set()
# Import data
mydata = pd.read_csv("dataset.csv")

class CreateNeuron():
    def __init__(self, numOfNeurons, numPerNeuron):
        self.synapticWeights = 2 * np.random.random((numPerNeuron, numOfNeurons)) - 1

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
            self.hiddenLayer.synapticWeights += 0.01 * hiddenLayerAdjustment
            self.outputLayer.synapticWeights += 0.01 * outputLayerAdjustment

    def evaluate(self, inputs):
        hiddenLayerOutput = self.sigmoid(np.dot(inputs, self.hiddenLayer.synapticWeights))
        outputLayerOutput = self.sigmoid(np.dot(hiddenLayerOutput, self.outputLayer.synapticWeights))

        return hiddenLayerOutput, outputLayerOutput
    
    def showWeights(self):
        print "Hidden Layer Weights:"
        print self.hiddenLayer.synapticWeights
        print "Output Layer Weights:"
        print self.outputLayer.synapticWeights

def cleandata(mydata):
    trainX = mydata.sample(frac=0.9)
    testX = mydata.drop(trainX.index)
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
    trainX = np.array((trainX-trainX.min())/(trainX.max()-trainX.min())) # Normalize to 0 
    testX  = testX.drop("Class", axis=1) # Remove the identifying class
    testX  = np.array((testX-testX.min())/(testX.max()-testX.min())) # Normalize to 0 

    return trainX, trainY, testX, testY


def visualise_data(mydata):
    # Visualise data

    print mydata.describe()
    print ("Empty values: {}.".format(mydata.isnull().values.any()))

    diabetes   = mydata[mydata.Class == "Diabetes"]
    nodiabetes = mydata[mydata.Class == "DR"] 

    data = [diabetes.PressureA, nodiabetes.PressureA]

    plt.figure()
    ax1 = plt.subplot(121)
    plt.boxplot(data, labels = ["Diabetes", "No Diabetes"])


    ax2 = plt.subplot(122)
    pt = sns.kdeplot(diabetes.Tortuosity, shade=True, label="Diabetes")
    pt = sns.kdeplot(nodiabetes.Tortuosity, shade=True, label="No Diabetes")

    plt.show()
    print mydata


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


def getAccuracy(groundTruth, predicted, KNN=False):
    percents = []
    for i, j in zip(groundTruth, predicted):
        distance = abs(i[0] - j[0]) if not KNN else abs(i[0] - j)
        percent = (1 - distance) * 100 
        percents.append(percent)
        print ("Ground Truth: %i -- Predicted -- %.2f -- Accuracy %.2f%%" % (i, j, percent))
    print("Overall accuracy %f" % np.mean(percents))
    return percents

def evaluateANN():
    trainX, trainY, testX, testY = cleandata(mydata)
    plt.figure(figsize=(16, 32))
    # create a mesh to plot in
    h = .02
    x_min, x_max = trainX[:, 0].min() - 1, trainX[:, 0].max() + 1
    y_min, y_max = trainX[:, 1].min() - 1, trainX[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    hidden_layer_dimensions = [2, 3, 4, 5, 10, 20, 50]
    for i, nn_hdim in enumerate(hidden_layer_dimensions):
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

if __name__ == "__main__":

    #Init
    # np.random.seed(1)

    # print "Initialising"
    # hiddenLayer = CreateNeuron(9, 10)
    # outputLayer = CreateNeuron(1, 9)

    # neuralNetwork = ANN(hiddenLayer, outputLayer)
    
    # neuralNetwork.showWeights()

    # trainX, trainY, testX, testY = cleandata(mydata)

    # print "Training"

    # neuralNetwork.trainNetwork(trainX, trainY, 60000)

    # neuralNetwork.showWeights()

    # print "Testing: "

    # hiddenData, outputData = neuralNetwork.evaluate(testX)
    # ANNPercent = getAccuracy(testY, outputData, KNN=False)
    #print testY
    # for i in outputData:
    #     print "%.15f" % i[0]
    evaluateANN()

    predictions = getKNN(trainX, trainY, testX, testY, 10)
    print "KNN Predictions"
    KNNPercent = getAccuracy(testY, predictions, KNN=True)
    #print predictions
