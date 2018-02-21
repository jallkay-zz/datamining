'''
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

    def __init__(self, degree = 1, regLambda = 1E-8):
        '''
        Constructor
        '''
        self.degree = degree
        self.regLambda = regLambda

    # THIS IS THE VANDER BIT
    def polyfeatures(self, X, degree):
        '''
        Expands the given X into an n * d array of polynomial features of
            degree d.
        Returns:
            A n-by-d numpy array, with each row comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not inlude the zero-th power.
        Arguments:
            X is an n-by-1 column numpy array
            degree is a positive integer
        '''
        n = len(X)
        result = np.empty((n, degree))

        for i in range(0, n):
            x = X[i]
            for j in range(0, degree):
                result[i, j] = x ** (j + 1)

        return result




    def fit(self, X, y):
        '''
            Trains the model
            Arguments:
                X is a n-by-1 array
                y is an n-by-1 array
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling
                at first
        '''

        n = len(X)
        X = self.polyfeatures(X, self.degree)

        self.standardizationValues(X)
        X = self.standardize(X)

        # Add 1 to beginning of X
        X = np.c_[np.ones([n, 1]), X]

        n,d = X.shape

        self.theta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)

        print "Calculated theta: "
        print self.theta

    def standardizationValues(self, X):
        n, d = X.shape


        result = np.ones([n, d])
        self.transformation = np.zeros([d, 2])

        for i in range(0, d):
            feature = np.empty(n)
            for j in range(0, n):
                feature[j] = X[j, i]
            mean = np.mean(feature)
            stdev = np.std(feature)

            self.transformation[i, 0] = mean
            self.transformation[i, 1] = stdev

    def standardize(self, X):

        n, d = X.shape

        result = np.zeros([n, d])

        for i in range(0, n):

            for j in range(0, d):
                if self.transformation[j, 1] == 0:
                    result[i, j] = 0
                else:
                    result[i, j] = (X[i, j] - self.transformation[j, 0])/self.transformation[j, 1]

        return result

        
    def predict(self, X):
        '''
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        '''
        n = len(X)

        X = self.polyfeatures(X, self.degree)
        X = self.standardize(X)
        
        # add 1s column
        X = np.c_[np.ones([n, 1]), X]

        

        return X.dot(self.theta)



#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------



def learningCurve(Xtrain, Ytrain, Xtest, Ytest, degree):
    '''
    Compute learning curve
        
    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        degree -- polynomial degree
        
    Returns:
        errorTrain -- errorTrain[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTest -- errorTrain[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]
        
    Note:
        errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    '''
    
    n = len(Xtrain)
    print n
    
    errorTrain = np.zeros((n))
    errorTest = np.zeros((n))

    for i in range(0, n):
        model = PolynomialRegression(degree=degree)
        model.fit(Xtrain[0:(i + 1)], Ytrain[0:(i + 1)])

        errorTrain[i] = computeError(model.predict(Xtrain[0:(i+1)]), Ytrain[0:(i+1)])
        errorTest[i] = computeError(model.predict(Xtest), Ytest)
    
    print errorTrain
    print errorTest
    
    return (errorTrain, errorTest)

def computeError(calculated, actual):
        n = len(calculated)

        total = 0
        for i in range(0, n):
            total += (calculated[i] - actual[i]) ** 2

        return total/n


df = pd.read_csv('regression_train_assignment2017.csv', sep=',')

trainSlice = int(round(df.count()[0] * 0.70))
testSlice = int(df.count()[0] - round(df.count()[0] * 0.30))


x_train = df['x'].as_matrix()
y_train = df['y'].as_matrix()
x_test  = df['x'].as_matrix()
y_test  = df['y'].as_matrix()

learningCurve(x_train, y_train, x_test, y_test, 5)
