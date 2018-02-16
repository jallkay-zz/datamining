import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def poly_regression(x_train, y_train, degree):
    ''' Function for getting the polynomial coefficients for fitting the test data '''
    degree = np.arange(1,degree+1)
    for iterDegree in degree:
        fit = np.poly1d(np.polyfit(x_train, y_train, iterDegree))
    return fit

def eval_poly_regression(parameters, x, y, degree):
    ''' Function for predicting values based on the fitted coefficients '''
    predict = parameters(x)
    error = math.sqrt(np.mean((predict-y)**2))
    print "Error with polynomial regression at degree %s is %s" %(degree, error)

df = pd.read_csv('regression_train_assignment2017.csv', sep=',')

trainSlice = int(round(df.count()[0] * 0.70))
testSlice = int(df.count()[0] - round(df.count()[0] * 0.30))


x_train = df['x'][trainSlice:].as_matrix()
y_train = df['y'][trainSlice:].as_matrix()

x_test = df['x'][:testSlice].as_matrix()
y_test = df['y'][:testSlice].as_matrix()

fit = poly_regression(x_train, y_train, 5)

print fit.coeffs

eval_poly_regression(fit, x_test, y_test, 5)



