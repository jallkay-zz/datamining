import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def poly_regression(x_train, y_train, degree):
    ''' Function for getting the polynomial coefficients for fitting the test data '''
    plot_config = ['-g', '-b', '-y', '-r', '-c']
    degree = np.arange(1,degree+1)
    for iterDegree in degree:
        fit = np.polyfit(x_train, y_train, iterDegree)
        print fit
    return fit

def eval_poly_regression(parameters, x, y, degree):
    ''' Function for predicting values based on the fitted coefficients '''
    #predict = parameters(x)
    #error = math.sqrt(np.mean((p56redict-y)**2))
    #print "Error with polynomial regression at degree %s is %s" %(degree, error)
    plt.plot(x, np.polyval(parameters, x), '-g', label="Fit "+str(degree-1)+ " degree poly")
    ##plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    ##                ncol=2, mode="expand", borderaxespad=0.)
    plt.show()
    return predict, error

df = pd.read_csv('regression_train_assignment2017.csv', sep=',')

trainSlice = int(round(df.count()[0] * 0.70))
testSlice = int(df.count()[0] - round(df.count()[0] * 0.30))


x_train = df['x'].as_matrix()
y_train = df['y'].as_matrix()
#x_test  = df['x'].as_matrix()
#y_test  = df['y'].as_matrix()


plt.plot(x_train, y_train, 'bo', label="Train")
#plt.xlim(-5, 5)
#plt.ylim(-5, 5)

#plt.plot(x_test, y_test, 'ro', label="Test")

fit = np.poly1d(np.polyfit(x_train, y_train, 3))

plt.plot(x_train, np.polyval(fit, x_train), '-g', label="Fit "+str(3)+ " degree poly")
plt.show()
#fit = poly_regression(x_train, y_train, 3)

#x_new = np.linspace(x_train[0], x_train[-1], 50)
#y_new = np.linspace(y_train[0], y_train[-1], 50)

#predict, error = eval_poly_regression(fit, x_train, y_train, 5)