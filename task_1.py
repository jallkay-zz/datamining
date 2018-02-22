import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def poly_regression(x_train, y_train, degree):

    parameters = np.poly1d(np.polyfit(x_train, y_train, degree))
    #parameters = np.poly1d(np.linalg.solve(x_train.T.dot(x_train), x_train.T.dot(y_train)))

    return parameters

def eval_poly_regression(parameters, x, y, degree):
    y_pred = parameters(x)
    rmse = np.sqrt(((y_pred - y) ** 2).mean())
    return rmse


df = pd.read_csv('regression_train_assignment2017.csv', sep=',')

trainSlice = int(round(df.count()[0] * 0.70))
testSlice = int(df.count()[0] - round(df.count()[0] * 0.30))

x_train = df['x'][:trainSlice].as_matrix()
y_train = df['y'][:trainSlice].as_matrix()

x_test  = df['x'][testSlice:].as_matrix()
y_test  = df['y'][testSlice:].as_matrix()

degrees = [0, 1, 2, 3, 5, 10]
plt.figure(1)
counter = 1
for deg in degrees:
    fit = poly_regression(x_train, y_train, deg)

    rmse = eval_poly_regression(fit, x_train, y_train, deg)
    print "RMSE TRAIN Deg: " + str(deg) + " -- : " + str(rmse)
    
    rmse = eval_poly_regression(fit, x_test, y_test, deg)
    print "RMSE TEST Deg: " + str(deg) + " -- : " + str(rmse)

    

    y_pred = fit(x_train)
    x_pred = np.linspace(x_train.min(), x_train.max(), 100)

    y_test_pred = fit(x_test)
    x_test_pred = np.linspace(x_test.min(), x_test.max(), 100)


    plt.subplot(4, 4, counter)
    plt.title('Training Degree: ' + str(deg))
    plt.plot(x_train, y_train, '.', label = 'original data')
    plt.plot(x_pred, fit(x_pred), '-', label = 'estimate')

    counter = counter + 1


    plt.subplot(4, 4, counter)
    plt.title('Test Degree: ' + str(deg))
    plt.plot(x_test, y_test, '.', label = 'original data')
    plt.plot(x_test_pred, fit(x_test_pred), '-', label = 'estimate')

    counter = counter + 1

plt.show()