import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def poly_regression(x_train, y_train, degree):

    X = np.empty((len(x_train), degree))
    # Add the nth degree feature expansion
    for i in range(0, len(x_train)):
        x = x_train[i]
        for j in range(0, degree):
            X[i, j] = x ** (j + 1)

    # Flip the array from back to front and add 1 to beginning of X
    X = np.fliplr(np.c_[np.ones([len(x_train), 1]), X])

    parameters = np.poly1d(np.linalg.solve(X.T.dot(X), X.T.dot(y_train)))

    return parameters

def eval_poly_regression(parameters, x, y, degree):
    y_pred = parameters(x)
    rmse = np.sqrt(((y_pred - y) ** 2).mean())
    return rmse

df = pd.read_csv('regression_train_assignment2017.csv', sep=',')

slice = int(round(df.count()[0] * 0.70))

x_train = df['x'][:slice].as_matrix()
y_train = df['y'][:slice].as_matrix()

x_test  = df['x'][slice:].as_matrix()
y_test  = df['y'][slice:].as_matrix()

degrees = [0, 1, 2, 3, 5, 10]
plt.figure(1)
counter = 1
for deg in degrees:
    # Get the parameters
    fit = poly_regression(x_train, y_train, deg)
    print fit
    # calculate the training set rmse
    rmse = eval_poly_regression(fit, x_train, y_train, deg)
    print "RMSE TRAIN Deg: " + str(deg) + " -- : " + str(rmse)
    # calculate the test set rmse
    rmse = eval_poly_regression(fit, x_test, y_test, deg)
    print "RMSE TEST Deg: " + str(deg) + " -- : " + str(rmse)
    # Apply the parameters to the training dataset
    y_pred = fit(x_train)
    # space out values
    x_pred = np.linspace(x_train.min(), x_train.max(), 100)
    # Apply the parameters to the test dataset
    y_test_pred = fit(x_test)
    x_test_pred = np.linspace(x_test.min(), x_test.max(), 100)
    # plot training output
    plt.subplot(4, 4, counter)
    plt.title('Training Degree: ' + str(deg))
    plt.plot(x_train, y_train, '.', label = 'original data')
    plt.plot(x_pred, fit(x_pred), '-', label = 'estimate')

    counter = counter + 1
    # plot test output
    plt.subplot(4, 4, counter)
    plt.title('Test Degree: ' + str(deg))
    plt.plot(x_test, y_test, '.', label = 'original data')
    plt.plot(x_test_pred, fit(x_test_pred), '-', label = 'estimate')

    counter = counter + 1

plt.show()