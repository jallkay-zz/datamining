import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
sns.set_style("darkgrid")

def poly_regression(x_train, y_train, degree):

    X = np.empty((len(x_train), degree+1))
    # Add the nth degree feature expansion
    for i in range(0, len(x_train)):
        iter = zip(range(0, degree + 1), range(degree, -1, -1))
        for j, k in iter:
            X[i, k] = x_train[i] ** (j)

    # Apply least squared solution
    parameters = np.poly1d(np.linalg.solve(X.T.dot(X), X.T.dot(y_train)))

    return parameters

def eval_poly_regression(parameters, x, y, degree):
    y_pred = parameters(x)
    rmse = np.sqrt(((y_pred - y) ** 2).mean())
    return rmse

df = pd.read_csv('regression_train_assignment2017.csv', sep=',')

slice = int(round(df.count()[0] * 0.70))

x_train = df['x'].as_matrix()
y_train = df['y'].as_matrix()

x_test  = df['x'][slice:].as_matrix()
y_test  = df['y'][slice:].as_matrix()

degrees = [0, 1, 2, 3, 5, 10]
plt.figure(1)
counter = 1

cols = sns.color_palette("muted", 5)
for deg in degrees:
    # Get the parameters
    fit = poly_regression(x_train, y_train, deg)
    print fit
    # calculate the training set rmse
    rmse_train = eval_poly_regression(fit, x_train, y_train, deg)
    print "RMSE TRAIN Deg: " + str(deg) + " -- : " + str(rmse_train)
    # calculate the test set rmse
    rmse_test = eval_poly_regression(fit, x_test, y_test, deg)
    print "RMSE TEST Deg: " + str(deg) + " -- : " + str(rmse_test)

    #plt.subplot(5, 4, 13)
    #plt.title('RMSE')
    #plt.plot(deg, rmse_train, '.b', label = 'Train')
    #plt.plot(deg, rmse_test, '.g', label = 'Test')

    # Apply the parameters to the training dataset
    y_pred = fit(x_train)
    # space out values
    x_pred = np.linspace(x_train.min(), x_train.max(), 100)
    # Apply the parameters to the test dataset
    y_test_pred = fit(x_test)
    x_test_pred = np.linspace(x_test.min(), x_test.max(), 100)
    # plot training output
    plt.subplot(3, 2, counter)
    plt.xticks(range(-5, 6))
    plt.title('Training Degree: ' + str(deg))
    plt.plot(x_train, y_train, color=cols[0], marker='.', ls='None', label = 'original data')
    plt.plot(x_pred, fit(x_pred), color=cols[3], label = 'estimate')

    counter = counter + 1
    # plot test output
    #plt.subplot(5, 4, counter)
    #plt.title('Test Degree: ' + str(deg))
    #plt.plot(x_test, y_test, '.', label = 'original data')
    #plt.plot(x_test_pred, fit(x_test_pred), '-', label = 'estimate')

    #counter = counter + 1

plt.show()