import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def poly_regression(x_train, y_train, degree):

    parameters = np.poly1d(np.polyfit(x_train, y_train, degree))
    #parameters = np.poly1d(np.linalg.solve(x_train.T.dot(x_train), x_train.T.dot(y_train)))

    print parameters

    return parameters



df = pd.read_csv('regression_train_assignment2017.csv', sep=',')

x_train = df['x'].as_matrix()
y_train = df['y'].as_matrix()

degrees = [0, 1, 2, 3, 5, 10]
plt.figure(1)
counter = 1
for deg in degrees:
    fit = poly_regression(x_train, y_train, deg)
    y_pred = fit(x_train)
    x_pred = np.linspace(x_train.min(), x_train.max(), 100)
    plotNum = str(33) + str(counter)
    plt.subplot(plotNum)
    plt.title('Degree: ' + str(deg))
    plt.plot(x_train, y_train, '.', label = 'original data')
    plt.plot(x_pred, fit(x_pred), '-', label = 'estimate')
    counter = counter + 1

plt.show()