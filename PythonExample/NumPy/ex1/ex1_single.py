import matplotlib.pyplot as plt
import numpy as np
# noinspection PyUnresolvedReferences
from gradientDescent import gradient_descent
# noinspection PyUnresolvedReferences
from plotData import plot_data

'''
    Linear regression with one variable
'''


# noinspection PyPep8Naming,PyPep8Naming,PyPep8Naming
def ex1_single():
    data = np.loadtxt('ex1data1.txt', delimiter=',')
    X = np.array(data[:, 0])
    y = np.array(data[:, 1])

    # plotting the data
    data_X = X
    data_y = y
    plot_data(data_X, data_y)

    # gradient descent
    m = len(X)
    X = np.column_stack((np.ones((m, 1)), X))
    y = y.reshape(len(y), 1)
    theta = np.zeros((2, 1))
    iterations = 1500
    alpha = 0.01

    theta, j = gradient_descent(X, y, theta, alpha, iterations)
    print('theta =[', theta[0, 0], ',', theta[1, 0], ']')

    # plotting the result
    plt.plot(X[:, 1], np.dot(X, theta), 'b-')
    plt.scatter(data_X, data_y, c='r', marker='x')
    plt.show()
