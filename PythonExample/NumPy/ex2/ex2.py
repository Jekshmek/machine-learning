import numpy as np
from scipy.optimize import minimize
# noinspection PyUnresolvedReferences
from plotData import plot_data
# noinspection PyUnresolvedReferences
from costFunction import sigmoid, cost_func
# noinspection PyUnresolvedReferences
from predict import predict
# noinspection PyUnresolvedReferences
from mapFeature import map_feature
# noinspection PyUnresolvedReferences
from costFunctionReg import cost_func_reg

# --------------------------------------------------------
# before regularized
data1 = np.loadtxt('./ex2data1.txt', delimiter=',')
X = np.array(data1[:, :2])
Y = np.array(data1[:, 2])

# plotting the data
# data1_X = X
# data1_y = Y
# plt1 = plot_data(data1_X, data1_y)
# plt1.xlabel('Exam 1 Score')
# plt1.ylabel('Exam 2 Score')
# plt1.legend(['Not admitted', 'Admitted'])
# plt1.show()

# predict formula
print(f'ans = {sigmoid(0)}')
print('--------------------------------------------------------')
# cost function
(m, n) = np.shape(X)
initial_theta = np.zeros((n + 1, 1))
X = np.column_stack((np.ones((m, 1)), X))
cost, grad = cost_func(initial_theta, X, Y)
print(f'Cost at initial theta (zeros): {cost}')
print(f'Gradient at initial theta (zeros): \n{grad}')
print('--------------------------------------------------------')


# gradient descent

def cost_function(theta, x, y):
    row, _ = x.shape
    j = (-np.dot(y.T, np.log(sigmoid(x.dot(theta)))) - np.dot((1 - y).T, np.log(1 - sigmoid(x.dot(theta))))) / row
    return j


def gradient(theta, x, y):
    row, col = x.shape
    theta = theta.reshape((col, 1))
    g = np.dot(x.T, sigmoid(x.dot(theta)) - y) / row
    return g.flatten()


# the minimize function in scipy.optimize is the same as fminunc in MatLab
# however the Y should be reshape before used in minimize function
# reference(Chinese): https://blog.csdn.net/csdn_inside/article/details/81558079
Y = Y.reshape((m, 1))
result = minimize(fun=cost_function, x0=initial_theta, args=(X, Y), method='TNC', jac=gradient)
print(f'fminunc result:\n{result}')
print('--------------------------------------------------------')
final_theta = np.array(result['x']).reshape((len(result['x']), 1))
print(f'theta:\n{final_theta}')
print('--------------------------------------------------------')

# evaluate logistic regression
prob = sigmoid(np.array([1, 45, 85]).dot(final_theta))
print(f'For a student with scores 45 and 85,\n we predict an admission probability of {prob[0]}\n')

p = predict(final_theta, X)
correct = 0
for i in range(m):
    if p[i] == Y[i]:
        correct += 1
print(f'Train Accuracy: {correct / m * 100}%')
print('--------------------------------------------------------')

# --------------------------------------------------------
# regularized
data2 = np.loadtxt('./ex2data2.txt', delimiter=',')
X = np.array(data2[:, :2])
Y = np.array(data2[:, 2])

# plotting the data
# data2_X = X
# data2_y = Y
# plt2 = plot_data(data2_X, data2_y)
# plt2.xlabel('Microchip Test 1')
# plt2.ylabel('Microchip Test 2')
# plt2.legend(['y = 1', 'y = 0'])
# plt2.show()

# feature mapping
X = map_feature(X[:, 0], X[:, 1])
print('--------------------------------------------------------')
# regularized cost function
(m, n) = np.shape(X)
initial_theta = np.zeros((n, 1))
l = 1
cost, grad = cost_func_reg(initial_theta, X, Y, l)
print(f'Cost at initial theta (zeros): {cost}')
print(f'Gradient at initial theta (zeros): \n{grad}')
print('--------------------------------------------------------')
