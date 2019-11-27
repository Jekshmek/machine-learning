import numpy as np

'''
function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

J = -1/m * sum(y' * log(sigmoid(X * theta)) + ...
    (1 - y)' * log(1 - sigmoid(X * theta)));

% grad = 1/m * sum((sigmoid(X * theta) - y)' * X);
[~, col] = size(X);
temp = zeros(col,1);
for j = 1: col
    for i = 1: m
        tmp = (sigmoid(X(i,:) * theta) - y(i)) * X(i,j);
        temp(j) = temp(j) + tmp;
    end
end
grad = (1 / m) * temp;

% =============================================================

end
'''


# predict formula
def sigmoid(s):
    return 1 / (1 + np.exp(-1 * s))


# cost function
# noinspection PyPep8Naming
def cost_func(initial_theta, X, y):
    m = len(y)
    temp1 = np.transpose([y]) * np.log(sigmoid(np.dot(X, initial_theta)))
    temp2 = np.transpose([1 - y]) * np.log(1 - sigmoid(np.dot(X, initial_theta)))
    cost = -1 / m * np.sum(temp1 + temp2)

    (_, col) = np.shape(X)
    grad = np.zeros((col, 1))
    for j in range(col):
        for i in range(m):
            grad[j] += (sigmoid(np.dot(X[i, :], initial_theta)) - y[i]) * X[i, j]
    grad /= m
    return cost, grad
