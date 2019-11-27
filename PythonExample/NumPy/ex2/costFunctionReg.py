import numpy as np

'''
function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

[~, col] = size(X);

J = -1/m * sum(y' * log(sigmoid(X * theta)) + ...
    (1 - y)' * log(1 - sigmoid(X * theta))) + ...
    lambda * sum((theta(2:col, 1)) .^ 2) / (2 * m);

temp = zeros(col,1);
for j = 1: col
    if j == 1
        for i = 1: m
            tmp = (sigmoid(X(i,:) * theta) - y(i)) * X(i,j);
            temp(j) = temp(j) + tmp;
        end
    else
        for i = 1: m
            tmp = (sigmoid(X(i,:) * theta) - y(i)) * X(i,j);
            temp(j) = temp(j) + tmp;
        end
        temp(j) = temp(j) + lambda * theta(j);
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
def cost_func_reg(initial_theta, X, y, l):
    m = len(y)
    temp1 = np.transpose([y]) * np.log(sigmoid(np.dot(X, initial_theta)))
    temp2 = np.transpose([1 - y]) * np.log(1 - sigmoid(np.dot(X, initial_theta)))
    cost = -1 / m * np.sum(temp1 + temp2) + l / (2 * m) * np.sum(initial_theta[1:] ** 2)

    (_, col) = np.shape(X)
    grad = np.zeros((col, 1))
    for j in range(col):
        for i in range(m):
            grad[j] += (sigmoid(np.dot(X[i, :], initial_theta)) - y[i]) * X[i, j]
        if j != 0:
            grad[j] += l * initial_theta[j]
    grad /= m
    return cost, grad
