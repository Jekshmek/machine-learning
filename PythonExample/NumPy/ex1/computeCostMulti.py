import numpy as np

'''
function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

J = sum((X * theta - y)' * (X * theta - y)) / (2 * m);

% =========================================================================

end
'''


# noinspection PyPep8Naming,PyPep8Naming
def compute_cost_multi(X, y, theta):
    m = len(y)

    temp = np.dot(X, theta) - y
    J = np.sum(np.transpose([temp]) * temp) / (2 * m)

    return J
