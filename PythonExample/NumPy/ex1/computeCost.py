import numpy as np

'''
function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

J = sum(((X * theta) - y) .^ 2) / (2 * m);

% =========================================================================

end
'''


# noinspection PyPep8Naming,PyPep8Naming
def compute_cost(X, y, theta):
    m = len(y)

    temp = np.dot(X, theta) - y
    J = np.sum(temp * temp) / (2 * m)

    return J
