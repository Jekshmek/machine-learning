import numpy as np
# noinspection PyUnresolvedReferences
from computeCostMulti import compute_cost_multi

'''
function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    error = (X * theta) - y;
    theta = theta - ((alpha / m) * X' * error);

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
'''


# noinspection PyPep8Naming,PyPep8Naming
def gradient_descent_multi(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros((num_iters, 1))
    
    for iter in range(num_iters):
        error = np.dot(X, theta) - y
        theta = theta - (alpha / m) * np.dot(X.T, error)
        #cof = (1/m)*alpha
        #theta = theta -cof * np.dot(X.T, error)
        J_history[iter] = compute_cost_multi(X, y, theta)

    return theta, J_history
