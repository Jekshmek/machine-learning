import numpy as np
# noinspection PyUnresolvedReferences
from computeCost import compute_cost

'''
function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
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
    %       of the cost function (computeCost) and gradient here.
    %
    
    theta = theta - alpha * (sum(((X * theta) - y) .* X))' / m;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
'''


# noinspection PyPep8Naming
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros((num_iters, 1))

    for iter in range(num_iters):
        temp_sum = np.sum((np.dot(X, theta) - y) * X, axis=0)
        theta = theta - alpha * np.transpose([temp_sum]) / m
        J_history[iter] = compute_cost(X, y, theta)

    return theta, J_history
