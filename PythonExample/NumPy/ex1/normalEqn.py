import numpy as np

'''
function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

theta = zeros(size(X, 2), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

% ---------------------- Sample Solution ----------------------


theta = pinv(X' * X) * X' * y;

% -------------------------------------------------------------


% ============================================================

end
'''


# noinspection PyPep8Naming
def normal_eqn(X, y):
    temp = np.dot(X.T, X)
    temp = np.linalg.pinv(temp)
    theta = np.dot(np.dot(temp, X.T), y)

    return theta
