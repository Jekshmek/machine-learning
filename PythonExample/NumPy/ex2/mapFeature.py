import numpy as np

'''
function out = mapFeature(X1, X2)
% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2) maps the two input features
%   to quadratic features used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   Inputs X1, X2 must be the same size
%

degree = 6;
out = ones(size(X1(:,1)));
for i = 1:degree
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end

end
'''


def map_feature(X1, X2):
    degree = 6
    out = np.ones((len(X1), 1))
    for i in range(degree):
        for j in range(i):
            out = np.column_stack((out, (X1 ** (i - j)) * (X2 ** j)))

    return out
