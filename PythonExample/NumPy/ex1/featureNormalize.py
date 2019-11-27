import numpy as np
 
'''
function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

mu = mean(X);
sigma = std(X); 

X_norm = (X_norm - mu) / sigma;

% ============================================================

end
'''

 
# noinspection PyPep8Naming,PyPep8Naming
def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


'''
def feature_normalize(X):
    X_norm = X
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    Mu_mat = np.tile(mu, (X.shape[0], 1)) # Extend mu to matrix of mx2
    Sigma_mat = np.tile(sigma, (X.shape[0], 1)) # Extend sigma to matrix of mx2
    X_norm = (X - Mu_mat) / Sigma_mat
    return X_norm, mu, sigma
'''

'''
def feature_normalize(X):

    #print(np.std( np.array([[1, 2], [3, 4]]) , axis=0))
    #print(np.mean(X, axis=0))
    n = X.shape[1]
 
    X_norm = X.astype('float32')
    mu = np.zeros(n)
    sigma = np.zeros(n)
    
    for i in range(n):
        feature = X_norm[:,i]
        mu[i] = np.mean(feature)
        sigma[i] = np.std(feature)
        X_norm[:,i] -= mu[i]
        X_norm[:,i] /= sigma[i]
    
    return X_norm, mu, sigma
'''