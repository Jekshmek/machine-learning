import numpy as np
import matplotlib.pyplot as plt

'''
function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

    pos = find(y==1); 
    neg = find(y == 0);
    
    plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
    plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y','MarkerSize', 7);

% =========================================================================



hold off;

end
'''


# noinspection PyPep8Naming
def plot_data(X, y):
    neg = np.where(y == 0)[0]
    pos = np.where(y == 1)[0]

    plt.figure()
    plt.scatter(X[neg, 0], X[neg, 1], c='r', marker='o')
    plt.scatter(X[pos, 0], X[pos, 1], c='b', marker='+')

    return plt
