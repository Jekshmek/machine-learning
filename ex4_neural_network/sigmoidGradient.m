function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) вычисляет градиент сигмовидной функции
% оценивается при z. Это должно работать независимо от того, является ли z матрицей или
% вектор. В частности, если z является вектором или матрицей, вы должны вернуть
% градиента для каждого элемента.

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).







g = sigmoid(z) .* (1 - sigmoid(z));






% =============================================================




end
