function plotData(X, y)
 
%PLOTDATA Наносит точки данных X и Y на новую фигуру
% PLOTDATA (x, y) отображает точки данных с `+` для положительных примеров
% и `о` для отрицательных примеров. Предполагается, что X является матрицей Mx2.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Инструкции: нанесите положительные и отрицательные примеры на
% 2D-график, используя опцию 'k+' для положительного
% examples и 'ko' для отрицательных примеров.
%



% format: plot (X, Y, PROPERTY, VALUE, ...)
% Positive values should be black/cross:
%    markerstyle: `+'  crosshair
%    color:       `k'  blacK
% Negative values should be yellow/circle:
%    markerstyle: `o'  circle
%    color:       ‘y’  yellow

% First find Indices of Positive and Negative Examples
pos = find(y == 1); % поступил
neg = find(y == 0); % не поступил

% Plot Postive Result: Черные точки
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);

% Plot Negative Result: Полые точки
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);






% =========================================================================



hold off;

end
