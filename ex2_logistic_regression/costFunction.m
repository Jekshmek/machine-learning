function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Вычислить стоимость и градиент для логистической регрессии
% J = COSTFUNCTION (theta, X, y) вычисляет стоимость использования theta в качестве
% параметр для логистической регрессии и градиент стоимости
% w.r.t. к параметрам.

% Инициализировать некоторые полезные значения
m = length(y); % number of training examples

% Вам нужно правильно вернуть следующие переменные 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Инструкции: Рассчитать стоимость конкретного выбора тета.
% Вы должны установить J в стоимость.
% Вычислить частные производные и установить граду частичное
% производных от стоимости в.р.т. каждый параметр в тета
%
% Примечание: grad должен иметь те же размеры, что и тета
%


% hypothesis
h = sigmoid(X * theta);

% Cost
J = 1 / m * ( -y' * log(h) - ( 1 - y )' * log(1 - h) );

% Gradient
grad = 1 / m * ((h - y)' * X)';






% =============================================================

end
