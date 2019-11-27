function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Вычислить стоимость и градиент для логистической регрессии с регуляризацией
% J = COSTFUNCTIONREG (theta, X, y, lambda) вычисляет стоимость использования
% тета как параметр для регуляризованной логистической регрессии и
% градиента стоимости с.р.т. к параметрам. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Инструкции: Рассчитать стоимость конкретного выбора theta.
% Вы должны установить J в стоимость.
% Вычислить частные производные и установить граду частичное
% производных от стоимости в.р.т. каждый параметр в theta



% hypothesis
h = sigmoid(X * theta);

% Функция затрат/стоимости
% отмечают, что theta(1) не должна быть упорядочена
% Таким образом, мы будем применять ниже функцию только theta[1], theta[2],...theta[n]
theta_1n = theta(2:size(theta));
theta_reg = [0;theta_1n];

J = 1 / m * ( -y' * log(h) - ( 1 - y )' * log(1 - h) ) + lambda / (2 * m) * theta_reg'*theta_reg;

% Gradient
grad = 1 / m * X' * (h - y) + lambda / m * theta_reg ;



% =============================================================

end
