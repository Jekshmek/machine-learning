function J = computeCost(X, y, theta)
 
%COMPUTECOST Вычислить стоимость для линейной регрессии
%   J = COMPUTECOST(X, y, theta) вычисляет стоимость использования тета в качестве
% параметра для линейной регрессии, чтобы соответствовать точкам данных в X и y

% Инициализируйте некоторые полезные значения
m = length(y); % number of training examples

% Вам нужно правильно вернуть следующие переменные
J = 0;

% ====================== YOUR CODE HERE ======================
% Инструкции: Рассчитать стоимость конкретного выбора тета
% Вы должны установить J в стоимость.

% Example:
%     |1 2104| 
% X = |1 1416| это входные данные
%     |1 1534|
%     |1 852|
%
% y = |300|
%     |240| примерные данные //для сравнения т.е. это верные данные по входным данным
%     |210|
%     |120|
%
%        |-40| это предполагаемые параметры для функции гипотезы
% theta =|0.25|  
%
% predictions/hypothesis:
% J(theta(x)) = theta.0 + theta.1 * x;
% J(theta(2104)) = 1*(-40) + 0.25 * 2104  % 1 это добавленный аргумент для возможности 
% перемножения матриц по линейной формуле
% J(theta(x)) = X * theta; % так будет проще и быстрее нежели в цикле

Predictions = X * theta; % на выходе матрица предпологаемых/предсказанных значений `y`

% считаем матрицу потерь т.е. разницу между верным и и предсказанным значением
% и возводим каждый елемент в квадрат для усреднения/точности
ErrorsPredictions = (Predictions - y).^ 2;

% теперь считаем сумму потерь для этих параметров и выводим среднюю потерю по всем входным данным
J = 1/(2*m) * sum(ErrorsPredictions);

% J это скалярное число потери по нашей вероятной функции предсказаний по текущим theta
% и theta следует корректировать изходя из величины потери, которая должна быть минимальной
% т.е. J=0. В функциях gradientDescent происходит корректировка theta.
% =========================================================================

endfunction