function [theta] = normalEqn(X, y)
%NORMALEQN Вычисляет замкнутую форму решения для линейной регрессии
% NORMALEQN (X, y) вычисляет решение в замкнутой форме для линейного
% регрессии с использованием нормальных уравнений.

theta = zeros(size(X, 2), 1);

% ====================== ВАШ КОД ЗДЕСЬ ======================
% Инструкции: завершите код для вычисления решения в закрытой форме
% к линейной регрессии и положить результат в тета.
%

% ---------------------- Sample Solution ----------------------


theta = pinv(X' * X) * X' * y;
% theta = (X' * X)^-1 * X' * y;
% note: can also be written with pinv function: pinv(X' * X) * X' * y
% -------------------------------------------------------------


% ============================================================

end
