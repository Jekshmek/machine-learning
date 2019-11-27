function plotData(x, y)
%PLOTDATA Plots the data points x and y into a new figure 
%   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
%   population and profit.

figure; % open a new figure window

% ====================== YOUR CODE HERE ======================
% Инструкции: нанесите тренировочные данные на фигуру, используя
% "figure" и "plot" команды. Установите метки осей, используя
% команды "xlabel" и "ylabel". Предположим,
% населения и данные о доходах были переданы в
% в качестве аргументов `x` и `y` этой функции.
%
% Подсказка: вы можете использовать опцию 'rx' с plot, чтобы иметь маркеры
% отображаются в виде красных крестиков. Кроме того, вы можете сделать
% маркеров больше при использовании plot(..., 'rx', 'MarkerSize', 10);

xlabel('Population');
ylabel('Revenue');
plot(x,y,'dg','markersize',5,'linewidth',2);



% ============================================================

endfunction
