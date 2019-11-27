function ex2_reg()

%% Machine Learning Online Class - Exercise 2: Logistic Regression
%
%  Инструкции
% ------------
%
% Этот файл содержит код, который поможет вам начать со второй части
% упражнения, которое охватывает регуляризацию с логистической регрессией.
%
% Вам нужно будет выполнить следующие функции в этом упражнении:
%
% sigmoid.m
% costFunction.m
% predict.m
% costFunctionReg.m
%
% Для этого упражнения вам не нужно изменять код в этом файле,
% или любые другие файлы, кроме упомянутых выше.

%% Initialization
clear ; close all; clc

%% Load Data
% Первые два столбца содержат значения X и третий столбец содержит метку (y).

  data = load('ex2data2.txt');
  X = data(:, [1, 2]); 
  y = data(:, 3);
 
  plotData(X, y);

  % Put some labels
  hold on;

  % Labels and Legend
  xlabel('Microchip Test 1')
  ylabel('Microchip Test 2')

  % Specified in plot order
  legend('y = 1', 'y = 0')
  hold off;


%% =========== Part 1: Regularized Logistic Regression =========================
% В этой части вы получаете набор данных с точками данных, которые 
% не являются линейно разделим. 
% Тем не менее, вы все равно хотели бы использовать логистику регрессии для 
% классификации точек данных.
%
% Для этого вы вводите больше функций для использования - 
% в частности, вы добавляете полиномиальных признаков к нашей матрице данных 
%(аналогично полиномиальным регрессии polynomial regression).
%


  % Add Polynomial Features

  % Обратите внимание, что mapFeature также добавляет столбец единиц для нас,
  % поэтому перехват термин обрабатывается
  X_mapFeature = mapFeature(X(:,1), X(:,2));

  % Initialize fitting parameters
  initial_theta = zeros(size(X_mapFeature, 2), 1);

  % Set regularization parameter lambda to 1
  lambda = 1;

  % Вычислить и отобразить начальную стоимость и градиент для упорядоченной логистики регрессии
  [cost, grad] = costFunctionReg(initial_theta, X_mapFeature, y, lambda);

  fprintf('Затраты при нулевой theta: %f == 0.693 (Ожидаемые затраты)\n', cost);
  fprintf('Градиент при нулевой theta (первые пять): [%f; %f; %f; %f; %f] == [0.0085; 0.0188; 0.0001; 0.0503; 0.0115](Ожидаемые градиенты)\n', grad(1:5));
  fprintf('\nПрограмма приостановлена. Нажмите ввод, чтобы продолжить.\n\n');
  pause;

  % Compute and display cost and gradient
  % with all-ones theta and lambda = 10
  test_theta = ones(size(X_mapFeature,2),1);
  [cost, grad] = costFunctionReg(test_theta, X_mapFeature, y, 10);

  fprintf('\nЗатраты при тестовой theta (с лямбда = 10): %f == 3.16(Ожидаемые затраты)\n', cost);
  fprintf('Градиент при нулевой theta (первые пять): [%f; %f; %f; %f; %f] == [0.3460; 0.1614; 0.1948; 0.2269; 0.0922] (Ожидаемые градиенты)\n',grad(1:5));
  fprintf('\nПрограмма приостановлена. Нажмите ввод, чтобы продолжить.\n\n');
  pause;

 

%% ============= Part 2: Regularization and Accuracies =========================
% Необязательное упражнение:
% В этой части вы попробуете разные значения лямбды и
% видят, как регуляризация влияет на границы решения
%
% Попробуйте следующие значения лямбда (0, 1, 10, 100).
%
% Как меняется граница решения, когда вы меняете лямбду? 
% Как точность тренировочного набора варьируется?
%

  % Initialize fitting parameters
  initial_theta = zeros(size(X_mapFeature, 2), 1);

  % Установите параметр регуляризации лямбда в 1 (вы должны изменить это)
  lambda = 1;
  %lambda = 0.5;		 
  %lambda = 0;		% no regularization => overfit
  %lambda = 10;		% более простая граница решения, но начинающая быть недостаточно подходящей
  %lambda = 100;  % действительно нижнее белье

  % Set Options
  options = optimset('GradObj', 'on', 'MaxIter', 400);

  % Optimize
  [theta, J, exit_flag] = ...
    fminunc(@(t)(costFunctionReg(t, X_mapFeature, y, lambda)), initial_theta, options);

  % Plot Boundary
  plotDecisionBoundary(theta, X_mapFeature, y);
  hold on;
  title(sprintf('lambda = %g', lambda))

  % Labels and Legend
  xlabel('Microchip Test 1')
  ylabel('Microchip Test 2')

  legend('y = 1', 'y = 0', 'Decision boundary')
  hold off;

  % Вычислите точность на нашем тренировочном наборе
  p = predict(theta, X_mapFeature);

  fprintf('Точность: %f == 83.1 (Ожидаемая точность с лямбда = %d)\n\n', mean(double(p == y)) * 100, lambda);
  fprintf('\nПрограмма приостановлена. Нажмите ввод, чтобы продолжить.\n\n');
  pause;
  clear ; close all; clc
endfunction 