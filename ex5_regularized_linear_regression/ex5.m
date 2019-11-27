function ex5()
  
%% Machine Learning Online Class
% Упражнение 5 | Регулярная линейная регрессия и смещение дисперсии
%
%  Инструкции
% ------------
%
% Этот файл содержит код, который поможет вам начать работу с
% упражнение. Вам нужно будет выполнить следующие функции:
%
% linearRegCostFunction.m
% learningCurve.m
% validationCurve.m
%
% Для этого упражнения вам не нужно изменять код в этом файле,
% или любые другие файлы, кроме упомянутых выше.
%

%% Initialization
clear ; close all; clc


%% =========== Part 1: Loading and Visualizing Data =============
% Мы начинаем упражнение с первой загрузки и визуализации набора данных.
% Следующий код загрузит набор данных в вашу среду и график
%  данные.
%

% Тренировка данных
fprintf('Loading and Visualizing Data ...\n')

% Загрузки из ex5data1:
% Вы будете иметь X, Y, Xval, Yval, Xtest, Ytest в вашей среде
load ('ex5data1.mat');

% m = Number of examples
m = size(X, 1);

% Plot training data
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');

fprintf('Программа приостановлена. Нажмите ввод, чтобы продолжить.\n\n');
pause;



%% =========== Part 3: Regularized Linear Regression Gradient =============
% Теперь вы должны реализовать градиент для регуляризованной линейной
% регрессии.
%

theta = [1 ; 1];
[J, grad] = linearRegCostFunction([ones(m, 1) X], y, theta, 1);

fprintf(['Gradient at theta = [1 ; 1]:  [%f; %f] '...
         '\n(это значение должно быть около [-15.303016; 598.250744])\n'], ...
         grad(1), grad(2));

fprintf('Программа приостановлена. Нажмите ввод, чтобы продолжить.\n\n');
pause;


%% =========== Part 4: Train Linear Regression =============
% После того, как вы правильно внедрили стоимость и градиент,
% Функция trainLinearReg будет использовать вашу функцию стоимости для обучения
% регуляризованной линейной регрессии.
%
% Write Note Примечание: данные нелинейные, поэтому это не даст
%                 поместиться.
%

% Поезд линейной регрессии с лямбда = 0
lambda = 0;
[theta] = trainLinearReg([ones(m, 1) X], y, lambda);

%  Plot fit over the data
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
hold on;
plot(X, [ones(m, 1) X]*theta, '--', 'LineWidth', 2)
hold off;

fprintf('Программа приостановлена. Нажмите ввод, чтобы продолжить.\n\n');
pause;


%% =========== Part 5: Кривая обучения для линейной регрессии =============
% Затем вы должны реализовать функцию learningCurve.
%
% Примечание  записи: поскольку модель не соответствует данным, мы ожидаем
% видят график с «высоким смещением» - Рисунок 3 в ex5.pdf
%

lambda = 0;
[error_train, error_val] = ...
    learningCurve([ones(m, 1) X], y, ...
                  [ones(size(Xval, 1), 1) Xval], yval, ...
                  lambda);

plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 150])

fprintf('# Примеры обучения \t Train Error \t Ошибка перекрестной проверки\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

fprintf('Программа приостановлена. Нажмите ввод, чтобы продолжить.\n\n');
pause;



%% =========== Part 6: Отображение объектов для полиномиальной регрессии =======
% Одним из решений этой проблемы является использование полиномиальной регрессии. 
% Вы должны сейчас выполнения полифункций для сопоставления каждого примера с его возможностями
%

p = 8;

% Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(m, 1), X_poly];                   % Add Ones

% Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p);
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones

% Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones

fprintf('Пример нормализованного обучения 1:\n');
fprintf('  %f  \n', X_poly(1, :));

fprintf('Программа приостановлена. Нажмите ввод, чтобы продолжить.\n\n');
pause;



%% =========== Часть 7. Кривая обучения для полиномиальной регрессии ===========
% Теперь вы можете экспериментировать с полиномиальной регрессией с несколькими
% значения лямбды. Код ниже выполняет полиномиальную регрессию с
% lambda = 0. Вы должны попробовать запустить код с разными значениями
% лямбда, чтобы увидеть, как изменяется форма и кривая обучения.
%

lambda = 3.8599;
[theta] = trainLinearReg(X_poly, y, lambda);

% Plot training data and fit
figure(1);
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
plotFit(min(X), max(X), mu, sigma, theta, p);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));

figure(2);
[error_train, error_val] = ...
    learningCurve(X_poly, y, X_poly_val, yval, lambda);
plot(1:m, error_train, 1:m, error_val);

title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Cross Validation')

fprintf('Полиномиальная регрессия (lambda = %f)\n\n', lambda);
fprintf('# Примеры обучения \t Train Error \t Ошибка перекрестной проверки\n');
for i = 1:m
    fprintf('\t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

fprintf('Программа приостановлена. Нажмите ввод, чтобы продолжить.\n\n');
pause;


%% =========== Часть 8. Проверка правильности выбора лямбды =============
% Теперь вы будете реализовывать validationCurve для проверки различных значений
% лямбда в проверочном наборе. Затем вы будете использовать это, чтобы выбрать
% «лучшее» значение лямбда
%

[lambda_vec, error_train, error_val] = ...
    validationCurve(X_poly, y, X_poly_val, yval);

close all;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('лямбда \t \tTrain Error \tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end

fprintf('Программа приостановлена. Нажмите ввод, чтобы продолжить.\n\n');
pause;

clear ; close all;
endfunction