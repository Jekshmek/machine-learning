function ex3()

%% Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

%  Instructions
%  ------------
%
% Этот файл содержит код, который поможет вам начать работу с
% линейных упражнений. 
%  Вам нужно будет выполнить следующие функции в этом упражнении:
%
%     lrCostFunction.m (logistic regression cost function) функция стоимости логистической регрессии
%     oneVsAll.m    Тренируйте классификатор «один против всех»
%     predictOneVsAll.m  Прогнозирование с использованием универсального классификатора «один против всех».
%     predict.m   функция прогнозирования нейронной сети
%
% Для этого упражнения вам не нужно изменять код в этом файле,
% или любые другие файлы, кроме упомянутых выше.
%

% В этом упражнении вы будете использовать логистическую регрессию и нейронные сети 
% для распознавания рукописных цифр (от От 0 до 9). 
% Автоматическое распознавание рукописных цифр широко используется сегодня - от распознавания
% почтовых индексов (почтовый коды) на почтовых конвертах для распознавания сумм, 
% написанных на банковских чеках. 
% Это упражнение покажет вам, как
% методы, которые вы изучили, могут быть использованы для этой задачи классификации. 
% В первой части упражнения вы будете расширять
% Ваша предыдущая реализация логистической регрессии и применить ее к классификации «все против всех»

  %% Initialization
  clear ; close all; clc


  %% Настройте параметры, которые вы будете использовать для этой части упражнения
  input_layer_size  = 400;  % 20x20 входные данные изображений цифр
  num_labels = 10;          % 10 labels, from 1 to 10
                            % (note that we have mapped "0" to label 10)
                            
  %% =========== Part 1: Loading and Visualizing Data =============
  % Мы начинаем упражнение с первой загрузки и визуализации набора данных.
  % Вы будете работать с набором данных, который содержит рукописные цифры.
  %

  % Load Training Data
  fprintf('Loading and Visualizing Data ...\n')

  #{
    В ex3data1.mat имеется 5000 обучающих примеров, где каждый обучающий пример 
    имеет размер 20 на 20 пикселей. полутоновое изображение цифры. 
    Каждый пиксель представлен числом с плавающей точкой, обозначающим шкалу серого
    Интенсивность в этом месте. 
    Сетка пикселей 20 на 20 «развернута» в 400-мерный вектор. 
    Каждый из них обучающие примеры становятся одной строкой в нашей матрице данных X. 
    Это дает нам матрицу 5000 на 400 X, где каждая
    строка представляет собой обучающий пример для изображения, написанного от руки.
    
    цифра «0» помечается как «10», а цифры «1» - «9» обозначаются как «1» - «9» в
их естественный порядок.
  #}
  load('ex3data1.mat'); % тренировочные данные хранятся в массивах X, y
  m = size(X, 1);

  % Случайно выберите 100 точек данных для отображения
  rand_indices = randperm(m);
  sel = X(rand_indices(1:100), :);

  displayData(sel);

  fprintf('Программа приостановлена. Нажмите Enter, чтобы продолжить.\n');
  pause;                     
                          
                          

%% ============ Part 2a: Vectorize Logistic Regression ============
% В этой части упражнения вы будете повторно использовать свою логистическую регрессию
% кода из последнего упражнения. Ваша задача здесь состоит в том, чтобы убедиться, что ваш
% регуляризованной логистической регрессии реализация векторизована. После
%, вы будете применять единую классификацию для всех рукописных цифра данных.
%

% Test case for lrCostFunction
fprintf('\nTesting lrCostFunction() with regularization');

theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3;
[J grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);

fprintf('\nЗатраты: %f == 2.534819 (Ожидаемые затраты)\n', J);

fprintf('Gradients:\n');
fprintf('Gradients [%f;%f;%f;%f] == [0.146561; -0.548558; 0.724722; 1.398003](Ожидаемые градиенты)\n', grad);
 

fprintf('Программа приостановлена. Нажмите Enter, чтобы продолжить.\n');
pause; 
   

%% ============ Part 2b: One-vs-All Training ============
fprintf('\nТренинг Логистическая регрессия «один против всех»...\n')

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

%fprintf('Первые 10 theta %f.\n',all_theta(0:10,:));
 


%% ================ Part 3: Predict for One-Vs-All ================

pred = predictOneVsAll(all_theta, X);

fprintf('\nТочность тренировочного набора: %f\n', mean(double(pred == y)) * 100);

 %pause; 
 %clear ; close all; clc                         
                       
 endfunction                         