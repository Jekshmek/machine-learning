function ex3_nn()
%% Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks

 
%  Инструкции
% ------------
%
% Этот файл содержит код, который поможет вам начать работу с
% линейных упражнений. Вам нужно будет выполнить следующие функции
% в этом упражнении:
%
% lrCostFunction.m (функция стоимости логистической регрессии)
% oneVsAll.m
% pregnoteOneVsAll.m
% predict.m
%
% Для этого упражнения вам не нужно изменять код в этом файле,
% или любые другие файлы, кроме упомянутых выше.
%

%% Initialization
clear ; close all; clc

%% Настройте параметры, которые вы будете использовать для этого упражнения
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10
% (обратите внимание, что мы присвоили «0» метке 10)

%% =========== Part 1: Loading and Visualizing Data =============
% Мы начинаем упражнение с первой загрузки и визуализации набора данных.
% Вы будете работать с набором данных, который содержит рукописные цифры.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

% Загружаются данные в матрицу X
% load('ex3data1.mat');
% Перенос сжатого формата в читабельный X.txt
% save X.txt X; и нужно вместо пробелов поставить запятую(кроме первого пробела)

% [indx_col,indx_row]=find(y == 10);
% fprintf('0: %d  шт\n',length(indx_col)) что бы найти количество 10-ток и т.д

% загрузим из файла с читабельными данными
% Это Сетка 20 на 20 пикселей «развернута» в 400-мерный вектор для каждой цифры со значениями от 0 до 1 это величина интенсивности серого цвета в ячейке
 X=load('ex3data1clone.txt');% формат .mat быстрее грузится

 m = size(X, 1);
  
 % сформируем `y` который содержит проверочные данные и они соответствуют набору входных данных X
 % т.е. первые 500 шт это оцифровка всех десяток (это 0) потом 500 шт единиц
 % по 500 шт начиная с 10(это 0) потом 1 далее 2 и так до 9  
 %
 y=zeros(m,1);
 k=1;
 for i=0:9,
      for j=1:500,
         if i==0,y(k)=10;
           else,y(k)=i;
         endif;
         k++;
      endfor;
      i++;
 endfor;
 
  




%fprintf('Размер матрицы: %d\n Количество аргументов обучающего набора (20x20 pixels) %d \n',m,length(X(1,:)))
%fprintf('Первый елемент матрицы X (ряд с 50 по 100) :\n')
%X(1,50:100)


% Случайно выберите 100 точек данных для отображения
sel = randperm(size(X, 1));
sel = sel(1:100);



displayData(X(sel, :));

%sel(1,50:100)'
%break;

fprintf('Программа приостановлена. Нажмите Enter, чтобы продолжить.\n');
pause;

%% ================ Part 2: Loading Pameters ================
% В этой части упражнения мы загружаем некоторые предварительно инициализированные
% параметров нейронной сети.

    fprintf('\nLoading Saved Neural Network Parameters ...\n')

% Загружаются веса в переменные:
% Theta1 matrix 25x400
% Theta2 matrix 25x400
load('ex3weights.mat');
% Перенос матриц в несжатом виде в файл
%save Theta1.txt Theta1;save Theta2.txt Theta2;break;


%% ================= Part 3: Implement Predict =================
% После обучения нейронной сети, мы хотели бы использовать ее для прогнозирования
% этикетки. 
% Теперь вы будете реализовывать функцию «прогнозирования predict», чтобы использовать
% нейронной сети для прогнозирования меток обучающего набора. 
% Это позволяет вы вычисляете точность тренировочного набора.
    pred = predict(Theta1, Theta2, X);

    fprintf('\nТочность тренировочного набора: %f\n', mean(double(pred == y)) * 100);

    fprintf('Программа приостановлена. Нажмите Enter, чтобы продолжить.\n');
    pause;
      
     % Чтобы дать вам представление о выходе сети, вы также можете запустить
     % через примеры по одному, чтобы увидеть, что он предсказывает.

     % Случайно переставлять примеры
    rp = randperm(m);

    for i = 1:m
      % Display
      fprintf('\nОтображение примера изображения\n');
      displayData(X(rp(i), :));

      pred = predict(Theta1, Theta2, X(rp(i),:));
      if pred==10,pred=0;endif;
      fprintf('\nПредсказание нейронной сети: %d (digit %d)\n', pred, mod(pred, 10));

      % Pause with quit option
      s = input('Приостановлено - нажмите Enter для продолжения, `q` для выхода:','s');
      if s == 'q'
       break
      endif
    
     endfor
     
endfunction     
     