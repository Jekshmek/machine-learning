function [theta, J_history,thetaBuf] = gradientDescent(X, y, theta, alpha, num_iters)
% Целью линейной регрессии является минимизация функции стоимости.

%GRADIENTDESCENT Выполняет градиентный спуск для изучения тета
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) обновляет тета
% выполнения шагов градиента num_iters с альфа-скоростью обучения

%  Инициализируйте некоторые полезные значения
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
lenTheta=length(theta);
thetaTemp = zeros(lenTheta,1);% вектор буффер для всех thata.0 thata.1 и т.д.
thetaBuf= zeros(num_iters, 2);

    for iter = 1:num_iters
% С каждым шагом градиентного спуска ваши параметры j приближаются к оптимальным
% значениям, которые достигнут самая низкая стоимость


% ====================== YOUR CODE HERE ======================
%  Инструкции: выполнить один шаг градиента для вектора параметров
 	% тета.
 	%
 	% Подсказка: при отладке может быть полезно распечатать значения
 	% от стоимости функции (computeCost) и градиента здесь.
%
 
   
   
% 1 variant ====================================================================
       % тут так же все theta обновляются одновременно т.е. при просчете следующего
       % значения используется старый вариант theta(i) и когда все просчитанны
       % они все обновляются
          %#{
         err = (X * theta) - y;			% m*1 vector
         cof = (1/m)*alpha;
    
         % theta = theta - cof * (X'*err);  
         % или таким образом считается sum
         theta = theta - cof * sum(err .* X)'; 
         
         thetaBuf(iter,1)= theta(1);
         thetaBuf(iter,2)= theta(2);
         %#}
 
% my variant ===================================================================
       % дольше работает из-за наличия цыкла
       % в этом варианте обновление theta(i) более наглядно, так жек одновременно
       % когда все theta(i) обновились они заменяют старые theta(i)
       #{
        
       cof = (1/m)*alpha;
       err = ((X*theta)-y)'; 
     
       for t_i = 1:lenTheta,
         thetaTemp(t_i,1) =  theta(t_i, 1) - cof * sum(err*(X(:, t_i))); % X(:, t_i) - это конкретно один тип/ряд входных аргументов
         thetaBuf(iter,t_i) = thetaTemp(t_i,1);       
       endfor;
       theta=thetaTemp;  
       #}
    
  %predictions = X * theta; %вектор ,матрица входных данных умноженная на вектор параметров функции-гипотезы      
  %errors = predictions - y; %вектор ошибок  
  %sumRowErrors = sum(errors .* X);%матрица ошибок, каждый елемент матрицы умножается на элемент вектора и суммируем ряды на выходе матрица размера "1 х size(X)"
      
     
% Сохранить стоимость J в каждой итерации

J_history(iter) = computeCost(X, y, theta);

endfor

endfunction