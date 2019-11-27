 %function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
  function [Buf_zero,Buf_one, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
% Целью линейной регрессии является минимизация функции стоимости.

%GRADIENTDESCENT Выполняет градиентный спуск для изучения тета
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) обновляет тета
% выполнения шагов градиента num_iters с альфа-скоростью обучения

%  Инициализируйте некоторые полезные значения
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
lenTheta=length(theta);
thetaBuf = zeros(lenTheta,1);% вектор буффер для всех thata.0 thata.1 и т.д.

 Buf_zero = zeros(num_iters,1);
 Buf_one = zeros(num_iters,1);
    for iter = 1:num_iters
% С каждым шагом градиентного спуска ваши параметры j приближаются к оптимальным
% значениям, которые достигнут самая низкая стоимость


% ====================== YOUR CODE HERE ======================
%  Инструкции: выполнить один шаг градиента для вектора параметров
 	% тета.
 	%
 	% Подсказка: при отладке может быть полезно распечатать значения
 	% от стоимости функции (computeCost) и градиента здесь.
%  surf([0.1 100 10000;0.1 100 10000; ]);
% surf([Q1,Q0,J_h]);



% 1 variant ============================================================
%      theta = theta - alpha/m*(X'*((X*theta)-y));
%      0.62843 1.01013
% 2 variant ============================================================
%     theta = theta - alpha*((X'*(X*theta-y))/m) ;
%      0.62843 1.01013
% 3 variant ============================================================
%       theta = theta - alpha/m*sum((X*theta-y) .* X)';
%       0.62843 1.01013
% 4 variant ============================================================
%      theta = theta - (alpha/m)*(X'*((X*theta)-y));
%       0.62843 1.01013
% 5 variant ============================================================
%     temp0 = theta(1, 1) - ((1/m)*(0.01)*(sum((X*theta)-y)));
%     temp1 = theta(2, 1) - ((1/m)*(0.01)*(sum(((X*theta)-y)'*(X(:, 2)))));
%     theta = [temp0 ; temp1] ;
%     0.62843 1.01013
% 6 variant ============================================================
%     predictions =  X * theta;
%     updates = X' * (predictions - y);
%     theta = theta - alpha * (1/m) * updates;
%     0.62843 1.01013
% - 7 variant ============================================================
%      theta(1) = theta(1) - alpha * (X*theta-y)' * X(:,1) / m
%      theta(2) = theta(2) - alpha * (X*theta-y)' * X(:,2) / m
% 8 variant ============================================================
%     predictions = X * theta;                  % hypothesis
%     errors = predictions - y;                 % hypothesis - y
%     delta = (1 / m) * sum(errors .* X(2));    % col1 is 1's
%     theta = theta - alpha * delta;
%      1.48185 0.48185
% 9 variant ============================================================
%       Errors = (X * theta) - y;			% m*1 vector
%       delta = (1/m * Errors' * X) ;		% 1*n vector
%       theta = theta - alpha * delta';
% 0.62843 1.01013
% 10 variant ===========================================================
%      i =1;
% 	   j = 0;
% 	   while i<=m
% 	     j = j + theta(1)*X(i,1) + theta(2)*X(i,2) - y(i) ;
% 		 i = i+1;
% 	   end
% 	   j = alpha * j;
% 	   j = j/m;
% 	   ans = j;
% 	   
% 	   i =1;
% 	   j = 0;
% 	   while i<=m
% 	     j = j + (theta(1)*X(i,1) + theta(2)*X(i,2) - y(i))*X(i,2); 
% 		 i = i+1;
% 	   end
% 	   j = alpha * j;
% 	   j = j/m;
% 	   
%      theta(1) =theta(1) -ans;
%      theta(2) =theta(2) - j; 
%        
%     0.62843 1.01013
% 11 variant ============================================================
    x = X(:,2);
    h = theta(1) + (theta(2) * x);
    
    new_theta_zero = theta(1) - alpha * (1/m) * sum(h-y);
    new_theta_one = theta(2) - alpha * (1/m) * sum((h-y) .* x);
    theta = [new_theta_zero0; new_theta_one];
    
    Buf_zero(iter)= new_theta_zero ;
    Buf_one(iter)=  new_theta_one ;
%    0.62843 1.01013
    
% 12 variant =============================================================
%    g = 0;
%    k = 0;
%    for i = 1 : m
%        k = k + theta(1) + theta(2) * X(i,2) - y(i);
%        g = g + (theta(1) + theta(2) * X(i,2) - y(i)) * X(i, 2);
%    end
%    theta(1) =theta(1) - alpha /m * k;
%    theta(2) = theta(2) - alpha /m * g;
%    0.62843 1.01013
    
    % my variant ===========================================================
    % X1 = [ones(20,1) (exp(1) + exp(2) * (0.1:0.1:2))'];
    % Y1 = X1(:,2) + sin(X1(:,1)) + cos(X1(:,2));
    % sprintf('%0.5f ', gradientDescent(X1, Y1, [0.5 -0.5]', 0.01, 10));

    %  Predictions = X * theta;
    %  ErrorsPredictions = (Predictions - y);
    %
    %  for thetaItem = 1:lenTheta,
    %    temp =  alpha * X;
    %    thetaBuf(thetaItem) = temp;
    %  endfor;
    % theta=thetaBuf;
     
% Сохранить стоимость J в каждой итерации

J_history(iter) = computeCost(X, y, theta);

endfor

endfunction

 
  
  