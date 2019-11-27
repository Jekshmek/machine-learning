function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Выполняет градиентный спуск для изучения тета
% theta = GRADIENTDESCENTMULTI (x, y, theta, alpha, num_iters) обновляет theta на
% выполнения шагов градиента num_iters с альфа-скоростью обучения

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

%lenTheta=length(theta);
%thetaTemp = zeros(lenTheta,1);
 
for iter = 1:num_iters

% ====================== YOUR CODE HERE ======================
% Инструкции: выполнить один шаг градиента для вектора параметров
% тета.
%
% Подсказка: при отладке может быть полезно распечатать значения
% от стоимости функции (computeCostMulti) и градиента здесь.
%


       %#{
         err = (X * theta) - y;			% m*1 vector
         cof = (1/m)*alpha;
         %theta = theta - cof * (X'*err);  
         % или таким образом считается sum
          theta = theta - cof * sum(err .* X)'; 
         %#}
       
       #{
         err = ((X*theta)-y)'; 
         cof = (1/m)*alpha;
         
         for t_i = 1:lenTheta,
           thetaTemp(t_i,1) =  theta(t_i, 1) - cof * sum(err*(X(:, t_i))); % X(:, t_i) - это конкретно один тип/ряд входных аргументов  
         endfor;
         theta=thetaTemp;
        #}
       
    
       #{
        for i=1:size(X,2)
            temp(i)=theta(i)-alpha*sum((X*theta-y).*X(:,i))/m;
        end
        theta=temp;
       #}

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

endfor

endfunction
