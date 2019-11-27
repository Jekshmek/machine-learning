function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Нормализует функции в X
% FEATURENORMALIZE (X) возвращает нормализованную версию X, где
% среднее значение каждого признака равно 0 и стандартное отклонение
% равно 1. Часто это хороший шаг предварительной обработки, когда
% работает с алгоритмами обучения.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Инструкции: сначала для каждого измерения объекта вычислите среднее
% от функции и вычесть ее из набора данных,
% хранения среднего значения в мю. Далее вычисляем
% стандартного отклонения каждого признака и деления
% каждой функции по стандартному отклонению, хранение
% стандартное отклонение в сигме.
%
% Обратите внимание, что X является матрицей, где каждый столбец является
% особенность, и каждая строка является примером. Тебе нужно
% выполнить нормализацию отдельно для
% каждой функции.
%
% Подсказка: вам могут пригодиться функции mean и std.
%       
 
  % Нормализация на среднее
     %#{
      mu = mean(X);% mean - среднее арифметическое. [дефолт]  `mean([10 20 30 40 50]) == 30`
      sigma = std(X); %std - Стандартное отклонение `std([10 20 30 40 50]) ==  15.811`
      %X_norm = (X .- mu) ./ sigma; % Возможен NaN если нет малейшего отклонения то 0/0=NaN
      
      X_norm = (X .- mu) ./ sigma;
     %#}  
     
     
% ============================================================
    % Тоже рабочие варианты
    #{
      mu = mean(X);
      X_norm = bsxfun(@minus, X, mu);
      sigma = std(X_norm);
      X_norm = bsxfun(@rdivide, X_norm, sigma);
    #}

    % Тоже рабочие варианты
    #{
     for i = 1: size(X,2)
        X_norm( :, i) = (X ( :, i) - mu (i)) / sigma(i) ;
     end
    #}

    % Тоже рабочие варианты
    #{
      for i = 1:size(X,2)
        mu(i) = mean(X(:,i));
        X_norm(:,i) = X_norm(:,i) - mu(i);
        sigma(i) = std(X_norm(:,i));
        X_norm(:,i) = X_norm(:,i) / sigma(i);
       %X_norm(:,i) = ( X(:,i) - mu(i) ) / sigma(i);
     endfor
    #}

     % Тоже рабочие варианты
    #{
      mu = mean(X);
      X_norm = bsxfun(@minus, X, mu);
      sigma = std(X_norm);
      X_norm = bsxfun(@rdivide, X_norm, sigma);
    #}

% ============================================================

%!test
%! [Xn mu sigma] = featureNormalize([1 ; 2 ; 3]);
%! Xn_expected = [ -1 0 1 ]';
%! mu_expected = 2;
%! sigma_expected = 1;
%! assert(Xn, Xn_expected, 1);
%! assert(mu, mu_expected, 1);
%! assert(sigma, sigma_expected, 1);

end
