function ex1_multi()

%% ================ Part 1: Масштабирование, нормировка среднего ================

    %% Clear and Close Figures
    clear ; close all; clc

    fprintf('Загрузка данных ...\n');

    %% Load Data
    %data = load('ex1data2.txt');
    %data = load('ex1data2test.txt');
    data = load('test2.txt');
    %data = load('test3.txt');
    X = data(:, 1:2);% сразу два столбца грузим
    y = data(:, 3);
    m = length(y);
   
    % График данных, с одним аргументом
    %plot(X(:, 1 ),y ,'dg','markersize',5,'linewidth',2),xlabel("Size, without N bathroom"),ylabel("Price");
   
   % График данных
    plot(X,y ,'dg','markersize',5,'linewidth',2),xlabel("Size, without N bathroom"),ylabel("Price");
   
    % Print out some data points
    fprintf('Первые 10 примеров из набора данных: \n');
    fprintf(' x = [%f %f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');
 
 
    % Scale features and set them to zero mean
    fprintf('Нормализующие функции ...\n');

    [X_normalize mu sigma] = featureNormalize(X);
     
    % mu - среднее арифметическое. [дефолт]
    % sigma - Стандартное отклонение
  
    fprintf('Первые 10 примеров из набора нормализованных данных: \n');
    fprintf(' x = [%f %f]  \n', [X_normalize(1:10,:)]');
    
     
    fprintf('Program paused. Press enter to continue.\n');
    pause;
    
    % Add intercept term to X
    X_normalize = [ones(m, 1) X_normalize];
     
    
    
 %% ================ Часть 2. Градиентный спуск ================

    % ====================== ВАШ КОД ЗДЕСЬ ======================
    % Инструкции: мы предоставили вам следующий стартер
    % кода, который запускает градиентный спуск с определенным
    % скорость обучения (альфа).
    %
    % Ваша задача - сначала убедиться, что ваши функции -
    % computeCost и GradientDescent уже работают с
    % этот стартовый код и поддержка нескольких переменных.
    %
    % После этого попробуйте запустить градиентный спуск с
    % разных значений альфа и посмотреть, какой из них дает
    % вам лучший результат.
    %
    % Наконец, вы должны завершить код в конце
    %, чтобы предсказать цену 1650 кв. футов, 3-х комн.
    %
    % Подсказка: с помощью команды «Hold on», вы можете построить несколько
    % графики на том же рисунке.
    %
    % Подсказка: при прогнозировании убедитесь, что вы выполняете ту же нормализацию функций.
    %

       
      fprintf ('Решение с помощью градиентного спуска ... \n');
      % Choose some alpha value
      alpha = 0.1;
      num_iters = 400;

      % Init Theta and Run Gradient Descent 
       theta = zeros(3, 1);% [0;0;0]
      [last_theta, J_history] = gradientDescentMulti(X_normalize, y, theta, alpha, num_iters);
 
 
 
      
      % График данных Size и прямой описывающей линейную функцию 
      %(Не точно так как прямая учитывает нормализованные данные и все аргументы,а тут один Size)
      plot( X(:,1),X_normalize*theta, '-') 
      hold on;
      plot(X(:, 1 ),y ,'dg','markersize',5,'linewidth',2),xlabel("Size, without N bathroom"),ylabel("Price");
      
 
      % Plot the convergence graph
      figure;
      plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);% вернуть индексы с 1-го елемента
      xlabel('Number of iterations');
      ylabel('Cost J');

      % Display gradient descent's result
      fprintf('Тета, рассчитанная по градиентному спуску: \n');
      fprintf(' %f \n', last_theta);
      fprintf('\n');

      % Оцените цену 1650 кв. Футов, 3-х комн. Дома
      % ====================== ВАШ КОД ЗДЕСЬ ======================
      % Напомним, что первый столбец X является единичным. Таким образом, это делает
      % не нужно нормализовать.
     
    
       
        % Детальный разбор денормализации для двух входных данных
       #{
         size_house = 17;
         br_house = 4;

        size_house_norm = (size_house-mu(1)) / sigma(1);
        br_house_norm = (br_house-mu(2)) / sigma(2);

        house_features = [17, 4];
        house_features_norm = [1, size_house_norm, br_house_norm];

        % Print out some data points
        fprintf('\n\nPrice to firecasat for: [%.3f %.3f],\n', mu);
        fprintf(' - SqFt: %.3f  ==> Normalized: %.3f,\n', size_house_norm);
        fprintf(' - BedRoom: %.3f ==> Normalized: %.3f,\n', br_house_norm);

        price = house_features_norm * last_theta;
        fprintf('Pice=%f\n\n',price);
       #}
       #{                                                                                                                                                            
          mu =  [8.0000   4.0000]
          sigma = [4.4721e+000  2.5820e-009]
      
          last_theta =
                     8.0000e+002
                     4.4721e+002
                    -3.4145e-013
      
        result = input[first second]-mu => [17   4] - [8.0000   4.0000] = [9   0]
        normal_result = result ./sigma => [9   0] ./ [4.4721e+000   2.5820e-009] = [2.01248  0.00000]
        normal_result*last_theta => [1   2.0125]*[8.0000e+002  4.4721e+002  -3.4145e-013] 
       #}
      
      % Входные данные для предсказания подвергнуть нормализации
      % количество входных данных должно соответствовать всему перечню входных данных
      % так как количество коефициентов `mu` и `sigma` именно столько их содержат.
      
      
      size_house = 17;%1650;% 17;
      br_house = 4;%3;% 4;
      features = [size_house br_house];
       % sigma [ 4.4721e+000  2.5820e-009]
        % (features-mu) = [9 0]
      normal_features = [1 (features-mu)./sigma];
      % normal_features [ 1.00000   2.01246  -0.25820]
      % last_theta  [8.0000e+002; 4.4721e+002; -3.4145e-013]
      
      price = normal_features * last_theta;
      
       % Показать входные данные до нормализации и после
         size_house_norm=normal_features(2); 
         br_house_norm=normal_features(3);
        fprintf('\n\nPrice to firecasat for: [%.3f %.3f],\n', mu);
        fprintf(' - SqFt: %.3f  ==> Normalized: %.3f,\n',size_house, size_house_norm);
        fprintf(' - BedRoom: %.3f ==> Normalized: %.3f,\n',br_house, br_house_norm);
        
      % features = [1 1650 3]; 
      % features = [1 17 4];
      % price = last_theta(1)*features(1)+last_theta(2)*(features(2)-mu(1))/sigma(1)+last_theta(3)*(features(3)-mu(2))/sigma(2);
      % ============================================================

       fprintf(['Прогнозируемая цена для дома с площадью %d кв. Футов, %d-х комн. \nСоставляет ' ...
                '(используя градиентный спуск):\n  $%f\n'],size_house,br_house, price);
 
      fprintf('Program paused. Press enter to continue.\n');
       pause;
     
%% ================ Часть 3. Нормальные уравнения ================
      clear ; close all; 
      fprintf ('---------------------------------------\n');
      fprintf ('Решение с помощью Normal Equations ... \n');
      % Странно , но тут градиентный спуск не нужен ?!

      % ====================== ВАШ КОД ЗДЕСЬ ======================
      % Инструкции: следующий код вычисляет закрытую форму
      % решение для линейной регрессии с использованием нормального
      % уравнений. Вы должны заполнить код в
      % normalEqn.m
      %
      % После этого вы должны завершить этот код
      %, чтобы предсказать цену 1650 кв. футов, 3-х комн.
      %


      %% Load Data
      data = load('ex1data2.txt');
      %data = load('test2.txt');
      X = data(:, 1:2);
      y = data(:, 3);
      m = length(y);

      % Add intercept term to X
      X = [ones(m, 1) X];

      % Calculate the parameters from the normal equation
      normal_theta = normalEqn(X, y);

      % Display normal equation's result
      fprintf('theta рассчитывается по normal equations: \n');
      fprintf(' %f \n', normal_theta);
      fprintf('\n');
 
      features = [1 1650 3];
      %features = [1 17 4];
      price = features * normal_theta;% то же самое  theta(1)*1+theta(2)*1650+theta(3)*3;
      % т.е. перемножить елементы и сложить
      % price = theta(1)*features(1)+theta(2)*features(2)+theta(3)*features(3);
      fprintf(['Прогнозируемая цена для дома с площадью %d кв. Футов, %d-х комн. \nСоставляет ' ...
               '(используя normal equations):\n $%f\n'],features(2),features(3), price);


  endfunction
