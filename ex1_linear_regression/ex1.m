function ex1()
 

%% ======================= Part 2: Plotting ====================================
#{
  fprintf('Plotting Data ...\n')
  data = load('ex1data1.txt');
  X = data(:, 1); y = data(:, 2);
  m = length(y); % number of training examples

  % Plot Data
  % Note: You have to complete the code in plotData.m
  plotData(X, y);
  clear data,X,m;

  fprintf('Program paused. Press enter to continue.\n');
  pause;
#}

%% =================== Part 3: Cost and Gradient descent =======================
  % initialization data 
  % data = load('ex1data1.txt'); 
  data = load('test1.txt');% Тестовые данные для theta=[0;1] и J = 0;
  y = data(:, 2);% обучающие данные
  m = length(y);
  X = [ones(m, 1), data(:,1)];% входные аргументы для обучающих данных
  theta = [0.0001;0.999];% zeros(2, 1); %[0;1] начальные параметры модели. Количество рядов должно соответствовать количеству колонок аргументов X
  alpha=0.001; % шаг,скорость обучения
  iterations=100;%2000; % количество итераций корректировки параметров
 
  
   % -------------------------
   % computeCost(X, y, theta);
   % -------------------------

  % Print out some data points
    fprintf('Первые 10 примеров из набора данных: \n');
    fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');
 
   [last_theta,J_cost_history,all_theta] = gradientDescent(X, y, theta, alpha, iterations);
   % такой вариант берет не последний результат итерации минимума, а минимум по индексу
   % так как возможен вариант проскочить минимум из-за большого шага alpha
   [indx_col,indx_row]=find(J_cost_history == min(J_cost_history));
   best_theta = [ all_theta(indx_col(1),1);all_theta(indx_col(1),2)];
   if length(indx_col) > 1,
     fprintf('Перегрузка! Обнаруженный минимум ниже не опускается\n'); 
   endif;
   if [ all_theta(indx_col(1),1);all_theta(indx_col(1),2)] != last_theta,
    fprintf('Пропуск минимума! Недостаточно итераций или большой шаг обучения\n'); 
   endif;
   fprintf('#2 Best theta %f:%f, Min errors %f\n',
   all_theta(indx_col(1),1),all_theta(indx_col(1),2),min(J_cost_history));
   #{
    [indx_col,indx_row]=find(J_cost_history == min(J_cost_history));
    fprintf('#2 Best theta %f:%f, Min errors %f\n',
    all_theta(indx_col(1),1),all_theta(indx_col(1),2),min(J_cost_history)); break;
   #}
  
  % Графики
  % Наложение графика данных на график прямой
  subplot(1,2,1)
  % график прямой ,  X(:,2) входные данные, X*last_theta это вектор данных умноженный на лучший параметр
  plot(X(:,2), X*best_theta , '-')   
  hold on;
  plot(X(:,2),y,'dg','markersize',5,'linewidth',2);% графика данных
  xlabel('Population'),ylabel('Revenue');
  subplot(1,2,2)
  title('Error predictions');
  plot(J_cost_history);
  hold off;
  
  
  % График скорости нахождения оптимальных параметров функции
  figure(2),surf([all_theta(:,1),all_theta(:,2),J_cost_history]),
  xlabel("theta 0"),ylabel("theta 1"), zlabel("J(theta N)");
  
  
 
 %% ============= Part 4: Visualizing J(theta_0, theta_1) ======================
 %#{
    % не корректно
    % Grid over which we will calculate J
    theta0_vals = linspace(-10, 10, 100);
    theta1_vals = linspace(-1, 4, 100);

    % initialize J_vals to a matrix of 0's
    J_vals = zeros(length(theta0_vals), length(theta1_vals));

    % Fill out J_vals
    for i = 1:length(theta0_vals)
        for j = 1:length(theta1_vals)
        t = [theta0_vals(i); theta1_vals(j)];
        J_vals(i,j) = computeCost(X, y, t);
        end
    end
  
    % Contour plot
    figure;
    % Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
    contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
    xlabel('\theta 0'); ylabel('\theta 1');
    hold on;
    plot(all_theta(:,2),all_theta(:,1), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
  %#}
 
 

  
endfunction

