function checkNNGradients(lambda)
%CHECKNNGRADIENTS Создает небольшую нейронную сеть для проверки
% обратного распространения градиентов
% CHECKNNGRADIENTS (лямбда) Создает небольшую нейронную сеть для проверки
% градиентов обратного распространения, выводятся аналитические градиенты
% произведено вашим кодом backprop и числовыми градиентами (вычислено
% используя computeNumericGradient). Эти два градиентных вычисления должны
% приводят к очень похожим значениям.
%

if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
end

input_layer_size = 3;
hidden_layer_size = 5;
num_labels = 3;
m = 5;

% We generate some 'random' test data
Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
Theta2 = debugInitializeWeights(num_labels, hidden_layer_size);
% Reusing debugInitializeWeights to generate X
X  = debugInitializeWeights(m, input_layer_size - 1);
y  = 1 + mod(1:m, num_labels)';

% Unroll parameters
nn_params = [Theta1(:) ; Theta2(:)];

% Short hand for cost function
costFunc = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, ...
                               num_labels, X, y, lambda);

[cost, grad] = costFunc(nn_params);
numgrad = computeNumericalGradient(costFunc, nn_params);

% Visually examine the two gradient computations.  The two columns
% you get should be very similar. 
disp([numgrad grad]);
fprintf(['Выше две колонки должны быть очень похожи.\n' ...
         '(Слева - ваш числовой градиент, справа - аналитический градиент)\n\n']);

% Оцените норму разницы между двумя решениями.
% Если у вас правильная реализация и предполагается, что вы использовали EPSILON = 0,0001
% в computeNumericGradient.m, тогда разница ниже должна быть меньше 1e-9
diff = norm(numgrad-grad)/norm(numgrad+grad);

fprintf(['Если ваша реализация обратного распространения верна, то \n' ...
         'относительная разница будет небольшой (менее 1e-9). \n' ...
         '\nОтносительная разница: %g\n'], diff);

end
