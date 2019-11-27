function [Predictions] = foo()

X=[1 0;
1 1;
1 2;
1 3;
1 4;
1 5;
1 6];
theta=[0;1];
y=[0;1;2;3;4;5;6];



m = length(y);
% ---------------------------------------------
Predictions = X * theta; % Predictions = [0;1;2;3;4;5;6];
ErrorsPredictions = (Predictions - y).^ 2; % ([0;1;2;3;4;5;6] - [0;1;2;3;4;5;6])  .^ 2 = [0;0;0;0;0;0;0]
J = 1/(2*m) * sum(ErrorsPredictions); % 1/(2*7) * sum([0;0;0;0;0;0;0])
% ---------------------------------------------



endfunction