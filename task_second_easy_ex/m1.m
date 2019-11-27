
function mukherjee()
  
data = load('ex1data1.dat');

X = data(:, 1); 
y = data(:, 2);
m = length(y); % number of training examples

% Plot Data
% Note: You have to complete the code in plotData.m
  plotData(X, y);

 
endfunction