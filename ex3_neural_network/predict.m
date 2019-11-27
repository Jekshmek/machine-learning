function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%#{
  a1 = [ones(m, 1), X];
  z1 = sigmoid(a1 * Theta1');
  a2 = [ones(m, 1), z1];
  z2 = sigmoid(a2 * Theta2');

  p = max(z2, [], 2);

  for i = 1:m
      for j = 1:num_labels
          if z2(i, j) == p(i)
              p(i) = j;
          end
      end
  end
%#}

% или
#{
  a1 = [ones(m, 1) X];
  %> size(a1)
  %    5000    401

  % hidden layer 2: activation unit 
  a2 = sigmoid (a1 * Theta1');
  % Add vector of 1s
  a2 = [ones(size(a2,1), 1) a2];

  % hidden layer 3: activation unit 
  a3 = sigmoid (a2 * Theta2');

  % using use max(A, [], 2) to obtain the max for each row.
  [val, index] = max(a3,[],2);

  p = index;

#}



% =========================================================================


end