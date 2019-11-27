function testSum()
    % Когда вектор умножаем на матрицу то сумма будет одинакова для двух вариантов:
    %  v=[1;2;3]; M=[ 1 2 10;2 4 20;5 4 30]; sum(v .* M)' == M'*v
  
X=[1 10;1 20;1 30];
theta=[1;2];
y=[2;4;5];
%                           X  * theta                               X  * theta - y
e=X*theta-y;% [1 10;1 20;1 30] * [1;2] = [1*1+2*10;1*1+2*201*1+2*30]=[21;41;61] - [2;4;5]=[19;37;56]
%   19
%   37
%   56

% (e.*X) => [19;37;56] * [1 10;1 20;1 30] => [1*19 10*19; 1*37 20*37;1*56 30*56] => [19 190;37 740;56 1680]
%     19    190
%     37    740
%     56   1680
%    sum(e .* X) =>  [112   2610]
%    sum(e .* X)' => [112;2610]
% или то жэ самое
% из [1 10;1 20;1 30]  => X' => [1 1 1;10 20 30]
%  (X'*e) => [1 1 1;10 20 30] * [19;37;56] = [1*19 + 1*37 +1*56 ; 10*19 + 20*37 + 30*56] =[ 112;2610]
  
  % т.е. sum(e .* X)' == X'*e   => sum([1;2;3] .* [10 20 30 40 50]) == ([10 20 30 40 50]')*[1;2;3]

 

r1= sum(e .* X)';
r2= X'*e;

 r1
 r2

endfunction