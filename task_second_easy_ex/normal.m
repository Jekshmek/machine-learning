function [result] = normal() 
  
load X.dat;
load y.dat;
  
result = 1 / (pinv(X)*X) * pinv(X) * y';
  
endfunction