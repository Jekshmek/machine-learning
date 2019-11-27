function [theta,J] = usedCost2()


load X.dat;
load y.dat;
%load theta.dat;

while 1,
  theta = [rand(1,1)*10;rand(1,1)*10];
  J = costTest(X,y,theta);
  if J < 0.0001,
    break;
  end;
end;

endfunction