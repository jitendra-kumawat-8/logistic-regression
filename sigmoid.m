function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));
l=size(z);
r=l(1); c=l(2);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
for i=1:r
  for j=1:c
    g(i,j)=1/(1+e^-z(i,j));
  endfor
endfor




% =============================================================

end
