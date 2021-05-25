function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
reg=0;
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
[r,c]=size(X);
z=X*theta;
g=sigmoid(z);
for i=1:m
  J=J+(-y(i)*log(g(i))-(1-y(i))*log(1-g(i)));
  grad=grad+(g(i)-y(i))*X(i,:);
endfor
J=J/m;
grad=grad/m;
grad=grad(1,:);
for i=2:c
  reg=reg+theta(i)^2;
endfor
reg=reg*lambda/(2*m);
J=J+reg;
c=length(grad);
for i=2:c
  grad(i)=grad(i)+(lambda/m)*theta(i);
endfor


% =============================================================

end
