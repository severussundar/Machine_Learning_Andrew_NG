function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

predictions = X*theta;
predictions = sigmoid(predictions);
cost1 = -1*(y .* log(predictions));
cost2 = -1*((1 - y) .* log(1 - predictions));
cost = (cost1+cost2)/m;
cost3 = sum(theta(2:end) .^2);
cost3 = (lambda*cost3)/(2*m);
J = sum(cost) + cost3;

grad = X' *(predictions-y);
grad = grad/m;
grad(2:end) = grad(2:end) + (lambda*theta(2:end))/m;

% =============================================================

end
