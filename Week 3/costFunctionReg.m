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


h = sigmoid(X*theta);

theta_j = theta(2:end);

J = 1/m * sum(-y.*log(h)-((-y+1).*log(-h+1))) + ...
    (lambda/(2*m) * sum(theta_j.^2));

%grad(1) = (1/m * sum((h(1)-y(1)).*X(1)));

grad = (1 / m) * sum( X .* repmat((sigmoid(X*theta) - y), 1, size(X,2)) );
grad(:,2:length(grad)) = grad(:,2:length(grad)) + (lambda/m)*theta(2:length(theta))';

%grad(2:end) = ((1/m) * (sum((h(2: end)-y(2:end)).*X(2:end))) + ((lambda/m)*theta_j));



% =============================================================

end
