function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% Set the values of C and sigma to very small numbers 
C = 0.001;
sigma = 0.001;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%

% The temp variable starts as a big number, as it will later be compared to
% the error of the prediction 
temp = 1000000

% Loops until optimal value of C and sigma are found
while true

    %Train and predict the model in the cross validation dataset
    model = svmTrain(Xval, yval, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    pred = svmPredict(model, Xval);
    
    error = mean(double(pred ~= yval));
    
    %Increases C and sigma by a factor of 10 if error is not optimal
    if error < temp
        temp = error;
        
        C = C * 10;
        sigma = sigma * 10;
    end 
    
    if error > temp
       C = C / 10;
       sigma = sigma / 10;
       break
    end
% =========================================================================

end
