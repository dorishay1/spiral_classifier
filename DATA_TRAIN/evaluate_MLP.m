function [err, acc] = evaluate_MLP(Y, Y0, labels,fun_loss)
%EVALUATE_MLP Evaluate a MLP
%   Get the squared error and accuracy of a MLP on a given dataset. 

% Get statistics
err         = fun_loss(Y',Y0);         % squared error
Y_labels	= output2labels(Y);
acc         = mean(double(Y_labels' == labels));	% accuracy

end

