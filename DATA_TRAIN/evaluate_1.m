function [err, acc] = evaluate_1(Y_out, Y,fun_loss)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

prediction_Y = Y_out > 0.5;
err = fun_loss(Y_out,Y);
%err = mean(sum(0.5.*(Y - Y_out).^2));      % squared error
acc = mean(prediction_Y==Y);
end

