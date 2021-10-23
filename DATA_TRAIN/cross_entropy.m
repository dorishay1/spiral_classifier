function [err] = cross_entropy(Y,Y0,Net,L)
%cross_entropy a function that do @crossentropy loss with a panelty
% parametrs
alpha = 0;
TOT_W =[];
Lamda = 2

for i=1:L
    W = reshape(Net(i).W,[],1);
    TOT_W = [TOT_W; W];
end 

% ridge
L2 = (Lamda/2)*sqrt(sum(TOT_W.^2));

%lasso
L1 = Lamda*sum(abs(TOT_W));

%penalty
panelty = alpha*L2+(1-alpha)*L1

err = -mean(Y0*ln(Y)+(1-Y0)*ln(1-Y))+panelty;
end

