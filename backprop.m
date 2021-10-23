function [Net] = backprop(s, Y, Y0, Net, eta, L, alpha, lambda, reg)
% Back propagation

delta = (Y - Y0).*s{L + 1}.der;

%for no momentum
if alpha == 0
    alpha = 1;
end

for l = L:-1:1
    
    dW = -eta.*(delta * s{l}.act').*alpha^(L-l) ;           % get the weights update
    delta = ((Net(l).W(:,(1:end-1)))'*delta).*s{l}.der;     % update delta
   
    %regolations
    if reg == 'L1'
        penalty = (eta*lambda)/(length(Net(l).W)).*sign(Net(l).W);
    elseif reg == 'L2'
        penalty = (eta*lambda)/(length(Net(l).W)).*Net(l).W;
    elseif reg == 0
        penalty = 0;
    end
    
        Net(l).W = Net(l).W + dW - penalty;                 % update the weights
    end
end

