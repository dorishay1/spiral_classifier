function [Net] = net_init(N, L, method, g_funcs)

if method == 'R'
    %Random
    Net = arrayfun(@(n, n1, g) struct('W', (0.1*randn(n1, n+1)), ...	% weights
        'g', g), ...                  % activation function
        N(1:L), N(2:L + 1), g_funcs);
    
elseif method == 'X'
    %Xavier
    mu = 0;
    M = 2/(N(1)+N(end));
    
    % Initialize the layers' weights and activation functions
    Net = arrayfun(@(n, n1, g) struct('W', M.*randn(n1, n+1)+mu, ...	% weights
        'g', g), ...                  % activation function
        N(1:L), N(2:L + 1), g_funcs);
end
