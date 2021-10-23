function [s, Y] = feedforward(X,Net,L)
      % Temporary neurons activities per layer
        s = cell(L + 1, 1);
        
        % Forward pass
        
        s{1} = struct('act', X, ...                     % set the input
                      'der', zeros(size(X)));           % for completeness
        for l = 1:L
            
            s{l}.act(size(s{l}.act,1)+1,:)=ones(1,size(s{l}.act,2));
            [g, gp] = Net(l).g(Net(l).W * s{l}.act);	% get next layer's activities 
                                                        % (and derivatives)
            s{l+1}  = struct('act', g, ...
                             'der', gp);                % save results per layer
        end
        
        Y = s{L + 1}.act;                               % get the output
end

