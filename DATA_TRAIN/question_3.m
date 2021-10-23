clear; close all; clc;

% question number 3 ex.2:

%% loading data
% train data
load('DATA_TRAIN.CSV');
% test data
load('DATA_valid.csv');

%% data

% train data 
train_data = DATA_TRAIN(:,1:2);
train_data(:,3)= atan(train_data(:,2)./train_data(:,1));
train_label =  DATA_TRAIN(:,3);
Y_train = (train_label+1 ==1:2); % one code teacher 

%test data
test_data = DATA_valid(:,1:2);
test_data(:,3)= atan(test_data(:,2)./test_data(:,1));
test_label =  DATA_valid(:,3);
Y_test = (test_label+1 ==1:2); % one code teacher 



% Network structure
N = [3, 30, 30, 2];	% number of neurons per layer
L = length(N) - 1;	% number of layers
fun_loss = @crossentropy;
w_init_param = @XavierInit;         %weights init technique

% training parameters
eta = 1e-3; % step size
batch_size = 100;
epoch = 120;
alpha = 0.4; 



% Activation functions (per layer)
g_funcs         = cell(1, L);
[g_funcs{L}]	= deal( @Funcs.Linear);         % all but last layer
[g_funcs{L-1}]	= (@Funcs.ReLU );
[g_funcs{1:L-2}]	= ( @Funcs.ReLU);




% Initialize the layers' weights and activation functions
Net = arrayfun(@(n, n1, g) struct('W', w_init_param(n)*randn(n1, n+1), ...	% weights
    'g', g), ...                  % activation function
    N(1:L), N(2:L + 1), g_funcs);






%% training
for i_epoch= 1:epoch
    perm = randperm(size(train_data, 1));
    train_data_rand = train_data(perm,:);
    train_label_rand = Y_train(perm,:);
    % input
    % Loop over all training samples (in mini-batches)
    for batch_start = 1:batch_size:length(perm)
        
        % Get the samples indices for the current batch
        batch_end = min(batch_start + batch_size - 1, length(perm));
        batch_idx = batch_start:batch_end;
        
        % Get the current batch data
        X   = train_data_rand(batch_idx,:);
        Y0  = train_label_rand(batch_idx,:);
        
        % feedforward
        %        s = cell(1,L+1); % cell of inputs
        %         derv = cell(1,L+1); % cell of derviation
        %         s{1} = X; % input of first layer
        %         derv{1} = 0;
        
        s = cell(L + 1, 1);
        
        % Forward pass
        s{1} = struct('act', X', ...                     % set the input
            'der', zeros(size(X')));
        
        for l = 1:L
            s{l}.act(end+1,:)=1;
            [g, gp] = Net(l).g(Net(l).W * (s{l}.act));	% get next layer's activities
            % (and derivatives)
            s{l+1}  = struct('act', g, ...
                'der', gp);                % save results per layer
        end
        
        Y = (s{L + 1}.act);
        Y = softmax(Y);
        
        % back propagation
%        delta = (Y-Y0').*(s{L + 1}.der);
        delta = (Y-Y0');
        
        dW = cell(L , 1);
        dW(:,1) = {0};

        for l = L:-1:1
            
            dW{l} = -eta*delta*(s{l}.act)'+alpha*dW{l};
            %dW = -eta*delta*(s{l}.act)'/batch_size  ;                   % get the weights update
            
            delta = ((Net(l).W(:,1:end-1))'*delta).*s{l}.der;        % update delta
            
            Net(l).W = Net(l).W + dW{l};	% update the weights
            
        end
        
        
        
    end
    
    % Get metrics for the training and validation sets
    [t_err, t_acc] = predict_evaluate_MLP(Net, ...
        train_data, Y_train, train_label, fun_loss);
    [v_err, v_acc] = predict_evaluate_MLP(Net, ...
        test_data, Y_test, test_label, fun_loss);
    
    
    history{i_epoch} = struct('train_err', t_err, ... % training error
        'train_acc', t_acc, ... % training accuracy
        'valid_err', v_err, ... % validation error
        'valid_acc', v_acc);    % validation accuracy
    
    % Command log
    fprintf('Epoch %d/%d, error = %0.3g, accuracy = %2.1f%%, time: %.1f[sec]. \n', ...
        i_epoch, epoch, v_err, 100*v_acc);
    
   
end


% weights init functions
function w_coef = HeInit(L_size)
    w_coef = sqrt(2/L_size);
end

function w_coef = XavierInit(L_size)
    w_coef = sqrt(1/L_size);
end
