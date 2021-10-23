clear; close all; clc;

%% Loading data
train = load('DATA_TRAIN.csv');
test = load('DATA_valid.csv');      %insert here test data
%% Net settings
N = [ 2, 30,30, 2];                 %number of nuerons in each layer
L = length(N) - 1;                  %number of layers

W_init = {'R', 'X'};                %R-random, X-xavier => choose in methods (1 or 2)
W_method = W_init{2};

n_epochs = 150;
batch_size = 100;
eta = 0.005;                        %learning rate
alpha = 1.5;                        %put 0 for no momentom
lambda = 0.5;
reg = 'L2';                         %regolation method ('L1'/'L2'/0(no reg))

%% W net & activation functions
% Activation functions (per layer)
g_funcs         = cell(1, L);
[g_funcs{1:L-1}]	= deal(@ActFuncs.Tanh);         % all but last layer
assert(length(N) == length(g_funcs) + 1, ...        % sanity check
    'The number of activation functions and the number of layers mismatch. ')
g_funcs{L} = @ActFuncs.Sigmoid;

Net = net_init(N,L,W_method,g_funcs);               %creating the Net
%% MLP
%train
data = train;

for epoch = 1:n_epochs
    
    % Get a random order of the samples
    perm = randperm(size(data, 1));
    
    % Loop over all training samples (in mini-batches)
    for batch_start = 1:batch_size:length(perm)
        
        % Get the samples' indices for the current batch
        batch_end = min(batch_start + batch_size - 1, length(perm));
        batch_ind = batch_start:batch_end;
        
        X = data(perm(batch_ind),1:2)';
        Y0 = data(perm(batch_ind),3)';
        
        %converting to one-hot. first raw is for spiral 1 and 2nd is for 0.
        Y0_C = zeros(2,batch_size);
        Y0_C(1,:) = Y0;
        Y0_C(2,:) = Y0_C(1,:)==Y0_C(2,:);
        
        Y0 = Y0_C;

        [s, Y] = feedforward(X,Net,L);
        
%         Y = softmax(Y,2);
        Net = backprop(s, Y, Y0, Net, eta, L, alpha, lambda, reg);
    end
end
%% valid
data = test;

X = data(:,1:2)';
Y0 = data(:,3)';

[s, Y] = feedforward(X,Net,L);
% Y = softmax(Y);

%back from one-hot to original display => 0 for 0 and 1 for 1.
Y_check = Y(1,:)>Y(2,:);

%accuracy check
correct = test(:,3) == Y_check';

percentError = (sum(correct)/length(correct))*100 ;
fprintf('Accuracy: %0.8f %%', percentError);
