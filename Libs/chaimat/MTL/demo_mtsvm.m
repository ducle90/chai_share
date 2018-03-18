% demo_mtsvm.m
% Created by Biqiao Zhang - on 2016/02/22
% This is a demo file to explain the usage of the multi-task SVM functions
% Note: LibSVM and its Matlab interface need to be installed for this code
% to run
% Other m-files required: calc_RBF_mtKernal.m, svm_customKernel.m

%% 1. load data.
% In this demo file, we randomly generate the data
clear; clc; close all
N = 500; % number of training instances
n = 500; % number of test instances
M = 20; % dimensionality of the features

% generate w (only for the purpose of generating y from x)
w_t1 = randn(M,1);
w_t2 = randn(M,1);

% generate features
Xtrain = randn(N,M);
Xtest = randn(n,M);

% generate tasks
Ttrain = ones(N,1); Ttrain(N/2+1:end) = Ttrain(N/2+1:end).*2;
Ttest = ones(n,1); Ttest(n/2+1:end) = Ttest(n/2+1:end).*2;

% generate labels
Ytrain = zeros(N,1);
Ytest = zeros(n,1);
Ytrain(1:N/2) = sign(Xtrain(1:N/2,:)*w_t1 + rand(N/2,1)./10);
Ytrain(N/2+1:end) = sign(Xtrain(N/2+1:end,:)*w_t2 + rand(N/2,1)./10);
Ytrain(find(Ytrain==0)) = 1;
Ytest(1:n/2) = sign(Xtest(1:n/2,:)*w_t1 + rand(N/2,1)./10);
Ytest(n/2+1:end) = sign(Xtest(n/2+1:end,:)*w_t2 + rand(N/2,1)./10);
Ytest(find(Ytest==0)) = 1;

%% 2. Calculate multi-task RBF kernal matrix
% In the demo we set gamma and rho to be arbitrary values, they need to be
% tuned in real use cases
gamma = 0.01;
rho = 10;
Ktrain  = calc_RBF_mtKernal( Xtrain, Xtrain, Ttrain, Ttrain, gamma, rho);
Ktest  = calc_RBF_mtKernal( Xtrain, Xtest, Ttrain, Ttest, gamma, rho);

%% Training and testing with multi-task SVM
% In the demo we set C to be arbitrary values, it need to be
% tuned in real use cases

C = 1;
[prediction, accuracy, dvalue] = svm_customKernel(Ktrain, Ktest, Ytrain, Ytest, C);

