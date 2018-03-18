% demo_mtsvm.m
% Created by Biqiao Zhang - on 2016/02/22
% This is a demo file to explain the usage of the multi-task feature learning
% and group multi-task learning functions
% Note: 
% 1. Liblinear and its Matlab interface need to be installed for this code
% to run
% 2. You need to download fastSqrtD.m and getFastGroupQ.m from http://www-scf.usc.edu/~zkang/research.html
% Other m-files required: stsvmlinear.m, mtfl.m, groupmtl.m, diagSqrtD.m, fastSqrtD.m, getFastGroupQ.m,
% calc_acc_mtl.m

%% 1. load data.
% In this demo file, we randomly generate the data
clear; clc; close all
N = 500; % number of training instances
n = 500; % number of test instances
d = 20; % dimensionality of the features
T = 5;

% generate w (only for the purpose of generating y from x)
W = zeros(d,T);
Xtrain = cell(T,1);
Xtest = cell(T,1);
Ytrain = cell(T,1);
Ytest = cell(T,1);
for i = 1:T
    W(:,i) = randn(d,1);
    Xtrain{i} = randn(N,d);
    Xtest{i} = randn(n,d);
    Ytrain{i} = sign(Xtrain{i}*W(:,i) + rand(N,1)./10);
    Ytest{i} = sign(Xtest{i}*W(:,i) + rand(N,1)./10);
    Ytrain{i}(find(Ytrain{i}==0)) = 1;
    Ytest{i}(find(Ytest{i}==0)) = 1;
end

%%
G = 2; % number of groups
epsilon = 0.01; % turbulance
gamma = 1; % regularization parameter
% call multi-task feature learning and calculate accuracy
What = mtfl( Xtrain, Ytrain, gamma, epsilon, 20, 2);
[accuracy, acc_t] = calc_acc_mtl(What, Xtest, Ytest);

% call group multi-task learning and calculate accuracy
[What_group,Q] = groupmtl( G, gamma, epsilon, Xtrain, Ytrain, 2);
[accuracy_group, acc_t_group] = calc_acc_mtl(What_group, Xtest, Ytest);

