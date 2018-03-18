function [ W, Q ] = groupmtl(G,gamma,epsilon,trainx,trainy,mode)
%%%%
% groupmtl - Group multi-task learning (Kang et al., 2011) with hinge loss
% Syntax: [ W, Q ] = groupmtl(G,gamma,epsilon,trainx,trainy,mode)
%
% Inputs:
%      G (scalar) - Number of groups
%      gamma (scalar) - Regularization parameter
%      epsilon (scalar) - Turbulance parameter to ensure convergence
%      trainx (T*1 cell, with Nt*d matrix in each cell) - Training features
%      trainy (T*1 cell, with Nt*1 vector in each cell) - Training labels,
%            labels need to be either -1 or 1
%      mode (1 or 2) - feature selection (1), feature learning (2)
%
% Outputs:
%    W (T*d matrix) - The weight matrix of multi-task learning. Each column
%    correspond to a task. 
%    Q (G*T matrix) - The grouping of the tasks, each row correspond to a
%    group. The ones are tasks belong to this group.
%
% Toolbox required: none
% Other m-files required: mtfl.m, stsvmlinear.m, fastSqrtD.m, diagSqrtD.m,
% getFastGroupQ.m
% Subfunctions: mtfl.m, stsvmlinear.m, fastSqrtD.m, diagSqrtD.m,
% getFastGroupQ.m
% MAT-files required: none
% Note: fastSqrtD.m and getFastGroupQ.m can be found at http://www-scf.usc.edu/~zkang/research.html
%
% References:
%    [1] Argyriou, Andreas, Theodoros Evgeniou, and Massimiliano Pontil. 
%        "Convex multi-task feature learning." Machine Learning 73.3 (2008): 243-272.
%    [2] Kang, Zhuoliang, Kristen Grauman, and Fei Sha. "Learning with whom 
%        to share in multi-task feature learning." Proceedings of the 28th International 
%        Conference on Machine Learning (ICML-11). 2011.
%
% Author: Biqiao Zhang
% University of Michigan, Department of Computer Science and Engineering
% Email: didizbq@umich.edu
% June 2015; Last revision: 06-March-2016 (Changed function and parameter names, added comments)
%%%%%%%%%%%%%%%%%%%%%%

T = length(trainx);
dim = size(trainx{1},2);
W = zeros(dim,T);

Q = zeros(G,T);
Q(1,:) = 1; % initialize the tasks into one group
maxIterQ = 5; % MAX Q iterations
maxIterW = 20; % MAX W iterations

% if every task is a group
if G == T
    C = 1/(2*gamma);
    for t = 1:T
        [W(:,t),~] = stsvmlinear(trainx{t},trainy{t},C,mode);
    end
    return
end

% if all tasks are in the same group
if G == 1
    W = mymtfl( trainx, trainy, gamma, epsilon, maxIterW, mode);
    return
end

% otherwise, need to learn Q and W together
for iter = 1:maxIterQ
    for g = 1:G
        tidx = find(Q(g,:)>0.5);
        nt = length(tidx);
        if nt>0
            X = cell(nt,1);
            Y = cell(nt,1);
            for i=1:nt
                X{i} = trainx{tidx(i)};
                Y{i} = trainy{tidx(i)};
            end
            W(:,tidx) = mtfl(X, Y, epsilon, gamma, maxIterW,mode);
        end
    end
    
    if iter~=maxIterQ
        [Q,Obj] = getFastGroupQ(W, epsilon, G);
    end
end

end

