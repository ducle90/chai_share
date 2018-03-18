function [ W ] = mtfl( trainx, trainy, gamma, epsilon, iterations, mode)
%%%%
% groupmtl - Multi-task feature learning (Argyriou et al., 2008) with
% hinge loss
% Syntax: [ W ] = mtfl(trainx,trainy,gamma,epsilon,mode)
%
% Inputs:
%      trainx (T*1 cell, with Nt*d matrix in each cell) - Training features
%      trainy (T*1 cell, with Nt*1 vector in each cell) - Training labels,
%            labels need to be either -1 or 1
%      gamma (scalar) - Regularization parameter
%      epsilon (scalar) - Turbulance parameter to ensure convergence
%      iterations (scalar) - maximum number of iterations
%      mode (1 or 2) - feature selection (1), feature learning (2)
%
% Outputs:
%    W (T*d matrix) - The weight matrix of multi-task learning. Each column
%    correspond to a task. 
%
% Toolbox required: none
% Other m-files required: stsvmlinear.m, fastSqrtD.m, diagSqrtD.m,
% Subfunctions: stsvmlinear.m, fastSqrtD.m, diagSqrtD.m,
% MAT-files required: none
% Note: fastSqrtD.m can be found at http://www-scf.usc.edu/~zkang/research.html
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

% gamma to C
C = 1/(2*gamma);

if T==1
    [W,~] = stsvmlinear(trainx{1},trainy{1},C,2);
    return
end

% initialize D = I/d, compute sqrt(D)
dim = size(trainx{1},2);
W = zeros(dim,T);
D = eye(dim)./dim;
D_sqrt = diag(sqrt(diag(D)));

for iter = 1:iterations
    for t = 1:T
        x = trainx{t};
        y = trainy{t};
        new_x = x * D_sqrt;
        [W(:,t),~] = stsvmlinear(new_x,y,C,2);
    end
    W = D_sqrt * W;

    if mode==1
        D_sqrt = diagSqrtD(W);
    else
        D_sqrt = fastSqrtD(W, epsilon);
    end
end

end

