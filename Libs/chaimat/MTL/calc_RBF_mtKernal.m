function [ mtK ] = calc_RBF_mtKernal( Xtrain, Xtest, Ttrain, Ttest, gamma, rho)
%calc_RBF_mtKernal - Calculate the multi-task RBF Gaussian Kernel
% Syntax: [mtK] = calc_RBF_mtKernal( Xtrain, Xtest, Ttrain, Ttest, gamma, rho)
%
% Inputs:
%    Xtrain (NxM matrix) - The feature matrix of the training data, N is
%    the number os instances, M is the dimensionality
%    Xtest (nxM matrix) - The feature matrix of the test data, n is
%    the number os instances, M is the dimensionality
%    Ttrain (Nx1 vector of integers) - The tasks of the training data 
%    Ttest (nx1 vector of integers) - The tasks of the test data
%    gamma (scalar) - Parameter controlling for the bandwidth of the RBF
%    Gaussian Kernel
%    rho (scaler) - Parameter controlling for the closeness between tasks
%    (if rho==infinity, the tasks are the same; if rho=0, the tasks are independent)
%
% Outputs:
%    mtK - The multi-task Gram Matrix
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% References:
%    [1] T. Evgeniou and M. Pontil, ?Regularized multi?task learning,? in
%        Proceedings of the tenth ACM SIGKDD international conference on
%        Knowledge discovery and data mining. ACM, 2004, pp. 109?117.
% Author: Biqiao Zhang
% University of Michigan, Department of Computer Science and Engineering
% Email: didizbq@umich.edu
% April 2015; Last revision: 22-Feb-2016 (Changed function and parameter names, added comments)
%%%%%%%%%%%%%%%%%%%%%%

% RBF kernel function
rbfKernel = @(A,B,gamma) exp(-gamma .* pdist2(A,B,'euclidean').^2);

% Calculate Gram matrix without task information
K = rbfKernel(Xtest,Xtrain,gamma);

ntest = length(Ttest);
ntrain = length(Ttrain);

% Calculate parameter matrix to be multiplied to Gram matrix
pmatrix = zeros(ntest,ntrain);
for i=1:ntest
    for j=1:ntrain
        if Ttest(i) == Ttrain(j)
            pmatrix(i,j)=1;
        else
            pmatrix(i,j)=rho/(1+rho);
        end
    end
end

% Calculate multi-task Gram matrix
mtK = pmatrix.*K;

end

