function [ sqrtD ] = diagSqrtD( W )
%%%%
%diagsqrtD - Calculate square root of D for mtfl and groupmtl (feature selection setting)
%
% Syntax: [ sqrtD ] = diagSqrtD( W )
%
% Inputs:
%    W (T*d matrix) - The weight matrix of multi-task learning. Each column
%    correspond to a task.   
%
% Outputs:
%    sqrtD(d*d matrix) - Square root of D
%
% Toolbox required: none
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% References:
%    [1] Argyriou, Andreas, Theodoros Evgeniou, and Massimiliano Pontil. 
%        "Convex multi-task feature learning." Machine Learning 73.3 (2008): 243-272.
%
% Author: Biqiao Zhang
% University of Michigan, Department of Computer Science and Engineering
% Email: didizbq@umich.edu
% June 2015; Last revision: 06-March-2016 (Changed function and parameter names, added comments)
%%%%%%%%%%%%%%%%%%%%%%

W2 = sqrt(sum(W.^2,2));
W21 = sum(W2);
D = W2./W21;
sqrtD = diag(sqrt(D));

end

