function [accuracy, acc_t] = calc_acc_mtl(W, X, Y)
%calc_acc_mtl - Calculate accuracy for mtfl and groupmtl
%
% Syntax: [accuracy, acc_t] = calc_acc_mtl(W, x, y)
%
% Inputs:
%    W (T*d matrix) - The weight matrix of multi-task learning. Each column
%    correspond to a task.
%    X (Tx1 cell, with N*d matrix in each cell) - The features of the testing data
%    Y (Tx1 cell, with N*1 vector in each cell) - The labels of the testing data
%
% Outputs:
%    accuracy (scalar) - Mean accuracy
%    acc_t (1*T vector) - The accuracy of each task
%
% Toolbox required: none
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% Author: Biqiao Zhang
% University of Michigan, Department of Computer Science and Engineering
% Email: didizbq@umich.edu
% June 2015; Last revision: 06-March-2016 (Changed function and parameter names, added comments)
%%%%%%%%%%%%%%%%%%%%%%

if iscell(Y)
    T = length(Y);
else
    T = 1;
end
acc_t = zeros(1,T);


for t= 1:T
    clear yhat
    yhat = sign(X{t}*W(:,t));
    acc_t(t) = length(find(Y{t}==yhat))/length(yhat);
end

accuracy = mean(acc_t);
end

