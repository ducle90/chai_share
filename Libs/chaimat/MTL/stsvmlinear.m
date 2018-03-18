function [w, model] = stsvmlinear(x,y,C,mode)
%stsvmlinear - Support Vector Machine Classification with Linear Kernel
%
% [w, model] = stsvmlinear(x,y,C,mode)
%
% Inputs:
%    x (Nxd vector) - The features of the training data
%    y (Nx1 vector) - The labels of the training data
%    C (scalar) - Cost parameter in SVM
%    mode (1 or 2): L1-regularization (1) or L2-regularization (2)
%
% Outputs:
%    w (dx1 vector) - The weight vector of the model
%    model (structure) - The model returned by Liblinear
%
% Toolbox required: Liblinear, with Matlab interface installed
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% References:
%    [1] Fan, Rong-En, et al. "LIBLINEAR: A library for large linear classification." 
%        The Journal of Machine Learning Research 9 (2008): 1871-1874.
%
% Author: Biqiao Zhang
% University of Michigan, Department of Computer Science and Engineering
% Email: didizbq@umich.edu
% June 2015; Last revision: 06-March-2016 (Changed function and parameter names, added comments)
%%%%%%%%%%%%%%%%%%%%%%

if mode == 1
    options = ['-q -s 5 -c ',num2str(C)];
else
    options = ['-q -s 2 -c ',num2str(C)];
end
model = train(y,sparse(x),options);
w = model.w';

end

