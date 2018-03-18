function [prediction, accuracy, dvalue] = svm_customKernel(Ktrain, Ktest, Ytrain, Ytest, C)
%svm_customKernel - Support Vector Machine Classification with custom Kernel
%
% Syntax: [prediction, accuracy, dvalue] = svm_customKernel(Ktrain, Ktest, Ytrain, Ytest, C)
%
% Inputs:
%    Ktrain (NxN matrix) - The gram matrix of the training data
%    Ktest (nxN matrix) - The gram matrix of the test and training data
%    Ytrain (Nx1 vector) - The labels of the training data
%    Ytest (nx1 vector) - The labels of the test data
%    C (scalar) - Cost parameter in SVM
%
% Outputs:
%    prediction (nx1 vector) - The predicted labels of the test data
%    accuracy (scalar) - The test accuracy
%    dvalue (nx1 vector) - The decision values of the test data
%
% Toolbox required: LibSVM, with Matlab interface installed
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% References:
%    [1] Chang, Chih-Chung, and Chih-Jen Lin. "LIBSVM: a library for support vector machines." 
%        ACM Transactions on Intelligent Systems and Technology (TIST) 2.3 (2011): 27.
%
% Author: Biqiao Zhang
% University of Michigan, Department of Computer Science and Engineering
% Email: didizbq@umich.edu
% April 2015; Last revision: 22-Feb-2016 (Changed function and parameter names, added comments)
%%%%%%%%%%%%%%%%%%%%%%

% Set first column to be sequence numbers
Ktrain = [(1:length(Ytrain))', Ktrain];
Ktest = [(1:length(Ytest))', Ktest];

% Call LibSVM training, with cost parameter C
model = svmtrain(Ytrain,Ktrain,['-q -s 0 -t 4 -c ', num2str(C)]);
[prediction, accuracy, dvalue]= svmpredict(Ytest, Ktest, model);

end

