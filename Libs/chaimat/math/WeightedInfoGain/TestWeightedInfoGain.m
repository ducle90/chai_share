%TESTDECLIPPING - Tests the included calcWeightedInfoGain method
%
% Other m-files required: calcWeightedInfoGain, calcWeightedEntropy, 
%                         calcWeightedEntropyGiven
% Subfunctions: none
% MAT-files required: none
%
% Author: John Gideon
% University of Michigan, Department of Computer Science and Engineering
% Email: gideonjn@umich.edu
% September 2015; Last revision: 29-September-2015
%
% See also: calcWeightedInfoGain,  calcWeightedEntropy,  calcWeightedEntropyGiven

%------------- BEGIN CODE --------------

%% Load Data
load ionosphere.mat;
Features = X;
labels = (strcmp(Y,'g')*2)-1;

%% Discretize features into 5 bins
Features = ceil((Features+1)*2.5);
Features(Features==0) = 1;

%% Determine unweighted info gain
unweightedIG = calcWeightedInfoGain(Features, labels);

%% Determine weighted info gain so not biased by label amounts
weights = zeros(numel(labels),1);
weights(labels==-1) = sum(labels==1)/numel(labels);
weights(labels==1) = sum(labels==-1)/numel(labels);
weightedIG = calcWeightedInfoGain(Features, labels, weights);

%------------- END OF CODE --------------