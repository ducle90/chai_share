function [ wInfoGain ] = calcWeightedInfoGain( Features, labels, weights )
%CALCWEIGHTEDINFOGAIN - Weighted version of information gain
%
% Syntax:  [wInfoGain] = calcWeightedInfoGain(Features,labels,weights)
%
% Inputs:
%    Features (NxM matrix) - The features already discretized into integers
%       starting at 1 in increments of 1
%    labels (Nx1 column vector) - The labels on which the features are
%       conditioned discretized into integers starting at 1 in increments 
%       of 1 (or just -1 and 1)
%    weights (Nx1 column vector) - The relative importance of each feature 
%       sample, which can be decimal [default is uniform weighting]
%
% Outputs:
%    wInfoGain (1xM row vector) - The weighted information gain of each 
%       column
%
% Examples: 
%    calcWeightedInfoGain([1 2 1 2]', [1 2 1 2]') = 1
%    calcWeightedInfoGain([1 2 1 2]', [1 1 2 2]') = 0
%    calcWeightedInfoGain([1 2 1 2]', [1 1 2 2]', [1 1 1 1]') = 0
%    calcWeightedInfoGain([1 2 1 2]', [1 1 2 2]', [.5 1 1 1]') = 0.0202
%
% Other m-files required: calcWeightedEntropy, calcWeightedEntropyGiven
% Subfunctions: none
% MAT-files required: none
%
% Author: John Gideon
% University of Michigan, Department of Computer Science and Engineering
% Email: gideonjn@umich.edu
% September 2015; Last revision: 20-September-2015
%
% See also: calcWeightedEntropy,  calcWeightedEntropyGiven

%------------- BEGIN CODE --------------

% Get sizes
nI = size(Features,1); % Number of instances

% Create uniform weights if not given
if ~exist('weights', 'var')
    weights = ones(nI,1);
end

% If there is a -1 label set it to 2 instead (compatability with -1 and 1)
labels(labels==-1) = 2;

% Info gain is entropy minus entropy conditioned on labels
wInfoGain = calcWeightedEntropy(Features, weights) - ...
    calcWeightedEntropyGiven(Features, labels, weights);

%------------- END OF CODE --------------