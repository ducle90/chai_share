function [ wEntropy ] = calcWeightedEntropyGiven( Features, labels, weights )
%CALCWEIGHTEDENTROPYGIVEN - Weighted version of entropy given a certain
%   label
%
% Syntax:  [wEntropy] = calcWeightedEntropyGiven(Features,labels,weights)
%
% Inputs:
%    Features (NxM matrix) - The features already discretized into integers
%       starting at 1 in increments of 1
%    labels (Nx1 column vector) - The labels on which the features are
%       conditioned discretized into integers starting at 1 in increments 
%       of 1
%    weights (Nx1 column vector) - The relative importance of each feature 
%       sample, which can be decimal [default is uniform weighting]
%
% Outputs:
%    wEntropy (1xM row vector) - The weighted conditional entropy of each 
%       column
%
% Examples: 
%    calcWeightedEntropyGiven([1 2 1 2]', [1 2 1 2]') = 0
%    calcWeightedEntropyGiven([1 2 1 2]', [1 1 2 2]') = 1
%    calcWeightedEntropyGiven([1 2 1 2]', [1 1 2 2]', [1 1 1 1]') = 1
%    calcWeightedEntropyGiven([1 2 1 2]', [1 1 2 2]', [.5 1 1 1]') = 0.965
%
% Other m-files required: calcWeightedEntropy
% Subfunctions: none
% MAT-files required: none
%
% References:
%    [1] Smieja, Marek. "Weighted approach to general entropy function." 
%        IMA Journal of Mathematical Control and Information (2014): dnt044.
%
% Author: John Gideon
% University of Michigan, Department of Computer Science and Engineering
% Email: gideonjn@umich.edu
% September 2015; Last revision: 20-September-2015
%
% See also: calcWeightedEntropy,  calcWeightedInfoGain

%------------- BEGIN CODE --------------

% Get sizes
nI = size(Features,1); % Number of instances
nA = size(Features,2); % Number of attributes
nLabelBins = max(labels); % Number of bins for labels

% Create uniform weights if not given
if ~exist('weights', 'var')
    weights = ones(nI,1);
end

% Loop through and condition on each label
wEntropy = zeros(1,nA);
totalWeight = sum(weights);
for l = 1:nLabelBins
    X_y = Features(labels==l,:);
    w_y = weights(labels==l);
    pL = sum(w_y) / totalWeight;
    wEntropy = wEntropy + (pL.*calcWeightedEntropy(X_y, w_y));
end

%------------- END OF CODE --------------