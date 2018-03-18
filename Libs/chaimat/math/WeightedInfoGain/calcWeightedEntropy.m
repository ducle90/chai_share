function [ wEntropy ] = calcWeightedEntropy( Features, weights )
%CALCWEIGHTEDENTROPY - Calculates a weighted version of entropy
%
% Syntax:  [wEntropy] = calcWeightedEntropy(Features,weights)
%
% Inputs:
%    Features (NxM matrix) - The features already discretized into integers
%       starting at 1 in increments of 1
%    weights (Nx1 column vector) - The relative importance of each feature 
%       sample, which can be decimal [default is uniform weighting]
%
% Outputs:
%    wEntropy (1xM row vector) - The weighted entropy of each column
%
% Examples: 
%    calcWeightedEntropy([1 2; 1 1]) = [0 1]
%    calcWeightedEntropy([1 2 1 3 1 2 4]') = 1.8424
%    calcWeightedEntropy([1 2 1 3 1 2 4]',[1 1 1 1 1 1 1]') = 1.8424
%    calcWeightedEntropy([1 2 1 3 1 2 4]',[1 3/2 1 3 1 3/2 3]') = 2
%    calcWeightedEntropy([1 2 1 2]', [.5 1 1 1]') = 0.9852
%
% Other m-files required: none
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
% See also: calcWeightedEntropyGiven,  calcWeightedInfoGain

%------------- BEGIN CODE --------------

% Get sizes
nI = size(Features,1); % Number of instances
nA = size(Features,2); % Number of attributes
nBins = max(max(Features)); % Number of discrete bins

% Create uniform weights if not given
if ~exist('weights', 'var')
    weights = ones(nI,1);
end

% Determine entropy by column
prob = zeros(nBins,nA);

% Count
for i = 1:nI
    for a = 1:nA
        d = Features(i,a);
        prob(d,a) = prob(d,a) + weights(i);
    end
end

% Normalize and calculate
prob = prob ./ sum(weights);
prob = prob.*log2(prob);
prob(isnan(prob)) = 0;
wEntropy = -sum(prob);

%------------- END OF CODE --------------