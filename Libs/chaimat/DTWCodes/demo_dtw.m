% demo_dtw.m
% Created by Yelin Kim - on 2015/10/4
% Modified by Yelin Kim - on 2016/3/5

% 1. Caculate DTW distance between two multi-dimensional trajectories with the
% same dimension and different lengths)
% assuming that there are 6 facial regions

n = 126; % number of feature dimensions


test{1} = randn(50, n);
ref{1} = randn(31, n);

% 2. calculate dtw distance from the two emotograms
DTWmatrix = calc_DTW_c2(test, ref);
        
        





