% demo_traclus.m
% Created by Yelin Kim - on 2015/10/4

% implementation of TRACLUS segmentation of the following paper: 
% Lee, Jae-Gil, Jiawei Han, and Kyu-Young Whang. "Trajectory clustering: a partition-and-group framework." Proceedings of the 2007 ACM SIGMOD international conference on Management of data. ACM, 2007.
clear; fclose('all'); addpath(genpath('.'))

% 1. load a sample multi-dimensional trajectory

m = 80; % number of frames
n = 6 ; % number of feature dimensions;

traj = randn(80,6);

MDL_ADVANTAGE = 3; % control need to be cross validated 

% 2. segment based on traclus 
[CP, realseg, new_traj] = TRACLUS_segment(traj, MDL_ADVANTAGE);


