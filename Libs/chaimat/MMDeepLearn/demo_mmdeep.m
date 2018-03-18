% demo_mmdeep.m
% Created by Yelin Kim - on 2015/10/4
% Modified by Yelin Kim - on 2016/3/6

%% This is a demo for how we can train 2-layer DBN with two modalities
% Structure is expalined in our paper: [Kim, Lee, and Mower Provost,
% ICASSP 2013] (DBN2 structure).


clear; fclose('all'); addpath(genpath('.'))

addpath(genpath('RBMLIB')) % add RBMLIB library


% example train data for multimodal 2-layer DBN
train = randn(500, 30);
audidx = 1:10;
vididx = 11:30;
numHiddenAudio = 20;
numHiddenVideo = 60;
numhid2 = 40;

% 2. run bb-RBM
[model1a, model1v, model2, finalHiddenTrainProb] = DBN2(train, audidx, vididx, numHiddenAudio, numHiddenVideo, numhid2);
