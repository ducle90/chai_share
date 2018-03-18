% Created by Yelin Kim - on 2013/10/31
% Modified by Yelin Kim - on 2016/3/6

%%
% INPUT:
% train: train data
% audidx: audio modality feature index
% vididx: video modality feature index

% numHiddenAudio : number of hidden nodes for 1st-layer audio RBM
% numHiddenVideo : number of hidden nodes for 1st-layer video RBM
% numhid2 : number of hidden nodes for 2nd layer
%%
% OUTPUT:
% model1a : learned RBM parameters of 1-st layer audio modality
% model1v : learned RBM parameters of 1-st layer video modality
% model2 : learned RBM parameters of 2-nd layer
% finalHiddenTrainProb : posterior prob of final layer output
%%

function [model1a, model1v, model2, finalHiddenTrainProb] = DBN2(train, audidx, vididx, numHiddenAudio, numHiddenVideo, numhid2)

TrainAudio = train(:, audidx);
TrainVideo = train(:, vididx);


% pretraining 1st layer for individual modalities
% sparse RBM was orignally used in our ICASSP 2013 paper (Kim, Lee, and
% Mower Provost) - this code use open source rbmBB code instead of the sparse RBM (code created by Honglak Lee; not open sourced)

display('-----------------Training 1st layer -----------------');

% train the Audio RBM
[model1a, ~] =rbmBB(TrainAudio,numHiddenAudio); % TODO maybe we need concatTrainProb>0.5 -- check this

% train the Video RBM
[model1v, ~] =rbmBB(TrainVideo,numHiddenVideo); % TODO maybe we need concatTrainProb>0.5 -- check this



% get the probability of hidden nodes in the first layer
probHiddenTrainAudio = logistic(TrainAudio*model1a.W + repmat(model1a.b,size(TrainAudio, 1),1));
probHiddenTrainVideo = logistic(TrainVideo*model1v.W + repmat(model1v.b,size(TrainVideo, 1),1));





% concatenated data
concatTrainProb=[probHiddenTrainAudio, probHiddenTrainVideo];


display('-----------------Training 2nd layer -----------------');
% concatenated RBM
% bernoulli bernoulli RBM
% the standard rbm
% rbmBB.m learns RBM with Bernoulli hidden and visible units
%OUTPUTS:
% model.type     ... Type of RBM (i.e. type of its visible and hidden units)
% model.W        ... The weights of the connections
% model.b        ... The biases of the hidden layer
% model.c        ... The biases of the visible layer
% model.top      ... The activity of the top layer, to be used when training
%               ... DBN's
% errors         ... The errors in reconstruction at every epoch
[model2, ~] =rbmBB(concatTrainProb,numhid2); 
finalHiddenTrainProb= logistic(concatTrainProb*model2.W + repmat(model2.b,size(concatTrainProb, 1),1));

end







