clc, clear all, close all;
% Load the saved after running usleet matlab code file
load(['person_feat_VIBeID_A2_1.mat'])

footstep_feat = [];
labels=[];
% List of dataset names
datasets = {'VIBeID_A1', 'VIBeID_A2_1','VIBeID_A2_2','VIBeID_A2_3','VIBeID_A3_1','VIBeID_A3_2','VIBeID_A3_3','VIBeID_A4_1'};
dataset = datasets{2};  % select which dataset entry number
for i = 1:30 %change the value to 100, 40, 30, 15 depending on the name of person used in dataset
    footstep_feat = [footstep_feat ; person_feat{i}];
    labels = [labels; i*ones(size(person_feat{i},1),1)];
    
    
end
footstep_feat = [footstep_feat, labels];

dataname = sprintf('footstep_feat_%s.mat', dataset);
save(dataname, 'footstep_feat')
