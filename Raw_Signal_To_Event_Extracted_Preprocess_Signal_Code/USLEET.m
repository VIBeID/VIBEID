clc , close all,   clear ;
%%




% Load Mat file for tarining model
load("E:\VIBeID\VIBeID_A2\A2_1\P1_full.mat")

Fs = 8000; % Frequency of the recorded signal
n = 100*Fs; 

geo_data = smooth(geo_data(1:n),5);  % smooth function to smooth out the recorded data
tm = (0:length(geo_data)-1)/Fs; % total time of the recorded data 

% Plot of signal for visualisation
figure(1)
plot(tm,geo_data)
grid on
xlabel('Time (sec)')
ylabel('Aplitude(V)')



tau = 1.2;
window = 0.350; % 350ms is taken as the window size
wndw_ovrlap = 0.40; % overlap
wndw_smpl = window*Fs;

num_seg=floor(1+(length(geo_data)-wndw_smpl)/(floor((1-wndw_ovrlap)*wndw_smpl)));

figure(2)

for i = 1:num_seg
    
    start = floor(wndw_smpl*(i-1)*(1-wndw_ovrlap) + 1);
    stop = floor(start + wndw_smpl -1);
    if stop >= length(geo_data)
        stop = length(geo_data);
    end
    wght_wndw = length(start:stop);  % window length
    weight = gausswin(wght_wndw,tau);  % calculating gaussian weights 
    w_diag = diag(weight);
    sig = w_diag*geo_data(start:stop);
 
    
    ovrlp_sig = [zeros(start-1,1); geo_data(start:stop) ; zeros((length(geo_data)-stop),1)];
    
    signal_feat(i,:) = Events_Features_Extraction(Fs,sig); %event and feature extraction from raw signal
   
end


signal_param = signal_feat;

Cluster_num  = 2; % Two cluster for event and noise class

[clust, cov_mat, mu_mat, phi] = GMM_EM(signal_param, Cluster_num);

[Y] = tsne(signal_param,'Algorithm','exact','Standardize',0,'Distance','cosine'); % tsne plot of data 

%% Creating and Traing the SVM Model 

c1_idx = clust{1,1};
c2_idx = clust{1,2};

figure(2)
plot(signal_param(c1_idx,1),signal_param(c1_idx,2),'ko','MarkerFaceColor','y','MarkerSize',7)

hold on
plot(signal_param(c2_idx,1),signal_param(c2_idx,2),'ko','MarkerFaceColor','g','MarkerSize',7)
legend('Cluster 1', 'Cluster 2')
hold off
grid on
xlabel('x\_1')
ylabel('x\_2')
title('Actual Distribution of the Dataset')

fprintf('Press Enter To Continue \n');
pause()

train_data = [signal_feat(c1_idx,1:end) ; signal_feat(c2_idx,1:end)];

if det(cov_mat(:,:,1)) > det(cov_mat(:,:,2))
    train_label = [ones(length(c1_idx),1); zeros(length(c2_idx),1)];
    lbl_clst1 = 1;
    lbl_clst2 = 0;
else
    train_label = [zeros(length(c1_idx),1); ones(length(c2_idx),1)];
    lbl_clst1 = 0;
    lbl_clst2 = 1;
end

%% Testing on the blind dataset

persons = {'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30'};
 % List of persons for blind training
person_feat = [];
% list of dataset
datasets = {'VIBeID_A1', 'VIBeID_A2_1','VIBeID_A2_2','VIBeID_A2_3','VIBeID_A3_1','VIBeID_A3_2','VIBeID_A3_3','VIBeID_A4_1'};
dataset = datasets{2}; % datset selection
for k = 1:30 % here 15 is the number of persons used for training

    load(sprintf('%s_full.mat', persons{k})) %load each file after merging
    geo_data = smooth(geo_data,5);

    fprintf('Events and Features are extracted from the person %d \n',k)

clear Evnt_Ind prdct_clst prob_clust

tm = (0:length(geo_data)-1)/Fs; 

wndw_smpl = window*Fs;

num_seg=floor(1+(length(geo_data)-wndw_smpl)/(floor((1-wndw_ovrlap)*wndw_smpl)));
i = 1;
iter = 1;


while iter < length(geo_data)
    
    start = floor(wndw_smpl*(i-1)*(1-wndw_ovrlap) + 1);
    stop = floor(start + wndw_smpl -1); 
    if stop >= length(geo_data)
        stop = length(geo_data);
    end
    gaussian_wndw = length(geo_data(start:stop));
   weight = gausswin(gaussian_wndw,tau);  % calculating gaussians weights 
    w_diag = diag(weight);
    sig = w_diag*geo_data(start:stop);
    
    Evnt_Ind(i,:) = [start , stop];
    
   
    
    test_data = Events_Features_Extraction(Fs,sig);
    prob_clust(i,1) = mvnpdf(test_data(:,1:end),mu_mat(1,:),cov_mat(:,:,1))*phi(1);
    prob_clust(i,2) = mvnpdf(test_data(:,1:end),mu_mat(2,:),cov_mat(:,:,2))*phi(2);
   
   
 i = i+1;
 iter = stop;


end



prob_clust=prob_clust./repmat(sum(prob_clust,2),1,size(prob_clust,2));

[c clust_assign] = max(prob_clust,[],2);

id = find(c<0.90);
c_idx = find(clust_assign == 1);
prdct_clst(c_idx,:) = lbl_clst1;
c_idx = find(clust_assign == 2);
prdct_clst(c_idx,:) = lbl_clst2;

prdct_clst(id,:) = 0;


Evnt_Ind = [Evnt_Ind , prdct_clst];

Sigma = 4.0; % Bandwidth parameter of the gaussian window
[Evnts_loc, footstep_feat] = Event_Extract(Evnt_Ind, geo_data, Sigma,k); % Event extraction for each person class

person_feat{k} = footstep_feat;         

end
%% Save person_feat variable from the workspace window
dataname = sprintf('person_feat_%s.mat', dataset);
save(dataname, 'person_feat')
