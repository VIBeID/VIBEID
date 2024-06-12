clc;
clear all;
close all;

% Dataset creation from raw files
mtr = 1.5;
% persons = {
%     'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10',
%     'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20',
%     'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30',
%     'P31', 'P32', 'P33', 'P34', 'P35', 'P36', 'P37', 'P38', 'P39', 'P40',
%     'P41', 'P42', 'P43', 'P44', 'P45', 'P46', 'P47', 'P48', 'P49', 'P50',
%     'P51', 'P52', 'P53', 'P54', 'P55', 'P56', 'P57', 'P58', 'P59', 'P60',
%     'P61', 'P62', 'P63', 'P64', 'P65', 'P66', 'P67', 'P68', 'P69', 'P70',
%     'P71', 'P72', 'P73', 'P74', 'P75', 'P76', 'P77', 'P78', 'P79', 'P80',
%     'P81', 'P82', 'P83', 'P84', 'P85', 'P86', 'P87', 'P88', 'P89', 'P90',
%     'P91', 'P92', 'P93', 'P94', 'P95', 'P96', 'P97', 'P98', 'P99', 'P100'
% };   % For VIBeID A1
persons = {'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 
    'P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30'};  % For VIBeID A2 {VIBeID A2.1, A2.2, and A2.3}
% persons = {'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 
%     'P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30',
%     'P31','P32','P33','P34','P35','P36','P37','P38','P39','P40'}; % For VIBeID A3 {VIBeID A3.1, A3.2, and A3.3}

dataset_original = [];

for j = 1:length(persons)
    person = persons{j};
    data = [];
    
    % Check if the directory exists
    if isfolder(person)
        % Get list of MAT files in the directory
        files = dir(fullfile(person, '*.mat'));
        
        for i = 1:3
            filename = fullfile(person, files(i).name);
            fprintf('Loading %s\n', filename);
            load(filename);
            
            tempdata = geo_data;
            data = [data; tempdata];
        end
        
        dataset_original = [dataset_original, data];
        
        geo_data = data;
        dataname = sprintf('%s_full.mat', person);
        save(dataname, 'geo_data');
    else
        fprintf('Directory %s does not exist.\n', person);
    end
end
