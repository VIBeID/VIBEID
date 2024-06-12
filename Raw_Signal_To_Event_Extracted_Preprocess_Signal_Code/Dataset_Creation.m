clc;
clear all;
close all;

% Dataset creation from raw files
mtr = 1.5;
persons = {'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30'};
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
