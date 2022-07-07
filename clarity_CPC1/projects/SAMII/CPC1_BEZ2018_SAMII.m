function CPC1_BEZ2018_SAMII(root_path, dataset, n_process)
% Compute the mutual information using the BEZ2018 model.
% 
% This function is the interface between the baseline provide by the
% clarity challenge and the proposed SAMII algorithm.
%
% Input:
% - root_path: Clarity root path
% - dataset: (train) for track 1 "close-training-set"
%            (train_indep) for track 2 "open-training-set"
%            (test) for track 1 "close-testing-set"
%            (test_indep) for track 2 "open-testing-set"
% - n_process: how many scenes do you want to process
%
% Output:
% - This will generate .json files with the transmitted information,
% perceived information and mutual information for each scene processed.
% Those files can be found in .../clarity_data/mutualinfo/$dataset§/
%
% Created by: Franklin Alvarez
% Alvarez and Nogueira (2022)
% First Clarity Prediction Challenge (2021-2022) and INTERSPEECH 2022

%% Paths
metadata_path = [root_path filesep 'data' filesep 'clarity_data' ...
    filesep 'metadata'];
scenes_data_path = [root_path filesep 'data' filesep 'clarity_data' ...
    filesep 'clarity_data' filesep 'scenes'];
ha_output_path = [root_path filesep 'data' filesep 'clarity_data' ...
    filesep 'clarity_data' filesep 'HA_outputs' filesep dataset];
output_path = [root_path filesep 'data' filesep 'clarity_data' ...
    filesep 'clarity_data' filesep 'mutualinfo' filesep dataset];

addpath([root_path filesep 'projects' filesep 'BEZ2018_CUDA'])

%% Read json files
% Listeners
flisteners = [metadata_path filesep 'listeners.CPC1_train.json']; 
listeners = readJSON(flisteners);

% Signals
fsignals = [metadata_path filesep 'CPC1.' dataset '.json']; 
signals = readJSON(fsignals);
n_process = str2num(n_process); %#ok<ST2NM> 
if n_process==0
    n_process = length(signals);
end
selected_signals = signals(1:n_process);

%% Loop over the data
for s=1:numel(selected_signals)
    disp(['*** File ' num2str(s) ':' num2str(numel(selected_signals)) ' ***'])
    signal = selected_signals(s);
    scene = signal.scene;
    listener_id = signal.listener;
    listener = listeners.(listener_id);
    l_audiogram = struct();
    l_audiogram.cfs = listener.audiogram_cfs';
    l_audiogram.left = listener.audiogram_levels_l;
    l_audiogram.right = listener.audiogram_levels_r;
    fname = [output_path filesep signal.signal '.json'];
    if isfile(fname)
        % If the scene has been already computed, skip it.
        continue
    end

    % Path to stimuli
    ha_stim_path = ...
        [ha_output_path filesep signal.signal '.wav'];
    anechoic_stim_path = ...
        [scenes_data_path filesep scene '_target_anechoic.wav'];

    % Run BEZ2018 model and calculate MI
    output = MI_BEZ2018(anechoic_stim_path, ha_stim_path, l_audiogram);

    % Create .json file
    txt = jsonencode(output,"PrettyPrint",true);
    fout = fname;
    fid = fopen(fout, 'wt');
    fprintf(fid,"%s",txt);
end

end

function data = readJSON(fname)
    fid = fopen(fname); 
    raw = fread(fid,inf); 
    str = char(raw'); 
    fclose(fid); 
    data = jsondecode(str);
end