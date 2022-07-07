function information = ...
    MI_BEZ2018(transmitted_path, perceived_path, audiogram)
%

%% Check GPU Availability
useGPU = true;
try
    gpuDevice;
catch
    useGPU = false;
end

%% BEZ2018 Parameters

species = 2; % Human species
ag_fs = [125 250 500 1e3 2e3 4e3 8e3];
ag_dbloss = [0 0 0 0 0 0 0]; % No hearing loss

if useGPU
    % If GPU is available, compute the the spike activity in 4000 fibers
    % distributed in 40 frequency bands between 125 Hz and 16 kHz
    numcfs = 40; % number of center frequencies (original in BEZ2018)
    CFs = logspace(log10(125),log10(16e3),numcfs); % (original in BEZ2018);
    numsponts_healthy = [16 23 61]; % Number of low-spont, medium-spont, 
                                    % and high-spont fibers at each CF in
                                    % a healthy ANF (100 ANFs)
                                    % (Well known ratio in literature,
                                    % Moore (2003))
else
    % If no GPU is available, compute the spike activity in 125 fibers
    % distributed in 25 frequency bands between 250 Hz and 16kHz
    numcfs = 25; % number of center frequencies (minimum)
    CFs   = logspace(log10(250),log10(8e3),numcfs);  % CF in Hz;
    numsponts_healthy = [1 1 3]; % Original BEZ2018 ratio.
end

nFibers = sum(numsponts_healthy, 'all');

%% Get spike trains
% read audio
[trans_audio, fs_t] = audioread(transmitted_path);
[per_audio, fs_p] = audioread(perceived_path);

% Audio signals should be both binaural or both monoaural 
%assert(size(trans_audio,2) == size(per_audio,2));
%assert(size(trans_audio,2) <= 2);
% is it binaural?
%binaural = false;
%if size(trans_audio,2) == 2
%    binaural = true;

% Obtain spike trains of target stimulus
spt_trans = struct();
spt_per = struct();

disp('Computing spike trains for transmitting information')

if useGPU
    [spt_trans.left,spt_trans.t] = BEZ2018_GPU(trans_audio(:,1), fs_t, ...
        species, ag_fs, ag_dbloss, CFs, numsponts_healthy);
    
    [spt_trans.right,~] = BEZ2018_GPU(trans_audio(:,2), fs_t, species, ...
        ag_fs, ag_dbloss, CFs, numsponts_healthy, useGPU);
    
    % Obtain spike trains of perceived stimulus with hearing loss
    disp('Computing spike trains for perceived information')
    [spt_per.left,spt_per.t] = BEZ2018_GPU(per_audio(:,1), fs_p, ...
        species, audiogram.cfs, audiogram.left, CFs, numsponts_healthy);
    
    [spt_per.right,~] = BEZ2018_GPU(per_audio(:,2), fs_p, species,...
        audiogram.cfs, audiogram.right, CFs, numsponts_healthy);

else
    [spt_trans.left,spt_trans.t] = BEZ2018(trans_audio(:,1), fs_t, ...
        species, ag_fs, ag_dbloss, CFs, numsponts_healthy);
    
    [spt_trans.right,~] = BEZ2018(trans_audio(:,2), fs_t, species, ...
        ag_fs, ag_dbloss, CFs, numsponts_healthy);
    
    % Obtain spike trains of perceived stimulus with hearing loss
    disp('Computing spike trains for perceived information')
    [spt_per.left,spt_per.t] = BEZ2018(per_audio(:,1), fs_p, species, ...
        audiogram.cfs, audiogram.left, CFs, numsponts_healthy);
    
    [spt_per.right,~] = BEZ2018(per_audio(:,2), fs_p, species, ...
        audiogram.cfs, audiogram.right, CFs, numsponts_healthy);
end

%% Get information from spike trains
if useGPU
    information = binauralMI_GPU(spt_trans, spt_per, numcfs, nFibers);
else
    information = binauralMI(spt_trans, spt_per, numcfs, nFibers);
end

