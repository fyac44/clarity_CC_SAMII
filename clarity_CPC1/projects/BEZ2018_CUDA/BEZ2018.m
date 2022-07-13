function [psth, t_psth] = ...
    BEZ2018(stim,Fs_stim,ag_fs,ag_dbloss,CFs,numsponts)
% This is a modification of the orignal function "generate_neurogram" from 
% the code provided with the publication Bruce et al., (2018).
%
% Input:
%  - stim:      Sound wave that reaches the ear [uPa]
%  - Fs_stim:   Sample frequency of the sound wave [Hz]
%  - ag_fs:     Listener's audiogram frequencies [Hz]
%  - ag_dbloss: Listener's audiogram losses [dB]
%  - CFs:       Center frequencies for the BEZ2018 model [Hz]
%  - numsponts: 1x3 array with the amount of low sapontaneous, mid 
%               spontaneous and high spontaneous rate fibers per CF.
%
% Output:
%  - psth:      Post Stimulus Time Histogram (Spikes grouped by CF)
%  - t_psth:    Time stamps for the psth [s]

species = 2; % Use human version of the BEZ2018Model
numcfs = length(CFs);

% Interpolate the loss across CFs from the listener audiogram
dbloss = interp1(ag_fs,ag_dbloss,CFs,'linear','extrap');
[cohcs,cihcs,~]=fitaudiogram2(CFs,dbloss,species);

% Generate/read the threshold values and refractory periods for the ANF
% population
if exist('ANpopulation.mat','file')
    load('ANpopulation.mat', 'sponts', 'tabss', 'trels');
    if (size(sponts.LS,2)<numsponts(1)) || ...
            (size(sponts.MS,2)<numsponts(2)) || ...
            (size(sponts.HS,2)<numsponts(3)) || ...
            (size(sponts.HS,1)<numcfs || ~exist('tabss','var'))
        [sponts,tabss,trels] = generateANpopulation(numcfs,numsponts);
    end
else
    [sponts,tabss,trels] = generateANpopulation(numcfs,numsponts);
end

implnt = 0;
noiseType = 1;  % 0 for fixed fGn (1 for variable fGn)

% stimulus parameters
Fs = 100e3;  % sampling rate in Hz (must be 100, 200 or 500 kHz)
stim100k = resample(stim,Fs,Fs_stim).';
nrep = 1; % ** THIS SHOULD ALWAYS BE ONE **
stimdur  = length(stim100k)/Fs;  % stimulus duration in seconds

pin = stim100k(:).';
clear stim100k

% prepare output ** If nrep is other than 1 this code will fail **
psth = zeros(numcfs, length(pin));

% Compute the spike trains per critical band
for CFlp = 1:numcfs
    
    CF = CFs(CFlp);     % Center frequency of the critical band
    cohc = cohcs(CFlp); % Outer hair cell loss
    cihc = cihcs(CFlp); % Inner hair cell loss
    
    % Thresholds and refractoriness from the fibers in the current CF
    sponts_concat = ...
        [sponts.LS(CFlp,1:numsponts(1)) sponts.MS(CFlp,1:numsponts(2)) ...
        sponts.HS(CFlp,1:numsponts(3))];
    tabss_concat = ...
        [tabss.LS(CFlp,1:numsponts(1)) tabss.MS(CFlp,1:numsponts(2)) ...
        tabss.HS(CFlp,1:numsponts(3))];
    trels_concat = ...
        [trels.LS(CFlp,1:numsponts(1)) trels.MS(CFlp,1:numsponts(2)) ...
        trels.HS(CFlp,1:numsponts(3))];
    
    % Obtain the relative transmembrane potential with the IHC model
    vihc = model_IHC_BEZ2018(pin,CF,nrep,1/Fs,stimdur,cohc,cihc,species);
    
    % Obtain the spike train for every ANFs in the CF
    for spontlp = 1:sum(numsponts)        
        spont = sponts_concat(spontlp);
        tabs = tabss_concat(spontlp);
        trel = trels_concat(spontlp);
        
        % Synapse model to obtain the spike train
        [psth_ft,~,~,~] = ...
            model_Synapse_BEZ2018(vihc,CF,nrep,1/Fs,noiseType,implnt,...
            spont,tabs,trel);

        % Critical band integration
        psth(CFlp,:) = psth(CFlp,:)+psth_ft;
    end
    
end

% time vector for the spike activity
t_psth = 0:1/Fs:(size(psth,2)-1)/Fs; 