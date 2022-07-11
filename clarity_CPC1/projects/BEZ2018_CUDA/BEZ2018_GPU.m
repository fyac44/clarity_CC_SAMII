function [psth, t_psth] = ...
    BEZ2018_GPU(stim,Fs_stim,ag_fs,ag_dbloss,CFs,numsponts)
% This is a modification of the orignal function "generate_neurogram" from 
% the code provided with the publication Bruce et al., (2018). Some remarks
% about the CUDA implementation:
% - The power law implementation will be always approximate.
% - The upsampling of the gaussian noise (Also for power law) is performed
%   with linear interpolation instead of using the "resample" algorithm
%   from MATLAB. This results in a sharp noise signal.
% - Is only one repetition of the whole stimulus, so nrep is always 1
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

% implnt = 0; CUDA code is optimized only for the approximated
% implementation of the power law function
noiseType = 1;  % 0 for fixed fGn (1 for variable fGn)

% stimulus parameters
Fs = 100e3;  % sampling rate in Hz (must be 100, 200 or 500 kHz)
stim100k = resample(stim,Fs,Fs_stim).';
stimdur  = length(stim100k)/Fs;  % stimulus duration in seconds

pin = stim100k(:).';
clear stim100k

% Apply the middle ear filter to the stim signal (MEX funtion)
meout = gpuArray(model_middleear(pin, species, 1/Fs));

% Parameters for the IHC model
ANFperCF =  sum(numsponts);
totalANFs = ANFperCF*numcfs;
cohcs = gpuArray(cohcs);
cihcs = gpuArray(cihcs);
CFs = gpuArray(CFs);
nsamples = length(meout);

vihc = gpuArray(zeros(nsamples, numcfs));
ihcouttmp = gpuArray(zeros(nsamples, numcfs));
tmpgain = gpuArray(zeros(nsamples, numcfs));

% Upload IHC model kernel
kernel_vihc = ...
    parallel.gpu.CUDAKernel('model_IHC_BEZ2018.ptx', ...
    'model_IHC_BEZ2018.cu');
kernel_vihc.ThreadBlockSize = ...
    [min(kernel_vihc.MaxThreadsPerBlock, numcfs),1,1]; 
kernel_vihc.GridSize = [ceil(numcfs/kernel_vihc.MaxThreadsPerBlock),1];

% Excecute IHC model with NVIDIA GPU
vihc = feval(kernel_vihc, vihc, ihcouttmp, tmpgain, meout, CFs, 1/Fs, ...
    nsamples, cohcs, cihcs, species, numcfs);

clear ihcouttmp tmpgain meout cohcs cihcs;

% Concatenate the thresholds and refractory period values and send them to
% the GPU
sponts_concat = ...
    [sponts.LS(:,1:numsponts(1)) sponts.MS(:,1:numsponts(2)) ...
    sponts.HS(:,1:numsponts(3))]';
tabss_concat = ...
    [tabss.LS(:,1:numsponts(1)) tabss.MS(:,1:numsponts(2)) ...
    tabss.HS(:,1:numsponts(3))]';
trels_concat = ...
    [trels.LS(:,1:numsponts(1)) trels.MS(:,1:numsponts(2)) ...
    trels.HS(:,1:numsponts(3))]';
maxSpikes = ceil(stimdur/min(tabss_concat,[],'All'));

sponts_concat = gpuArray(sponts_concat(:)');
tabss_concat = gpuArray(tabss_concat(:)');
trels_concat = gpuArray(trels_concat(:)');

% Parameters for the synapsys model
nSites = 4;      % Number of synpatic release sites 

plSampFreq = 10e3;
hurstIndex = 0.9;
resamp = ceil(Fs/plSampFreq);

% Upload the synapse model kernel
kernel_synapse = ...
    parallel.gpu.CUDAKernel('model_Synapse_BEZ2018.ptx', ...
    'model_Synapse_BEZ2018.cu');
kernel_synapse.ThreadBlockSize = ...
    [min(kernel_synapse.MaxThreadsPerBlock, totalANFs),1,1]; 
kernel_synapse.GridSize = ...
    [ceil(totalANFs/kernel_synapse.MaxThreadsPerBlock),1];

% Configure arrays needed for the synapse model
tSpikes = gpuArray(zeros(maxSpikes, totalANFs)); % Spike times
spCount = gpuArray(zeros(1, totalANFs)); % Spike count
eventsSize = maxSpikes*4+6*nSites+1;
eventsRand = gpuArray(rand(eventsSize, totalANFs)); % Random numbers for 
                                                    % needed in every event
                                                    % (Spikes, redocking, 
                                                    % refractory periods)
delaypoints = floor(7500./(CFs./1e3));
max_delaypoint = gather(max(delaypoints));
max_ndownsampled = ceil((nsamples+2*max_delaypoint)*plSampFreq/Fs);

% Generate noise
resampN = ceil(1e-1*plSampFreq);
maxRandNums = ceil(max_ndownsampled/resampN)+1; 
if (maxRandNums<10)
    maxRandNums = 10;
end
nsRandNums = gpuArray(zeros(1, numcfs)); % How many random points per cf
randResamps = gpuArray(zeros(1, numcfs)); % Resample for those points
randNums = gpuArray(zeros(maxRandNums,totalANFs)); % Random points for 
                                                   % gaussian noise in the 
                                                   % Power-law adaptation

% Obtain the random points for the Gaussian noise for every center
% frequency
% NOTE: In this implementation a linear regression is performed instead of 
% using the resample function from matlab. This will end in a sharper shape
% of the noise.
for cf=1:numcfs
    anfEnd = cf*ANFperCF;
    anfStart = anfEnd - ANFperCF + 1;
    ndownsampled = ...
        gather(ceil((nsamples+2*delaypoints(cf))*plSampFreq/Fs));
    Nend = ceil(ndownsampled/resampN)+1; 
    if (Nend<10)
        Nend = 10;
    end
    % Generate a Gaussian shape noise
    [randNums(1:Nend, anfStart:anfEnd), nsRandNums(cf), ...
        randResamps(cf)] = ffGn_gpu(ndownsampled, ANFperCF, ...
        1/plSampFreq, hurstIndex, noiseType);
end

% Excecute the synapse model kernel
[tSpikes, spCount] = feval(kernel_synapse, tSpikes, spCount, vihc, CFs, ...
    randNums, nsRandNums, randResamps, eventsRand, sponts_concat, ...
    tabss_concat, trels_concat, delaypoints, nsamples, resamp, 1/Fs, ...
    plSampFreq, nSites, maxRandNums, maxSpikes, eventsSize, ANFperCF, ...
    totalANFs);

clear randNums vihc eventRands sponts_concat tabss_concat trels_concat ...
    randResamps nsRandNums delaypoints eventsRand synout CFs

% Prepare output
psth = zeros(numcfs, nsamples);
t_psth = 0:1/Fs:(nsamples-1)*1/Fs; 

% Move from GPU to CPU
spCount = gather(spCount);
tSpikes = gather(tSpikes);

% Obtain the spike train from every ANF and sum them by center frequency
for CFlp = 1:numcfs
    for spontlp = 1:ANFperCF
        anf = (CFlp-1)*ANFperCF+spontlp;
        sps = spCount(anf);
        if sps <0
            disp(['error in anf: ' num2str(anf)])
        else
            for i=1:sps
                ipst = 1+round(nsamples*tSpikes(i, anf)/stimdur);
                psth(CFlp,ipst) = psth(CFlp,ipst) + 1;
            end
        end
    end
end

