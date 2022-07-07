%function [psth,t_ft] = generate_neurogram_BEZ2018_CUDA(stim,Fs_stim,species,ag_fs,ag_dbloss,CFs,numsponts_healthy)
function [neurogram_ft,neurogram_mr,t_ft,t_mr,CFs] = Copy_of_generate_neurogram_BEZ2018_CUDA(stim,Fs_stim,species,ag_fs,ag_dbloss)
% model fiber parameters

numcfs = 40;
CFs   = logspace(log10(250),log10(16e3),numcfs);  % CF in Hz;

%numcfs = length(CFs);
% cohcs  = ones(1,numcfs);  % normal ohc function
% cihcs  = ones(1,numcfs);  % normal ihc function

dbloss = interp1(ag_fs,ag_dbloss,CFs,'linear','extrap');

% mixed loss
[cohcs,cihcs,OHC_Loss]=fitaudiogram2(CFs,dbloss,species);

% OHC loss
% [cohcs,cihcs,OHC_Loss]=fitaudiogram(CFs,dbloss,species,dbloss);

% IHC loss
% [cohcs,cihcs,OHC_Loss]=fitaudiogram(CFs,dbloss,species,zeros(size(CFs)));

numsponts_healthy = [16 23 61]; % Number of low-spont, medium-spont, and high-spont fibers at each CF in a healthy AN

if exist('ANpopulation.mat','file')
    load('ANpopulation.mat');
    if (size(sponts.LS,2)<numsponts_healthy(1))||(size(sponts.MS,2)<numsponts_healthy(2))||(size(sponts.HS,2)<numsponts_healthy(3))||(size(sponts.HS,1)<numcfs||~exist('tabss','var'))
        [sponts,tabss,trels] = generateANpopulation(numcfs,numsponts_healthy);
    end
else
    [sponts,tabss,trels] = generateANpopulation(numcfs,numsponts_healthy);
end

implnt = 0;    % "0" for approximate or "1" for actual implementation of the power-law functions in the Synapse
noiseType = 1;  % 0 for fixed fGn (1 for variable fGn)

% stimulus parameters
Fs = 100e3;  % sampling rate in Hz (must be 100, 200 or 500 kHz)
stim100k = resample(stim,Fs,Fs_stim).';
T  = length(stim100k)/Fs;  % stimulus duration in seconds

% PSTH parameters
nrep = 1;
psthbinwidth_mr = 100e-6; % mean-rate binwidth in seconds;
windur_ft=32;
smw_ft = hamming(windur_ft);
windur_mr=128;
smw_mr = hamming(windur_mr);

pin = stim100k(:).';

clear stim100k

simdur = ceil(T*1.2/psthbinwidth_mr)*psthbinwidth_mr;

meout = gpuArray(model_middleear(pin, species, 1/Fs, simdur));

numsponts = round([1 1 1].*numsponts_healthy); % Healthy AN
%     numsponts = round([0.5 0.5 0.5].*numsponts_healthy); % 50% fiber loss of all types
%     numsponts = round([0 1 1].*numsponts_healthy); % Loss of all LS fibers
%     numsponts = round([cihc 1 cihc].*numsponts_healthy); % loss of LS and HS fibers proportional to IHC impairment
ANFperCF =  sum(numsponts);
totalANFs = ANFperCF*numcfs;
cohcs = gpuArray(cohcs);
cihcs = gpuArray(cihcs);
CFs = gpuArray(CFs);

nsamplesrep = length(meout);
nsamples = nrep*nsamplesrep;

vihc = gpuArray(zeros(nsamples, numcfs));

ihcouttmp = gpuArray(zeros(nsamples, numcfs));
tmpgain = gpuArray(zeros(nsamplesrep, numcfs));

kernel_vihc = ...
    parallel.gpu.CUDAKernel('model_IHC_BEZ2018.ptx', 'model_IHC_BEZ2018.cu');
kernel_vihc.ThreadBlockSize = [min(kernel_vihc.MaxThreadsPerBlock, numcfs),1,1]; 
kernel_vihc.GridSize = [ceil(numcfs/kernel_vihc.MaxThreadsPerBlock),1];

vihc = feval(kernel_vihc, vihc, ihcouttmp, tmpgain, meout, CFs, nrep, 1/Fs, nsamplesrep, cohcs, cihcs, species, numcfs);

clear ihcouttmp tmpgain meout cohcs cihcs;

sponts_concat = [sponts.LS(:,1:numsponts(1)) sponts.MS(:,1:numsponts(2)) sponts.HS(:,1:numsponts(3))]';
tabss_concat = [tabss.LS(:,1:numsponts(1)) tabss.MS(:,1:numsponts(2)) tabss.HS(:,1:numsponts(3))]';
trels_concat = [trels.LS(:,1:numsponts(1)) trels.MS(:,1:numsponts(2)) trels.HS(:,1:numsponts(3))]';
maxSpikes = ceil(simdur/min(tabss_concat,[],'All'));

sponts_concat = gpuArray(sponts_concat(:)');
tabss_concat = gpuArray(tabss_concat(:)');
trels_concat = gpuArray(trels_concat(:)');

nSites = 4;      % Number of synpatic release sites 

plSampFreq = 10e3;
hurstIndex = 0.9;
resamp = ceil(Fs/plSampFreq);

kernel_synapse = ...
    parallel.gpu.CUDAKernel('model_Synapse_BEZ2018.ptx', 'model_Synapse_BEZ2018.cu');
kernel_synapse.ThreadBlockSize = [min(kernel_synapse.MaxThreadsPerBlock, totalANFs),1,1]; 
kernel_synapse.GridSize = [ceil(totalANFs/kernel_synapse.MaxThreadsPerBlock),1];

% SpikeTimes
tSpikes = gpuArray(zeros(maxSpikes, totalANFs));
spCount = gpuArray(zeros(1, totalANFs));
eventsSize = maxSpikes*4+6*nSites+1;
eventsRand = gpuArray(rand(eventsSize, totalANFs));
delaypoints = floor(7500./(CFs./1e3));
max_delaypoint = gather(max(delaypoints));
max_ndownsampled = ceil((nsamples+2*max_delaypoint)*plSampFreq/Fs);
%sout1 = gpuArray(zeros(max_ndownsampled, totalANFs));
%sout2 = gpuArray(zeros(max_ndownsampled, totalANFs));

resampN = ceil(1e-1*plSampFreq);
maxRandNums = ceil(max_ndownsampled/resampN)+1; 
if (maxRandNums<10)
    maxRandNums = 10;
end
nsRandNums = gpuArray(zeros(1, numcfs));
randResamps = gpuArray(zeros(1, numcfs));
randNums = gpuArray(zeros(maxRandNums,totalANFs));

for cf=1:numcfs
    anf_end = cf*ANFperCF;
    anf_start = anf_end - ANFperCF + 1;
    ndownsampled = gather(ceil((nsamples+2*delaypoints(cf))*plSampFreq/Fs));
    Nend = ceil(ndownsampled/resampN)+1; 
    if (Nend<10)
        Nend = 10;
    end
    [randNums(1:Nend, anf_start:anf_end), nsRandNums(cf), randResamps(cf)] = ...
        ffGn_gpu(ndownsampled, ANFperCF, 1/plSampFreq, hurstIndex, noiseType);
end
    
[tSpikes, spCount] = feval(kernel_synapse, tSpikes, spCount, vihc, ...
    CFs, randNums, nsRandNums, randResamps, eventsRand, sponts_concat, tabss_concat, trels_concat, ...
    delaypoints, nsamples, resamp, 1/Fs, plSampFreq, ...
    nSites, maxRandNums, maxSpikes, eventsSize, ANFperCF, totalANFs);

clear randNums vihc eventRands sponts_concat tabss_concat trels_concat ranResamps nsRandNums delaypoints enventsRand synout %CFs

CFs = gather(CFs);

psth = zeros(totalANFs, nsamplesrep);
spCount = gather(spCount);
tSpikes = gather(tSpikes);

for anf=1:totalANFs
    sps = spCount(anf);
    if sps <0
        disp(['error in anf: ' num2str(anf)])
    else
        for i=1:sps
            ipst = 1+mod(round( nsamples*tSpikes(i, anf)/(simdur*nrep)),nsamplesrep);
            psth(anf, ipst) = psth(anf, ipst) + 1;
        end
    end
end

%neurogram_ft = zeros(numcfs, nsamplesrep);
%t_ft = 0:1/Fs:(nsamplesrep-1)*1/Fs; 

for CFlp = 1:numcfs
    for spontlp = 1:ANFperCF
        psth_ft = psth((CFlp-1)*ANFperCF+spontlp,:);
        psthbins = round(psthbinwidth_mr*Fs);  % number of psth_ft bins per psth bin
        psth_mr = sum(reshape(psth_ft,psthbins,length(psth_ft)/psthbins));
        if spontlp == 1
            neurogram_ft(CFlp,:) = filter(smw_ft,1,psth_ft);
            %neurogram_Sout(CFlp,:) = synout;            
            neurogram_mr(CFlp,:) = filter(smw_mr,1,psth_mr);
        else
            
            neurogram_ft(CFlp,:) = neurogram_ft(CFlp,:)+filter(smw_ft,1,psth_ft);
            %neurogram_Sout(CFlp,:) = neurogram_Sout(CFlp,:)+synout;
            neurogram_mr(CFlp,:) = neurogram_mr(CFlp,:)+filter(smw_mr,1,psth_mr);
        end
    end
end

% 
neurogram_ft = neurogram_ft(:,1:windur_ft/2:end); % 50% overlap in Hamming window
t_ft = 0:windur_ft/2/Fs:(size(neurogram_ft,2)-1)*windur_ft/2/Fs; % time vector for the fine-timing neurogram
neurogram_mr = neurogram_mr(:,1:windur_mr/2:end); % 50% overlap in Hamming window
t_mr = 0:windur_mr/2*psthbinwidth_mr:(size(neurogram_mr,2)-1)*windur_mr/2*psthbinwidth_mr; % time vector for the mean-rate neurogram
%neurogram_Sout = squeeze(sum(reshape(gather(synout'), [ANFperCF, numcfs, nsamples]),1));
% t_Sout = 0:1/Fs:(size(neurogram_Sout,2)-1)/Fs; % time vector for the synapse output neurogram
% %t_mr = [0 t_Sout(end)];

