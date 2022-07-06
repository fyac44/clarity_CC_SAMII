function BEZ_model(root_path, dataset, n_process)
% Implementation of the BEZ2018 to obtain the spike activity and  

%% Parameters
metadata_path = [root_path filesep 'data' filesep 'clarity_data' ...
    filesep 'metadata'];
scenes_data_path = [root_path filesep 'data' filesep 'clarity_data' ...
    filesep 'clarity_data' filesep 'scenes'];
ha_output_path = [root_path filesep 'data' filesep 'clarity_data' ...
    filesep 'clarity_data' filesep 'HA_outputs' filesep dataset];
output_path = [root_path filesep 'data' filesep 'clarity_data' ...
    filesep 'clarity_data' filesep 'mutualinfo' filesep dataset];
addpath([root_path filesep 'projects' filesep 'mi'])
n_process = str2num(n_process); %#ok<ST2NM> 
species = 2;
ag_fs = [125 250 500 1e3 2e3 4e3 8e3];
ag_dbloss = [0 0 0 0 0 0 0]; % No hearing loss
% stimdb = 65; % normalize all wav files to 65 dB

%% Read json files
% Scenes
%fscenes = [metadata_path filesep 'scenes.CPC1_train.json']; 
%scene_lists = readJSON(fscenes);
% Listeners
flisteners = [metadata_path filesep 'listeners.CPC1_train.json']; 
listeners = readJSON(flisteners);
% Signals
fsignals = [metadata_path filesep 'CPC1.' dataset '.json']; 
signals = readJSON(fsignals);
if n_process==0
    n_process = length(signals);
end
selected_signals = signals(1:n_process);

% Parameters
maxITD = 1e-3; % Maximun interaural time difference (As in MBSTOI)
numcfs = 40; % number of center frequencies (as in BEZ2018)
CFs   = logspace(log10(125),log10(16e3),numcfs);  % CF in Hz;
%numsponts_healthy = [10 10 30]; % As in BEZ2018
numsponts_healthy = [16 23 61]; % Number of low-spont, medium-spont, 
                                % and high-spont fibers at each CF in
                                % a healthy ANF (100 ANFs)
                                % (Known ratio between those fibers)
nFibers = sum(numsponts_healthy, 'all');
winL = 20e-3; % Take snippets of 20ms of the spike activity
trefmin = 208.5e-6+131e-6; % Minimun refractory period possible in BEZ2018.
                           % As the sum of min(tabs) + min(trel).

for s=1:numel(selected_signals)
    disp(['*** File ' num2str(s) ':' num2str(numel(selected_signals)) ' ***'])
    signal = selected_signals(s);
    scene = signal.scene;
    listener_id = signal.listener;
    listener = listeners.(listener_id);
    fname = [output_path filesep signal.signal '.json'];
    if isfile(fname)
        continue
    end
    %system = signal.system;

    % Path to stimuli
    ha_stim_path = ...
        [ha_output_path filesep signal.signal '.wav'];
    anechoic_stim_path = ...
        [scenes_data_path filesep scene '_target_anechoic.wav'];

    % read stimuli
    [ha_stim, Fs_ha_stim] = audioread(ha_stim_path);
    [an_stim, Fs_an_stim] = audioread(anechoic_stim_path);
    
    % Obtain spike activity with hearing loss + hearing aids
    disp('Computing HA L')
    [ha_l_spikes,t_ha_spikes] = ...
        generate_neurogram_BEZ2018_CUDA(ha_stim(:,1), Fs_ha_stim, species,...
        listener.audiogram_cfs', listener.audiogram_levels_l, CFs, ...
        numsponts_healthy);
    disp('Computing HA R')
    [ha_r_spikes,~] = ...
        generate_neurogram_BEZ2018_CUDA(ha_stim(:,2), Fs_ha_stim, species,...
        listener.audiogram_cfs', listener.audiogram_levels_r, CFs, ...
        numsponts_healthy);

    % Obtain spike activity of anechoic stimuli
    disp('Computing AN L')
    [an_l_spikes,t_an_spikes] = ...
        generate_neurogram_BEZ2018_CUDA(an_stim(:,1), Fs_an_stim, species,...
        ag_fs, ag_dbloss, CFs, numsponts_healthy);
    disp('Computing AN R')
    [an_r_spikes,~] = ...
        generate_neurogram_BEZ2018_CUDA(an_stim(:,2), Fs_an_stim, species,...
        ag_fs, ag_dbloss, CFs, numsponts_healthy);

    % solve any missmatch in shape between hearing aids and anechoic
    % signals by trimming the signals to the shorter lenght
    disp('Computing Mutual Information')
    size_ha = size(t_ha_spikes,2);
    size_an = size(t_an_spikes,2);
    min_size = min(size_ha, size_an);

    % Look at the Mutual information per every frame of the neurogram
    T = t_an_spikes(2); %Periodicity of the spike activity signal
    itd_frames = ceil(maxITD/T);
    itds = -itd_frames:itd_frames; % interaual time differece for binaural
    i_frames = floor(trefmin/T); % Integration frames (2 spikes in the same 
                                 % AN fiber is not possible within this 
                                 % frame window)
    N = i_frames*floor(winL/(i_frames*T)); % Total samples in winL
    hop = round(N/2); % Hop size
    total_frames = ceil(min_size/hop); % Total time frames

    left_mi = gpuArray(zeros(total_frames, numcfs));
    left_ti = gpuArray(zeros(total_frames, numcfs));
    left_pi = gpuArray(zeros(total_frames, numcfs));
    right_mi = gpuArray(zeros(total_frames, numcfs));
    right_ti = gpuArray(zeros(total_frames, numcfs));
    right_pi = gpuArray(zeros(total_frames, numcfs));
    binaural_mi = gpuArray(zeros(total_frames, numcfs));
    binaural_ti = gpuArray(zeros(total_frames, numcfs));
    binaural_pi = gpuArray(zeros(total_frames, numcfs));
    an_delay = zeros(1, total_frames);
    ha_delay = zeros(1, total_frames);

    for n=1:total_frames
        % Get frames
        init_frame = 1+(n-1)*hop;
        end_frame = min(init_frame+N-1, min_size);
    
        % Best itd
        lowest_diff_an = 1e27;
        lowest_diff_ha = 1e27;
        for itd=itds
            if init_frame + itd < 1
                continue
            end
            if end_frame + itd <= size_an
                diff_an = sqrt(sum(sum((...
                    an_l_spikes(:,init_frame+itd:end_frame+itd) - ...
                    an_r_spikes(:,init_frame:end_frame)).^2)));
                 if diff_an < lowest_diff_an
                    delay_an = itd;
                    lowest_diff_an = diff_an;
                end 
            end
            if end_frame + itd <= size_ha
                diff_ha = sqrt(sum(sum((...
                    ha_l_spikes(:,init_frame+itd:end_frame+itd) - ...
                    ha_r_spikes(:,init_frame:end_frame)).^2)));
    
                if diff_ha < lowest_diff_ha
                    delay_ha = itd;
                    lowest_diff_ha = diff_ha;
                end
            end
        end
        % Delay in seconds for the best mi (negative values means that
        % ha signal is ahead of the anechoic signal)
        an_delay(n) = delay_an;
        ha_delay(n) = delay_ha;
    end
    
    an_delay = gpuArray(an_delay);
    ha_delay = gpuArray(ha_delay);
    
    an_l_spikes = gpuArray(an_l_spikes');
    an_r_spikes = gpuArray(an_r_spikes');
    ha_l_spikes = gpuArray(ha_l_spikes');
    ha_r_spikes = gpuArray(ha_r_spikes');
    
    kernel_mi = ...
        parallel.gpu.CUDAKernel('mutual_info.ptx', 'mutual_info.cu');
    kernel_mi.ThreadBlockSize = [min(kernel_mi.MaxThreadsPerBlock, numcfs*total_frames),1,1]; 
    kernel_mi.GridSize = [ceil(numcfs*total_frames/kernel_mi.MaxThreadsPerBlock),1];
    
    [left_mi, right_mi, binaural_mi, left_pi, right_pi, binaural_pi, left_ti, right_ti, binaural_ti] = ...
        feval(kernel_mi, left_mi, right_mi, binaural_mi, left_pi, right_pi, binaural_pi, left_ti, right_ti, binaural_ti, ...
        ha_l_spikes, ha_r_spikes, an_l_spikes, an_r_spikes, ha_delay, an_delay, ...
        nFibers, N, hop, i_frames, min_size, numcfs, total_frames);
    
    % Output structure
    output = struct();
    output.CFs = CFs;
    output.timeStamps = (0:total_frames-1)*hop*T;
    output.left = struct();
    output.left.mi = gather(left_mi');
    output.left.ti = gather(left_ti');
    output.left.pi = gather(left_pi');
    output.right = struct();
    output.right.mi = gather(right_mi');
    output.right.ti = gather(right_ti');
    output.right.pi = gather(right_pi');
    output.binaural = struct();
    output.binaural.an_delay = T*gather(an_delay);
    output.binaural.ha_delay = T*gather(ha_delay);
    output.binaural.mi = gather(binaural_mi');
    output.binaural.ti = gather(binaural_ti');
    output.binaural.pi = gather(binaural_pi');

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

% function ent = my_entropy(a,nF)
%     rho_s = sum(a,'all')/(numel(a)*nF);
%     if rho_s == 0 || rho_s == 1
%         ent = 0;
%     else
%         ent = - (rho_s*log2(rho_s) + (1-rho_s)*log2(1-rho_s));
%     end
% end
% 
% function mi = my_mutualinfo(x, y, nF)
%     assert(length(x)==length(y))
%     N = length(x);
%     jpdf = zeros(2,2);
%     for n=1:N
%         o11 = min(x(n), y(n)); % Ocurrences of [1 1]
%         o01 = max(0,y(n)-x(n)); % Ocurrences of [0 1]
%         o10 = max(0,x(n)-y(n)); % Ocurrences of [1 0]
%         o00 = nF - o11 - o01 - o10; % Ocurrences of [0 0]
%         jpdf = jpdf + [o00, o01 ; ...
%                        o10, o11];
%     end
%     jpdf = jpdf/(N*nF);
%     ent_x = my_entropy(x, nF);
%     ent_y = my_entropy(y, nF);
%     jent_xy = - [jpdf(1,1)*log2(jpdf(1,1)), ...
%                  jpdf(1,2)*log2(jpdf(1,2)), ...
%                  jpdf(2,1)*log2(jpdf(2,1)), ...
%                  jpdf(2,2)*log2(jpdf(2,2))];
%     jent_xy(isnan(jent_xy)) = 0;
%     mi = ent_x + ent_y - sum(jent_xy, 'all');
% end
% 
% function reduced_spikes = reduce(spikes, frames)
%     N = ceil(length(spikes)/frames);
%     reduced_spikes = zeros(1, N);
%     for n=0:N-1
%         initf = n*frames + 1;
%         endf = min(initf+frames-1, length(spikes));
%         reduced_spikes(n+1) = sum(spikes(initf:endf));
%     end
% end