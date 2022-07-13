function information = binauralMI_GPU(spt_S, spt_R, numcfs, nFibers)

% Solve any mismatch due to different audio sample frequencies
size_S = length(spt_S.t);
size_R = length(spt_R.t);
min_size = min(size_R, size_S);

awinL = 20e-3; % Analysis window [s]
trefmin = 208.5e-6; % Minimun refractory period possible in BEZ2018 [s].

% Analysis window and integration window sizes in frames
samplesT = spt_S.t(2); % Periodicity of the spike train samples
iSamples = floor(trefmin/samplesT); % Samples in an integration frame 
                                    % (2 spikes cannot belong to the same
                                    % AN fiber within these samples)
awSamples = iSamples*floor(awinL/(iSamples*samplesT)); % Total samples in 
                                                       % an analysis window
awHop = round(awSamples/2); % Hop size
awTotal = ceil(min_size/awHop); % Total of analysis windows

%% Binaural delays for transmitted and perceived signal.
maxITD = 1e-3; % Maximun interaural time difference for binaural representation (As in MBSTOI)
itd_frames = ceil(maxITD/samplesT);
itds = -itd_frames:itd_frames; % interaual time differece for binaural
S_delay = zeros(1, awTotal);
R_delay = zeros(1, awTotal);

for n=1:awTotal
    % Get frames
    init_frame = 1+(n-1)*awHop;
    end_frame = min(init_frame+awSamples-1, min_size);

    % Best itd
    lowest_diff_S = 1e27;
    lowest_diff_R = 1e27;
    for itd=itds
        if init_frame + itd < 1
            continue
        end
        if end_frame + itd <= size_S
            diff_S = sqrt(sum(sum((...
                spt_S.left(:,init_frame+itd:end_frame+itd) - ...
                spt_S.right(:,init_frame:end_frame)).^2)));
             if diff_S < lowest_diff_S
                delay_S = itd;
                lowest_diff_S = diff_S;
            end 
        end
        if end_frame + itd <= size_R
            diff_R = sqrt(sum(sum((...
                spt_R.left(:,init_frame+itd:end_frame+itd) - ...
                spt_R.right(:,init_frame:end_frame)).^2)));

            if diff_R < lowest_diff_R
                delay_R = itd;
                lowest_diff_R = diff_R;
            end
        end
    end
    % Delay in seconds for the best binaural representation 
    S_delay(n) = delay_S;
    R_delay(n) = delay_R;
end

%% Memory allocation in the GPU
% Output information
%  - mi: Mutual information
%  - Si: Target information (i.e. information in clean speech)
%  - Ri: Perceived information (i.e. information in degraded speech)
%  - l, r and b: left, right and binaural
mi_l = gpuArray(zeros(awTotal, numcfs));
Si_l = gpuArray(zeros(awTotal, numcfs));
Ri_l = gpuArray(zeros(awTotal, numcfs));
mi_r = gpuArray(zeros(awTotal, numcfs));
Si_r = gpuArray(zeros(awTotal, numcfs));
Ri_r = gpuArray(zeros(awTotal, numcfs));
mi_b = gpuArray(zeros(awTotal, numcfs));
Si_b = gpuArray(zeros(awTotal, numcfs));
Ri_b = gpuArray(zeros(awTotal, numcfs));

% Spike trains per critical band
S_l = gpuArray(spt_S.left');
S_r = gpuArray(spt_S.right');
R_l = gpuArray(spt_R.left');
R_r = gpuArray(spt_R.right');

% Delay for binaural representation
S_delay = gpuArray(S_delay);
R_delay = gpuArray(R_delay);

%% Use the GPU

% Upload kernel
kernel_mi = ...
    parallel.gpu.CUDAKernel('mutual_info.ptx', 'mutual_info.cu');
kernel_mi.ThreadBlockSize = ...
    [min(kernel_mi.MaxThreadsPerBlock, numcfs*awTotal),1,1]; 
kernel_mi.GridSize = [ceil(numcfs*awTotal/kernel_mi.MaxThreadsPerBlock),1];

% Excecute kernel
[mi_l, mi_r, mi_b, Ri_l, Ri_r, Ri_b, Si_l, Si_r, Si_b] = ...
    feval(kernel_mi, mi_l, mi_r, mi_b, Ri_l, Ri_r, Ri_b, Si_l, Si_r, ...
    Si_b, R_l, R_r, S_l, S_r, R_delay, S_delay, nFibers, awSamples, ...
    awHop, iSamples, min_size, numcfs, awTotal);

%% Gather output structure
information = struct();
information.CFs = CFs;
information.timeStamps = (0:awTotal-1)*awHop*samplesT;
information.left = struct();
information.left.mi = gather(mi_l');
information.left.Si = gather(Si_l');
information.left.Ri = gather(Ri_l');
information.right = struct();
information.right.mi = gather(mi_r');
information.right.Si = gather(Si_r');
information.right.Ri = gather(Ri_r');
information.binaural = struct();
information.binaural.S_delay = samplesT*gather(S_delay);
information.binaural.R_delay = samplesT*gather(R_delay);
information.binaural.mi = gather(mi_b');
information.binaural.Si = gather(Si_b');
information.binaural.Ri = gather(Ri_b');

end

