function information = binauralMI(spt_trans, spt_per, numcfs, nFibers)

% Solve any mismatch due to different audio sample frequencies
size_trans = length(spt_trans.t);
size_per = length(spt_per.t);
min_size = min(size_per, size_trans);

%% MI Parameters
awinL = 20e-3; % Analysis window [s]
trefmin = 208.5e-6; % Minimun refractory period possible in BEZ2018 [s].

% Analysis window and integration window sizes in frames
samplesT = spt_trans.t(2); % Periodicity of the spike train samples
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
an_delay = zeros(1, awTotal);
ha_delay = zeros(1, awTotal);

for n=1:awTotal
    % Get frames
    init_frame = 1+(n-1)*awHop;
    end_frame = min(init_frame+awSamples-1, min_size);

    % Best itd
    lowest_diff_an = 1e27;
    lowest_diff_ha = 1e27;
    for itd=itds
        if init_frame + itd < 1
            continue
        end
        if end_frame + itd <= size_trans
            diff_an = sqrt(sum(sum((...
                spt_trans.left(:,init_frame+itd:end_frame+itd) - ...
                spt_trans.right(:,init_frame:end_frame)).^2)));
             if diff_an < lowest_diff_an
                delay_an = itd;
                lowest_diff_an = diff_an;
            end 
        end
        if end_frame + itd <= size_per
            diff_ha = sqrt(sum(sum((...
                spt_per.left(:,init_frame+itd:end_frame+itd) - ...
                spt_per.right(:,init_frame:end_frame)).^2)));

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

left_mi = gpuArray(zeros(awTotal, numcfs));
left_ti = gpuArray(zeros(awTotal, numcfs));
left_pi = gpuArray(zeros(awTotal, numcfs));
right_mi = gpuArray(zeros(awTotal, numcfs));
right_ti = gpuArray(zeros(awTotal, numcfs));
right_pi = gpuArray(zeros(awTotal, numcfs));
binaural_mi = gpuArray(zeros(awTotal, numcfs));
binaural_ti = gpuArray(zeros(awTotal, numcfs));
binaural_pi = gpuArray(zeros(awTotal, numcfs));

an_delay = gpuArray(an_delay);
ha_delay = gpuArray(ha_delay);

spt_trans_l = gpuArray(spt_trans.left');
spt_trans_r = gpuArray(spt_trans.right');
spt_per_l = gpuArray(spt_per.left');
spt_per_r = gpuArray(spt_per.right');

kernel_mi = ...
    parallel.gpu.CUDAKernel('mutual_info.ptx', 'mutual_info.cu');
kernel_mi.ThreadBlockSize = [min(kernel_mi.MaxThreadsPerBlock, numcfs*awTotal),1,1]; 
kernel_mi.GridSize = [ceil(numcfs*awTotal/kernel_mi.MaxThreadsPerBlock),1];

[left_mi, right_mi, binaural_mi, left_pi, right_pi, binaural_pi, left_ti, right_ti, binaural_ti] = ...
    feval(kernel_mi, left_mi, right_mi, binaural_mi, left_pi, right_pi, binaural_pi, left_ti, right_ti, binaural_ti, ...
    spt_per_l, spt_per_r, spt_trans_l, spt_trans_r, ha_delay, an_delay, ...
    nFibers, awSamples, awHop, iSamples, min_size, numcfs, awTotal);

% Output structure
information = struct();
information.CFs = CFs;
information.timeStamps = (0:awTotal-1)*awHop*samplesT;
information.left = struct();
information.left.mi = gather(left_mi');
information.left.ti = gather(left_ti');
information.left.pi = gather(left_pi');
information.right = struct();
information.right.mi = gather(right_mi');
information.right.ti = gather(right_ti');
information.right.pi = gather(right_pi');
information.binaural = struct();
information.binaural.an_delay = samplesT*gather(an_delay);
information.binaural.ha_delay = samplesT*gather(ha_delay);
information.binaural.mi = gather(binaural_mi');
information.binaural.ti = gather(binaural_ti');
information.binaural.pi = gather(binaural_pi');

end

