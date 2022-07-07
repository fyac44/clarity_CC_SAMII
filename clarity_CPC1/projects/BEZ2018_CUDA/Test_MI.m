clear
clc
an_l_spikes = randi([0,2], 3, 30);
an_r_spikes = randi([0,2], 3, 30);
ha_l_spikes = randi([0,2], 3, 30);
ha_r_spikes = randi([0,2], 3, 30);

% solve any missmatch in shape between hearing aids and anechoic
% signals by trimming the signals to the shorter lenght
size_ha = 30;
size_an = 30;
min_size = min(size_ha, size_an);

% Look at the Mutual information per every frame of the neurogram
T = 1; %Periodicity of the spike activity signal
itd_frames = 2;
itds = -itd_frames:itd_frames; % interaual time differece for binaural
i_frames = 2; % Integration frames (2 spikes in the same 
                             % AN fiber is not possible within this 
                             % frame window)
N = 10; % Total samples in winL
hop = round(N/2); % Hop size
total_frames = ceil(min_size/hop); % Total time frames

CFs = [250 500 1000];
numcfs = 3;

nF =  30;
nFibers = 30;

% Output structure
output = struct();
output.CFs = CFs;
output.timeStamps = (0:total_frames-1)*hop*T;
output.left = struct();
output.left.mi = zeros(numcfs, total_frames);
output.left.ti = zeros(numcfs, total_frames);
output.left.pi = zeros(numcfs, total_frames);
output.right = struct();
output.right.mi = zeros(numcfs, total_frames);
output.right.ti = zeros(numcfs, total_frames);
output.right.pi = zeros(numcfs, total_frames);
output.binaural = struct();
output.binaural.an_delay = zeros(1, total_frames);
output.binaural.ha_delay = zeros(1, total_frames);
output.binaural.mi = zeros(numcfs, total_frames);
output.binaural.ti = zeros(numcfs, total_frames);
output.binaural.pi = zeros(numcfs, total_frames);

disp('Computing Mutual Information')

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
    output.binaural.an_delay(n) = delay_an*T;
    output.binaural.ha_delay(n) = delay_ha*T;

    an_b_spikes = ...
        an_l_spikes(:,init_frame+delay_an:end_frame+delay_an) + ...
        an_r_spikes(:,init_frame:end_frame);
    ha_b_spikes = ...
        ha_l_spikes(:,init_frame+delay_ha:end_frame+delay_ha) + ...
        ha_r_spikes(:,init_frame:end_frame);

    for cf=1:numcfs
        % Reduced representations via integration frames
        an_l = reduce(an_l_spikes(cf,init_frame:end_frame), i_frames);
        ha_l = reduce(ha_l_spikes(cf,init_frame:end_frame), i_frames);
        an_r = reduce(an_r_spikes(cf,init_frame:end_frame), i_frames);
        ha_r = reduce(ha_r_spikes(cf,init_frame:end_frame), i_frames);
        an_b = reduce(an_b_spikes(cf,:), i_frames);
        ha_b = reduce(ha_b_spikes(cf,:), i_frames);

        % Mutual information for every cf (left, rigth and binaural)
        output.left.mi(cf,n) = my_mutualinfo(an_l, ha_l, nFibers);
        output.right.mi(cf,n) = my_mutualinfo(an_r, ha_r, nFibers);
        output.binaural.mi(cf,n) = my_mutualinfo(an_b, ha_b,2*nFibers);

        % Transmited information of the anechoic signal
        output.left.ti(cf,n) = my_entropy(an_l, nFibers);
        output.right.ti(cf,n) = my_entropy(an_r, nFibers);
        output.binaural.ti(cf,n) = my_entropy(an_b, 2*nFibers);

        % Perceived information of the ha signa
        output.left.pi(cf,n) = my_entropy(ha_l, nFibers);
        output.right.pi(cf,n) = my_entropy(ha_r, nFibers);
        output.binaural.pi(cf,n) = my_entropy(ha_b, 2*nFibers);
    end    
end

left_mi = gpuArray(zeros(total_frames, numcfs));
left_ti = gpuArray(zeros(total_frames, numcfs));
left_pi = gpuArray(zeros(total_frames, numcfs));
right_mi = gpuArray(zeros(total_frames, numcfs));
right_ti = gpuArray(zeros(total_frames, numcfs));
right_pi = gpuArray(zeros(total_frames, numcfs));
binaural_mi = gpuArray(zeros(total_frames, numcfs));
binaural_ti = gpuArray(zeros(total_frames, numcfs));
binaural_pi = gpuArray(zeros(total_frames, numcfs));

an_delay = gpuArray(output.binaural.an_delay);
ha_delay = gpuArray(output.binaural.ha_delay);

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

function ent = my_entropy(a,nF)
    rho_s = sum(a,'all')/(numel(a)*nF);
    if rho_s == 0 || rho_s == 1
        ent = 0;
    else
        ent = - (rho_s*log2(rho_s) + (1-rho_s)*log2(1-rho_s));
    end
end

function mi = my_mutualinfo(x, y, nF)
    assert(length(x)==length(y))
    N = length(x);
    jpdf = zeros(2,2);
    for n=1:N
        o11 = min(x(n), y(n)); % Ocurrences of [1 1]
        o01 = max(0,y(n)-x(n)); % Ocurrences of [0 1]
        o10 = max(0,x(n)-y(n)); % Ocurrences of [1 0]
        o00 = nF - o11 - o01 - o10; % Ocurrences of [0 0]
        jpdf = jpdf + [o00, o01 ; ...
                       o10, o11];
    end
    jpdf = jpdf/(N*nF);
    ent_x = my_entropy(x, nF);
    ent_y = my_entropy(y, nF);
    jent_xy = - [jpdf(1,1)*log2(jpdf(1,1)), ...
                 jpdf(1,2)*log2(jpdf(1,2)), ...
                 jpdf(2,1)*log2(jpdf(2,1)), ...
                 jpdf(2,2)*log2(jpdf(2,2))];
    jent_xy(isnan(jent_xy)) = 0;
    mi = ent_x + ent_y - sum(jent_xy, 'all');
end

function reduced_spikes = reduce(spikes, frames)
    N = ceil(length(spikes)/frames);
    reduced_spikes = zeros(1, N);
    for n=0:N-1
        initf = n*frames + 1;
        endf = min(initf+frames-1, length(spikes));
        reduced_spikes(n+1) = sum(spikes(initf:endf));
    end
end