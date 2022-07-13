function information = binauralMI(spt_S, spt_R, numcfs, nFibers)

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

% Output structure
information = struct();
information.CFs = CFs;
information.timeStamps = (0:total_frames-1)*hop*T;
information.left = struct();
information.left.mi = zeros(numcfs, total_frames);
information.left.Si = zeros(numcfs, total_frames);
information.left.Ri = zeros(numcfs, total_frames);
information.right = struct();
information.right.mi = zeros(numcfs, total_frames);
information.right.Si = zeros(numcfs, total_frames);
information.right.Ri = zeros(numcfs, total_frames);
information.binaural = struct();
information.binaural.S_delay = zeros(1, total_frames);
information.binaural.R_delay = zeros(1, total_frames);
information.binaural.mi = zeros(numcfs, total_frames);
information.binaural.Si = zeros(numcfs, total_frames);
information.binaural.Ri = zeros(numcfs, total_frames);

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
    information.binaural.S_delay(n) = delay_S*samplesT;
    information.binaural.R_delay(n) = delay_R*samplesT;

    % Binaural representation in this specific analysis window
    spt_S.binaural = ...
        spt_S.left(:,init_frame+delay_an:end_frame+delay_S) + ...
        spt_S.right(:,init_frame:end_frame);
    spt_R.binaural = ...
        spt_R.left(:,init_frame+delay_ha:end_frame+delay_R) + ...
        spt_R.right(:,init_frame:end_frame);

    for cf=1:numcfs
        % Time integration
        S_l = reduce(spt_S.left(cf,init_frame:end_frame), i_frames);
        R_l = reduce(spt_R.left(cf,init_frame:end_frame), i_frames);

        S_r = reduce(spt_S.right(cf,init_frame:end_frame), i_frames);
        R_r = reduce(spt_R.right(cf,init_frame:end_frame), i_frames);

        S_b = reduce(spt_S.binaural(cf,:), i_frames);
        R_b = reduce(spt_R.binaural(cf,:), i_frames);

        % Mutual information for every cf (left, rigth and binaural)
        information.left.mi(cf,n) = my_mutualinfo(S_l, R_l, nFibers);
        information.right.mi(cf,n) = my_mutualinfo(S_r, R_r, nFibers);
        information.binaural.mi(cf,n) = my_mutualinfo(S_b, R_b,2*nFibers);

        % Transmited information of the anechoic signal
        information.left.Si(cf,n) = my_entropy(S_l, nFibers);
        information.right.Si(cf,n) = my_entropy(S_r, nFibers);
        information.binaural.Si(cf,n) = my_entropy(S_b, 2*nFibers);

        % Perceived information of the ha signa
        information.left.Ri(cf,n) = my_entropy(R_l, nFibers);
        information.right.Ri(cf,n) = my_entropy(R_r, nFibers);
        information.binaural.Ri(cf,n) = my_entropy(R_b, 2*nFibers);
    end

end

end

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