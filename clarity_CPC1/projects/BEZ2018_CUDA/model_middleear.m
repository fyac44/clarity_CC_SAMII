% Middle ear filter implemented in the BEZ2018 IHC model.
%
% USAGE:
%     meout = model_middleear(pin,species,dt);
%
% INPUT:
%  - pin: Sound wave [Pa] sampled at the appropriate sampling rate 
%         (see instructions below)
%  - species: "1" for cat, "2" for human with BM tuning from Shera et al. 
%             (PNAS 2002), or "3" for human BM tuning from Glasberg & Moore
%             (Hear. Res. 1990)
%  - dt: Binsize [s], i.e., the reciprocal of the sampling rate
%        (see instructions below)
%
% OUTPUT:
% - meout: Sound wave entering the cochlea [Pa]
%
% NOTE ON SAMPLING RATE:
% Since version 2 of the BEZ2018 code, it is possible to run the model at a
% range of sampling rates between 100 kHz and 500 kHz.
% It is recommended to run the model at 100 kHz for CFs up to 20 kHz, and
% at 200 kHz for CFs> 20 kHz to 40 kHz.