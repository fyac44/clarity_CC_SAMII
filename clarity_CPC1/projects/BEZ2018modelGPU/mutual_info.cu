/* This is the BEZ2018 version of the code for auditory periphery model from the Carney, Bruce and Zilany labs.
 * 
 * This release implements the version of the model described in:
 *
 *   Bruce, I.C., Erfani, Y., and Zilany, M.S.A. (2018). "A Phenomenological
 *   model of the synapse between the inner hair cell and auditory nerve: 
 *   Implications of limited neurotransmitter release sites," to appear in
 *   Hearing Research. (Special Issue on "Computational Models in Hearing".)
 *
 * Please cite this paper if you publish any research
 * results obtained with this code or any modified versions of this code.
 *
 * See the file readme.txt for details of compiling and running the model.
 *
 * %%% Ian C. Bruce (ibruce@ieee.org), Yousof Erfani (erfani.yousof@gmail.com),
 *     Muhammad S. A. Zilany (msazilany@gmail.com) - December 2017 %%%
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAXSPIKES 1000000
#ifndef TWOPI
#define TWOPI 6.28318530717959
#endif

#ifndef __max
#define __max(a,b) (((a) > (b))? (a): (b))
#endif

#ifndef __min
#define __min(a,b) (((a) < (b))? (a): (b))
#endif

__global__ void mutualInfo( double *mi_L,
                            double *mi_R,
                            double *mi_B,
                            double *pi_L,
                            double *pi_R,
                            double *pi_B,
                            double *ti_L,
                            double *ti_R,
                            double *ti_B,
                            const double *spikesha_L,
                            const double *spikesha_R,
                            const double *spikesan_L,
                            const double *spikesan_R,
                            const double *ha_delays,
                            const double *an_delays,
                            const int nF,
                            const int N,
                            const int hop,
                            const int i_frames,
                            const int in_sample_limit,
                            const int numcfs,
                            const int totalsamps)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numcfs*totalsamps) return;
    
    unsigned int index_cf = floor((double)index/totalsamps); // Center frequency index from 0 to numcfs-1
    unsigned int index_frame = (int) fmod((double) index, (double) totalsamps); // Frame index from 0 to totalsamps-1
    unsigned int stride_output = index_cf*totalsamps; // Stride at the output for every cf
    unsigned int stride_input = index_cf*in_sample_limit; // Stride at the input for every cf
    
    // Delay applied to the current frame for binaural calculations
    int delay_an = an_delays[index_frame];
    int delay_ha = ha_delays[index_frame];

    // Frame indexes for the input matrix
    int init_frame = index_frame*hop + stride_input; // 4277050
    int end_frame = __min(init_frame + N, (index_cf + 1)*in_sample_limit); //min(4279050, 4278000)

    // Functions
    __device__ double entropy(const double *, int, int, int, int);
    __device__ double mutualinfo(const double *,const double *, int, int, int, int);
    __device__ double entropy_b(const double *,const double *, int, int, int, int, int);
    __device__ double mutualinfo_b(const double *, const double *, const double *, const double *, int, int, int, int, int, int);

    // Output arrays
    mi_L[index_frame + stride_output] = mutualinfo(spikesan_L, spikesha_L, init_frame, end_frame, nF, i_frames);
    mi_R[index_frame + stride_output] = mutualinfo(spikesan_R, spikesha_R, init_frame, end_frame, nF, i_frames);
    mi_B[index_frame + stride_output] = mutualinfo_b(spikesan_L, spikesan_R, spikesha_L, spikesha_R, delay_an, delay_ha, init_frame, end_frame, nF, i_frames);

    ti_L[index_frame + stride_output] = entropy(spikesan_L, init_frame, end_frame, nF, i_frames);
    ti_R[index_frame + stride_output] = entropy(spikesan_R, init_frame, end_frame, nF, i_frames);
    ti_B[index_frame + stride_output] = entropy_b(spikesan_L, spikesan_R, delay_an, init_frame, end_frame, nF, i_frames);

    pi_L[index_frame + stride_output] = entropy(spikesha_L, init_frame, end_frame, nF, i_frames);
    pi_R[index_frame + stride_output] = entropy(spikesha_R, init_frame, end_frame, nF, i_frames);
    pi_B[index_frame + stride_output] = entropy_b(spikesha_L, spikesha_R, delay_ha, init_frame, end_frame, nF, i_frames);
            
    return;  
}

__device__ double entropy(const double *x, int initf, int endf, int nF, int i_frames)
{
    int n_iwindows, count, i, size;
    double rho_s, ent;
    
    size = endf-initf;
    n_iwindows = (int) ceil((double)size/i_frames); // number of integration windows in this frame
            
    // count spikes in the frame
    count = 0;
    for (i=initf;i<endf;i++)
    {
        count = count + (int) x[i];
    }

    // calculate probability of a spike for every fiber an integration window
    rho_s = (double)count/((double) n_iwindows*nF);
    if (rho_s == 0 || rho_s == 1)
        ent = 0.0; // avoid log(0)
    else
        ent = - (rho_s*log2(rho_s) + (1-rho_s)*log2(1-rho_s)); // entropy
    return ent;
}


__device__ double entropy_b(const double *l, const double *r, int delay, int initf, int endf, int nF, int i_frames)
{
    int n_iwindows, count, i, size;
    double rho_s, ent;
    
    size = endf-initf;
    n_iwindows = (int) ceil((double)size/i_frames); // number of integration windows in this frame
            
    // count spikes in the frame
    count = 0;
    for (i=initf;i<endf;i++)
    {
        count = count + (int) (l[i+delay] + r[i]);
    }

    // calculate probability of a spike for every fiber an integration window
    rho_s = (double)count/((double) n_iwindows*2*nF);
    if (rho_s == 0 || rho_s == 1)
        ent = 0.0; // avoid log(0)
    else
        ent = - (rho_s*log2(rho_s) + (1-rho_s)*log2(1-rho_s)); // entropy
    return ent;
}

__device__ double mutualinfo(const double *x, const double *y, int initf, int endf, int nF, int i_frames)
{
    int n_iwindows, iw, i, countx, county, i_init, i_end, size;
    int o00, o01, o10, o11;
    double jpdf00, jpdf01, jpdf10, jpdf11, info00, info01, info10, info11;
    double ent_x, ent_y, jent_xy, mi;

    size = endf - initf;
    n_iwindows = (int) ceil((double)size/i_frames); // number of integration windows in this frame (48)
    jpdf00 = 0;
    jpdf01 = 0;
    jpdf10 = 0;
    jpdf11 = 0;
    
    // loop per integration window
    for (iw=0; iw<n_iwindows; iw++)
    {
        countx = 0;
        county = 0;
        i_init = iw*i_frames; // 940
        i_end = __min(size, i_init+i_frames); // min(950, 960)
        // count spikes for each integration window for both signals
        for (i=i_init; i<i_end; i++)
        {
            countx = countx + (int) x[i+initf];
            county = county + (int) y[i+initf];
        }

        // spikes ocurrencies
        o11 = __min(countx, county); // 11=fibers spiking in both signals
        o01 = __max(0, county-countx); //  01=fiber spiking only in y
        o10 = __max(0, countx-county); // 10=fibers spiking only in x
        o00 = nF - o01 - o10 - o11; // 00=fibers not spiking in any

        // joint probability distribution function (per integration window per fiber)
        jpdf00 = jpdf00 + (double)o00/(n_iwindows*nF);
        jpdf01 = jpdf01 + (double)o01/(n_iwindows*nF);
        jpdf10 = jpdf10 + (double)o10/(n_iwindows*nF);
        jpdf11 = jpdf11 + (double)o11/(n_iwindows*nF);
    }

    // Entropy of the input signal
    ent_x = entropy(x, initf, endf, nF, i_frames);
    ent_y = entropy(y, initf, endf, nF, i_frames);

    // Information given by every combination of spiking (avoiding log(0))
    if (jpdf00 == 0)
        info00 = 0.0;
    else
        info00 = jpdf00*log2(jpdf00);
        
    if (jpdf01 == 0)
        info01 = 0.0;
    else
        info01 = jpdf01*log2(jpdf01);
        
    if (jpdf10 == 0)
        info10 = 0.0;
    else
        info10 = jpdf10*log2(jpdf10);

    if (jpdf11 == 0)
        info11 = 0.0;
    else
        info11 = jpdf11*log2(jpdf11);
        
    // joint entropy as the summed information for every spiking case 
    jent_xy = - (info00 + info01 + info10 + info11);

    // mutual information as Hx + Hy - Hxy
    mi = ent_x + ent_y - jent_xy;
    return mi;
}

__device__ double mutualinfo_b(const double *xl, const double *xr, const double *yl, const double *yr, int delayx, int delayy, int initf, int endf, int nF, int i_frames)
{
    int n_iwindows, iw, i, countx, county, i_init, i_end, size;
    int o00, o01, o10, o11;
    double jpdf00, jpdf01, jpdf10, jpdf11, info00, info01, info10, info11;
    double ent_x, ent_y, jent_xy, mi;

    size = endf - initf;
    n_iwindows = (int) ceil((double)size/i_frames); // number of integration windows in this frame (48)
    jpdf00 = 0;
    jpdf01 = 0;
    jpdf10 = 0;
    jpdf11 = 0;
    
    // loop per integration window
    for (iw=0; iw<n_iwindows; iw++)
    {
        countx = 0;
        county = 0;
        i_init = iw*i_frames; // 940
        i_end = __min(size, i_init+i_frames); // min(950, 960)
        // count spikes for each integration window for both signals
        for (i=i_init; i<i_end; i++)
        {
            countx = countx + (int) (xl[i+initf+delayx] + xr[i+initf]);
            county = county + (int) (yl[i+initf+delayy] + yr[i+initf]);
        }

        // spikes ocurrencies
        o11 = __min(countx, county); // 11=fibers spiking in both signals
        o01 = __max(0, county-countx); //  01=fiber spiking only in y
        o10 = __max(0, countx-county); // 10=fibers spiking only in x
        o00 = 2*nF - o01 - o10 - o11; // 00=fibers not spiking in any

        // joint probability distribution function (per integration window per fiber)
        jpdf00 = jpdf00 + (double)o00/(n_iwindows*2*nF);
        jpdf01 = jpdf01 + (double)o01/(n_iwindows*2*nF);
        jpdf10 = jpdf10 + (double)o10/(n_iwindows*2*nF);
        jpdf11 = jpdf11 + (double)o11/(n_iwindows*2*nF);
    }

    // Entropy of the input signal
    ent_x = entropy_b(xl, xr, delayx, initf, endf, nF, i_frames);
    ent_y = entropy_b(yl, yr, delayy, initf, endf, nF, i_frames);

    // Information given by every combination of spiking (avoiding log(0))
    if (jpdf00 == 0)
        info00 = 0.0;
    else
        info00 = jpdf00*log2(jpdf00);
        
    if (jpdf01 == 0)
        info01 = 0.0;
    else
        info01 = jpdf01*log2(jpdf01);
        
    if (jpdf10 == 0)
        info10 = 0.0;
    else
        info10 = jpdf10*log2(jpdf10);

    if (jpdf11 == 0)
        info11 = 0.0;
    else
        info11 = jpdf11*log2(jpdf11);
        
    // joint entropy as the summed information for every spiking case 
    jent_xy = - (info00 + info01 + info10 + info11);

    // mutual information as Hx + Hy - Hxy
    mi = ent_x + ent_y - jent_xy;
    return mi;
}