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

__global__ void SpikeGenerator(double *sptime,
                               double *spCount,
                               double *preRelease_initialGuessTimeBins,
                               double *unitRateInterval,
                               double *elapsed_time,
                               double *previous_release_times,
                               double *current_release_times,
                               double *oneSiteRedock,
                               double* Xsum,
                               double *synout,
                               double *randNums,
                               const double tdres,
                               const double t_rd_rest,
                               const double *t_rd_inits,
                               const double tau,
                               const double t_rd_jump,
                               const int nSites,
                               const double *tabss,
                               const double *trels,
                               const int nsamples,
                               const int MaxArraySizeSpikes,
                               const int MaxEventsSize,
                               const int totalANFs)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= totalANFs) return;
    unsigned int stride = nSites*index;
    unsigned int stride_synout = nsamples*index;
    unsigned int stride_sptime = MaxArraySizeSpikes*index;
    unsigned int stride_events = MaxEventsSize*index;

    double t_rd_init = t_rd_inits[index];
    double tabs = tabss[index];
    double trel = trels[index];
    
    int k, kInit;  /*the loop starts from kInit */
    
    unsigned int i, siteNo, nRand, j;
    double Tref, current_refractory_period, trel_k, tmp;
    int t_rd_decay, rd_first;
    
    double previous_redocking_period,  current_redocking_period;
    int oneSiteRedock_rounded, elapsed_time_rounded ;

    nRand = stride_events;
    /* Initial < redocking time associated to nSites release sites */
    for (i=stride; i<stride+nSites; i++)
    {
        oneSiteRedock[i]=-t_rd_init*log( randNums[nRand++] );
    }
    
    /* Initial  preRelease_initialGuessTimeBins  associated to nsites release sites */
    
    for (i=stride; i<stride+nSites; i++)
    {
        preRelease_initialGuessTimeBins[i]= __max(-nsamples,ceil ((nSites/__max(synout[stride_synout],0.1) + t_rd_init)*log( randNums[nRand++] ) / tdres));   
    }
    
    for (i = 0; i < nSites-1; i++)
    {
        // Last i elements are already in place 
        for (j = stride; j < stride+nSites-i-1; j++)
        {
            if (preRelease_initialGuessTimeBins[j] > preRelease_initialGuessTimeBins[j+1])
            {
                tmp = preRelease_initialGuessTimeBins[j];
                preRelease_initialGuessTimeBins[j] = preRelease_initialGuessTimeBins[j+1];
                preRelease_initialGuessTimeBins[j+1] = tmp;
            }
        }
    } 
    
    /* Consider the inital previous_release_times to be  the preReleaseTimeBinsSorted *tdres */
    for (i=stride; i<stride+nSites; i++)
    {
        previous_release_times[i] = ((double)preRelease_initialGuessTimeBins[i])*tdres;
    }
    
    /* The position of first spike, also where the process is started- continued from the past */
    kInit = (int) preRelease_initialGuessTimeBins[stride];
    
    
    /* Current refractory time */
    Tref = tabs - trel*log( randNums[nRand++] );
    
    /*initlal refractory regions */
    current_refractory_period = (double) kInit*tdres;
    
    spCount[index] = 0; /* total numebr of spikes fired */
    k = kInit;  /*the loop starts from kInit */
    
    /* set dynamic mean redocking time to initial mean redocking time  */
    previous_redocking_period = t_rd_init;
    current_redocking_period = previous_redocking_period;
    t_rd_decay = 1; /* Logical "true" as to whether to decay the value of current_redocking_period at the end of the time step */
    rd_first = 0; /* Logical "false" as to whether to a first redocking event has occurred */
    
    /* a loop to find the spike times for all the totalstim*nrep */
    while (k < nsamples){
        
        for (siteNo = stride; siteNo<stride+nSites; siteNo++)
        {
            
            if ( k > preRelease_initialGuessTimeBins[siteNo] )
            {
            
                /* redocking times do not necessarily occur exactly at time step value - calculate the
                 * number of integer steps for the elapsed time and redocking time */
                oneSiteRedock_rounded =  (int) floor(oneSiteRedock[siteNo]/tdres);
                elapsed_time_rounded =  (int) floor(elapsed_time[siteNo]/tdres);
                if ( oneSiteRedock_rounded == elapsed_time_rounded )
                {
                    /* Jump  trd by t_rd_jump if a redocking event has occurred   */
                    current_redocking_period  =   previous_redocking_period  + t_rd_jump;
                    previous_redocking_period =   current_redocking_period;
                    t_rd_decay = 0; /* Don't decay the value of current_redocking_period if a jump has occurred */
                    rd_first = 1; /* Flag for when a jump has first occurred */
                }
                
                /* to be sure that for each site , the code start from its
                 * associated  previus release time :*/
                elapsed_time[siteNo] = elapsed_time[siteNo] + tdres;
            };
            
            
            /*the elapsed time passes  the one time redock (the redocking is finished),
             * In this case the synaptic vesicle starts sensing the input
             * for each site integration starts after the redockinging is finished for the corresponding site)*/
            if ( elapsed_time[siteNo] >= oneSiteRedock [siteNo] )
            {
                Xsum[siteNo] = Xsum[siteNo] + synout[__max(0,k) + stride_synout] / nSites;
                
                /* There are  nSites integrals each vesicle senses 1/nosites of  the whole rate */
            }
            
            
            
            if  ( (Xsum[siteNo]  >=  unitRateInterval[siteNo]) &&  ( k >= preRelease_initialGuessTimeBins[siteNo] ) )
            {  /* An event- a release  happened for the siteNo*/
                
                oneSiteRedock[siteNo]  = -current_redocking_period*log( randNums[nRand++]);
                current_release_times[siteNo] = previous_release_times[siteNo]  + elapsed_time[siteNo];
                elapsed_time[siteNo] = 0;               
                
                if ( (current_release_times[siteNo] >= current_refractory_period) )
                {  /* A spike occured for the current event- release
                 * spike_times[(int)(current_release_times[siteNo]/tdres)-kInit+1 ] = 1;*/
                    
                    /*Register only non negative spike times */
                    if (current_release_times[siteNo] >= 0)
                    {
                        sptime[(unsigned int)spCount[index] + stride_sptime] = current_release_times[siteNo]; spCount[index] = spCount[index] + 1;
                    }
                    
                    trel_k = __min(trel*100/synout[__max(0,k) + stride_synout],trel);

                    Tref = tabs-trel_k*log( randNums[nRand++] );   /*Refractory periods */
                    
                    current_refractory_period = current_release_times[siteNo] + Tref;
                    
                }
                
                previous_release_times[siteNo] = current_release_times[siteNo];
                
                Xsum[siteNo] = 0;
                unitRateInterval[siteNo] = (int) (-log( randNums[nRand++] ) / tdres);
                
            };
            /* Error Catching */
            if ( (spCount[index]+1)>MaxArraySizeSpikes  || (nRand+1 )> MaxEventsSize )
            {     /* mexPrintf ("--------Array for spike times or random Buffer length not large enough, Rerunning the function.-----\n\n"); */
                spCount[index] = -1;
                k = nsamples;
                siteNo = nSites;
            }
            
        };
        
        /* Decay the adapative mean redocking time towards the resting value if no redocking events occurred in this time step */
        if ( (t_rd_decay==1) && (rd_first==1) )
        {
            current_redocking_period =   previous_redocking_period  - (tdres/tau)*( previous_redocking_period-t_rd_rest );
            previous_redocking_period =  current_redocking_period;
        }
        else
        {
            t_rd_decay = 1;
        }

        k = k+1;

    };
    
    return;  
}