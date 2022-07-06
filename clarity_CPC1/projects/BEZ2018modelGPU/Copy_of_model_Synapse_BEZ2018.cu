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

#include "complex.cu"

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

__constant__ double alpha1 = 1.5e-6*100e3;
__constant__ double beta1 = 5e-4;
__constant__ double alpha2 = 1e-2*100e3;
__constant__ double beta2 = 1e-1;

__constant__ double t_rd_rest = 14.0e-3;
__constant__ double t_rd_jump = 0.4e-3;

__constant__ double tau =  60.0e-3;

/* --------------------------------------------------------------------------------------------*/
__global__ void Synapse(double *tSpikes,
                        double *spCount,
                        double *ihcout,
                        double *sout1,
                        double *sout2,
                        const double *cfs,
                        const double *randNums,
                        const double *nsRandNums,
                        const double *randResamps,
                        const double *randEvents,
                        const double *sponts,
                        const double *tabss,
                        const double *trels,
                        const double *delaypoints,
                        const int nsamples,
                        const int max_ndownsampled,
                        const int resamp,
                        const int implnt,
                        const double tdres,
                        const double sampFreq,
                        const int nSites,
                        const int maxRandNums,
                        const int maxSpikes,
                        const int maxEvents,
                        const int ANFperCF,
                        const int totalANFs)
{
    unsigned int anf_index = blockIdx.x * blockDim.x + threadIdx.x; // ANF index within the CF
    if (anf_index >= totalANFs) return;
    unsigned int cf_index = floor((double)anf_index/ANFperCF);
    unsigned int stride_input = nsamples*cf_index; // ihcout
    unsigned int stride_ndownsampled = max_ndownsampled*anf_index; // sout1, sout2
    unsigned int stride_randnums = maxRandNums*anf_index; // randNums
    unsigned int stride_spikes = maxSpikes*anf_index; // tSpikes
    unsigned int stride_events = maxEvents*anf_index; // randEvents

    double spont = sponts[anf_index];
    double tabs = tabss[anf_index];
    double trel = trels[anf_index];

    int delaypoint = (int) delaypoints[cf_index];
    int nRandNums = (int) nsRandNums[cf_index];
    int randResamp = (int) randResamps[cf_index];
    double cf = cfs[cf_index];

    double I1, I2, binwidth, incr;
    int    k,j,indx,i,b,resamppl,rindex, mult, nsamplesdelaypl, ndownsampled;

    double cf_factor,cfslope,cfsat,cfconst,multFac, sigma, randNum, mappingOut, powerLawIn, synSampOut0, synSampOut1;

    double m10, m11, m12, m20, m21, m22, m30, m31, m32, m40, m41, m42, m50, m51, m52;
    double n10, n11, n12, n20, n21, n22, n30, n31, n32;

    double t_rd_init, Tref, current_refractory_period;
    double previous_redocking_period, current_redocking_period, trel_k, synout;
    int kInit, s, t_rd_decay, rd_first, siteNo, oneSiteRedock_rounded, elapsed_time_rounded, tmp, aux, count;
    unsigned int eventIndex;

    double *oneSiteRedock = new double[nSites];
    double *elapsed_time = new double[nSites];
    double *previous_release_times = new double[nSites];
    double *current_release_times = new double[nSites];
    double *Xsum = new double[nSites];
    
    int *unitRateInterval = new int[nSites];
    int *preRelease_initialGuessTimeBins = new int[nSites];

    binwidth = 1/sampFreq;
    I1 = 0;
    I2 = 0;

    /*----------------------------------------------------------*/
    /*----- Mapping Function from IHCOUT to input to the PLA ----------------------*/
    /*----------------------------------------------------------*/
    cfslope = powf(spont,0.19)*powf(10.0,-0.87);
    cfconst = 0.1*powf(log10(spont),2)+0.56*log10(spont)-0.84;
    cfsat = powf(10.0,(cfslope*8965.5/1e3 + cfconst));
    cf_factor = __min(cfsat,powf(10.0,cfslope*cf/1e3 + cfconst))*2.0;

    multFac = __max(2.95*__max(1.0,1.5-spont/100),4.3-0.2*cf/1e3);

    /*----------------------------------------------------------*/
    /*----- Running Power-law Adaptation -----------------------*/
    /*----------------------------------------------------------*/
    nsamplesdelaypl = nsamples+3*delaypoint;
    ndownsampled = ceil((nsamples+2*delaypoint)*sampFreq*tdres);;
    resamppl = __min(resamp, floor((double)(nsamplesdelaypl-1)/(ndownsampled-1)));
    mappingOut = powf(10.0,(0.9*log10(fabs(ihcout[stride_input])*cf_factor))+ multFac);
    if (ihcout[stride_input]<0) mappingOut = - mappingOut;
    if (spont>=20) sigma=spont/2;
    else if (spont>=0.2) sigma=10.0;
    else sigma = 1.0;
    powerLawIn = 0.0;
    mult = 1.0;
    for (k=0; k<ndownsampled; k++)
    {
        if ((k*resamppl>delaypoint) && (k*resamppl<nsamples+delaypoint))
        {
            indx = k*resamppl-delaypoint;
            mappingOut = powf(10.0,(0.9*log10(fabs(ihcout[indx+stride_input])*cf_factor))+ multFac);
            if (ihcout[indx+stride_input]<0) mappingOut = - mappingOut;
        }
        else if (k*resamppl>=nsamples+delaypoint)
        {
            mult = k*resamppl - __max((k-1)*resamppl,nsamples+delaypoint-1);
            mappingOut = powerLawIn;
        }
        powerLawIn = mappingOut+3.0*spont*mult;

        rindex = floor((double)k/randResamp);
        if (rindex>nRandNums-1) rindex = nRandNums -1;
        incr = (randNums[rindex+1+stride_randnums]-randNums[rindex+stride_randnums]) / (double)randResamp;
        randNum = randNums[rindex+stride_randnums] + incr*fmod((double) k, (double) randResamp);
        sout1[k + stride_ndownsampled]  = __max( 0, powerLawIn + sigma*randNum - alpha1*I1);
        sout2[k + stride_ndownsampled]  = __max( 0, powerLawIn - alpha2*I2);
        
        if (implnt==1)    /* ACTUAL Implementation */
        {
            I1 = 0; I2 = 0;
            for (j=0; j<k+1; ++j)
            {
                I1 += (sout1[j + stride_ndownsampled])*binwidth/((k-j)*binwidth + beta1);
                I2 += (sout2[j + stride_ndownsampled])*binwidth/((k-j)*binwidth + beta2);
            }
        } /* end of actual */
        
        if (implnt==0)    /* APPROXIMATE Implementation */
        {
            if (k==0)
            {
                n10 = 1.0e-3*sout2[stride_ndownsampled];
                n20 = n10; n30 = n20;
                n11 = n10; n21 = n20; n31 = n30;
            }
            else if (k==1)
            {
                n10 = 1.992127932802320*n11 + 1.0e-3*(sout2[k+ stride_ndownsampled] - 0.994466986569624*sout2[k-1+ stride_ndownsampled]);
                n20 = 1.999195329360981*n21 + n10 - 1.997855276593802*n11;
                n30 =-0.798261718183851*n31 + n20 + 0.798261718184977*n21;
                n12 = n11; n22 = n21; n32 = n31;
                n11 = n10; n21 = n20; n31 = n30;
            }
            else
            {
                n10 = 1.992127932802320*n11 - 0.992140616993846*n12 + 1.0e-3*(sout2[k+ stride_ndownsampled] - 0.994466986569624*sout2[k-1+ stride_ndownsampled] + 0.000000000002347*sout2[k-2+ stride_ndownsampled]);
                n20 = 1.999195329360981*n21 - 0.999195402928777*n22 + n10 - 1.997855276593802*n11 + 0.997855827934345*n12;
                n30 =-0.798261718183851*n31 - 0.199131619873480*n32 + n20 + 0.798261718184977*n21 + 0.199131619874064*n22;
                n12 = n11; n22 = n21; n32 = n31;
                n11 = n10; n21 = n20; n31 = n30;
            }
            I2 = n30;
            
            if (k==0)
            {
                m10 = 0.2*sout1[stride_ndownsampled];
                m20 = m10;	m30 = m20;
                m40 = m30;	m50 = m40;
                m11 = m10; m21 = m20; m31 = m30; m41 = m40; m51 = m50;
            }
            else if (k==1)
            {
                m10 = 0.491115852967412*m11 + 0.2*(sout1[k + stride_ndownsampled] - 0.173492003319319*sout1[k-1+ stride_ndownsampled]);
                m20 = 1.084520302502860*m21 + m10 - 0.803462163297112*m11;
                m30 = 1.588427084535629*m31 + m20 - 1.416084732997016*m21;
                m40 = 1.886287488516458*m41 + m30 - 1.830362725074550*m31;
                m50 = 1.989549282714008*m51 + m40 - 1.983165053215032*m41;
                m12 = m11; m22 = m21; m32 = m31; m42 = m41; m52 = m51;
                m11 = m10; m21 = m20; m31 = m30; m41 = m40; m51 = m50;
            }
            else
            {
                m10 = 0.491115852967412*m11 - 0.055050209956838*m12 + 0.2*(sout1[k + stride_ndownsampled] - 0.173492003319319*sout1[k-1 + stride_ndownsampled] + 0.000000172983796*sout1[k-2 + stride_ndownsampled]);
                m20 = 1.084520302502860*m21 - 0.288760329320566*m22 + m10 - 0.803462163297112*m11 + 0.154962026341513*m12;
                m30 = 1.588427084535629*m31 - 0.628138993662508*m32 + m20 - 1.416084732997016*m21 + 0.496615555008723*m22;
                m40 = 1.886287488516458*m41 - 0.888972875389923*m42 + m30 - 1.830362725074550*m31 + 0.836399964176882*m32;
                m50 = 1.989549282714008*m51 - 0.989558985673023*m52 + m40 - 1.983165053215032*m41 + 0.983193027347456*m42;
                m12 = m11; m22 = m21; m32 = m31; m42 = m41; m52 = m51;
                m11 = m10; m21 = m20; m31 = m30; m41 = m40; m51 = m50;
            }
            I1 = m50;
        } /* end of approximate implementation */
        
        synSampOut1 = sout1[k + stride_ndownsampled] + sout2[k + stride_ndownsampled];
        if (k==0)
        {
            /* Pass the output of Synapse model through the Spike Generator */
            t_rd_init = t_rd_rest+0.02e-3*spont-t_rd_jump;
        
            eventIndex = stride_events;

            /* Initial < redocking time associated to nSites release sites */
            for (i=0; i<nSites; i++)
            {
                oneSiteRedock[i]=-t_rd_init*log(randEvents[eventIndex++]);
                elapsed_time[i] = 0;
                current_release_times[i] = 0;
                Xsum[i] = 0;
                unitRateInterval[i] = 0;
            }
        
            /* Initial  preRelease_initialGuessTimeBins  associated to nsites release sites */
            for (i=0; i<nSites; i++)
            {
                aux = (int) ceil(((double)nSites/__max(synSampOut1,0.1) + t_rd_init)*log(randEvents[eventIndex++] ) / tdres);
                preRelease_initialGuessTimeBins[i] = __max(-nsamples, aux);
            }
        
            for (i = 0; i < nSites-1; i++)
            {
                // Last i elements are already in place 
                for (j = 0; j < nSites-i-1; j++)
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
            for (i=0; i<nSites; i++)
            {
                previous_release_times[i] = ((double)preRelease_initialGuessTimeBins[i])*tdres;
            }
        
            kInit = preRelease_initialGuessTimeBins[0];
        
            /* Current refractory time */
            Tref = tabs - trel*log( randEvents[eventIndex++] );
            
            /*initlal refractory regions */
            current_refractory_period = (double) kInit*tdres;
            
            spCount[anf_index] = 0; /* total numebr of spikes fired */
            s = kInit;  /*the loop starts from kInit */
            
            /* set dynamic mean redocking time to initial mean redocking time  */
            previous_redocking_period = t_rd_init;
            current_redocking_period = previous_redocking_period;
            t_rd_decay = 1; /* Logical "true" as to whether to decay the value of current_redocking_period at the end of the time step */
            rd_first = 0; /* Logical "false" as to whether to a first redocking event has occurred */

            /* a loop to find the spike times for all the totalstim*nrep */
            while (s < 0){
                
                for (siteNo = 0; siteNo<nSites; siteNo++)
                {
                    
                    if ( s > preRelease_initialGuessTimeBins[siteNo] )
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
                        Xsum[siteNo] = Xsum[siteNo] + synSampOut1 / nSites;
                        
                        /* There are  nSites integrals each vesicle senses 1/nosites of  the whole rate */
                    }
                    
                                        
                    if  ( (Xsum[siteNo]  >= (double) unitRateInterval[siteNo]) &&  ( s >= preRelease_initialGuessTimeBins[siteNo] ) )
                    {  /* An event- a release  happened for the siteNo*/
                        
                        oneSiteRedock[siteNo]  = -current_redocking_period*log(randEvents[eventIndex++]);
                        current_release_times[siteNo] = previous_release_times[siteNo]  + elapsed_time[siteNo];
                        elapsed_time[siteNo] = 0;               
                        
                        if ( (current_release_times[siteNo] >= current_refractory_period) )
                        {  /* A spike occured for the current event- release
                         * spike_times[(int)(current_release_times[siteNo]/tdres)-kInit+1 ] = 1;*/
                            
                            /*Register only non negative spike times */
                            if (current_release_times[siteNo] >= 0)
                            {
                                count= (int)spCount[anf_index];
                                tSpikes[count + stride_spikes] = current_release_times[siteNo]; spCount[anf_index] = spCount[anf_index] + 1;
                            }
                            
                            trel_k = __min(trel*100/synSampOut1,trel);
        
                            Tref = tabs-trel_k*log( randEvents[eventIndex++] );   /*Refractory periods */
                            
                            current_refractory_period = current_release_times[siteNo] + Tref;
                            
                        }
                        
                        previous_release_times[siteNo] = current_release_times[siteNo];
                        
                        Xsum[siteNo] = 0;
                        unitRateInterval[siteNo] = (int) (-log(randEvents[eventIndex++]) / tdres);
                        
                    };
                    /* Error Catching */
                    if ( ((int)spCount[anf_index]+1)>maxSpikes  || (eventIndex+1-stride_events)>maxEvents  )
                    {     /* mexPrintf ("--------Array for spike times or random Buffer length not large enough, Rerunning the function.-----\n\n"); */
                        spCount[anf_index] = -1;
                        k = ndownsampled;
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
                
                s = s+1;
            };






        }
        else
        {
            incr = (synSampOut1-synSampOut0)/(double)resamp;
            for(b=0; b<resamp; ++b)
            {
                s = (k-1)*resamp+b - delaypoint;
                
                if ((s >= 0) && (s < nsamples))
                {
                    synout = synSampOut0 + b*incr;
                        
                    for (siteNo = 0; siteNo<nSites; siteNo++)
                    {
                        
                        if ( s > preRelease_initialGuessTimeBins[siteNo] )
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
                            Xsum[siteNo] = Xsum[siteNo] + synout/ nSites;
                            
                            /* There are  nSites integrals each vesicle senses 1/nosites of  the whole rate */
                        }
                        
                        
                        
                        if  ( (Xsum[siteNo]  >= (double) unitRateInterval[siteNo]) &&  ( s >= preRelease_initialGuessTimeBins[siteNo] ) )
                        {  /* An event- a release  happened for the siteNo*/
                            
                            oneSiteRedock[siteNo]  = -current_redocking_period*log(randEvents[eventIndex++]);
                            current_release_times[siteNo] = previous_release_times[siteNo]  + elapsed_time[siteNo];
                            elapsed_time[siteNo] = 0;               
                            
                            if ( (current_release_times[siteNo] >= current_refractory_period) )
                            {  /* A spike occured for the current event- release
                             * spike_times[(int)(current_release_times[siteNo]/tdres)-kInit+1 ] = 1;*/
                                
                                /*Register only non negative spike times */
                                if (current_release_times[siteNo] >= 0)
                                {
                                    count= (int)spCount[anf_index];
                                    tSpikes[count + stride_spikes] = current_release_times[siteNo]; spCount[anf_index] = spCount[anf_index] + 1;
                                }
                                
                                trel_k = __min(trel*100/synout,trel);
            
                                Tref = tabs-trel_k*log( randEvents[eventIndex++] );   /*Refractory periods */
                                
                                current_refractory_period = current_release_times[siteNo] + Tref;
                                
                            }
                            
                            previous_release_times[siteNo] = current_release_times[siteNo];
                            
                            Xsum[siteNo] = 0;
                            unitRateInterval[siteNo] = (int) (-log(randEvents[eventIndex++]) / tdres);
                            
                        };
                        /* Error Catching */
                        if ( ((int)spCount[anf_index]+1)>maxSpikes  || (eventIndex+1-stride_events)>maxEvents  )
                        {     /* mexPrintf ("--------Array for spike times or random Buffer length not large enough, Rerunning the function.-----\n\n"); */
                            spCount[anf_index] = -1;
                            k = ndownsampled;
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
                        






                }
            }
        }
        synSampOut0 = synSampOut1;
    }   /* end of all samples */
            
    delete[] oneSiteRedock;
    delete[] preRelease_initialGuessTimeBins;
    delete[] elapsed_time;
    delete[] previous_release_times;
    delete[] current_release_times;
    delete[] Xsum;
    delete[] unitRateInterval;
}
/* ------------------------------------------------------------------------------------ */


