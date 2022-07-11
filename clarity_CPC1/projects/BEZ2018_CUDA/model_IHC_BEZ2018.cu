/* This is a modified version of the IHC Model originally found in the 
 * BEZ2018 model of the auditory periphery model from the Carney, Bruce 
 * and Zilany labs.
 * 
 * This release implements the version of the model described in:
 *
 *  Bruce, I.C., Erfani, Y., and Zilany, M.S.A. (2018). "A Phenomenological
 *  model of the synapse between the inner hair cell and auditory nerve: 
 *  Implications of limited neurotransmitter release sites," to appear in
 *  Hearing Research. (Special Issue on "Computational Models in Hearing".)
 *
 * This code was modified to be used in this work:
 *  
 *  Alvarez and Nogueira (2022). "Predicting Speech Intelligibility using
 *  the Spike Activity Mutual Information Index". INTERSPEECH 2022.
 *
 * Please cite these papers if you publish any research
 * results obtained with this code or any modified versions of this code.
 *
 * See the file readme.txt for details of compiling and running the model.
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
        
__global__ void IHCAN(double *ihcout,
                      double *ihcouttmp,
                      double *tmpgain,
                      const double *meout, 
                      const double *cfs,
                      const double tdres,
                      const int totalstim,
                      const double *cohcs,
                      const double *cihcs,        
                      const int species,
                      const int numcfs)
{    
    /*parallel variables*/
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numcfs) return;
    double cf = cfs[index];
    double cohc = cohcs[index];
    double cihc = cihcs[index];

	/*variables for the signal-path, control-path and onward */
	int    grd;

    double bmplace,centerfreq,gain,taubm,ratiowb,bmTaubm,fcohc,TauWBMax,TauWBMin,tauwb;
    double Taumin,Taumax,bmTaumin,bmTaumax,ratiobm,lasttmpgain,wbgain,ohcasym,ihcasym,delay;
    double c1filterouttmp,c2filterouttmp,c1vihctmp,c2vihctmp;
	int    i,n,delaypoint,grdelay,bmorder,wborder;
	double wbout1,wbout,ohcnonlinout,ohcout,tmptauc1,tauc1,rsigma,wb_gain;

    /* Chirp, Lowpass and wbGammaTone Filters */
    double C1gain_norm, C1initphase, C2gain_norm, C2initphase, wbphase; 
    double C1input[11][3], C1output[11][3], C2input[11][3], C2output[11][3], ohc[4], ohcl[4], ihc[8], ihcl[8];
    COMPLEX wbgtf[4], wbgtfl[4];
    
            
    /* Declarations of the functions used in the program */
	__device__ double C1ChirpFilt(double, double,double, int, double, double, double &, double &, double C1input[11][3], double C1output[11][3]);
	__device__ double C2ChirpFilt(double, double,double, int, double, double, double &, double &, double C2input[11][3], double C2output[11][3]);
    __device__ double WbGammaTone(double, double, double, int, double, double, int, double &, COMPLEX wbgtf[4], COMPLEX wbgtfl[4]);

    __device__ double Get_tauwb(double, int, int, double &, double &);
	__device__ double Get_taubm(double, int, double, double &, double &, double &);
    __device__ double gain_groupdelay(double, double, double, double, int &);
    __device__ double delay_cat(double cf);
    __device__ double delay_human(double cf);

    __device__ double OhcLowPass(double, double, double, int, double, int, double ohc[4], double ohcl[4]);
    __device__ double IhcLowPass(double, double, double, int, double, int, double ihc[8], double ihcl[8]);
	__device__ double Boltzman(double, double, double, double, double);
    __device__ double NLafterohc(double, double, double, double);
	__device__ double ControlSignal(double, double, double, double, double);

    __device__ double NLogarithm(double, double, double, double);

	/** Calculate the center frequency for the control-path wideband filter
	    from the location on basilar membrane, based on Greenwood (JASA 1990) */

	if (species==1) /* for cat */
    {
        /* Cat frequency shift corresponding to 1.2 mm */
        bmplace = 11.9 * log10(0.80 + cf / 456.0); /* Calculate the location on basilar membrane from CF */
        centerfreq = 456.0*(powf(10.0,(bmplace+1.2)/11.9)-0.80); /* shift the center freq */
    }

	if (species>1) /* for human */
    {
        /* Human frequency shift corresponding to 1.2 mm */
        bmplace = (35/2.1) * log10(1.0 + cf / 165.4); /* Calculate the location on basilar membrane from CF */
        centerfreq = 165.4*(powf(10.0,(bmplace+1.2)/(35/2.1))-1.0); /* shift the center freq */
    }

	/*==================================================================*/
	/*====== Parameters for the gain ===========*/
    
	if(species==1) gain = 52.0/2.0*(tanh(2.2*log10(cf/0.6e3)+0.15)+1.0); /* for cat */
    if(species>1) gain = 52.0/2.0*(tanh(2.2*log10(cf/0.6e3)+0.15)+1.0); /* for human */
    /*gain = 52/2*(tanh(2.2*log10(cf/1e3)+0.15)+1);*/
    if(gain>60.0) gain = 60.0;  
    if(gain<15.0) gain = 15.0;
    
	/*====== Parameters for the control-path wideband filter =======*/
	bmorder = 3;
	Get_tauwb(cf,species,bmorder,Taumax,Taumin);
	taubm   = cohc*(Taumax-Taumin)+Taumin;
	ratiowb = Taumin/Taumax;
	/*====== Parameters for the signal-path C1 filter ======*/
	Get_taubm(cf,species,Taumax,bmTaumax,bmTaumin,ratiobm);
	bmTaubm  = cohc*(bmTaumax-bmTaumin)+bmTaumin;
	fcohc    = bmTaumax/bmTaubm;
    /*====== Parameters for the control-path wideband filter =======*/
	wborder  = 3;
    TauWBMax = Taumin+0.2*(Taumax-Taumin);
	TauWBMin = TauWBMax/Taumax*Taumin;
    tauwb    = TauWBMax+(bmTaubm-bmTaumax)*(TauWBMax-TauWBMin)/(bmTaumax-bmTaumin);
	
	wbgain = gain_groupdelay(tdres,centerfreq,cf,tauwb,grdelay);
	tmpgain[index*totalstim]   = wbgain; 
	lasttmpgain  = wbgain;
  	/*===============================================================*/
    /* Nonlinear asymmetry of OHC function and IHC C1 transduction function*/
	ohcasym  = 7.0;    
	ihcasym  = 3.0;
  	/*===============================================================*/
    /*===============================================================*/
    /* Prewarping and related constants for the middle ear */

    for (n=0;n<totalstim;n++) /* Start of the loop */
    {    
        /* Control-path filter */

        wbout1 = WbGammaTone(meout[n],tdres,centerfreq,n,tauwb,wbgain,wborder,wbphase,wbgtf,wbgtfl);
        wbout  = powf((tauwb/TauWBMax),wborder)*wbout1*10e3*__max(1,cf/5e3);

        ohcnonlinout = Boltzman(wbout,ohcasym,12.0,5.0,5.0); /* pass the control signal through OHC Nonlinear Function */
        ohcout = OhcLowPass(ohcnonlinout,tdres,600,n,1.0,2,ohc,ohcl);/* lowpass filtering after the OHC nonlinearity */

        tmptauc1 = NLafterohc(ohcout,bmTaumin,bmTaumax,ohcasym); /* nonlinear function after OHC low-pass filter */
        tauc1    = cohc*(tmptauc1-bmTaumin)+bmTaumin;  /* time -constant for the signal-path C1 filter */
        rsigma   = 1/tauc1-1/bmTaumax; /* shift of the location of poles of the C1 filter from the initial positions */

        //if (1/tauc1<0.0) mexErrMsgTxt("The poles are in the right-half plane; system is unstable.\n");

        tauwb = TauWBMax+(tauc1-bmTaumax)*(TauWBMax-TauWBMin)/(bmTaumax-bmTaumin);

        wb_gain = gain_groupdelay(tdres,centerfreq,cf,tauwb,grdelay);

        grd = grdelay; 

        if ((grd+n)<totalstim)
             tmpgain[grd+n+index*totalstim] = wb_gain;

        if (tmpgain[n+index*totalstim] == 0)
            tmpgain[n+index*totalstim] = lasttmpgain;	

        wbgain      = tmpgain[n+index*totalstim];
        lasttmpgain = wbgain;

        /*====== Signal-path C1 filter ======*/

         c1filterouttmp = C1ChirpFilt(meout[n], tdres, cf, n, bmTaumax, rsigma, C1gain_norm, C1initphase, C1input, C1output); /* C1 filter output */


        /*====== Parallel-path C2 filter ======*/

         c2filterouttmp  = C2ChirpFilt(meout[n], tdres, cf, n, bmTaumax, 1/ratiobm, C2gain_norm, C2initphase, C2input, C2output); /* parallel-filter output*/

        /*=== Run the inner hair cell (IHC) section: NL function and then lowpass filtering ===*/

        c1vihctmp  = NLogarithm(cihc*c1filterouttmp,0.1,ihcasym,cf);

        c2vihctmp = -NLogarithm(c2filterouttmp*fabs(c2filterouttmp)*cf/10*cf/2e3,0.2,1.0,cf); /* C2 transduction output */

        ihcouttmp[n+index*totalstim] = IhcLowPass(c1vihctmp+c2vihctmp,tdres,3000,n,1.0,7,ihc,ihcl);
    };  /* End of the loop */
     
   	/* Adjust total path delay to IHC output signal */
    if (species==1)
        delay      = delay_cat(cf);
    if (species>1)
    {/*    delay      = delay_human(cf); */
        delay      = delay_cat(cf); /* signal delay changed back to cat function for version 5.2 */
    };
    delaypoint =__max(0,(int) ceil(delay/tdres));
            
    for(i=delaypoint;i<totalstim;i++)
	{        
		ihcout[i+index*totalstim] = ihcouttmp[i - delaypoint + index*totalstim];       
  	};   

} /* End of the SingleAN function */
/* -------------------------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------------------------- */
/** Get TauMax, TauMin for the tuning filter. The TauMax is determined by the bandwidth/Q10
    of the tuning filter at low level. The TauMin is determined by the gain change between high
    and low level */

__device__ double Get_tauwb(double cf, int species, int order, double &taumax, double &taumin)
{
  double Q10,bw,gain,ratio;
    
  if(species==1) gain = 52.0/2.0*(tanh(2.2*log10(cf/0.6e3)+0.15)+1.0); /* for cat */
  if(species>1) gain = 52.0/2.0*(tanh(2.2*log10(cf/0.6e3)+0.15)+1.0); /* for human */
  /*gain = 52/2*(tanh(2.2*log10(cf/1e3)+0.15)+1);*/ /* older values */

  if(gain>60.0) gain = 60.0;  
  if(gain<15.0) gain = 15.0;
   
  ratio = powf(10.0,(-gain/(20.0*order)));       /* ratio of TauMin/TauMax according to the gain, order */
  if (species==1) /* cat Q10 values */
  {
    Q10 = powf(10.0,0.4708*log10(cf/1e3)+0.4664);
  }
  if (species==2) /* human Q10 values from Shera et al. (PNAS 2002) */
  {
    Q10 = powf((cf/1000),0.3)*12.7*0.505+0.2085;
  }
  if (species==3) /* human Q10 values from Glasberg & Moore (Hear. Res. 1990) */
  {
    Q10 = cf/24.7/(4.37*(cf/1000)+1)*0.505+0.2085;
  }
  bw     = cf/Q10;
  taumax = 2.0/(TWOPI*bw);
   
  taumin   = taumax*ratio;
  
  return 0;
}
/* -------------------------------------------------------------------------------------------- */
__device__ double Get_taubm(double cf, int species, double taumax,double &bmTaumax,double &bmTaumin, double &ratio)
{
  double gain,factor,bwfactor;
    
  if(species==1) gain = 52.0/2.0*(tanh(2.2*log10(cf/0.6e3)+0.15)+1.0); /* for cat */
  if(species>1) gain = 52.0/2.0*(tanh(2.2*log10(cf/0.6e3)+0.15)+1.0); /* for human */
  /*gain = 52/2*(tanh(2.2*log10(cf/1e3)+0.15)+1);*/ /* older values */

 
  if(gain>60.0) gain = 60.0;  
  if(gain<15.0) gain = 15.0;

  bwfactor = 0.7;
  factor   = 2.5;

  ratio  = powf(10.0,(-gain/(20.0*factor))); 

  bmTaumax = taumax/bwfactor;
  bmTaumin = bmTaumax*ratio;     
  return 0;
}
/* -------------------------------------------------------------------------------------------- */
/** Pass the signal through the signal-path C1 Tenth Order Nonlinear Chirp-Gammatone Filter */

__device__ double C1ChirpFilt(double x, double tdres,double cf, int n, double taumax, double rsigma, double &C1gain_norm, double &C1initphase, double C1input[11][3], double C1output[11][3])
{
    double ipw, ipb, rpa, pzero, rzero;
	double sigma0,fs_bilinear,CF,norm_gain,phase,c1filterout;
	int i,r,order_of_pole,half_order_pole,order_of_zero;
	double temp, dy, preal, pimg;

	COMPLEX p[10]; 
	
	/* Defining initial locations of the poles and zeros */
	/*======== setup the locations of poles and zeros =======*/
	  sigma0 = 1/taumax;
	  ipw    = 1.01*cf*TWOPI-50;
	  ipb    = 0.2343*TWOPI*cf-1104;
	  rpa    = powf(10.0, log10(cf)*0.9 + 0.55)+ 2000;
	  pzero  = powf(10.0,log10(cf)*0.7+1.6)+500;

	/*===============================================================*/     
         
     order_of_pole    = 10;             
     half_order_pole  = order_of_pole/2;
     order_of_zero    = half_order_pole;

	 fs_bilinear = TWOPI*cf/tan(TWOPI*cf*tdres/2);
     rzero       = -pzero;
	 CF          = TWOPI*cf;
   
   if (n==0)
   {		  
	p[0].x = -sigma0;     

    p[0].y = ipw;

	p[4].x = p[0].x - rpa; p[4].y = p[0].y - ipb;

    p[2].x = (p[0].x + p[4].x) * 0.5; p[2].y = (p[0].y + p[4].y) * 0.5;

    p[1]   = compconj(p[0]);    p[3] = compconj(p[2]); p[5] = compconj(p[4]);

    p[6]   = p[0]; p[7] = p[1]; p[8] = p[4]; p[9]= p[5];

	   C1initphase = 0.0;
       for (i=0;i<half_order_pole;i++)          
	   {
           preal     = p[i*2].x;
		   pimg      = p[i*2].y;
	       C1initphase = C1initphase + atan(CF/(-rzero))-atan((CF-pimg)/(-preal))-atan((CF+pimg)/(-preal));
	   };

	/*===================== Initialize C1input & C1output =====================*/

      for (i=0;i<(half_order_pole+1);i++)          
      {
		   C1input[i][2] = 0; 
		   C1input[i][1] = 0; 
		   C1input[i][0] = 0;
		   C1output[i][2] = 0; 
		   C1output[i][1] = 0; 
		   C1output[i][0] = 0;
      }

	/*===================== normalize the gain =====================*/
    
      C1gain_norm = 1.0;
      for (r=0; r<order_of_pole; r++)
		   C1gain_norm = C1gain_norm*(powf((CF - p[r].y),2) + p[r].x*p[r].x);
      
   };
     
    norm_gain= sqrt(C1gain_norm)/powf(sqrt(CF*CF+rzero*rzero),order_of_zero);
	
	p[0].x = -sigma0 - rsigma;

	// if (p[1].x>0.0) mexErrMsgTxt("The system becomes unstable.\n");
	
	p[0].y = ipw;

	p[4].x = p[0].x - rpa; p[4].y = p[0].y - ipb;

    p[2].x = (p[0].x + p[4].x) * 0.5; p[2].y = (p[0].y + p[4].y) * 0.5;

    p[1] = compconj(p[0]); p[3] = compconj(p[2]); p[5] = compconj(p[4]);

    p[6] = p[0]; p[7] = p[1]; p[8] = p[4]; p[9]= p[5];

    phase = 0.0;
    for (i=0;i<half_order_pole;i++)          
    {
           preal = p[i*2].x;
		   pimg  = p[i*2].y;
	       phase = phase-atan((CF-pimg)/(-preal))-atan((CF+pimg)/(-preal));
	};

	rzero = -CF/tan((C1initphase-phase)/order_of_zero);

    // if (rzero>0.0) mexErrMsgTxt("The zeros are in the right-half plane.\n");
	 
   /*%==================================================  */
	/*each loop below is for a pair of poles and one zero */
   /*%      time loop begins here                         */
   /*%==================================================  */
 
       C1input[0][2]=C1input[0][1]; 
	   C1input[0][1]=C1input[0][0]; 
	   C1input[0][0]= x;

       for (i=0;i<half_order_pole;i++)          
       {
           preal = p[i*2].x;
		   pimg  = p[i*2].y;
		  	   
           temp  = powf((fs_bilinear-preal),2)+ powf(pimg,2);
		   

           /*dy = (input[i][1] + (1-(fs_bilinear+rzero)/(fs_bilinear-rzero))*input[i][2]
                                 - (fs_bilinear+rzero)/(fs_bilinear-rzero)*input[i][3] );
           dy = dy+2*output[i][1]*(fs_bilinear*fs_bilinear-preal*preal-pimg*pimg);

           dy = dy-output[i][2]*((fs_bilinear+preal)*(fs_bilinear+preal)+pimg*pimg);*/
		   
	       dy = C1input[i][0]*(fs_bilinear-rzero) - 2*rzero*C1input[i][1] - (fs_bilinear+rzero)*C1input[i][2]
                 +2*C1output[i][0]*(fs_bilinear*fs_bilinear-preal*preal-pimg*pimg)
			     -C1output[i][1]*((fs_bilinear+preal)*(fs_bilinear+preal)+pimg*pimg);

		   dy = dy/temp;

		   C1input[i+1][2] = C1output[i][1]; 
		   C1input[i+1][1] = C1output[i][0]; 
		   C1input[i+1][0] = dy;

		   C1output[i][1] = C1output[i][0]; 
		   C1output[i][0] = dy;
       }

	   dy = C1output[half_order_pole-1][0]*norm_gain;  /* don't forget the gain term */
	   c1filterout= dy/4.0;   /* signal path output is divided by 4 to give correct C1 filter gain */
	                   
     return (c1filterout);
}  

/* -------------------------------------------------------------------------------------------- */
/** Parallelpath C2 filter: same as the signal-path C1 filter with the OHC completely impaired */

__device__ double C2ChirpFilt(double xx, double tdres,double cf, int n, double taumax, double fcohc, double &C2gain_norm, double &C2initphase, double C2input[11][3], double C2output[11][3])
{
   
	double ipw, ipb, rpa, pzero, rzero;

	double sigma0,fs_bilinear,CF,norm_gain,phase,c2filterout;
	int    i,r,order_of_pole,half_order_pole,order_of_zero;
	double temp, dy, preal, pimg;

	COMPLEX p[10]; 	
    
    /*================ setup the locations of poles and zeros =======*/

	  sigma0 = 1/taumax;
	  ipw    = 1.01*cf*TWOPI-50;
      ipb    = 0.2343*TWOPI*cf-1104;
	  rpa    = powf(10.0, log10(cf)*0.9 + 0.55)+ 2000;
	  pzero  = powf(10.0,log10(cf)*0.7+1.6)+500;
	/*===============================================================*/     
         
     order_of_pole    = 10;             
     half_order_pole  = order_of_pole/2;
     order_of_zero    = half_order_pole;

	 fs_bilinear = TWOPI*cf/tan(TWOPI*cf*tdres/2);
     rzero       = -pzero;
	 CF          = TWOPI*cf;
   	    
    if (n==0)
    {		  
	p[0].x = -sigma0;     

    p[0].y = ipw;

	p[4].x = p[0].x - rpa; p[4].y = p[0].y - ipb;

    p[2].x = (p[0].x + p[4].x) * 0.5; p[2].y = (p[0].y + p[4].y) * 0.5;

    p[1] = compconj(p[0]); p[3] = compconj(p[2]); p[5] = compconj(p[4]);

    p[6] = p[0]; p[7] = p[1]; p[8] = p[4]; p[9]= p[5];

	   C2initphase = 0.0;
       for (i=0;i<half_order_pole;i++)         
	   {
           preal     = p[i*2].x;
		   pimg      = p[i*2].y;
	       C2initphase = C2initphase + atan(CF/(-rzero))-atan((CF-pimg)/(-preal))-atan((CF+pimg)/(-preal));
	   };

	/*===================== Initialize C2input & C2output =====================*/

      for (i=0;i<(half_order_pole+1);i++)          
      {
		   C2input[i][2] = 0; 
		   C2input[i][1] = 0; 
		   C2input[i][0] = 0;
		   C2output[i][2] = 0; 
		   C2output[i][1] = 0; 
		   C2output[i][0] = 0;
      }
    
    /*===================== normalize the gain =====================*/
    
     C2gain_norm = 1.0;
     for (r=0; r<order_of_pole; r++)
		   C2gain_norm = C2gain_norm*(powf((CF - p[r].y),2) + p[r].x*p[r].x);
    };
     
    norm_gain= sqrt(C2gain_norm)/powf(sqrt(CF*CF+rzero*rzero),order_of_zero);
    
	p[0].x = -sigma0*fcohc;

	// if (p[1].x>0.0) mexErrMsgTxt("The system becomes unstable.\n");
	
	p[0].y = ipw;

	p[4].x = p[0].x - rpa; p[4].y = p[0].y - ipb;

    p[2].x = (p[0].x + p[4].x) * 0.5; p[2].y = (p[0].y + p[4].y) * 0.5;

    p[1] = compconj(p[0]); p[3] = compconj(p[2]); p[5] = compconj(p[4]);

    p[6] = p[0]; p[7] = p[1]; p[8] = p[4]; p[9]= p[5];

    phase = 0.0;
    for (i=0;i<half_order_pole;i++)          
    {
           preal = p[i*2].x;
		   pimg  = p[i*2].y;
	       phase = phase-atan((CF-pimg)/(-preal))-atan((CF+pimg)/(-preal));
	};

	rzero = -CF/tan((C2initphase-phase)/order_of_zero);	
    // if (rzero>0.0) mexErrMsgTxt("The zeros are in the right-hand plane.\n");
   /*%==================================================  */
   /*%      time loop begins here                         */
   /*%==================================================  */

       C2input[0][2]=C2input[0][1]; 
	   C2input[0][1]=C2input[0][0]; 
	   C2input[0][0]= xx;

      for (i=0;i<half_order_pole;i++)          
      {
           preal = p[i*2].x;
		   pimg  = p[i*2].y;
		  	   
           temp  = powf((fs_bilinear-preal),2)+ powf(pimg,2);
		   
           /*dy = (input[i][1] + (1-(fs_bilinear+rzero)/(fs_bilinear-rzero))*input[i][2]
                                 - (fs_bilinear+rzero)/(fs_bilinear-rzero)*input[i][3] );
           dy = dy+2*output[i][1]*(fs_bilinear*fs_bilinear-preal*preal-pimg*pimg);

           dy = dy-output[i][2]*((fs_bilinear+preal)*(fs_bilinear+preal)+pimg*pimg);*/
		   
	      dy = C2input[i][0]*(fs_bilinear-rzero) - 2*rzero*C2input[i][1] - (fs_bilinear+rzero)*C2input[i][2]
                 +2*C2output[i][0]*(fs_bilinear*fs_bilinear-preal*preal-pimg*pimg)
			     -C2output[i][1]*((fs_bilinear+preal)*(fs_bilinear+preal)+pimg*pimg);

		   dy = dy/temp;

		   C2input[i+1][2] = C2output[i][1]; 
		   C2input[i+1][1] = C2output[i][0]; 
		   C2input[i+1][0] = dy;

		   C2output[i][1] = C2output[i][0]; 
		   C2output[i][0] = dy;

       };

	  dy = C2output[half_order_pole-1][0]*norm_gain;
	  c2filterout= dy/4.0;
	  
	  return (c2filterout); 
}   

/* -------------------------------------------------------------------------------------------- */
/** Pass the signal through the Control path Third Order Nonlinear Gammatone Filter */

__device__ double WbGammaTone(double x,double tdres,double centerfreq, int n, double tau,double gain,int order, double &wbphase, COMPLEX wbgtf[4], COMPLEX wbgtfl[4])
{
  double delta_phase,dtmp,c1LP,c2LP,out;
  int i,j;
  
  if (n==0)
  {
      wbphase = 0;
      for(i=0; i<=order;i++)
      {
            wbgtfl[i] = compmult(0,compexp(0));
            wbgtf[i]  = compmult(0,compexp(0));
      }
  }
  
  delta_phase = -TWOPI*centerfreq*tdres;
  wbphase += delta_phase;
  
  dtmp = tau*2.0/tdres;
  c1LP = (dtmp-1)/(dtmp+1);
  c2LP = 1.0/(dtmp+1);
  wbgtf[0] = compmult(x,compexp(wbphase));                 /* FREQUENCY SHIFT */
  
  for(j = 1; j <= order; j++)                              /* IIR Bilinear transformation LPF */
  wbgtf[j] = comp2sum(compmult(c2LP*gain,comp2sum(wbgtf[j-1],wbgtfl[j-1])),
      compmult(c1LP,wbgtfl[j]));
  out = REAL(compprod(compexp(-wbphase), wbgtf[order])); /* FREQ SHIFT BACK UP */
  
  for(i=0; i<=order;i++) wbgtfl[i] = wbgtf[i];
  return(out);
}

/* -------------------------------------------------------------------------------------------- */
/** Calculate the gain and group delay for the Control path Filter */

__device__ double gain_groupdelay(double tdres,double centerfreq, double cf, double tau,int &grdelay)
{ 
  double tmpcos,dtmp2,c1LP,c2LP,tmp1,tmp2,wb_gain;

  tmpcos = cos(TWOPI*(centerfreq-cf)*tdres);
  dtmp2 = tau*2.0/tdres;
  c1LP = (dtmp2-1)/(dtmp2+1);
  c2LP = 1.0/(dtmp2+1);
  tmp1 = 1+c1LP*c1LP-2*c1LP*tmpcos;
  tmp2 = 2*c2LP*c2LP*(1+tmpcos);
  
  wb_gain = powf(tmp1/tmp2, 1.0/2.0);
  
  grdelay = (int)floor((0.5-(c1LP*c1LP-c1LP*tmpcos)/(1+c1LP*c1LP-2*c1LP*tmpcos)));

  return(wb_gain);
}
/* -------------------------------------------------------------------------------------------- */
/** Calculate the delay (basilar membrane, synapse, etc. for cat) */
__device__ double delay_cat(double cf)
{  
  double A0,A1,x,delay;

  A0    = 3.0;  
  A1    = 12.5;
  x     = 11.9 * log10(0.80 + cf / 456.0);      /* cat mapping */
  delay = A0 * exp( -x/A1 ) * 1e-3;
  
  return(delay);
}

/* Calculate the delay (basilar membrane, synapse, etc.) for human, based
        on Harte et al. (JASA 2009) */
__device__ double delay_human(double cf) 
{  
  double A,B,delay;

  A    = -0.37;  
  B    = 11.09/2;
  delay = B * powf(cf * 1e-3,A)*1e-3;
  
  return(delay);
}

/* -------------------------------------------------------------------------------------------- */
/* Get the output of the OHC Nonlinear Function (Boltzman Function) */

__device__ double Boltzman(double x, double asym, double s0, double s1, double x1)
  {
	double shift,x0,out1,out;

    shift = 1.0/(1.0+asym);  /* asym is the ratio of positive Max to negative Max*/
    x0    = s0*log((1.0/shift-1)/(1+exp(x1/s1)));
	    
    out1 = 1.0/(1.0+exp(-(x-x0)/s0)*(1.0+exp(-(x-x1)/s1)))-shift;
	out = out1/(1-shift);

    return(out);
  }  /* output of the nonlinear function, the output is normalized with maximum value of 1 */
  
/* -------------------------------------------------------------------------------------------- */
/* Get the output of the OHC Low Pass Filter in the Control path */

__device__ double OhcLowPass(double x,double tdres,double Fc, int n,double gain,int order, double ohc[4], double ohcl[4])
{
  double c,c1LP,c2LP;
  int i,j;

  if (n==0)
  {
      for(i=0; i<(order+1);i++)
      {
          ohc[i] = 0;
          ohcl[i] = 0;
      }
  }    
  
  c = 2.0/tdres;
  c1LP = ( c - TWOPI*Fc ) / ( c + TWOPI*Fc );
  c2LP = TWOPI*Fc / (TWOPI*Fc + c);
  
  ohc[0] = x*gain;
  for(i=0; i<order;i++)
    ohc[i+1] = c1LP*ohcl[i+1] + c2LP*(ohc[i]+ohcl[i]);
  for(j=0; j<=order;j++) ohcl[j] = ohc[j];
  return(ohc[order]);
}
/* -------------------------------------------------------------------------------------------- */
/* Get the output of the IHC Low Pass Filter  */

__device__ double IhcLowPass(double x,double tdres,double Fc, int n,double gain,int order, double ihc[8], double ihcl[8])
{
  double C,c1LP,c2LP;
  int i,j;

  if (n==0)
  {
      for(i=0; i<(order+1);i++)
      {
          ihc[i] = 0;
          ihcl[i] = 0;
      }
  }     
  
  C = 2.0/tdres;
  c1LP = ( C - TWOPI*Fc ) / ( C + TWOPI*Fc );
  c2LP = TWOPI*Fc / (TWOPI*Fc + C);
  
  ihc[0] = x*gain;
  for(i=0; i<order;i++)
    ihc[i+1] = c1LP*ihcl[i+1] + c2LP*(ihc[i]+ihcl[i]);
  for(j=0; j<=order;j++) ihcl[j] = ihc[j];
  return(ihc[order]);
}
/* -------------------------------------------------------------------------------------------- */
/* Get the output of the Control path using Nonlinear Function after OHC */

__device__ double NLafterohc(double x,double taumin, double taumax, double asym)
{    
	double R,dc,R1,s0,x1,out,minR;

	minR = 0.05;
    R  = taumin/taumax;
    
	if(R<minR) minR = 0.5*R;
    else       minR = minR;
    
    dc = (asym-1)/(asym+1.0)/2.0-minR;
    R1 = R-minR;

    /* This is for new nonlinearity */
    s0 = -dc/log(R1/(1-minR));
	
    x1  = fabs(x);
    out = taumax*(minR+(1.0-minR)*exp(-x1/s0));
	if (out<taumin) out = taumin; 
    if (out>taumax) out = taumax;
    return(out);
}
/* -------------------------------------------------------------------------------------------- */
/* Get the output of the IHC Nonlinear Function (Logarithmic Transduction Functions) */

__device__ double NLogarithm(double x, double slope, double asym, double cf)
{
	double corner,strength,xx,splx,asym_t;
	    
    corner    = 80; 
    strength  = 20.0e6/powf(10.0,corner/20);
            
    xx = log(1.0+strength*fabs(x))*slope;
    
    if(x<0)
	{
		splx   = 20*log10(-x/20e-6);
		asym_t = asym -(asym-1)/(1+exp(splx/5.0));
		xx = -1/asym_t*xx;
	};   
    return(xx);
}
/* -------------------------------------------------------------------------------------------- */