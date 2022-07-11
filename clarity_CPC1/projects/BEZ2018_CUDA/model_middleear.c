/* This is the BEZ2018 version of the code for the middle ear filter from 
 * the Carney, Bruce and Zilany labs.
 * 
 * Please refer to this work which contains the original work:
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
 * Please cite these papers if you publish any research results obtained 
 * with this code or any modified versions of this code.
 *
 * See the file readme.txt for details of compiling and running the model.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mex.h>
#include <time.h>

#define MAXSPIKES 1000000
#ifndef TWOPI
#define TWOPI 6.28318530717959
#endif

/* This function is the MEX "wrapper", to pass the input and output variables between the .mex* file and Matlab or Octave */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	
	double *px, tdres, reptime;
	int    species, pxbins, lp;
    mwSize outsize[2];
    
	double *pxtmp, *speciestmp, *tdrestmp;
    double *ihcout;
   
	void   model_ME(double *, int, double, int, double *);
	
	/* Check for proper number of arguments */
	
	if (nrhs != 3) 
	{
		mexErrMsgTxt("model_middleear requires 3 input arguments.");
	}; 

	if (nlhs !=1)  
	{
		mexErrMsgTxt("model_middleear requires 1 output argument.");
	};
	
	/* Assign pointers to the inputs */

	pxtmp		= mxGetPr(prhs[0]);
    speciestmp	= mxGetPr(prhs[1]);
    tdrestmp	= mxGetPr(prhs[2]);
		
	/* Check individual input arguments */

	pxbins = (int) mxGetN(prhs[0]);
	if (pxbins==1)
		mexErrMsgTxt("px must be a row vector\n");

    species = (int) speciestmp[0];
	if (speciestmp[0]!=species)
		mexErrMsgTxt("species must an integer.\n");
	if (species<1 || species>3)
		mexErrMsgTxt("Species must be 1 for cat, or 2 or 3 for human.\n");

    tdres = tdrestmp[0];

    px = (double*)mxCalloc(pxbins,sizeof(double)); 

	/* Put stimulus waveform into pressure waveform */

	for (lp=0; lp<pxbins; lp++)
			px[lp] = pxtmp[lp];
	
	/* Create an array for the return argument */
	
    outsize[0] = 1;
	outsize[1] = pxbins;
    
	plhs[0] = mxCreateNumericArray(2, outsize, mxDOUBLE_CLASS, mxREAL);
	
	/* Assign pointers to the outputs */
	
	ihcout  = mxGetPr(plhs[0]);
		
	/* run the model */

	model_ME(px,species,tdres,pxbins,ihcout);

 mxFree(px);

}

void model_ME(double *px, int species, double tdres, int totalstim, double *meout)
{	
    
    /*variables for middle-ear model */
	double megainmax;
    double *mey1, *mey2, *mey3;
    double fp,C,m11,m12,m13,m14,m15,m16,m21,m22,m23,m24,m25,m26,m31,m32,m33,m34,m35,m36;
    
    int n;
	    
	mey1 = (double*)mxCalloc(totalstim,sizeof(double));
	mey2 = (double*)mxCalloc(totalstim,sizeof(double));
	mey3 = (double*)mxCalloc(totalstim,sizeof(double));
    
  	/*===============================================================*/
    /*===============================================================*/
    /* Prewarping and related constants for the middle ear */
     fp = 1e3;  /* prewarping frequency 1 kHz */
     C  = TWOPI*fp/tan(TWOPI/2*fp*tdres);
     if (species==1) /* for cat */
     {
         /* Cat middle-ear filter - simplified version from Bruce et al. (JASA 2003) */
         m11 = C/(C + 693.48);                    m12 = (693.48 - C)/C;            m13 = 0.0;
         m14 = 1.0;                               m15 = -1.0;                      m16 = 0.0;
         m21 = 1/(pow(C,2) + 11053*C + 1.163e8);  m22 = -2*pow(C,2) + 2.326e8;     m23 = pow(C,2) - 11053*C + 1.163e8; 
         m24 = pow(C,2) + 1356.3*C + 7.4417e8;    m25 = -2*pow(C,2) + 14.8834e8;   m26 = pow(C,2) - 1356.3*C + 7.4417e8;
         m31 = 1/(pow(C,2) + 4620*C + 909059944); m32 = -2*pow(C,2) + 2*909059944; m33 = pow(C,2) - 4620*C + 909059944;
         m34 = 5.7585e5*C + 7.1665e7;             m35 = 14.333e7;                  m36 = 7.1665e7 - 5.7585e5*C;
         megainmax=41.1405;
     };
     if (species>1) /* for human */
     {
         /* Human middle-ear filter - based on Pascal et al. (JASA 1998)  */
         m11=1/(pow(C,2)+5.9761e+003*C+2.5255e+007);m12=(-2*pow(C,2)+2*2.5255e+007);m13=(pow(C,2)-5.9761e+003*C+2.5255e+007);m14=(pow(C,2)+5.6665e+003*C);             m15=-2*pow(C,2);					m16=(pow(C,2)-5.6665e+003*C);
         m21=1/(pow(C,2)+6.4255e+003*C+1.3975e+008);m22=(-2*pow(C,2)+2*1.3975e+008);m23=(pow(C,2)-6.4255e+003*C+1.3975e+008);m24=(pow(C,2)+5.8934e+003*C+1.7926e+008); m25=(-2*pow(C,2)+2*1.7926e+008);	m26=(pow(C,2)-5.8934e+003*C+1.7926e+008);
         m31=1/(pow(C,2)+2.4891e+004*C+1.2700e+009);m32=(-2*pow(C,2)+2*1.2700e+009);m33=(pow(C,2)-2.4891e+004*C+1.2700e+009);m34=(3.1137e+003*C+6.9768e+008);     m35=2*6.9768e+008;				m36=(-3.1137e+003*C+6.9768e+008);
         megainmax=2;
     };
  	for (n=0;n<totalstim;n++) /* Start of the loop */
    {    
        if (n==0)  /* Start of the middle-ear filtering section  */
		{
	    	mey1[0]  = m11*px[0];
            if (species>1) mey1[0] = m11*m14*px[0];
            mey2[0]  = mey1[0]*m24*m21;
            mey3[0]  = mey2[0]*m34*m31;
            meout[0] = mey3[0]/megainmax ;
        }
            
        else if (n==1)
		{
            mey1[1]  = m11*(-m12*mey1[0] + px[1]       - px[0]);
            if (species>1) mey1[1] = m11*(-m12*mey1[0]+m14*px[1]+m15*px[0]);
			mey2[1]  = m21*(-m22*mey2[0] + m24*mey1[1] + m25*mey1[0]);
            mey3[1]  = m31*(-m32*mey3[0] + m34*mey2[1] + m35*mey2[0]);
            meout[1] = mey3[1]/megainmax;
		}
	    else 
		{
            mey1[n]  = m11*(-m12*mey1[n-1]  + px[n]         - px[n-1]);
            if (species>1) mey1[n]= m11*(-m12*mey1[n-1]-m13*mey1[n-2]+m14*px[n]+m15*px[n-1]+m16*px[n-2]);
            mey2[n]  = m21*(-m22*mey2[n-1] - m23*mey2[n-2] + m24*mey1[n] + m25*mey1[n-1] + m26*mey1[n-2]);
            mey3[n]  = m31*(-m32*mey3[n-1] - m33*mey3[n-2] + m34*mey2[n] + m35*mey2[n-1] + m36*mey2[n-2]);
            meout[n] = mey3[n]/megainmax;
		}; 	/* End of the middle-ear filtering section */   
     
   };  /* End of the loop */
    mxFree(mey1); mxFree(mey2); mxFree(mey3);	
}