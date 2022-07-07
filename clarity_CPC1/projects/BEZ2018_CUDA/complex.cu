/* 
complex.cpp includes all of the COMPLEX math functions needed for model programs
*/

#include <stdlib.h>
#include <math.h>
    
/* COMPLEX.HPP header file		
 * use for complex arithmetic in C 
 (part of them are from "C Tools for Scientists and Engineers" by L. Baker)
*/

/* Structure of the complex */
struct __COMPLEX{ double x,y; };
/* structure COMPLEX same as __COMPLEX */
typedef struct __COMPLEX COMPLEX;

/* for below, X, Y are complex structures, and one is returned */

/* real part of the complex multiplication */
#define CMULTR(X,Y) ((X).x*(Y).x-(X).y*(Y).y)
/* image part of the complex multiplication */
#define CMULTI(X,Y) ((X).y*(Y).x+(X).x*(Y).y)
/* used in the Division : real part of the division */
#define CDRN(X,Y) ((X).x*(Y).x+(Y).y*(X).y)
/* used in the Division : image part of the division */
#define CDIN(X,Y) ((X).y*(Y).x-(X).x*(Y).y)
/*  used in the Division : denumerator of the division */
#define CNORM(X) ((X).x*(X).x+(X).y*(X).y)
/* real part of the complex */
#define CREAL(X) (double((X).x))
/* conjunction value */
#define CONJG(z,X) {(z).x=(X).x;(z).y= -(X).y;}
/* conjunction value */
#define CONJ(X) {(X).y= -(X).y;}
/* muliply : z could not be same variable as X or Y, same rule for other Macro */
#define CMULT(z,X,Y) {(z).x=CMULTR((X),(Y));(z).y=CMULTI((X),(Y));}
/* division */
#define CDIV(z,X,Y){double d=CNORM(Y); (z).x=CDRN(X,Y)/d; (z).y=CDIN(X,Y)/d;}
/* addition */
#define CADD(z,X,Y) {(z).x=(X).x+(Y).x;(z).y=(X).y+(Y).y;}
/* subtraction */
#define CSUB(z,X,Y) {(z).x=(X).x-(Y).x;(z).y=(X).y-(Y).y;}
/* assign */
#define CLET(to,from) {(to).x=(from).x;(to).y=(from).y;}
/* abstract value(magnitude) */
/* #define cabs(X) sqrt((X).y*(X).y+(X).x*(X).x) comment out for lcc compiler -not used in code anyway.  */
/* real to complex */
#define CMPLX(X,real,imag) {(X).x=(real);(X).y=(imag);}
/* multiply with real*/
#define CTREAL(z,X,real) {(z).x=(X).x*(real);(z).y=(X).y*(real);}


/* implementation using function : for compatibility */
/* divide */
__device__ COMPLEX compdiv(COMPLEX ne,COMPLEX de);
/* this returns a complex number equal to exp(i*theta) */
__device__ COMPLEX compexp(double theta);
/* Multiply a complex number by a scalar */
__device__ COMPLEX compmult(double scalar,COMPLEX compnum);
/* Find the product of 2 complex numbers */
__device__ COMPLEX compprod(COMPLEX compnum1, COMPLEX compnum2);
/* add 2 complex numbers */
__device__ COMPLEX comp2sum(COMPLEX summand1, COMPLEX summand2);
/* add three complex numbers */
__device__ COMPLEX comp3sum(COMPLEX summand1, COMPLEX summand2, COMPLEX summand3);
/* subtraction: complexA - complexB */
__device__ COMPLEX compsubtract(COMPLEX complexA, COMPLEX complexB);
/* Get the real part of the complex */
__device__ double  REAL(COMPLEX compnum); /*{double(compnum.x);};*/
/* Get the imaginary part of the complex */
__device__ double IMAG(COMPLEX compnum);
/* Get the conjugate of the complex  */
__device__ COMPLEX compconj(COMPLEX complexA);

/* divide */
__device__ COMPLEX compdiv(COMPLEX ne,COMPLEX de)
{
  double d;
  COMPLEX z;
  d=de.x*de.x+de.y*de.y;
  z.x=(ne.x*de.x+ne.y*de.y)/d;
  z.y=(ne.y*de.x-ne.x*de.y)/d;
  return(z);
}
/* this returns a complex number equal to exp(i*theta) */
__device__ COMPLEX compexp(double theta)
{
  COMPLEX dummy;
  dummy.x = cos(theta);
  dummy.y = sin(theta);
  return dummy;
}
/* Multiply a complex number by a scalar */
__device__ COMPLEX compmult(double scalar, COMPLEX compnum)
{
 COMPLEX answer;
 answer.x = scalar * compnum.x;
 answer.y = scalar * compnum.y;
 return answer;
}
/* Find the product of 2 complex numbers */
__device__ COMPLEX compprod(COMPLEX compnum1, COMPLEX compnum2)
{
 COMPLEX answer;
 answer.x = (compnum1.x * compnum2.x) - (compnum1.y * compnum2.y);
 answer.y = (compnum1.x * compnum2.y) + (compnum1.y * compnum2.x);
 return answer;
}
/* add 2 complex numbers */
__device__ COMPLEX comp2sum(COMPLEX summand1, COMPLEX summand2)
{
 COMPLEX answer;
 answer.x = summand1.x + summand2.x;
 answer.y = summand1.y + summand2.y;
 return answer;
}
/* add three complex numbers */
__device__ COMPLEX comp3sum(COMPLEX summand1, COMPLEX summand2, COMPLEX summand3)
{
 COMPLEX answer;
 answer.x = summand1.x + summand2.x + summand3.x;
 answer.y = summand1.y + summand2.y + summand3.y;
 return answer;
}

/* subtraction: complexA - complexB */
__device__ COMPLEX compsubtract(COMPLEX complexA, COMPLEX complexB)
{
 COMPLEX answer;
 answer.x = complexA.x - complexB.x;
 answer.y = complexA.y - complexB.y;
 return answer;
}
/* Get the real part of the complex */
__device__ double REAL(COMPLEX compnum)
{ return(compnum.x); } 

/* Get the imaginary part of the complex */
__device__ double IMAG(COMPLEX compnum)
{ return(compnum.y); } 

/* Get the conjugate of the complex signal */
__device__ COMPLEX compconj(COMPLEX complexA)
{
  COMPLEX answer;
  answer.x = complexA.x;
  answer.y = -complexA.y;
  return (answer);
}