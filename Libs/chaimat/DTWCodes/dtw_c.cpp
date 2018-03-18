/**
 *
 * Yelin Kim
 * Feb 19, 2014
 * Modified the code of Quan Wang (copyright as follows)
 * Multi-dimensional version
 *
 *Usage:
 *
 * d=dtw_c(s,t)  or  d=dtw_c(s,t,w)
 * where s is D * M, t is D * N, both D-dimensional signals, and w is window parameter
 *
 * Copyright (C) 2013 Quan Wang <wangq10@rpi.edu>,
 * Signal Analysis and Machine Perception Laboratory,
 * Department of Electrical, Computer, and Systems Engineering,
 * Rensselaer Polytechnic Institute, Troy, NY 12180, USA
 */

/** 
 * This is the C/MEX code of dynamic time warping of two signals
 *
 * compile: 
 *     mex dtw_c.c
 *
 * usage:
 *     d=dtw_c(s,t)  or  d=dtw_c(s,t,w)
 *     where s is signal 1, t is signal 2, w is window parameter 
 */

#include "mex.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#define S(i,j) s[(i)+(j)*dim]
#define T(i,j) t[(i)+(j)*dim]

double dtw_c(double *s, double *t, int w, int ns, int nt, int dim)
{
    double d=0;
    int sizediff=ns-nt>0 ? ns-nt : nt-ns;
    double ** D;
    int i,j;
    int j1,j2;
    double cost,temp;
    
    // printf("ns=%d, nt=%d, w=%d, s[0]=%f, t[0]=%f\n",ns,nt,w,s[0],t[0]);
    
    
    if(w!=-1 && w<sizediff) w=sizediff; // adapt window size
    
    // create D
    D=(double **)malloc((ns+1)*sizeof(double *));
    for(i=0;i<ns+1;i++)
    {
        D[i]=(double *)malloc((nt+1)*sizeof(double));
    }
    
    // initialization
    for(i=0;i<ns+1;i++)
    {
        for(j=0;j<nt+1;j++)
        {
            D[i][j]=-1;
        }
    }
    D[0][0]=0;
    
    // dynamic programming
    for(i=1;i<=ns;i++)
    {
        if(w==-1)
        {
            j1=1;
            j2=nt;
        }
        else
        {
            j1= i-w>1 ? i-w : 1;
            j2= i+w<nt ? i+w : nt;
        }
        for(j=j1;j<=j2;j++)
        {
//            cost= fabs(s[i-1]-t[j-1]);
           
            cost = 0;
            for(int pq=0; pq<dim; pq++)
            {
  //              cost+= pow( (s[(i-1)+ pq*dim] - t[ (j-1) + pq*dim]), 2);
            //      cost+= pow( (S(i-1, pq) - T(j-1, pq)), 2);
                 cost+= pow( (S(pq, i-1) - T(pq, j-1)), 2); // This works correctly!
            }

            /* Find smallest among step pattern recursion */
            temp=D[i-1][j];
            if(D[i][j-1]!=-1) 
            {
                if(temp==-1 || D[i][j-1]<temp) temp=D[i][j-1];
            }
            D[i][j]=cost+temp;

            if(D[i-1][j-1]!=-1) 
            {
                if(temp==-1 || (2*cost+ D[i-1][j-1]) < D[i][j]) 
                {
                    D[i][j]=2*cost+D[i-1][j-1];

                }// for this case we need to multiply 2 on cost
            }
            /*******************************************/
            
        }
    }
    
    
    d=D[ns][nt];
    
    /* view matrix D */
    /*
    for(i=0;i<ns+1;i++)
    {
        for(j=0;j<nt+1;j++)
        {
            printf("%f  ",D[i][j]);
        }
        printf("\n");
    }
    */ 
    
    // free D
    for(i=0;i<ns+1;i++)
    {
        free(D[i]);
    }
    free(D);
    
    return d;
}

/* the gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    double *s,*t; /* pointers to input matrices */
    int w;
    int ns,nt, dim;
    double *dp;
    
    /*  check for proper number of arguments */
    if(nrhs!=2&&nrhs!=3)
    {
        mexErrMsgIdAndTxt( "MATLAB:dtw_c:invalidNumInputs",
                "Two or three inputs required.");
    }
    if(nlhs>1)
    {
        mexErrMsgIdAndTxt( "MATLAB:dtw_c:invalidNumOutputs",
                "dtw_c: One output required.");
    }
    
    /* check to make sure w is a scalar */
    if(nrhs==2)
    {
        w=-1;
    }
    else if(nrhs==3)
    {
        if( !mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) ||
                mxGetN(prhs[2])*mxGetM(prhs[2])!=1 )
        {
            mexErrMsgIdAndTxt( "MATLAB:dtw_c:wNotScalar",
                    "dtw_c: Input w must be a scalar.");
        }
        
        /*  get the scalar input w */
        w = (int) mxGetScalar(prhs[2]);
    }
    
    
    /*  create a pointer to the input matrix s */
    s = mxGetPr(prhs[0]);
    
    /*  create a pointer to the input matrix t */
    t = mxGetPr(prhs[1]);
    
    /*  get the length of the matrix input s */
    ns = mxGetN(prhs[0]); // number of rows in matlab = number of columns in C
    
    /*  get the length of the matrix input t */
    nt = mxGetN(prhs[1]);
    /*  get the dim of the matrix input s, t */
    
    dim = mxGetM(prhs[0]);
    
    /* Validate input arguments */
    if (dim != mxGetM(prhs[1])) {
        mexErrMsgIdAndTxt("MATLAB:dtw_c:differentdims",
            "Dimensions of matrices do not match.");
    }
    
    
    /*  set the output pointer to the output matrix */
    plhs[0] = mxCreateDoubleMatrix( 1, 1, mxREAL);
    
    /*  create a C pointer to a copy of the output matrix */
    dp = mxGetPr(plhs[0]);
    
    /*  call the C subroutine */
    dp[0]=dtw_c(s,t,w,ns,nt, dim);
    
    return;
    
}
