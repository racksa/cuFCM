#include <cufft.h>

#ifndef CONFIG_H
#define CONFIG_H


#define PI 3.14159265358979
#define PIsqrt 1.7724538509055159
#define PI2 6.28318530717959
#define PI2sqrt 2.5066282746310002
#define TWOoverPIsqrt 0.7978845608028654
#define PI2sqrt_inv 0.3989422804014327

#define SOLVER_MODE 1
// 0 = FCM
// 1 = Fast FCM

#define TOL 0
// 0 = 10^-3
// 1 = 10^-4
// 2 = 10^-5
// 3 = 10^-6

#define ENABLE_REPEAT 1
// 0 = Dont repeat
// 1 = Reapeat measurement

// #define RH 0.008089855908506678
#define RH 0.02609300415934458

#if SOLVER_MODE == 1

    #define NPTS 256.0
    #define DX (PI2/NPTS)

    #if TOL == 0
        #define ALPHA 0.9352    /* SIGMA/dx */
        #define BETA 9.706      /* (NGD.DX)/SIGMA */
        #define GAMMA 5.573     /* RC/SIGMA */
    #elif TOL == 1
        #define ALPHA 1.0509    /* SIGMA/dx */
        #define BETA 10.873     /* (NGD.DX)/SIGMA */
        #define GAMMA 6.373     /* RC/SIGMA */
    #elif TOL == 2
        #define ALPHA 1.1545    /* SIGMA/dx */
        #define BETA 11.773     /* (NGD.DX)/SIGMA */
        #define GAMMA 7.090     /* RC/SIGMA */
    #elif TOL == 3
        #define ALPHA 1.2526    /* SIGMA/dx */
        #define BETA 12.639     /* (NGD.DX)/SIGMA */
        #define GAMMA 7.717     /* RC/SIGMA */
    #endif
    // #define ALPHA 1.3450    /* SIGMA/dx */
    // #define BETA 13.410     /* (NGD.DX)/SIGMA */
    // #define GAMMA 8.326     /* RC/SIGMA */

    #define SIGMA_FCM (RH/PIsqrt)
    #define SIGMA (DX*ALPHA)
    #define SIGMA_FAC (SIGMA/SIGMA_FCM)
    #define NGD (int)(BETA*ALPHA)
    #define RREF_FAC (GAMMA*ALPHA)

#elif SOLVER_MODE == 0

    #define SIGMA (RH/PIsqrt)
    #define SIGMA_FAC 1

    #if TOL == 0
        #define ALPHA 1.0409    /* SIGMA/dx */
        #define BETA 10.873     /* (NGD.DX)/SIGMA */
    #elif TOL == 1
        #define ALPHA 1.1545    /* SIGMA/dx */
        #define BETA 11.773     /* (NGD.DX)/SIGMA */
    #elif TOL == 2
        #define ALPHA 1.2526    /* SIGMA/dx */
        #define BETA 12.639     /* (NGD.DX)/SIGMA */
    #elif TOL == 3
        #define ALPHA 1.3850    /* SIGMA/dx */
        #define BETA 13.410     /* (NGD.DX)/SIGMA */
    #endif
    
    #define DX_C (SIGMA/ALPHA)
    #define NPTS ((int)(PI2/DX_C)%2 == 0? (int)(PI2/DX_C) : (int)(PI2/DX_C) + 1)
    #define DX (PI2/NPTS)
    #define NGD (int)(BETA*ALPHA)

    // #define NPTS 500
    // #define DX (PI2/NPTS)
    // #define NGD 20

    #define RREF_FAC 5.21186960

#endif

#define NX NPTS
#define NY NPTS
#define NZ NPTS


#define GRID_SIZE (int)(NX*NY*NZ)
#define FFT_GRID_SIZE ((NX/2+1)*NY*NZ)

#define THREADS_PER_BLOCK 32

#define BATCH 10

#define RANK 1

#define HASH_ENCODE_FUNC linear_encode

#define INIT_FROM_FILE 1
// 0 = Generate random packing
// 1 = Read initial data from file

#define GRIDDING_TYPE 2
// 0 = Thread per particle (TPP) register
// 1 = Thread per particle (TPP) recompute
// 2 = Block per particle (BPP) shared  **default
// 3 = Block per particle (BPP) recompute

#define SPATIAL_HASHING 2
// 0 = Spatial hashing for correction, but without sorting
// 1 = Spatial hashing and sorting
// 2 = Spatial hashing and sorting (GPU)    **default

#define CORRECTION_TYPE 1
// 0 = Linklist
// 1 = Spatial hashing (TPP)  (must have Spatial hashing == 2)    **default

#define SORT_BACK 1
// 0 = Do not sort back
// 1 = Sort back    **default

#define OUTPUT_TO_FILE 0
// 0 = Dont write to file
// 1 = Write to file

#define CHECK_ERROR 1
// 0 = Dont check error
// 1 = Check error

#define USE_DOUBLE_PRECISION false

#if USE_DOUBLE_PRECISION
    typedef double Real;
    typedef cufftDoubleReal myCufftReal;
    typedef cufftDoubleComplex myCufftComplex;
    #define cufftComplex2Real CUFFT_Z2D
    #define cufftReal2Complex CUFFT_D2Z
    #define cufftExecReal2Complex cufftExecD2Z
    #define cufftExecComplex2Real cufftExecZ2D
#else
    typedef float Real;
    typedef cufftReal myCufftReal;
    typedef cufftComplex myCufftComplex;
    #define cufftComplex2Real CUFFT_C2R
    #define cufftReal2Complex CUFFT_R2C
    #define cufftExecReal2Complex cufftExecR2C
    #define cufftExecComplex2Real cufftExecC2R
#endif

#endif