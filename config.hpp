#include <cufft.h>
#include <cuda.h>

#ifndef CONFIG_H
#define CONFIG_H


#define PI 3.14159265358979
#define PIsqrt 1.7724538509055159
#define PI2 6.28318530717959
#define PI2sqrt 2.5066282746310002
#define sqrt2oPI 0.7978845608028654
#define TWOoverPIsqrt 0.7978845608028654
#define PI2sqrt_inv 0.3989422804014327
#define SQRT2 1.4142135623730951

#define THREADS_PER_BLOCK 32

#define SOLVER_MODE 1
// 0 = FCM
// 1 = Fast FCM

#define INIT_FROM_FILE 1
// 0 = Generate random packing
// 1 = Read initial data from file

#define SPREAD_TYPE 4
// 0 = Thread per particle (TPP) register   **deprecated
// 1 = Thread per particle (TPP) recompute  **deprecated
// 2 = Block per particle (BPP) shared      **deprecated
// 3 = Block per particle (BPP) recompute
// 4 = Block per particle (BPP) shared dynamic  **default

#define GATHER_TYPE 4
// 0 = Thread per particle (TPP) register   **deprecated
// 1 = Thread per particle (TPP) recompute  **deprecated
// 2 = Block per particle (BPP) shared      **deprecated
// 3 = Block per particle (BPP) recompute
// 4 = Block per particle (BPP) shared dynamic  **default

#define CORRECTION_TYPE 1
// 0 = Linklist     **deprecated
// 1 = Spatial hashing (TPP)   **default

#define SORT_BACK 1
// 0 = Do not sort back
// 1 = Sort back    **default

#define OUTPUT_TO_FILE 1
// 0 = Dont write to file
// 1 = Write to file    **default

#define CHECK_ERROR 0
// 0 = Dont check error     **default
// 1 = Check error from file

#define USE_DOUBLE_PRECISION false

#if USE_DOUBLE_PRECISION
    typedef double Real;
    typedef long Integer;
    typedef cufftDoubleReal myCufftReal;
    typedef cufftDoubleComplex myCufftComplex;
    #define cufftComplex2Real CUFFT_Z2D
    #define cufftReal2Complex CUFFT_D2Z
    #define cufftExecReal2Complex cufftExecD2Z
    #define cufftExecComplex2Real cufftExecZ2D
    #define my_rint rint
    #define my_exp exp
    #define my_floor floor
#else
    typedef float Real;
    typedef int Integer;
    typedef cufftReal myCufftReal;
    typedef cufftComplex myCufftComplex;
    #define cufftComplex2Real CUFFT_C2R
    #define cufftReal2Complex CUFFT_R2C
    #define cufftExecReal2Complex cufftExecR2C
    #define cufftExecComplex2Real cufftExecC2R
    #define my_rint rintf
    #define my_exp expf
    #define my_floor floorf
    
#endif

#define NGD_UAMMD 9
#define NGD 9

struct Pars
    {
        int N, nx, ny, nz, repeat, prompt;
        Real rh, alpha, beta, eta, boxsize;
    };

struct Random_Pars
    {
        int N, nx, ny, nz, repeat, prompt;
        Real rh, alpha, beta, eta, boxsize;
        Real dt, Fref;
    };
    
#endif