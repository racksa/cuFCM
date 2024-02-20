#include <cufft.h>
#include <cuda.h>

#ifndef CONFIG_H
#define CONFIG_H


#define PI 3.14159265358979323846
#define PIsqrt 1.7724538509055159
#define PI2 6.28318530717959
#define PI2sqrt 2.5066282746310002
#define sqrt2oPI 0.7978845608028654
#define TWOoverPIsqrt 0.7978845608028654
#define PI2sqrt_inv 0.3989422804014327
#define SQRT2 1.4142135623730951

#ifdef USE_REGULARFCM
#endif

// Options
#define FCM_THREADS_PER_BLOCK 32 // default value is 32

#define ROTATION 1
// 0 = No rotation
// 1 = Rotation     **default

#define SPREAD_TYPE 4   // Deprecated option
// 0 = Thread per particle (TPP) register   **deprecated
// 1 = Thread per particle (TPP) recompute  **deprecated
// 2 = Block per particle (BPP) shared      **deprecated
// 3 = Block per particle (BPP) recompute
// 4 = Block per particle (BPP) shared dynamic  **default

#define GATHER_TYPE 4   // Deprecated option
// 0 = Thread per particle (TPP) register   **deprecated
// 1 = Thread per particle (TPP) recompute  **deprecated
// 2 = Block per particle (BPP) shared      **deprecated
// 3 = Block per particle (BPP) recompute   **deprecated
// 4 = Block per particle (BPP) shared dynamic  **default

#define CORRECTION_TYPE 1   // Deprecated option
// 0 = Linklist     **deprecated
// 1 = Spatial hashing (TPP)   **default

#define SORT_BACK 1
// 0 = Do not sort back
// 1 = Sort back    **default

#ifdef USE_DOUBLE_PRECISION
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
    #define my_fmod fmod
    #define my_sqrt sqrt
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
    #define my_fmod fmodf
    #define my_sqrt sqrtf
    
#endif

struct Pars
    {
        int N, nx, ny, nz, repeat, prompt;
        Real rh, alpha, beta, eta, boxsize, checkerror;
    };

struct Random_Pars
    {
        int N, nx, ny, nz, repeat, prompt;
        Real rh, alpha, beta, eta, boxsize, checkerror;
        Real dt, Fref;
    };
    
#endif