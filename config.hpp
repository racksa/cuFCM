#include <cufft.h>

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

#define THREADS_PER_BLOCK 32

#define HASH_ENCODE_FUNC linear_encode

#define INIT_FROM_FILE 1
// 0 = Generate random packing
// 1 = Read initial data from file

#define SPREAD_TYPE 4
// 0 = Thread per particle (TPP) register
// 1 = Thread per particle (TPP) recompute
// 2 = Block per particle (BPP) shared  **default
// 3 = Block per particle (BPP) recompute
// 4 = Block per particle (BPP) shared dynamic

#define GATHER_TYPE 4
// 0 = Thread per particle (TPP) register
// 1 = Thread per particle (TPP) recompute
// 2 = Block per particle (BPP) shared  **default
// 3 = Block per particle (BPP) recompute
// 4 = Block per particle (BPP) shared dynamic

#define SPATIAL_HASHING 2
// 0 = Spatial hashing for correction, but without sorting
// 1 = Spatial hashing and sorting
// 2 = Spatial hashing and sorting (GPU)    **default

#if SOLVER_MODE == 1 and SPATIAL_HASHING != 2
    #define CORRECTION_TYPE 0
#else
    #define CORRECTION_TYPE 1
// 0 = Linklist
// 1 = Spatial hashing (TPP)  (must have SPATIAL_HASHING == 2)    **default
#endif

#define SORT_BACK 1
// 0 = Do not sort back
// 1 = Sort back    **default

#define OUTPUT_TO_FILE 1
// 0 = Dont write to file
// 1 = Write to file    **default

#define CHECK_ERROR 1
// 0 = Dont check error
// 1 = Check error from file  **default
// 2 = Check error from random sample

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

#define NGD_UAMMD 9
#define NGD 9
#endif