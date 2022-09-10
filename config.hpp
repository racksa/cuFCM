#include <cufft.h>

#ifndef CONFIG_H
#define CONFIG_H


#define PI 3.14159265358979
#define PI2 6.28318530717959
#define PI2sqrt 2.5066282746310002
#define TWOoverPIsqrt 0.7978845608028654
#define PI2sqrt_inv 0.3989422804014327

#define SOLVER_MODE 1
// 0 = FCM
// 1 = Fast FCM

// #define RH 0.008089855908506678
#define RH 0.02609300415934458

#define NGD 9
#define SIGMA_FAC 1.55917641
#define RREF_FAC 5.21186960


#define NX 256.0
#define NY 256.0
#define NZ 256.0
#define NPTS 256.0

// #define NGD 11
// #define SIGMA_FAC 1.75207280;
// #define RREF_FAC 6.69738570;

// #define NGD 13
// #define SIGMA_FAC 1.92479594;
// #define RREF_FAC 8.18540500;

// #define NGD 15
// #define SIGMA_FAC 2.08834941;
// #define RREF_FAC 9.66631420;

// #define NGD 18
// #define SIGMA_FAC 2.24239977;
// #define RREF_FAC 11.19847000;

#define GRID_SIZE (NX*NY*NZ)
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

#define SORT_BACK 1
// 0 = Do not sort back
// 1 = Sort back    **default

#define CORRECTION_TYPE 1
// 0 = Linklist
// 1 = Spatial hashing (TPP)  (must have Spatial hashing == 2)    **default

#define OUTPUT_TO_FILE 1
// 0 = Dont write to file
// 1 = Write to file

#define USE_DOUBLE_PRECISION true

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