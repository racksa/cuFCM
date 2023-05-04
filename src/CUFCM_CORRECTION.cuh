#pragma once
#include "config.hpp"

__device__ __host__
Real A_old(Real r, Real rsq, Real sigma, Real sigmasq, Real expS, Real erfS);

__device__ __host__
Real B_old(Real r, Real rsq, Real sigma, Real sigmasq, Real expS, Real erfS);

__device__ __host__
Real f_old(Real r, Real rsq, Real sigma, Real sigmasq, Real expS, Real erfS);

__device__ __host__
Real dfdr_old(Real r, Real rsq, Real sigma, Real sigmasq, Real expS, Real erfS);

__device__ __host__
Real dAdr(Real r, Real rsq, Real sigma, Real sigmasq, Real expS, Real erfS);

__device__ __host__
Real dBdr(Real r, Real rsq, Real sigma, Real sigmasq, Real expS, Real erfS);



__device__ __host__
Real S_I(Real r, Real rsq, Real sigma, Real sigmasq, Real gaussgam, Real erfS);

__device__ __host__
Real S_xx(Real r, Real rsq, Real sigma, Real sigmasq, Real gaussgam, Real erfS);

__device__ __host__
Real f(Real r, Real rsq, Real sigma, Real sigmasq, Real gaussgam, Real erfS);

__device__ __host__
Real dfdr(Real r, Real rsq, Real sigma, Real sigmasq, Real gaussgam, Real erfS);

__device__ __host__
Real Q_I(Real r, Real rsq, Real sigma, Real sigmasq, Real gaussgam, Real erfS);

__device__ __host__
Real Q_xx(Real r, Real rsq, Real sigma, Real sigmasq, Real gaussgam, Real erfS);

__device__ __host__
Real P_I(Real r, Real rsq, Real sigma, Real sigmasq, Real gaussgam, Real erfS);

__device__ __host__
Real P_xx(Real r, Real rsq, Real sigma, Real sigmasq, Real gaussgam, Real erfS);

__device__ __host__
Real T_I(Real r, Real rsq, Real sigma, Real sigmasq, Real gaussgam);

__device__ __host__
Real T_xx(Real r, Real rsq, Real sigma, Real sigmasq, Real gaussgam);

__device__ __host__
Real K(Real r, Real rsq, Real sigma, Real sigmasq, Real gaussgam);

__global__
void cufcm_pair_correction(Real* Y, Real* V, Real* W, Real* F, Real* T, int N, Real Lx, Real Ly, Real Lz,
                    int *particle_cellindex, int *cell_start, int *cell_end,
                    int *map,
                    int ncell, Real Rrefsq,
                    Real sigma,
                    Real sigmaFCM,
                    Real sigmaFCMdip);

__global__
void cufcm_self_correction(Real* V, Real* W, Real* F, Real* T, int N,
                                Real StokesMob, Real ModStokesMob,
                                Real PDStokesMob, Real BiLapMob,
                                Real WT1Mob, Real WT2Mob);

// __global__
// void cufcm_compute_formula(Real* Y, Real* V, Real* W, Real* F, Real* T, int N,
//                     Real sigmaFCM, 
//                     Real sigmaFCMdip,
//                     Real StokesMob,
//                     Real WT1Mob);