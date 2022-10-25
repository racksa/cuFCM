#pragma once
#include "config.hpp"

__device__ __host__
Real f(Real r, Real rsq, Real sigma, Real sigmasq, Real expS, Real erfS);

__device__ __host__
Real dfdr(Real r, Real rsq, Real sigma, Real sigmasq, Real expS, Real erfS);

__device__ __host__
Real A(Real r, Real rsq, Real sigma, Real sigmasq, Real expS, Real erfS);

__device__ __host__
Real B(Real r, Real rsq, Real sigma, Real sigmasq, Real expS, Real erfS);

__device__ __host__
Real dAdr(Real r, Real rsq, Real sigma, Real sigmasq, Real expS, Real erfS);

__device__ __host__
Real dBdr(Real r, Real rsq, Real sigma, Real sigmasq, Real expS, Real erfS);

__device__ __host__
Real C(Real r, Real rsq, Real sigma, Real sigmasq, Real gaussgam, Real erfS);

__device__ __host__
Real D(Real r, Real rsq, Real sigma, Real sigmasq, Real gaussgam, Real erfS);

__device__ __host__
Real P(Real r, Real rsq, Real sigma, Real sigmasq, Real gaussgam);

__device__ __host__
Real Q(Real r, Real rsq, Real sigma, Real sigmasq, Real gaussgam);

__global__
void cufcm_self_correction(Real* V, Real* W, Real* F, Real* T, int N, Real boxsize,
                                Real StokesMob, Real ModStokesMob,
                                Real PDStokesMob, Real BiLapMob,
                                Real WT1Mob, Real WT2Mob);

__global__
void cufcm_pair_correction_spatial_hashing_tpp(Real* Y, Real* V, Real* W, Real* F, Real* T, int N, Real boxsize,
                    int *particle_cellindex, int *cell_start, int *cell_end,
                    int *map,
                    int ncell, Real Rrefsq,
                    Real pdmag,
                    Real sigma, Real sigmasq,
                    Real sigmaFCM, Real sigmaFCMsq,
                    Real sigmaFCMdip, Real sigmaFCMdipsq);

__global__
void cufcm_compute_formula(Real* Y, Real* V, Real* W, Real* F, Real* T, int N, int N_truncate,
                    Real sigmaFCM, 
                    Real sigmaFCMdip,
                    Real StokesMob,
                    Real WT1Mob);