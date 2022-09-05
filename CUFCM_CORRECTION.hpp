#pragma once

__device__ __host__
double f(double r, double rsq, double sigma, double sigmasq, double expS, double erfS);

__device__ __host__
double dfdr(double r, double rsq, double sigma, double sigmasq, double expS, double erfS);

__device__ __host__
double A(double r, double rsq, double sigma, double sigmasq, double expS, double erfS);

__device__ __host__
double B(double r, double rsq, double sigma, double sigmasq, double expS, double erfS);

__device__ __host__
double dAdr(double r, double rsq, double sigma, double sigmasq, double expS, double erfS);

__device__ __host__
double dBdr(double r, double rsq, double sigma, double sigmasq, double expS, double erfS);

__device__ __host__
double C(double r, double rsq, double sigma, double sigmasq, double gaussgam, double erfS);

__device__ __host__
double D(double r, double rsq, double sigma, double sigmasq, double gaussgam, double erfS);

__device__ __host__
double P(double r, double rsq, double sigma, double sigmasq, double gaussgam);

__device__ __host__
double Q(double r, double rsq, double sigma, double sigmasq, double gaussgam);

__global__
void cufcm_pair_correction(double* Y, double* V, double* W, double* F, double* T, int N,
                    int *map, int *head, int *list,
                    int ncell, double Rrefsq,
                    double pdmag,
                    double sigma, double sigmasq,
                    double sigmaFCM, double sigmaFCMsq,
                    double sigmaFCMdip, double sigmaFCMdipsq);

__global__
void cufcm_self_correction(double* V, double* W, double* F, double* T, int N,
                                double StokesMob, double ModStokesMob,
                                double PDStokesMob, double BiLapMob,
                                double WT1Mob, double WT2Mob);

__global__
void cufcm_pair_correction_spatial_hashing(double* Y, double* V, double* W, double* F, double* T, int N,
                    int *map, int *head, int *list,
                    int ncell, double Rrefsq,
                    double pdmag,
                    double sigma, double sigmasq,
                    double sigmaFCM, double sigmaFCMsq,
                    double sigmaFCMdip, double sigmaFCMdipsq);

void cufcm_pair_correction_loop(double* Y, double* V, double* W, double* F, double* T, int N,
                    int *map, int *head, int *list,
                    int ncell, double Rrefsq,
                    double pdmag,
                    double sigma, double sigmasq,
                    double sigmaFCM, double sigmaFCMsq,
                    double sigmaFCMdip, double sigmaFCMdipsq);

void cufcm_self_correction_loop(double* V, double* W, double* F, double* T, int N,
                                double StokesMob, double ModStokesMob,
                                double PDStokesMob, double BiLapMob,
                                double WT1Mob, double WT2Mob);