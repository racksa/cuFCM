
#pragma once
#include "config.hpp"

__global__
void cufcm_mono_distribution_single_fx(Real *fx, Real *Y,
              Real *F, int N, int ngd, 
              Real sigmasq,
              Real anorm, Real anorm2,
              Real dx, int npts);

__global__
void cufcm_mono_distribution_single_fx_recompute(Real *fx, Real *Y,
              Real *F, int N, int ngd, 
              Real sigmasq,
              Real anorm, Real anorm2,
              Real dx, int npts);

__global__
void cufcm_mono_distribution_regular_fxfyfz(myCufftReal *fx, myCufftReal *fy, myCufftReal *fz,
              Real *Y,
              Real *F, int N, int ngd, 
              Real sigmasq,
              Real anorm, Real anorm2,
              Real dx, int npts);