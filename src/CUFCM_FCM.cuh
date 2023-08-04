#pragma once
#include "config.hpp"

///////////////////////////////////////////////////////////////////////////////
// Fast FCM
///////////////////////////////////////////////////////////////////////////////
// __global__
// void cufcm_precompute_gauss(int N, int ngd, Real* Y,
//                     Real* gaussx_, Real* gaussy, Real* gaussz,
//                     Real* grad_gaussx_dip, Real* grad_gaussy_dip, Real* grad_gaussz_dip,
//                     Real* gaussgrid,
//                     Real* xdis, Real* ydis, Real* zdis,
//                     int* indx, int* indy, int* indz,
//                     Real sigmadipsq, Real anorm, Real anorm2, Real dx, Real nx, Real ny, Real nz);

// __global__
// void cufcm_mono_dipole_distribution_tpp_register(myCufftReal *fx, myCufftReal *fy, myCufftReal *fz, int N,
//               Real *T, Real *F, Real pdmag, Real sigmasq, 
//               Real *gaussx, Real *gaussy, Real *gaussz,
//               Real *grad_gaussx_dip, Real *grad_gaussy_dip, Real *grad_gaussz_dip,
//               Real *xdis, Real *ydis, Real *zdis,
//               int *indx, int *indy, int *indz,
//               int ngd, Real nx, Real ny, Real nz);

// __global__
// void cufcm_mono_dipole_distribution_tpp_recompute(myCufftReal *fx, myCufftReal *fy, myCufftReal *fz,
//               Real *Y, Real *T, Real *F,
//               int N, int ngd, 
//               Real pdmag, Real sigmasq, Real sigmadipsq,
//               Real anorm, Real anorm2,
//               Real dx, Real nx, Real ny, Real nz);

// __global__
// void cufcm_mono_dipole_distribution_bpp_shared(myCufftReal *fx, myCufftReal *fy, myCufftReal *fz,
//               Real *Y, Real *T, Real *F,
//               int N, int ngd, 
//               Real pdmag, Real sigmasq, Real sigmadipsq,
//               Real anorm, Real anorm2,
//               Real dx, Real nx, Real ny, Real nz);

// __global__
// void cufcm_mono_dipole_distribution_bpp_recompute(myCufftReal *fx, myCufftReal *fy, myCufftReal *fz,
//               Real *Y, Real *T, Real *F,
//               int N, int ngd, 
//               Real sigma, Real Sigma,
//               Real dx, Real nx, Real ny, Real nz);

__global__
void cufcm_mono_dipole_distribution_bpp_shared_dynamic(myCufftReal *fx, myCufftReal *fy, myCufftReal *fz,
              Real *Y, Real *T, Real *F,
              int N, int ngd, 
              Real sigma, Real sigmadip, Real Sigma,
              Real dx, int nx, int ny, int nz,
              int *particle_index, int start, int end,
              int rotation);

// __global__
// void cufcm_mono_dipole_distribution_selection(myCufftReal *fx, myCufftReal *fy, myCufftReal *fz, Real *Y,
//               Real *T, Real *F, int N, int ngd, 
//               Real sigma, Real Sigma,
//               Real dx, double nx, double ny, double nz,
//               int *particle_index, int start, int end);

// __global__
// void cufcm_mono_dipole_distribution_mono(myCufftReal *fx, myCufftReal *fy, myCufftReal *fz,
//               Real *Y, Real *F,
//               int N, int ngd, 
//               Real sigma, Real Sigma,
//               Real dx, double nx, double ny, double nz);

// __global__
// void cufcm_mono_dipole_distribution_mono_selection(myCufftReal *fx, myCufftReal *fy, myCufftReal *fz,
//               Real *Y, Real *F,
//               int N, int ngd, 
//               Real sigma, Real Sigma,
//               Real dx, double nx, double ny, double nz,
//               int *particle_index, int start, int end);
    

__global__
void cufcm_flow_solve(myCufftComplex* fk_x, myCufftComplex* fk_y, myCufftComplex* fk_z,
                      myCufftComplex* uk_x, myCufftComplex* uk_y, myCufftComplex* uk_z,
                      int nx, int ny, int nz, Real Lx_reduced, Real Ly_reduced, Real Lz_reduced);

// __global__
// void cufcm_particle_velocities_tpp_register(myCufftReal *ux, myCufftReal *uy, myCufftReal *uz, int N,
//                                Real *VTEMP, Real *WTEMP,
//                                Real pdmag, Real sigmasq, 
//                                Real *gaussx, Real *gaussy, Real *gaussz,
//                                Real *grad_gaussx_dip, Real *grad_gaussy_dip, Real *grad_gaussz_dip,
//                                Real *xdis, Real *ydis, Real *zdis,
//                                int *indx, int *indy, int *indz,
//                                int ngd, Real dx, Real nx, Real ny, Real nz);

// __global__
// void cufcm_particle_velocities_tpp_recompute(myCufftReal *ux, myCufftReal *uy, myCufftReal *uz,
//                                 Real *Y,
//                                 Real *VTEMP, Real *WTEMP,
//                                 int N, int ngd, 
//                                 Real pdmag, Real sigmasq, Real sigmadipsq,
//                                 Real anorm, Real anorm2,
//                                 Real dx, Real nx, Real ny, Real nz);

// __global__
// void cufcm_particle_velocities_bpp_shared(myCufftReal *ux, myCufftReal *uy, myCufftReal *uz,
//                                 Real *Y,
//                                 Real *VTEMP, Real *WTEMP,
//                                 int N, int ngd, 
//                                 Real pdmag, Real sigmasq, Real sigmadipsq,
//                                 Real anorm, Real anorm2,
//                                 Real dx, Real nx, Real ny, Real nz);

// __global__
// void cufcm_particle_velocities_bpp_recompute(myCufftReal *ux, myCufftReal *uy, myCufftReal *uz,
//                                 Real *Y,
//                                 Real *VTEMP, Real *WTEMP,
//                                 int N, int ngd, 
//                                 Real sigma, Real Sigma,
//                                 Real dx, Real nx, Real ny, Real nz);

__global__
void cufcm_particle_velocities_bpp_shared_dynamic(myCufftReal *ux, myCufftReal *uy, myCufftReal *uz,
                                Real *Y,
                                Real *VTEMP, Real *WTEMP,
                                int N, int ngd, 
                                Real sigma, Real sigmadip, Real Sigma,
                                Real dx, int nx, int ny, int nz,
                                int *particle_index, int start, int end,
                                int rotation);

// __global__
// void cufcm_particle_velocities_selection(myCufftReal *ux, myCufftReal *uy, myCufftReal *uz,
//                                 Real *Y, Real *VTEMP, Real *WTEMP,
//                                 int N, int ngd, 
//                                 Real sigma, Real Sigma,
//                                 Real dx, Real nx, Real ny, Real nz, 
//                                 int *particle_index, int start, int end);

// __global__
// void cufcm_particle_velocities_mono(myCufftReal *ux, myCufftReal *uy, myCufftReal *uz,
//                                 Real *Y,
//                                 Real *VTEMP,
//                                 int N, int ngd, 
//                                 Real sigma, Real Sigma,
//                                 Real dx, Real nx, Real ny, Real nz);

// __global__
// void cufcm_particle_velocities_mono_selection(myCufftReal *ux, myCufftReal *uy, myCufftReal *uz,
//                                 Real *Y,
//                                 Real *VTEMP,
//                                 int N, int ngd, 
//                                 Real sigma, Real Sigma,
//                                 Real dx, Real nx, Real ny, Real nz,
//                                 int *particle_index, int start, int end);
