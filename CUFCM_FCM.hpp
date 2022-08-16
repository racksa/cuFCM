#pragma once



__global__
void cufcm_test_force(cufftReal* fx, cufftReal* fy, cufftReal* fz);

__global__
void cufcm_gaussian_setup(int N, int ngd, double* Y,
                    double* gaussx_, double* gaussy, double* gaussz,
                    double* grad_gaussx_dip, double* grad_gaussy_dip, double* grad_gaussz_dip,
                    double* gaussgrid,
                    double* xdis, double* ydis, double* zdis,
                    int* indx, int* indy, int* indz,
                    double sigmadipsq, double anorm, double anorm2, double dx);

__global__
void GA_setup(double *GA, double *T, int N);

__global__
void cufcm_mono_dipole_distribution(cufftReal *fx, cufftReal *fy, cufftReal *fz, int N,
              double *GA, double *F, double pdmag, double sigmasq, 
              double *gaussx, double *gaussy, double *gaussz,
              double *grad_gaussx_dip, double *grad_gaussy_dip, double *grad_gaussz_dip,
              double *xdis, double *ydis, double *zdis,
              int *indx, int *indy, int *indz,
              int ngd);

__global__
void cufcm_flow_solve(cufftComplex* fk_x, cufftComplex* fk_y, cufftComplex* fk_z,
                      cufftComplex* uk_x, cufftComplex* uk_y, cufftComplex* uk_z,
                      double* q, double* qpad, double* qsq, double* qpadsq);

__global__
void normalise_array(cufftReal* ux, cufftReal* uy, cufftReal* uz);

__global__
void cufcm_particle_velocities(cufftReal *ux, cufftReal *uy, cufftReal *uz, int N,
                               double *VTEMP, double *WTEMP,
                               double pdmag, double sigmasq, 
                               double *gaussx, double *gaussy, double *gaussz,
                               double *grad_gaussx_dip, double *grad_gaussy_dip, double *grad_gaussz_dip,
                               double *xdis, double *ydis, double *zdis,
                               int *indx, int *indy, int *indz,
                               int ngd, double dx);

void cufcm_test_force_loop(cufftReal* fx, cufftReal* fy, cufftReal* fz);

void cufcm_gaussian_setup_loop(int N, int ngd, double* Y,
                    double* gaussx_, double* gaussy, double* gaussz,
                    double* grad_gaussx_dip, double* grad_gaussy_dip, double* grad_gaussz_dip,
                    double* gaussgrid,
                    double* xdis, double* ydis, double* zdis,
                    int* indx, int* indy, int* indz,
                    double sigmadipsq, double anorm, double anorm2, double dx);

void GA_setup_loop(double *GA, double *T, int N);

void cufcm_mono_dipole_distribution_loop(cufftReal *fx, cufftReal *fy, cufftReal *fz, int N,
              double *GA, double *F, double pdmag, double sigmasq, 
              double *gaussx, double *gaussy, double *gaussz,
              double *grad_gaussx_dip, double *grad_gaussy_dip, double *grad_gaussz_dip,
              double *xdis, double *ydis, double *zdis,
              int *indx, int *indy, int *indz,
              int ngd);

void cufcm_particle_velocities_loop(cufftReal *ux, cufftReal *uy, cufftReal *uz, int N,
                               double *VTEMP, double *WTEMP,
                               double pdmag, double sigmasq, 
                               double *gaussx, double *gaussy, double *gaussz,
                               double *grad_gaussx_dip, double *grad_gaussy_dip, double *grad_gaussz_dip,
                               double *xdis, double *ydis, double *zdis,
                               int *indx, int *indy, int *indz,
                               int ngd, double dx);


__device__ __host__
double int_pow(double base, int power);