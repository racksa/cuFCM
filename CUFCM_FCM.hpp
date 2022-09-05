#pragma once



__global__
void cufcm_test_force(cufftDoubleReal* fx, cufftDoubleReal* fy, cufftDoubleReal* fz);

__global__
void cufcm_precompute_gauss(int N, int ngd, double* Y,
                    double* gaussx_, double* gaussy, double* gaussz,
                    double* grad_gaussx_dip, double* grad_gaussy_dip, double* grad_gaussz_dip,
                    double* gaussgrid,
                    double* xdis, double* ydis, double* zdis,
                    int* indx, int* indy, int* indz,
                    double sigmadipsq, double anorm, double anorm2, double dx);

__global__
void GA_setup(double *GA, double *T, int N);

__global__
void cufcm_mono_dipole_distribution_tpp_register(cufftDoubleReal *fx, cufftDoubleReal *fy, cufftDoubleReal *fz, int N,
              double *GA, double *F, double pdmag, double sigmasq, 
              double *gaussx, double *gaussy, double *gaussz,
              double *grad_gaussx_dip, double *grad_gaussy_dip, double *grad_gaussz_dip,
              double *xdis, double *ydis, double *zdis,
              int *indx, int *indy, int *indz,
              int ngd);

__global__
void cufcm_mono_dipole_distribution_tpp_recompute(cufftDoubleReal *fx, cufftDoubleReal *fy, cufftDoubleReal *fz,
              double *Y, double *GA, double *F,
              int N, int ngd, 
              double pdmag, double sigmasq, double sigmadipsq,
              double anorm, double anorm2,
              double dx);

__global__
void cufcm_mono_dipole_distribution_bpp(cufftDoubleReal *fx, cufftDoubleReal *fy, cufftDoubleReal *fz, double *Y,
              double *GA, double *F, int N, int ngd, 
              double pdmag, double sigmasq, double sigmadipsq,
              double anorm, double anorm2,
              double dx);

__global__
void cufcm_flow_solve(cufftDoubleComplex* fk_x, cufftDoubleComplex* fk_y, cufftDoubleComplex* fk_z,
                      cufftDoubleComplex* uk_x, cufftDoubleComplex* uk_y, cufftDoubleComplex* uk_z,
                      double* q, double* qpad, double* qsq, double* qpadsq);

__global__
void normalise_array(cufftDoubleReal* ux, cufftDoubleReal* uy, cufftDoubleReal* uz);

__global__
void cufcm_particle_velocities_tpp_register(cufftDoubleReal *ux, cufftDoubleReal *uy, cufftDoubleReal *uz, int N,
                               double *VTEMP, double *WTEMP,
                               double pdmag, double sigmasq, 
                               double *gaussx, double *gaussy, double *gaussz,
                               double *grad_gaussx_dip, double *grad_gaussy_dip, double *grad_gaussz_dip,
                               double *xdis, double *ydis, double *zdis,
                               int *indx, int *indy, int *indz,
                               int ngd, double dx);

__global__
void cufcm_particle_velocities_tpp_recompute(cufftDoubleReal *ux, cufftDoubleReal *uy, cufftDoubleReal *uz,
                                double *Y,
                                double *VTEMP, double *WTEMP,
                                int N, int ngd, 
                                double pdmag, double sigmasq, double sigmadipsq,
                                double anorm, double anorm2,
                                double dx);

__global__
void cufcm_particle_velocities_bpp(cufftDoubleReal *ux, cufftDoubleReal *uy, cufftDoubleReal *uz,
                                double *Y,
                                double *VTEMP, double *WTEMP,
                                int N, int ngd, 
                                double pdmag, double sigmasq, double sigmadipsq,
                                double anorm, double anorm2,
                                double dx);

void cufcm_test_force_loop(cufftDoubleReal* fx, cufftDoubleReal* fy, cufftDoubleReal* fz);

void cufcm_precompute_gauss_loop(int N, int ngd, double* Y,
                    double* gaussx_, double* gaussy, double* gaussz,
                    double* grad_gaussx_dip, double* grad_gaussy_dip, double* grad_gaussz_dip,
                    double* gaussgrid,
                    double* xdis, double* ydis, double* zdis,
                    int* indx, int* indy, int* indz,
                    double sigmadipsq, double anorm, double anorm2, double dx);

void GA_setup_loop(double *GA, double *T, int N);

void cufcm_mono_dipole_distribution_tpp_loop(cufftDoubleReal *fx, cufftDoubleReal *fy, cufftDoubleReal *fz, int N,
              double *GA, double *F, double pdmag, double sigmasq, 
              double *gaussx, double *gaussy, double *gaussz,
              double *grad_gaussx_dip, double *grad_gaussy_dip, double *grad_gaussz_dip,
              double *xdis, double *ydis, double *zdis,
              int *indx, int *indy, int *indz,
              int ngd);

void cufcm_particle_velocities_loop(cufftDoubleReal *ux, cufftDoubleReal *uy, cufftDoubleReal *uz, int N,
                               double *VTEMP, double *WTEMP,
                               double pdmag, double sigmasq, 
                               double *gaussx, double *gaussy, double *gaussz,
                               double *grad_gaussx_dip, double *grad_gaussy_dip, double *grad_gaussz_dip,
                               double *xdis, double *ydis, double *zdis,
                               int *indx, int *indy, int *indz,
                               int ngd, double dx);

__device__ __host__
double int_pow(double base, int power);