#include <cmath>
#include <iostream>
// Include CUDA runtime and CUFFT
#include <cufft.h>

#include "config.hpp"
#include "CUFCM_FCM.cuh"
#include <cub/cub.cuh>


///////////////////////////////////////////////////////////////////////////////
// Fast FCM
///////////////////////////////////////////////////////////////////////////////

__global__
void cufcm_mono_dipole_distribution_bpp_shared_dynamic(myCufftReal *fx, myCufftReal *fy, myCufftReal *fz,
              Real *Y, Real *T, Real *F,
              int N, int ngd, 
              Real sigma, Real sigmadip, Real Sigma,
              Real dx, double nx, double ny, double nz){
    
    int ngdh = ngd/2;

    extern __shared__ Integer s[];
    Integer *indx_shared = s;
    Integer *indy_shared = (Integer*)&indx_shared[ngd];
    Integer *indz_shared = (Integer*)&indy_shared[ngd];
    #if SOLVER_MODE == 0
        Real *gaussx_shared = (Real*)&indz_shared[ngd]; 
        Real *gaussy_shared = (Real*)&gaussx_shared[ngd];
        Real *gaussz_shared = (Real*)&gaussy_shared[ngd];
        Real *gaussx_dip_shared = (Real*)&gaussz_shared[ngd]; 
        Real *gaussy_dip_shared = (Real*)&gaussx_dip_shared[ngd];
        Real *gaussz_dip_shared = (Real*)&gaussy_dip_shared[ngd];
        Real *grad_gaussx_dip_shared = (Real*)&gaussz_dip_shared[ngd];
    #elif SOLVER_MODE == 1
        Real *xdis_shared = (Real*)&indz_shared[ngd];    
        Real *ydis_shared = (Real*)&xdis_shared[ngd];
        Real *zdis_shared = (Real*)&ydis_shared[ngd];
        Real *gaussx_shared = (Real*)&zdis_shared[ngd]; 
        Real *gaussy_shared = (Real*)&gaussx_shared[ngd];
        Real *gaussz_shared = (Real*)&gaussy_shared[ngd];
        Real *grad_gaussx_dip_shared = (Real*)&gaussz_shared[ngd];
    #endif
    Real *grad_gaussy_dip_shared = (Real*)&grad_gaussx_dip_shared[ngd];
    Real *grad_gaussz_dip_shared = (Real*)&grad_gaussy_dip_shared[ngd];
    Real *Y_shared = (Real*)&grad_gaussz_dip_shared[ngd];
    Real *F_shared = (Real*)&Y_shared[3];
    Real *g_shared = (Real*)&F_shared[3];

    
    #if SOLVER_MODE == 0
        Real sigmasq = sigma*sigma;
        Real sigmadipsq = sigmadip*sigmadip;
        Real Sigmadipsq = sigmadipsq;
        Real anorm = Real(1.0)/my_sqrt(Real(2.0)*Real(PI)*sigmasq);
        Real anormdip = Real(1.0)/my_sqrt(Real(2.0)*Real(PI)*sigmadipsq);
    #elif SOLVER_MODE == 1
        Real Sigmasq = Sigma*Sigma;
        Real Sigmadipsq = Sigmasq;
        Real Anorm = Real(1.0)/my_sqrt(Real(PI2)*Sigmasq);
        Real width2 = (Real(2.0)*Sigmasq);
        Real pdmag = sigma*sigma - Sigmasq;
    #endif
    

    for(int np = blockIdx.x; np < N; np += gridDim.x){

        if(threadIdx.x == 0){
            Y_shared[0] = Y[3*np + 0];
            Y_shared[1] = Y[3*np + 1];
            Y_shared[2] = Y[3*np + 2];

            F_shared[0] = F[3*np + 0];
            F_shared[1] = F[3*np + 1];
            F_shared[2] = F[3*np + 2];

            // anti-symmetric G_lk = 0.5*epsilon_lkp*T_p
            g_shared[0] = + Real(0.0);
            g_shared[1] = + Real(0.0);
            g_shared[2] = + Real(0.0);
            g_shared[3] = + Real(0.5)*T[3*np + 2];
            g_shared[4] = - Real(0.5)*T[3*np + 2];
            g_shared[5] = + Real(-0.5)*T[3*np + 1];
            g_shared[6] = - Real(-0.5)*T[3*np + 1];
            g_shared[7] = + Real(0.5)*T[3*np + 0];
            g_shared[8] = - Real(0.5)*T[3*np + 0];
        }
        __syncthreads();

        for(int i = threadIdx.x; i < 4*ngd; i += blockDim.x){
            Real xg = my_rint(Y_shared[0]/dx) - ngdh + my_fmod(Real(i), Real(ngd));
            Real yg = my_rint(Y_shared[1]/dx) - ngdh + my_fmod(Real(i), Real(ngd));
            Real zg = my_rint(Y_shared[2]/dx) - ngdh + my_fmod(Real(i), Real(ngd));

            Real xx = xg*dx - Y_shared[0];
            Real yy = yg*dx - Y_shared[1];
            Real zz = zg*dx - Y_shared[2];
            /* dis */
            if(i<ngd){
                #if SOLVER_MODE == 0
                    gaussx_shared[i] = anorm*my_exp(-xx*xx/(Real(2.0)*sigmasq));
                    gaussy_shared[i] = anorm*my_exp(-yy*yy/(Real(2.0)*sigmasq));
                    gaussz_shared[i] = anorm*my_exp(-zz*zz/(Real(2.0)*sigmasq));
                #elif SOLVER_MODE == 1
                    xdis_shared[i] = xx;
                    ydis_shared[i] = yy;
                    zdis_shared[i] = zz;
                #endif
            }
            /* gauss */
            if(i>=ngd && i<2*ngd){
                #if SOLVER_MODE == 0
                    gaussx_dip_shared[i-ngd] = anormdip*my_exp(-xx*xx/(Real(2.0)*sigmadipsq));
                    gaussy_dip_shared[i-ngd] = anormdip*my_exp(-yy*yy/(Real(2.0)*sigmadipsq));
                    gaussz_dip_shared[i-ngd] = anormdip*my_exp(-zz*zz/(Real(2.0)*sigmadipsq));
                #elif SOLVER_MODE == 1
                    gaussx_shared[i-ngd] = Anorm*my_exp(-xx*xx/width2);
                    gaussy_shared[i-ngd] = Anorm*my_exp(-yy*yy/width2);
                    gaussz_shared[i-ngd] = Anorm*my_exp(-zz*zz/width2);
                #endif
            }
            /* grad_gauss */
            if(i>=2*ngd && i<3*ngd){
                grad_gaussx_dip_shared[i-2*ngd] = - xx / Sigmadipsq;
                grad_gaussy_dip_shared[i-2*ngd] = - yy / Sigmadipsq;
                grad_gaussz_dip_shared[i-2*ngd] = - zz / Sigmadipsq;
            }
            /* ind */
            if(i>=3*ngd){
                indx_shared[i-3*ngd] = xg - nx * my_floor( xg / nx );
                indy_shared[i-3*ngd] = yg - ny * my_floor( yg / ny );
                indz_shared[i-3*ngd] = zg - nz * my_floor( zg / nz );
            }
        }
        __syncthreads();
        
        for(int t = threadIdx.x; t < ngd*ngd*ngd; t += blockDim.x){
            const int k = t/(ngd*ngd);
            const int j = (t - k*ngd*ngd)/ngd;
            const int i = t - k*ngd*ngd - j*ngd;

            Real gradx = grad_gaussx_dip_shared[i];
            Real grady = grad_gaussy_dip_shared[j];
            Real gradz = grad_gaussz_dip_shared[k];

            int ind = indx_shared[i] + indy_shared[j]*nx + indz_shared[k]*nx*ny;

            #if SOLVER_MODE == 0
                Real temp = gaussx_shared[i]*gaussy_shared[j]*gaussz_shared[k];
                Real tempdip = gaussx_dip_shared[i]*gaussy_dip_shared[j]*gaussz_dip_shared[k];
            #elif SOLVER_MODE == 1
                Real r2 = xdis_shared[i]*xdis_shared[i] + ydis_shared[j]*ydis_shared[j] + zdis_shared[k]*zdis_shared[k];
                Real temp1 = gaussx_shared[i]*gaussy_shared[j]*gaussz_shared[k];
                Real temp2 = Real(0.5) * pdmag / Sigmasq;
                Real temp3 = temp2 / Sigmasq;
                Real temp4 = Real(3.0)*temp2;
                Real temp = temp1*( Real(1.0) + temp3*r2 - temp4);
                Real tempdip = temp1;
            #endif

            atomicAdd(&fx[ind], F_shared[0]*temp + (g_shared[0]*gradx + g_shared[3]*grady + g_shared[5]*gradz)*tempdip);
            atomicAdd(&fy[ind], F_shared[1]*temp + (g_shared[4]*gradx + g_shared[1]*grady + g_shared[7]*gradz)*tempdip);
            atomicAdd(&fz[ind], F_shared[2]*temp + (g_shared[6]*gradx + g_shared[8]*grady + g_shared[2]*gradz)*tempdip);
        }
    }
}


// __global__
// void cufcm_mono_dipole_distribution_selection(myCufftReal *fx, myCufftReal *fy, myCufftReal *fz, Real *Y,
//               Real *T, Real *F, int N, int ngd, 
//               Real sigma, Real Sigma,
//               Real dx, double nx, double ny, double nz,
//               int *particle_index, int start, int end){
    
//     // TODO: GPU is more comfortable computing FP2 (double) than integer
//     int ngdh = ngd/2;

//     extern __shared__ Integer s[];
//     Integer *indx_shared = s;
//     Integer *indy_shared = (Integer*)&indx_shared[ngd];
//     Integer *indz_shared = (Integer*)&indy_shared[ngd];
//     Real *xdis_shared = (Real*)&indz_shared[ngd];    
//     Real *ydis_shared = (Real*)&xdis_shared[ngd];
//     Real *zdis_shared = (Real*)&ydis_shared[ngd];
//     Real *gaussx_shared = (Real*)&zdis_shared[ngd]; 
//     Real *gaussy_shared = (Real*)&gaussx_shared[ngd];
//     Real *gaussz_shared = (Real*)&gaussy_shared[ngd];
//     Real *grad_gaussx_dip_shared = (Real*)&gaussz_shared[ngd];
//     Real *grad_gaussy_dip_shared = (Real*)&grad_gaussx_dip_shared[ngd];
//     Real *grad_gaussz_dip_shared = (Real*)&grad_gaussy_dip_shared[ngd];
//     Real *Y_shared = (Real*)&grad_gaussz_dip_shared[ngd];
//     Real *F_shared = (Real*)&Y_shared[3];
//     Real *g_shared = (Real*)&F_shared[3];

//     Real Sigmasq = Sigma*Sigma;
//     Real Sigmadipsq = Sigmasq;
//     Real Anorm = Real(1.0)/my_sqrt(Real(PI2)*Sigmasq);
//     Real width2 = (Real(2.0)*Sigmasq);
//     Real pdmag = sigma*sigma - Sigmasq;

//     for(int np = blockIdx.x; (np < N) && (particle_index[np] >= start && particle_index[np] < end); np += gridDim.x){

//         if(threadIdx.x == 0){
//             Y_shared[0] = Y[3*np + 0];
//             Y_shared[1] = Y[3*np + 1];
//             Y_shared[2] = Y[3*np + 2];

//             F_shared[0] = F[3*np + 0];
//             F_shared[1] = F[3*np + 1];
//             F_shared[2] = F[3*np + 2];

//             // anti-symmetric G_lk = 0.5*epsilon_lkp*T_p
//             g_shared[0] = + Real(0.0);
//             g_shared[1] = + Real(0.0);
//             g_shared[2] = + Real(0.0);
//             g_shared[3] = + Real(0.5)*T[3*np + 2];
//             g_shared[4] = - Real(0.5)*T[3*np + 2];
//             g_shared[5] = + Real(-0.5)*T[3*np + 1];
//             g_shared[6] = - Real(-0.5)*T[3*np + 1];
//             g_shared[7] = + Real(0.5)*T[3*np + 0];
//             g_shared[8] = - Real(0.5)*T[3*np + 0];
//         }
//         __syncthreads();

//         for(int i = threadIdx.x; i < 4*ngd; i += blockDim.x){
//             Real xg = my_rint(Y_shared[0]/dx) - ngdh + my_fmod(Real(i), Real(ngd));
//             Real yg = my_rint(Y_shared[1]/dx) - ngdh + my_fmod(Real(i), Real(ngd));
//             Real zg = my_rint(Y_shared[2]/dx) - ngdh + my_fmod(Real(i), Real(ngd));

//             Real xx = xg*dx - Y_shared[0];
//             Real yy = yg*dx - Y_shared[1];
//             Real zz = zg*dx - Y_shared[2];
//             /* dis */
//             if(i<ngd){ 
//                 xdis_shared[i] = xx;
//                 ydis_shared[i] = yy;
//                 zdis_shared[i] = zz;
//             }
//             /* gauss */
//             if(i>=ngd && i<2*ngd){
//                 gaussx_shared[i-ngd] = Anorm*my_exp(-xx*xx/width2);
//                 gaussy_shared[i-ngd] = Anorm*my_exp(-yy*yy/width2);
//                 gaussz_shared[i-ngd] = Anorm*my_exp(-zz*zz/width2);
//             }
//             /* grad_gauss */
//             if(i>=2*ngd && i<3*ngd){
//                 grad_gaussx_dip_shared[i-2*ngd] = - xx / Sigmadipsq;
//                 grad_gaussy_dip_shared[i-2*ngd] = - yy / Sigmadipsq;
//                 grad_gaussz_dip_shared[i-2*ngd] = - zz / Sigmadipsq;
//             }
//             /* ind */
//             if(i>=3*ngd){
//                 indx_shared[i-3*ngd] = xg - nx * my_floor( xg / nx );
//                 indy_shared[i-3*ngd] = yg - ny * my_floor( yg / ny );
//                 indz_shared[i-3*ngd] = zg - nz * my_floor( zg / nz );
//             }
//         }
//         __syncthreads();
        
//         for(int t = threadIdx.x; t < ngd*ngd*ngd; t += blockDim.x){
//             const int k = t/(ngd*ngd);
//             const int j = (t - k*ngd*ngd)/ngd;
//             const int i = t - k*ngd*ngd - j*ngd;

//             Real gradx = grad_gaussx_dip_shared[i];
//             Real grady = grad_gaussy_dip_shared[j];
//             Real gradz = grad_gaussz_dip_shared[k];

//             int ind = indx_shared[i] + indy_shared[j]*nx + indz_shared[k]*nx*ny;
//             Real r2 = xdis_shared[i]*xdis_shared[i] + ydis_shared[j]*ydis_shared[j] + zdis_shared[k]*zdis_shared[k];
//             Real temp = gaussx_shared[i]*gaussy_shared[j]*gaussz_shared[k];
//             Real temp2 = Real(0.5) * pdmag / Sigmasq;
//             Real temp3 = temp2 / Sigmasq;
//             Real temp4 = Real(3.0)*temp2;
//             Real temp5 = temp*( Real(1.0) + temp3*r2 - temp4);

//             atomicAdd(&fx[ind], F_shared[0]*temp5 + (g_shared[0]*gradx + g_shared[3]*grady + g_shared[5]*gradz)*temp);
//             atomicAdd(&fy[ind], F_shared[1]*temp5 + (g_shared[4]*gradx + g_shared[1]*grady + g_shared[7]*gradz)*temp);
//             atomicAdd(&fz[ind], F_shared[2]*temp5 + (g_shared[6]*gradx + g_shared[8]*grady + g_shared[2]*gradz)*temp);

//         }
//     }
// }

__global__
void cufcm_mono_dipole_distribution_mono(myCufftReal *fx, myCufftReal *fy, myCufftReal *fz,
              Real *Y, Real *F,
              int N, int ngd, 
              Real sigma, Real Sigma,
              Real dx, double nx, double ny, double nz){
    int ngdh = ngd/2;

    extern __shared__ Integer s[];
    Integer *indx_shared = s;
    Integer *indy_shared = (Integer*)&indx_shared[ngd];
    Integer *indz_shared = (Integer*)&indy_shared[ngd];
    Real *xdis_shared = (Real*)&indz_shared[ngd];    
    Real *ydis_shared = (Real*)&xdis_shared[ngd];
    Real *zdis_shared = (Real*)&ydis_shared[ngd];
    Real *gaussx_shared = (Real*)&zdis_shared[ngd]; 
    Real *gaussy_shared = (Real*)&gaussx_shared[ngd];
    Real *gaussz_shared = (Real*)&gaussy_shared[ngd];
    Real *Y_shared = (Real*)&gaussz_shared[ngd];
    Real *F_shared = (Real*)&Y_shared[3];

    Real Sigmasq = Sigma*Sigma;
    Real Anorm = Real(1.0)/my_sqrt(Real(PI2)*Sigmasq);
    Real width2 = (Real(2.0)*Sigmasq);
    Real pdmag = sigma*sigma - Sigmasq;

    for(int np = blockIdx.x; np < N; np += gridDim.x){

        if(threadIdx.x == 0){
            Y_shared[0] = Y[3*np + 0];
            Y_shared[1] = Y[3*np + 1];
            Y_shared[2] = Y[3*np + 2];

            F_shared[0] = F[3*np + 0];
            F_shared[1] = F[3*np + 1];
            F_shared[2] = F[3*np + 2];
        }
        __syncthreads();

        for(int i = threadIdx.x; i < 4*ngd; i += blockDim.x){
            Real xg = my_rint(Y_shared[0]/dx) - ngdh + my_fmod(Real(i), Real(ngd));
            Real yg = my_rint(Y_shared[1]/dx) - ngdh + my_fmod(Real(i), Real(ngd));
            Real zg = my_rint(Y_shared[2]/dx) - ngdh + my_fmod(Real(i), Real(ngd));

            Real xx = xg*dx - Y_shared[0];
            Real yy = yg*dx - Y_shared[1];
            Real zz = zg*dx - Y_shared[2];
            /* dis */
            if(i<ngd){ 
                xdis_shared[i] = xx;
                ydis_shared[i] = yy;
                zdis_shared[i] = zz;
            }
            /* gauss */
            if(i>=ngd && i<2*ngd){
                gaussx_shared[i-ngd] = Anorm*my_exp(-xx*xx/width2);
                gaussy_shared[i-ngd] = Anorm*my_exp(-yy*yy/width2);
                gaussz_shared[i-ngd] = Anorm*my_exp(-zz*zz/width2);
            }
            /* grad_gauss */
            if(i>=2*ngd && i<3*ngd){
            }
            /* ind */
            if(i>=3*ngd){
                indx_shared[i-3*ngd] = xg - nx * my_floor( xg / nx );
                indy_shared[i-3*ngd] = yg - ny * my_floor( yg / ny );
                indz_shared[i-3*ngd] = zg - nz * my_floor( zg / nz );
            }
        }
        __syncthreads();
        
        for(int t = threadIdx.x; t < ngd*ngd*ngd; t += blockDim.x){
            const int k = t/(ngd*ngd);
            const int j = (t - k*ngd*ngd)/ngd;
            const int i = t - k*ngd*ngd - j*ngd;

            int ind = indx_shared[i] + indy_shared[j]*nx + indz_shared[k]*nx*ny;
            Real r2 = xdis_shared[i]*xdis_shared[i] + ydis_shared[j]*ydis_shared[j] + zdis_shared[k]*zdis_shared[k];
            Real temp = gaussx_shared[i]*gaussy_shared[j]*gaussz_shared[k];
            Real temp2 = Real(0.5) * pdmag / Sigmasq;
            Real temp3 = temp2 / Sigmasq;
            Real temp4 = Real(3.0)*temp2;
            Real temp5 = temp*( Real(1.0) + temp3*r2 - temp4);

            atomicAdd(&fx[ind], F_shared[0]*temp5);
            atomicAdd(&fy[ind], F_shared[1]*temp5);
            atomicAdd(&fz[ind], F_shared[2]*temp5);

        }
    }
}

// __global__
// void cufcm_mono_dipole_distribution_mono_selection(myCufftReal *fx, myCufftReal *fy, myCufftReal *fz,
//               Real *Y, Real *F,
//               int N, int ngd, 
//               Real sigma, Real Sigma,
//               Real dx, double nx, double ny, double nz,
//               int *particle_index, int start, int end){
//     int ngdh = ngd/2;

//     extern __shared__ Integer s[];
//     Integer *indx_shared = s;
//     Integer *indy_shared = (Integer*)&indx_shared[ngd];
//     Integer *indz_shared = (Integer*)&indy_shared[ngd];
//     Real *xdis_shared = (Real*)&indz_shared[ngd];    
//     Real *ydis_shared = (Real*)&xdis_shared[ngd];
//     Real *zdis_shared = (Real*)&ydis_shared[ngd];
//     Real *gaussx_shared = (Real*)&zdis_shared[ngd]; 
//     Real *gaussy_shared = (Real*)&gaussx_shared[ngd];
//     Real *gaussz_shared = (Real*)&gaussy_shared[ngd];
//     Real *Y_shared = (Real*)&gaussz_shared[ngd];
//     Real *F_shared = (Real*)&Y_shared[3];

//     Real Sigmasq = Sigma*Sigma;
//     Real Anorm = Real(1.0)/my_sqrt(Real(PI2)*Sigmasq);
//     Real width2 = (Real(2.0)*Sigmasq);
//     Real pdmag = sigma*sigma - Sigmasq;

//     for(int np = blockIdx.x; (np < N) && (particle_index[np] >= start && particle_index[np] < end); np += gridDim.x){

//         if(threadIdx.x == 0){
//             Y_shared[0] = Y[3*np + 0];
//             Y_shared[1] = Y[3*np + 1];
//             Y_shared[2] = Y[3*np + 2];

//             F_shared[0] = F[3*np + 0];
//             F_shared[1] = F[3*np + 1];
//             F_shared[2] = F[3*np + 2];
//         }
//         __syncthreads();

//         for(int i = threadIdx.x; i < 4*ngd; i += blockDim.x){
//             Real xg = my_rint(Y_shared[0]/dx) - ngdh + my_fmod(Real(i), Real(ngd));
//             Real yg = my_rint(Y_shared[1]/dx) - ngdh + my_fmod(Real(i), Real(ngd));
//             Real zg = my_rint(Y_shared[2]/dx) - ngdh + my_fmod(Real(i), Real(ngd));

//             Real xx = xg*dx - Y_shared[0];
//             Real yy = yg*dx - Y_shared[1];
//             Real zz = zg*dx - Y_shared[2];
//             /* dis */
//             if(i<ngd){ 
//                 xdis_shared[i] = xx;
//                 ydis_shared[i] = yy;
//                 zdis_shared[i] = zz;
//             }
//             /* gauss */
//             if(i>=ngd && i<2*ngd){
//                 gaussx_shared[i-ngd] = Anorm*my_exp(-xx*xx/width2);
//                 gaussy_shared[i-ngd] = Anorm*my_exp(-yy*yy/width2);
//                 gaussz_shared[i-ngd] = Anorm*my_exp(-zz*zz/width2);
//             }
//             /* grad_gauss */
//             if(i>=2*ngd && i<3*ngd){
//             }
//             /* ind */
//             if(i>=3*ngd){
//                 indx_shared[i-3*ngd] = xg - nx * my_floor( xg / nx );
//                 indy_shared[i-3*ngd] = yg - ny * my_floor( yg / ny );
//                 indz_shared[i-3*ngd] = zg - nz * my_floor( zg / nz );
//             }
//         }
//         __syncthreads();
        
//         for(int t = threadIdx.x; t < ngd*ngd*ngd; t += blockDim.x){
//             const int k = t/(ngd*ngd);
//             const int j = (t - k*ngd*ngd)/ngd;
//             const int i = t - k*ngd*ngd - j*ngd;

//             int ind = indx_shared[i] + indy_shared[j]*nx + indz_shared[k]*nx*ny;
//             Real r2 = xdis_shared[i]*xdis_shared[i] + ydis_shared[j]*ydis_shared[j] + zdis_shared[k]*zdis_shared[k];
//             Real temp = gaussx_shared[i]*gaussy_shared[j]*gaussz_shared[k];
//             Real temp2 = Real(0.5) * pdmag / Sigmasq;
//             Real temp3 = temp2 / Sigmasq;
//             Real temp4 = Real(3.0)*temp2;
//             Real temp5 = temp*( Real(1.0) + temp3*r2 - temp4);

//             atomicAdd(&fx[ind], F_shared[0]*temp5);
//             atomicAdd(&fy[ind], F_shared[1]*temp5);
//             atomicAdd(&fz[ind], F_shared[2]*temp5);

//         }
//     }
// }

__global__
void cufcm_flow_solve(myCufftComplex* fk_x, myCufftComplex* fk_y, myCufftComplex* fk_z,
                      myCufftComplex* uk_x, myCufftComplex* uk_y, myCufftComplex* uk_z,
                      int nx, int ny, int nz, Real Lx, Real Ly, Real Lz){
    const int i = threadIdx.x + blockIdx.x*blockDim.x;

    int fft_nx = nx/2 + 1;
    Real grid_size = nx*ny*nz;
    int fft_grid_size = fft_nx*ny*nz;

    if(i < fft_grid_size){
        const int indk = i/(ny*fft_nx);
        const int indj = (i - indk*(ny*fft_nx))/fft_nx;
        const int indi = i - (indj + indk*ny)*fft_nx;
        
        Real q1 = ( (indi <= nx/2)? Real(indi) : Real(indi - nx) ) * (Real(PI2)/Lx);
        Real q2 = ( (indj <= ny/2)? Real(indj) : Real(indj - ny) ) * (Real(PI2)/Ly);
        Real q3 = ( (indk <= nz/2)? Real(indk) : Real(indk - nz) ) * (Real(PI2)/Lz);
        Real qq = q1*q1 + q2*q2 + q3*q3;
        Real qq_inv = (Real)1.0/(qq);

        Real f1_re = fk_x[i].x;
        Real f1_im = fk_x[i].y;
        Real f2_re = fk_y[i].x;
        Real f2_im = fk_y[i].y;
        Real f3_re = fk_z[i].x;
        Real f3_im = fk_z[i].y;

        if(i==0){
            f1_re = (Real)0.0;
            f1_im = (Real)0.0;
            f2_re = (Real)0.0;
            f2_im = (Real)0.0;
            f3_re = (Real)0.0;
            f3_im = (Real)0.0;
        }

        Real kdotf_re = (q1*f1_re+q2*f2_re+q3*f3_re)*qq_inv;
        Real kdotf_im = (q1*f1_im+q2*f2_im+q3*f3_im)*qq_inv;
        Real norm = qq_inv / grid_size;
        
        // printf("ik_x[%d] = [%.4f %.4f]\n", i, uk_x[i].x, uk_y[i].y);

        uk_x[i].x = norm*(f1_re-q1*(kdotf_re));
        uk_x[i].y = norm*(f1_im-q1*(kdotf_im));
        uk_y[i].x = norm*(f2_re-q2*(kdotf_re));
        uk_y[i].y = norm*(f2_im-q2*(kdotf_im));
        uk_z[i].x = norm*(f3_re-q3*(kdotf_re));
        uk_z[i].y = norm*(f3_im-q3*(kdotf_im));

        if(i==0){
            uk_x[0].x = (Real)0.0;
            uk_x[0].y = (Real)0.0;
            uk_y[0].x = (Real)0.0;
            uk_y[0].y = (Real)0.0;
            uk_z[0].x = (Real)0.0;
            uk_z[0].y = (Real)0.0;
        }
    }
    return;
}

__global__
void cufcm_particle_velocities_bpp_shared_dynamic(myCufftReal *ux, myCufftReal *uy, myCufftReal *uz,
                                Real *Y, Real *VTEMP, Real *WTEMP,
                                int N, int ngd, 
                                Real sigma, Real sigmadip, Real Sigma,
                                Real dx, Real nx, Real ny, Real nz){
    
    int ngdh = ngd/2;
    Real norm = dx*dx*dx;
    Real Vx = (Real) 0.0, Vy = (Real) 0.0, Vz = (Real) 0.0, Wx = (Real) 0.0, Wy = (Real) 0.0, Wz = (Real) 0.0;

    extern __shared__ Integer s[];
    Integer *indx_shared = s;
    Integer *indy_shared = (Integer*)&indx_shared[ngd];
    Integer *indz_shared = (Integer*)&indy_shared[ngd];
    #if SOLVER_MODE == 0
        Real *gaussx_shared = (Real*)&indz_shared[ngd]; 
        Real *gaussy_shared = (Real*)&gaussx_shared[ngd];
        Real *gaussz_shared = (Real*)&gaussy_shared[ngd];
        Real *gaussx_dip_shared = (Real*)&gaussz_shared[ngd]; 
        Real *gaussy_dip_shared = (Real*)&gaussx_dip_shared[ngd];
        Real *gaussz_dip_shared = (Real*)&gaussy_dip_shared[ngd];
        Real *grad_gaussx_dip_shared = (Real*)&gaussz_dip_shared[ngd];
    #elif SOLVER_MODE == 1
        Real *xdis_shared = (Real*)&indz_shared[ngd];    
        Real *ydis_shared = (Real*)&xdis_shared[ngd];
        Real *zdis_shared = (Real*)&ydis_shared[ngd];
        Real *gaussx_shared = (Real*)&zdis_shared[ngd]; 
        Real *gaussy_shared = (Real*)&gaussx_shared[ngd];
        Real *gaussz_shared = (Real*)&gaussy_shared[ngd];
        Real *grad_gaussx_dip_shared = (Real*)&gaussz_shared[ngd];
    #endif
    Real *grad_gaussy_dip_shared = (Real*)&grad_gaussx_dip_shared[ngd];
    Real *grad_gaussz_dip_shared = (Real*)&grad_gaussy_dip_shared[ngd];
    Real *Y_shared = (Real*)&grad_gaussz_dip_shared[ngd];

    #if SOLVER_MODE == 0
        Real sigmasq = sigma*sigma;
        Real sigmadipsq = sigmadip*sigmadip;
        Real Sigmadipsq = sigmadipsq;
        Real anorm = Real(1.0)/my_sqrt(Real(2.0)*Real(PI)*sigmasq);
        Real anormdip = Real(1.0)/my_sqrt(Real(2.0)*Real(PI)*sigmadipsq);
    #elif SOLVER_MODE == 1
        Real Sigmasq = Sigma*Sigma;
        Real Sigmadipsq = Sigmasq;
        Real Anorm = Real(1.0)/my_sqrt(Real(PI2)*Sigmasq);
        Real width2 = (Real(2.0)*Sigmasq);
        Real pdmag = sigma*sigma - Sigmasq;
    #endif
    

    // Specialize BlockReduce
    typedef cub::BlockReduce<Real, FCM_THREADS_PER_BLOCK> BlockReduce;
    // Allocate shared memory for BlockReduce
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    for(int np = blockIdx.x; np < N; np += gridDim.x){
        if(threadIdx.x == 0){
            Y_shared[0] = Y[3*np + 0];
            Y_shared[1] = Y[3*np + 1];
            Y_shared[2] = Y[3*np + 2];
        }
        __syncthreads();

        for(int i = threadIdx.x; i < 4*ngd; i += blockDim.x){
            Real xg = my_rint(Y_shared[0]/dx) - ngdh + my_fmod(Real(i), Real(ngd));
            Real yg = my_rint(Y_shared[1]/dx) - ngdh + my_fmod(Real(i), Real(ngd));
            Real zg = my_rint(Y_shared[2]/dx) - ngdh + my_fmod(Real(i), Real(ngd));

            Real xx = xg*dx - Y_shared[0];
            Real yy = yg*dx - Y_shared[1];
            Real zz = zg*dx - Y_shared[2];
            /* dis */
            if(i<ngd){
                #if SOLVER_MODE == 0
                    gaussx_shared[i] = anorm*my_exp(-xx*xx/(Real(2.0)*sigmasq));
                    gaussy_shared[i] = anorm*my_exp(-yy*yy/(Real(2.0)*sigmasq));
                    gaussz_shared[i] = anorm*my_exp(-zz*zz/(Real(2.0)*sigmasq));
                #elif SOLVER_MODE == 1
                    xdis_shared[i] = xx;
                    ydis_shared[i] = yy;
                    zdis_shared[i] = zz;
                #endif
            }
            /* gauss */
            if(i>=ngd && i<2*ngd){
                #if SOLVER_MODE == 0
                    gaussx_dip_shared[i-ngd] = anormdip*my_exp(-xx*xx/(Real(2.0)*sigmadipsq));
                    gaussy_dip_shared[i-ngd] = anormdip*my_exp(-yy*yy/(Real(2.0)*sigmadipsq));
                    gaussz_dip_shared[i-ngd] = anormdip*my_exp(-zz*zz/(Real(2.0)*sigmadipsq));
                #elif SOLVER_MODE == 1
                    gaussx_shared[i-ngd] = Anorm*my_exp(-xx*xx/width2);
                    gaussy_shared[i-ngd] = Anorm*my_exp(-yy*yy/width2);
                    gaussz_shared[i-ngd] = Anorm*my_exp(-zz*zz/width2);
                #endif
            }
            /* grad_gauss */
            if(i>=2*ngd && i<3*ngd){
                grad_gaussx_dip_shared[i-2*ngd] = - xx / Sigmadipsq;
                grad_gaussy_dip_shared[i-2*ngd] = - yy / Sigmadipsq;
                grad_gaussz_dip_shared[i-2*ngd] = - zz / Sigmadipsq;
            }
            /* ind */
            if(i>=3*ngd){
                indx_shared[i-3*ngd] = xg - nx * my_floor( xg / nx);
                indy_shared[i-3*ngd] = yg - ny * my_floor( yg / ny);
                indz_shared[i-3*ngd] = zg - nz * my_floor( zg / nz);
            }
        }
        __syncthreads();

        for(int t = threadIdx.x; t < ngd*ngd*ngd; t += blockDim.x){
            const int k = t/(ngd*ngd);
            const int j = (t - k*ngd*ngd)/ngd;
            const int i = t - k*ngd*ngd - j*ngd;

            Real gradx = grad_gaussx_dip_shared[i];
            Real grady = grad_gaussy_dip_shared[j];
            Real gradz = grad_gaussz_dip_shared[k];

            int ind = indx_shared[i] + indy_shared[j]*int(nx) + indz_shared[k]*int(nx)*int(ny);
            
            #if SOLVER_MODE == 0
                Real temp = gaussx_shared[i]*gaussy_shared[j]*gaussz_shared[k]*norm;
                Real tempdip = gaussx_dip_shared[i]*gaussy_dip_shared[j]*gaussz_dip_shared[k]*norm;
            #elif SOLVER_MODE == 1
                Real r2 = xdis_shared[i]*xdis_shared[i] + ydis_shared[j]*ydis_shared[j] + zdis_shared[k]*zdis_shared[k];
                Real temp1 = gaussx_shared[i]*gaussy_shared[j]*gaussz_shared[k]*norm;
                Real temp2 = Real(0.5) * pdmag / Sigmasq;
                Real temp3 = temp2 /Sigmasq;
                Real temp4 = Real(3.0)*temp2;
                Real temp5 = ( Real(1.0) + temp3*r2 - temp4);
                Real temp = temp1*temp5;
                Real tempdip = temp1;
            #endif

            Vx += ux[ind]*temp;
            Vy += uy[ind]*temp;
            Vz += uz[ind]*temp;
            Wx += (Real)-0.5*(uz[ind]*grady - uy[ind]*gradz)*tempdip;
            Wy += (Real)-0.5*(ux[ind]*gradz - uz[ind]*gradx)*tempdip;
            Wz += (Real)-0.5*(uy[ind]*gradx - ux[ind]*grady)*tempdip; 
        }
        
        // Reduction
        Real total_Vx = BlockReduce(temp_storage).Sum(Vx);
        Real total_Vy = BlockReduce(temp_storage).Sum(Vy);
        Real total_Vz = BlockReduce(temp_storage).Sum(Vz);
        Real total_Wx = BlockReduce(temp_storage).Sum(Wx);
        Real total_Wy = BlockReduce(temp_storage).Sum(Wy);
        Real total_Wz = BlockReduce(temp_storage).Sum(Wz);
    
        if(threadIdx.x==0){
            VTEMP[3*np + 0] = total_Vx;  
            VTEMP[3*np + 1] = total_Vy;
            VTEMP[3*np + 2] = total_Vz;
            WTEMP[3*np + 0] = total_Wx;
            WTEMP[3*np + 1] = total_Wy;
            WTEMP[3*np + 2] = total_Wz;
        }
    }
}

// __global__
// void cufcm_particle_velocities_selection(myCufftReal *ux, myCufftReal *uy, myCufftReal *uz,
//                                 Real *Y, Real *VTEMP, Real *WTEMP,
//                                 int N, int ngd, 
//                                 Real sigma, Real Sigma,
//                                 Real dx, Real nx, Real ny, Real nz, 
//                                 int *particle_index, int start, int end){
    
//     int ngdh = ngd/2;
//     Real norm = dx*dx*dx;
//     Real Vx = (Real) 0.0, Vy = (Real) 0.0, Vz = (Real) 0.0, Wx = (Real) 0.0, Wy = (Real) 0.0, Wz = (Real) 0.0;

//     extern __shared__ Integer s[];
//     Integer *indx_shared = s;
//     Integer *indy_shared = (Integer*)&indx_shared[ngd];
//     Integer *indz_shared = (Integer*)&indy_shared[ngd];
//     Real *xdis_shared = (Real*)&indz_shared[ngd];    
//     Real *ydis_shared = (Real*)&xdis_shared[ngd];
//     Real *zdis_shared = (Real*)&ydis_shared[ngd];
//     Real *gaussx_shared = (Real*)&zdis_shared[ngd]; 
//     Real *gaussy_shared = (Real*)&gaussx_shared[ngd];
//     Real *gaussz_shared = (Real*)&gaussy_shared[ngd];
//     Real *grad_gaussx_dip_shared = (Real*)&gaussz_shared[ngd];
//     Real *grad_gaussy_dip_shared = (Real*)&grad_gaussx_dip_shared[ngd];
//     Real *grad_gaussz_dip_shared = (Real*)&grad_gaussy_dip_shared[ngd];
//     Real *Y_shared = (Real*)&grad_gaussz_dip_shared[ngd];

//     Real Sigmasq = Sigma*Sigma;
//     Real Sigmadipsq = Sigmasq;
//     Real Anorm = Real(1.0)/my_sqrt(Real(PI2)*Sigmasq);
//     Real width2 = (Real(2.0)*Sigmasq);
//     Real pdmag = sigma*sigma - Sigmasq;

//     // Specialize BlockReduce
//     typedef cub::BlockReduce<Real, FCM_THREADS_PER_BLOCK> BlockReduce;
//     // Allocate shared memory for BlockReduce
//     __shared__ typename BlockReduce::TempStorage temp_storage;
    
//     for(int np = blockIdx.x; (np < N) && (particle_index[np] >= start && particle_index[np] < end); np += gridDim.x){
//         if(threadIdx.x == 0){
//             Y_shared[0] = Y[3*np + 0];
//             Y_shared[1] = Y[3*np + 1];
//             Y_shared[2] = Y[3*np + 2];
//         }
//         __syncthreads();

//         for(int i = threadIdx.x; i < 4*ngd; i += blockDim.x){
//             Real xg = my_rint(Y_shared[0]/dx) - ngdh + my_fmod(Real(i), Real(ngd));
//             Real yg = my_rint(Y_shared[1]/dx) - ngdh + my_fmod(Real(i), Real(ngd));
//             Real zg = my_rint(Y_shared[2]/dx) - ngdh + my_fmod(Real(i), Real(ngd));

//             Real xx = xg*dx - Y_shared[0];
//             Real yy = yg*dx - Y_shared[1];
//             Real zz = zg*dx - Y_shared[2];
//             /* dis */
//             if(i<ngd){ 
//                 xdis_shared[i] = xx;
//                 ydis_shared[i] = yy;
//                 zdis_shared[i] = zz;
//             }
//             /* gauss */
//             if(i>=ngd && i<2*ngd){
//                 gaussx_shared[i-ngd] = Anorm*my_exp(-xx*xx/width2);
//                 gaussy_shared[i-ngd] = Anorm*my_exp(-yy*yy/width2);
//                 gaussz_shared[i-ngd] = Anorm*my_exp(-zz*zz/width2);
//             }
//             /* grad_gauss */
//             if(i>=2*ngd && i<3*ngd){
//                 grad_gaussx_dip_shared[i-2*ngd] = - xx / Sigmadipsq;
//                 grad_gaussy_dip_shared[i-2*ngd] = - yy / Sigmadipsq;
//                 grad_gaussz_dip_shared[i-2*ngd] = - zz / Sigmadipsq;
//             }
//             /* ind */
//             if(i>=3*ngd){
//                 indx_shared[i-3*ngd] = xg - nx * my_floor( xg / nx);
//                 indy_shared[i-3*ngd] = yg - ny * my_floor( yg / ny);
//                 indz_shared[i-3*ngd] = zg - nz * my_floor( zg / nz);
//             }
//         }
//         __syncthreads();

//         for(int t = threadIdx.x; t < ngd*ngd*ngd; t += blockDim.x){
//             const int k = t/(ngd*ngd);
//             const int j = (t - k*ngd*ngd)/ngd;
//             const int i = t - k*ngd*ngd - j*ngd;

//             Real gradx = grad_gaussx_dip_shared[i];
//             Real grady = grad_gaussy_dip_shared[j];
//             Real gradz = grad_gaussz_dip_shared[k];

//             int ind = indx_shared[i] + indy_shared[j]*int(nx) + indz_shared[k]*int(nx)*int(ny);
//             Real r2 = xdis_shared[i]*xdis_shared[i] + ydis_shared[j]*ydis_shared[j] + zdis_shared[k]*zdis_shared[k];
//             Real temp = gaussx_shared[i]*gaussy_shared[j]*gaussz_shared[k]*norm;
//             Real temp2 = Real(0.5) * pdmag / Sigmasq;
//             Real temp3 = temp2 /Sigmasq;
//             Real temp4 = Real(3.0)*temp2;
//             Real temp5 = ( Real(1.0) + temp3*r2 - temp4);

//             Real ux_temp = ux[ind]*temp;
//             Real uy_temp = uy[ind]*temp;
//             Real uz_temp = uz[ind]*temp;

//             Vx += ux_temp*temp5;
//             Vy += uy_temp*temp5;
//             Vz += uz_temp*temp5;
//             Wx += Real(-0.5)*(uz_temp*grady - uy_temp*gradz);
//             Wy += Real(-0.5)*(ux_temp*gradz - uz_temp*gradx);
//             Wz += Real(-0.5)*(uy_temp*gradx - ux_temp*grady);
//         }
        
//         // Reduction
//         Real total_Vx = BlockReduce(temp_storage).Sum(Vx);
//         Real total_Vy = BlockReduce(temp_storage).Sum(Vy);
//         Real total_Vz = BlockReduce(temp_storage).Sum(Vz);
//         Real total_Wx = BlockReduce(temp_storage).Sum(Wx);
//         Real total_Wy = BlockReduce(temp_storage).Sum(Wy);
//         Real total_Wz = BlockReduce(temp_storage).Sum(Wz);
    
//         if(threadIdx.x==0){
//             VTEMP[3*np + 0] += total_Vx;
//             VTEMP[3*np + 1] += total_Vy;
//             VTEMP[3*np + 2] += total_Vz;
//             WTEMP[3*np + 0] += total_Wx;
//             WTEMP[3*np + 1] += total_Wy;
//             WTEMP[3*np + 2] += total_Wz;
//         }
//     }
// }

__global__
void cufcm_particle_velocities_mono(myCufftReal *ux, myCufftReal *uy, myCufftReal *uz,
                                Real *Y,
                                Real *VTEMP,
                                int N, int ngd, 
                                Real sigma, Real Sigma,
                                Real dx, Real nx, Real ny, Real nz){
    int ngdh = ngd/2;
    Real norm = dx*dx*dx;
    Real Vx = (Real) 0.0, Vy = (Real) 0.0, Vz = (Real) 0.0;

    extern __shared__ Integer s[];
    Integer *indx_shared = s;
    Integer *indy_shared = (Integer*)&indx_shared[ngd];
    Integer *indz_shared = (Integer*)&indy_shared[ngd];
    Real *xdis_shared = (Real*)&indz_shared[ngd];    
    Real *ydis_shared = (Real*)&xdis_shared[ngd];
    Real *zdis_shared = (Real*)&ydis_shared[ngd];
    Real *gaussx_shared = (Real*)&zdis_shared[ngd]; 
    Real *gaussy_shared = (Real*)&gaussx_shared[ngd];
    Real *gaussz_shared = (Real*)&gaussy_shared[ngd];
    Real *Y_shared = (Real*)&gaussz_shared[ngd];

    Real Sigmasq = Sigma*Sigma;
    Real Anorm = Real(1.0)/my_sqrt(Real(PI2)*Sigmasq);
    Real width2 = (Real(2.0)*Sigmasq);
    Real pdmag = sigma*sigma - Sigmasq;

    // Specialize BlockReduce
    typedef cub::BlockReduce<Real, FCM_THREADS_PER_BLOCK> BlockReduce;
    // Allocate shared memory for BlockReduce
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    for(int np = blockIdx.x; np < N; np += gridDim.x){
        if(threadIdx.x == 0){
            Y_shared[0] = Y[3*np + 0];
            Y_shared[1] = Y[3*np + 1];
            Y_shared[2] = Y[3*np + 2];
        }
        __syncthreads();

        for(int i = threadIdx.x; i < 4*ngd; i += blockDim.x){
            Real xg = my_rint(Y_shared[0]/dx) - ngdh + my_fmod(Real(i), Real(ngd));
            Real yg = my_rint(Y_shared[1]/dx) - ngdh + my_fmod(Real(i), Real(ngd));
            Real zg = my_rint(Y_shared[2]/dx) - ngdh + my_fmod(Real(i), Real(ngd));

            Real xx = xg*dx - Y_shared[0];
            Real yy = yg*dx - Y_shared[1];
            Real zz = zg*dx - Y_shared[2];
            /* dis */
            if(i<ngd){ 
                xdis_shared[i] = xx;
                ydis_shared[i] = yy;
                zdis_shared[i] = zz;
            }
            /* gauss */
            if(i>=ngd && i<2*ngd){
                gaussx_shared[i-ngd] = Anorm*my_exp(-xx*xx/width2);
                gaussy_shared[i-ngd] = Anorm*my_exp(-yy*yy/width2);
                gaussz_shared[i-ngd] = Anorm*my_exp(-zz*zz/width2);
            }
            /* grad_gauss */
            if(i>=2*ngd && i<3*ngd){
            }
            /* ind */
            if(i>=3*ngd){
                indx_shared[i-3*ngd] = xg - nx * my_floor( xg / nx);
                indy_shared[i-3*ngd] = yg - ny * my_floor( yg / ny);
                indz_shared[i-3*ngd] = zg - nz * my_floor( zg / nz);
            }
        }
        __syncthreads();

        for(int t = threadIdx.x; t < ngd*ngd*ngd; t += blockDim.x){
            const int k = t/(ngd*ngd);
            const int j = (t - k*ngd*ngd)/ngd;
            const int i = t - k*ngd*ngd - j*ngd;

            int ind = indx_shared[i] + indy_shared[j]*int(nx) + indz_shared[k]*int(nx)*int(ny);
            Real r2 = xdis_shared[i]*xdis_shared[i] + ydis_shared[j]*ydis_shared[j] + zdis_shared[k]*zdis_shared[k];
            Real temp = gaussx_shared[i]*gaussy_shared[j]*gaussz_shared[k]*norm;
            Real temp2 = Real(0.5) * pdmag / Sigmasq;
            Real temp3 = temp2 /Sigmasq;
            Real temp4 = Real(3.0)*temp2;
            Real temp5 = ( Real(1.0) + temp3*r2 - temp4);

            Real ux_temp = ux[ind]*temp;
            Real uy_temp = uy[ind]*temp;
            Real uz_temp = uz[ind]*temp;

            Vx += ux_temp*temp5;
            Vy += uy_temp*temp5;
            Vz += uz_temp*temp5;
        }
        
        // Reduction
        Real total_Vx = BlockReduce(temp_storage).Sum(Vx);
        Real total_Vy = BlockReduce(temp_storage).Sum(Vy);
        Real total_Vz = BlockReduce(temp_storage).Sum(Vz);
    
        if(threadIdx.x==0){
            VTEMP[3*np + 0] = total_Vx;  
            VTEMP[3*np + 1] = total_Vy;
            VTEMP[3*np + 2] = total_Vz;
        }
    }                                  
}

// __global__
// void cufcm_particle_velocities_mono_selection(myCufftReal *ux, myCufftReal *uy, myCufftReal *uz,
//                                 Real *Y,
//                                 Real *VTEMP,
//                                 int N, int ngd, 
//                                 Real sigma, Real Sigma,
//                                 Real dx, Real nx, Real ny, Real nz,
//                                 int *particle_index, int start, int end){
//     int ngdh = ngd/2;
//     Real norm = dx*dx*dx;
//     Real Vx = (Real) 0.0, Vy = (Real) 0.0, Vz = (Real) 0.0;

//     extern __shared__ Integer s[];
//     Integer *indx_shared = s;
//     Integer *indy_shared = (Integer*)&indx_shared[ngd];
//     Integer *indz_shared = (Integer*)&indy_shared[ngd];
//     Real *xdis_shared = (Real*)&indz_shared[ngd];    
//     Real *ydis_shared = (Real*)&xdis_shared[ngd];
//     Real *zdis_shared = (Real*)&ydis_shared[ngd];
//     Real *gaussx_shared = (Real*)&zdis_shared[ngd]; 
//     Real *gaussy_shared = (Real*)&gaussx_shared[ngd];
//     Real *gaussz_shared = (Real*)&gaussy_shared[ngd];
//     Real *Y_shared = (Real*)&gaussz_shared[ngd];

//     Real Sigmasq = Sigma*Sigma;
//     Real Anorm = Real(1.0)/my_sqrt(Real(PI2)*Sigmasq);
//     Real width2 = (Real(2.0)*Sigmasq);
//     Real pdmag = sigma*sigma - Sigmasq;

//     // Specialize BlockReduce
//     typedef cub::BlockReduce<Real, FCM_THREADS_PER_BLOCK> BlockReduce;
//     // Allocate shared memory for BlockReduce
//     __shared__ typename BlockReduce::TempStorage temp_storage;
    
//     for(int np = blockIdx.x; (np < N) && (particle_index[np] >= start && particle_index[np] < end); np += gridDim.x){
//         if(threadIdx.x == 0){
//             Y_shared[0] = Y[3*np + 0];
//             Y_shared[1] = Y[3*np + 1];
//             Y_shared[2] = Y[3*np + 2];
//         }
//         __syncthreads();

//         for(int i = threadIdx.x; i < 4*ngd; i += blockDim.x){
//             Real xg = my_rint(Y_shared[0]/dx) - ngdh + my_fmod(Real(i), Real(ngd));
//             Real yg = my_rint(Y_shared[1]/dx) - ngdh + my_fmod(Real(i), Real(ngd));
//             Real zg = my_rint(Y_shared[2]/dx) - ngdh + my_fmod(Real(i), Real(ngd));

//             Real xx = xg*dx - Y_shared[0];
//             Real yy = yg*dx - Y_shared[1];
//             Real zz = zg*dx - Y_shared[2];
//             /* dis */
//             if(i<ngd){ 
//                 xdis_shared[i] = xx;
//                 ydis_shared[i] = yy;
//                 zdis_shared[i] = zz;
//             }
//             /* gauss */
//             if(i>=ngd && i<2*ngd){
//                 gaussx_shared[i-ngd] = Anorm*my_exp(-xx*xx/width2);
//                 gaussy_shared[i-ngd] = Anorm*my_exp(-yy*yy/width2);
//                 gaussz_shared[i-ngd] = Anorm*my_exp(-zz*zz/width2);
//             }
//             /* grad_gauss */
//             if(i>=2*ngd && i<3*ngd){
//             }
//             /* ind */
//             if(i>=3*ngd){
//                 indx_shared[i-3*ngd] = xg - nx * my_floor( xg / nx);
//                 indy_shared[i-3*ngd] = yg - ny * my_floor( yg / ny);
//                 indz_shared[i-3*ngd] = zg - nz * my_floor( zg / nz);
//             }
//         }
//         __syncthreads();

//         for(int t = threadIdx.x; t < ngd*ngd*ngd; t += blockDim.x){
//             const int k = t/(ngd*ngd);
//             const int j = (t - k*ngd*ngd)/ngd;
//             const int i = t - k*ngd*ngd - j*ngd;

//             int ind = indx_shared[i] + indy_shared[j]*int(nx) + indz_shared[k]*int(nx)*int(ny);
//             Real r2 = xdis_shared[i]*xdis_shared[i] + ydis_shared[j]*ydis_shared[j] + zdis_shared[k]*zdis_shared[k];
//             Real temp = gaussx_shared[i]*gaussy_shared[j]*gaussz_shared[k]*norm;
//             Real temp2 = Real(0.5) * pdmag / Sigmasq;
//             Real temp3 = temp2 /Sigmasq;
//             Real temp4 = Real(3.0)*temp2;
//             Real temp5 = ( Real(1.0) + temp3*r2 - temp4);

//             Real ux_temp = ux[ind]*temp;
//             Real uy_temp = uy[ind]*temp;
//             Real uz_temp = uz[ind]*temp;

//             Vx += ux_temp*temp5;
//             Vy += uy_temp*temp5;
//             Vz += uz_temp*temp5;
//         }
        
//         // Reduction
//         Real total_Vx = BlockReduce(temp_storage).Sum(Vx);
//         Real total_Vy = BlockReduce(temp_storage).Sum(Vy);
//         Real total_Vz = BlockReduce(temp_storage).Sum(Vz);
    
//         if(threadIdx.x==0){
//             VTEMP[3*np + 0] += total_Vx;  
//             VTEMP[3*np + 1] += total_Vy;
//             VTEMP[3*np + 2] += total_Vz;
//         }
//     }
// }

///////////////////////////////////////////////////////////////////////////////
// Regular FCM
///////////////////////////////////////////////////////////////////////////////
// __global__
// void cufcm_mono_dipole_distribution_regular_fcm(myCufftReal *fx, myCufftReal *fy, myCufftReal *fz,
//               Real *Y, Real *T, Real *F,
//               int N, int ngd,
//               Real sigma, Real sigmadip,
//               Real dx, Real nx, Real ny, Real nz){

//     int ngdh = ngd/2;

//     extern __shared__ Integer ls[];
//     Integer *indx_shared = (Integer*)ls;
//     Integer *indy_shared = (Integer*)&indx_shared[ngd];
//     Integer *indz_shared = (Integer*)&indy_shared[ngd];
//     Real *gaussx_shared = (Real*)&indz_shared[ngd]; 
//     Real *gaussy_shared = (Real*)&gaussx_shared[ngd];
//     Real *gaussz_shared = (Real*)&gaussy_shared[ngd];
//     Real *gaussx_dip_shared = (Real*)&gaussz_shared[ngd]; 
//     Real *gaussy_dip_shared = (Real*)&gaussx_dip_shared[ngd];
//     Real *gaussz_dip_shared = (Real*)&gaussy_dip_shared[ngd];
//     Real *grad_gaussx_dip_shared = (Real*)&gaussz_dip_shared[ngd];
//     Real *grad_gaussy_dip_shared = (Real*)&grad_gaussx_dip_shared[ngd];
//     Real *grad_gaussz_dip_shared = (Real*)&grad_gaussy_dip_shared[ngd];
//     Real *Y_shared = (Real*)&grad_gaussz_dip_shared[ngd];
//     Real *F_shared = (Real*)&Y_shared[3];
//     Real *g_shared = (Real*)&F_shared[3];

//     Real sigmasq = sigma*sigma;
//     Real sigmadipsq = sigmadip*sigmadip;
//     Real anorm = Real(1.0)/my_sqrt(Real(2.0)*Real(PI)*sigmasq);
//     Real anormdip = Real(1.0)/my_sqrt(Real(2.0)*Real(PI)*sigmadipsq);
    
//     for(int np = blockIdx.x; np < N; np += gridDim.x){

//         if(threadIdx.x == 0){
//             Y_shared[0] = Y[3*np + 0];
//             Y_shared[1] = Y[3*np + 1];
//             Y_shared[2] = Y[3*np + 2];

//             F_shared[0] = F[3*np + 0];
//             F_shared[1] = F[3*np + 1];
//             F_shared[2] = F[3*np + 2];

//             g_shared[0] = + Real(0.0);
//             g_shared[1] = + Real(0.0);
//             g_shared[2] = + Real(0.0);
//             g_shared[3] = + Real(0.5)*T[3*np + 2];
//             g_shared[4] = - Real(0.5)*T[3*np + 2];
//             g_shared[5] = + Real(-0.5)*T[3*np + 1];
//             g_shared[6] = - Real(-0.5)*T[3*np + 1];
//             g_shared[7] = + Real(0.5)*T[3*np + 0];
//             g_shared[8] = - Real(0.5)*T[3*np + 0];
//         }

//         __syncthreads();

//         for(int i = threadIdx.x; i < 4*ngd; i += blockDim.x){
//             Real xg = my_rint(Y_shared[0]/dx) - ngdh + my_fmod(Real(i), Real(ngd));
//             Real yg = my_rint(Y_shared[1]/dx) - ngdh + my_fmod(Real(i), Real(ngd));
//             Real zg = my_rint(Y_shared[2]/dx) - ngdh + my_fmod(Real(i), Real(ngd));

//             Real xx = xg*dx - Y_shared[0];
//             Real yy = yg*dx - Y_shared[1];
//             Real zz = zg*dx - Y_shared[2];
            
//             // gauss
//             if(i<ngd){ 
//                 gaussx_shared[i] = anorm*my_exp(-xx*xx/(Real(2.0)*sigmasq));
//                 gaussy_shared[i] = anorm*my_exp(-yy*yy/(Real(2.0)*sigmasq));
//                 gaussz_shared[i] = anorm*my_exp(-zz*zz/(Real(2.0)*sigmasq));
//             }
//             // gauss dip
//             if(i>=ngd && i<2*ngd){
//                 gaussx_dip_shared[i-ngd] = anormdip*my_exp(-xx*xx/(Real(2.0)*sigmadipsq));
//                 gaussy_dip_shared[i-ngd] = anormdip*my_exp(-yy*yy/(Real(2.0)*sigmadipsq));
//                 gaussz_dip_shared[i-ngd] = anormdip*my_exp(-zz*zz/(Real(2.0)*sigmadipsq));
//             }
//             // grad_gauss
//             if(i>=2*ngd && i<3*ngd){
//                 grad_gaussx_dip_shared[i-2*ngd] = - xx / sigmadipsq;
//                 grad_gaussy_dip_shared[i-2*ngd] = - yy / sigmadipsq;
//                 grad_gaussz_dip_shared[i-2*ngd] = - zz / sigmadipsq;
//             }
//             // ind
//             if(i>=3*ngd){
//                 indx_shared[i-3*ngd] = xg - nx * my_floor( xg / nx);
//                 indy_shared[i-3*ngd] = yg - ny * my_floor( yg / ny);
//                 indz_shared[i-3*ngd] = zg - nz * my_floor( zg / nz);
//             }
//         }
//         __syncthreads();
        
//         for(int t = threadIdx.x; t < ngd*ngd*ngd; t += blockDim.x){
//             const int k = t/(ngd*ngd);
//             const int j = (t - k*ngd*ngd)/ngd;
//             const int i = t - k*ngd*ngd - j*ngd;

//             Real gradx = grad_gaussx_dip_shared[i];
//             Real grady = grad_gaussy_dip_shared[j];
//             Real gradz = grad_gaussz_dip_shared[k];

//             int ind = indx_shared[i] + indy_shared[j]*int(nx) + indz_shared[k]*int(nx)*int(ny);
//             Real temp = gaussx_shared[i]*gaussy_shared[j]*gaussz_shared[k];
//             Real tempdip = gaussx_dip_shared[i]*gaussy_dip_shared[j]*gaussz_dip_shared[k];

//             atomicAdd(&fx[ind], F_shared[0]*temp + (g_shared[0]*gradx + g_shared[3]*grady + g_shared[5]*gradz)*tempdip);
//             atomicAdd(&fy[ind], F_shared[1]*temp + (g_shared[4]*gradx + g_shared[1]*grady + g_shared[7]*gradz)*tempdip);
//             atomicAdd(&fz[ind], F_shared[2]*temp + (g_shared[6]*gradx + g_shared[8]*grady + g_shared[2]*gradz)*tempdip);
//         }
//     }
// }

// __global__
// void cufcm_particle_velocities_regular_fcm(myCufftReal *ux, myCufftReal *uy, myCufftReal *uz,
//                                 Real *Y,
//                                 Real *VTEMP, Real *WTEMP,
//                                 int N, int ngd, 
//                                 Real sigma, Real sigmadip,
//                                 Real dx, Real nx, Real ny, Real nz){
//     int ngdh = ngd/2;
//     Real norm = dx*dx*dx;
//     Real Vx = (Real) 0.0, Vy = (Real) 0.0, Vz = (Real) 0.0, Wx = (Real) 0.0, Wy = (Real) 0.0, Wz = (Real) 0.0;

//     extern __shared__ Integer ls[];
//     Integer *indx_shared = ls;
//     Integer *indy_shared = (Integer*)&indx_shared[ngd];
//     Integer *indz_shared = (Integer*)&indy_shared[ngd];
//     Real *gaussx_shared = (Real*)&indz_shared[ngd]; 
//     Real *gaussy_shared = (Real*)&gaussx_shared[ngd];
//     Real *gaussz_shared = (Real*)&gaussy_shared[ngd];
//     Real *gaussx_dip_shared = (Real*)&gaussz_shared[ngd]; 
//     Real *gaussy_dip_shared = (Real*)&gaussx_dip_shared[ngd];
//     Real *gaussz_dip_shared = (Real*)&gaussy_dip_shared[ngd];
//     Real *grad_gaussx_dip_shared = (Real*)&gaussz_dip_shared[ngd];
//     Real *grad_gaussy_dip_shared = (Real*)&grad_gaussx_dip_shared[ngd];
//     Real *grad_gaussz_dip_shared = (Real*)&grad_gaussy_dip_shared[ngd];
//     Real *Y_shared = (Real*)&grad_gaussz_dip_shared[ngd];

//     Real sigmasq = sigma*sigma;
//     Real sigmadipsq = sigmadip*sigmadip;
//     Real anorm = Real(1.0)/my_sqrt(Real(2.0)*Real(PI)*sigmasq);
//     Real anormdip = Real(1.0)/my_sqrt(Real(2.0)*Real(PI)*sigmadipsq);

//     // Specialize BlockReduce
//     typedef cub::BlockReduce<Real, FCM_THREADS_PER_BLOCK> BlockReduce;
//     // Allocate shared memory for BlockReduce
//     __shared__ typename BlockReduce::TempStorage temp_storage;

//     for(int np = blockIdx.x; np < N; np += gridDim.x){
//         if(threadIdx.x == 0){
//             Y_shared[0] = Y[3*np + 0];
//             Y_shared[1] = Y[3*np + 1];
//             Y_shared[2] = Y[3*np + 2];
//         }
//         __syncthreads();

//         for(int i = threadIdx.x; i < 4*ngd; i += blockDim.x){
//             Real xg = my_rint(Y_shared[0]/dx) - ngdh + my_fmod(Real(i), Real(ngd));
//             Real yg = my_rint(Y_shared[1]/dx) - ngdh + my_fmod(Real(i), Real(ngd));
//             Real zg = my_rint(Y_shared[2]/dx) - ngdh + my_fmod(Real(i), Real(ngd));

//             Real xx = xg*dx - Y_shared[0];
//             Real yy = yg*dx - Y_shared[1];
//             Real zz = zg*dx - Y_shared[2];
            
//             // gauss
//             if(i<ngd){ 
//                 gaussx_shared[i] = anorm*my_exp(-xx*xx/(Real(2.0)*sigmasq));
//                 gaussy_shared[i] = anorm*my_exp(-yy*yy/(Real(2.0)*sigmasq));
//                 gaussz_shared[i] = anorm*my_exp(-zz*zz/(Real(2.0)*sigmasq));
//             }
//             // gauss dip
//             if(i>=ngd && i<2*ngd){
//                 gaussx_dip_shared[i-ngd] = anormdip*my_exp(-xx*xx/(Real(2.0)*sigmadipsq));
//                 gaussy_dip_shared[i-ngd] = anormdip*my_exp(-yy*yy/(Real(2.0)*sigmadipsq));
//                 gaussz_dip_shared[i-ngd] = anormdip*my_exp(-zz*zz/(Real(2.0)*sigmadipsq));
//             }
//             // grad_gauss
//             if(i>=2*ngd && i<3*ngd){
//                 grad_gaussx_dip_shared[i-2*ngd] = - xx / sigmadipsq;
//                 grad_gaussy_dip_shared[i-2*ngd] = - yy / sigmadipsq;
//                 grad_gaussz_dip_shared[i-2*ngd] = - zz / sigmadipsq;
//             }
//             // ind
//             if(i>=3*ngd){
//                 indx_shared[i-3*ngd] = xg - nx * my_floor( xg / nx);
//                 indy_shared[i-3*ngd] = yg - ny * my_floor( yg / ny);
//                 indz_shared[i-3*ngd] = zg - nz * my_floor( zg / nz);
//             }
//         }
//         __syncthreads();

//         for(int t = threadIdx.x; t < ngd*ngd*ngd; t += blockDim.x){
//             const int k = t/(ngd*ngd);
//             const int j = (t - k*ngd*ngd)/ngd;
//             const int i = t - k*ngd*ngd - j*ngd;

//             Real gradx = grad_gaussx_dip_shared[i];
//             Real grady = grad_gaussy_dip_shared[j];
//             Real gradz = grad_gaussz_dip_shared[k];
            
//             int ind = indx_shared[i] + indy_shared[j]*int(nx) + indz_shared[k]*int(nx)*int(ny);
//             Real temp = gaussx_shared[i]*gaussy_shared[j]*gaussz_shared[k]*norm;
//             Real tempdip = gaussx_dip_shared[i]*gaussy_dip_shared[j]*gaussz_dip_shared[k]*norm;

//             Vx += ux[ind]*temp;
//             Vy += uy[ind]*temp;
//             Vz += uz[ind]*temp;
//             Wx += (Real)-0.5*(uz[ind]*grady - uy[ind]*gradz)*tempdip;
//             Wy += (Real)-0.5*(ux[ind]*gradz - uz[ind]*gradx)*tempdip;
//             Wz += (Real)-0.5*(uy[ind]*gradx - ux[ind]*grady)*tempdip; 
//         }
//         // Reduction
//         Real total_Vx = BlockReduce(temp_storage).Sum(Vx);
//         Real total_Vy = BlockReduce(temp_storage).Sum(Vy);
//         Real total_Vz = BlockReduce(temp_storage).Sum(Vz);
//         Real total_Wx = BlockReduce(temp_storage).Sum(Wx);
//         Real total_Wy = BlockReduce(temp_storage).Sum(Wy);
//         Real total_Wz = BlockReduce(temp_storage).Sum(Wz);
    
//         if(threadIdx.x==0){
//             VTEMP[3*np + 0] = total_Vx;  
//             VTEMP[3*np + 1] = total_Vy;
//             VTEMP[3*np + 2] = total_Vz;
//             WTEMP[3*np + 0] = total_Wx;
//             WTEMP[3*np + 1] = total_Wy;
//             WTEMP[3*np + 2] = total_Wz;
//         }
//     }
// }
