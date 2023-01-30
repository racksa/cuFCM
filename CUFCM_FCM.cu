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
/*
__global__
void cufcm_mono_dipole_distribution_tpp_recompute(myCufftReal *fx, myCufftReal *fy, myCufftReal *fz, 
              Real *Y, Real *T, Real *F, 
              int N, int ngd, 
              Real pdmag, Real sigmasq, Real sigmadipsq,
              Real anorm, Real anorm2,
              Real dx, Real nx, Real ny, Real nz){

    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    int xc, yc, zc;
    int xg, yg, zg;
    int i, j, k, ii, jj, kk;
    Real xx, yy, zz, r2;
    Real xx2, yy2, zz2;
    Real g11, g22, g33, g12, g21, g13, g31, g23, g32;
    Real gx, gy, gz, Fx, Fy, Fz;
    Real g11xx, g22yy, g33zz, g12yy, g21xx, g13zz, g31xx, g23zz, g32yy;
    Real temp;
    Real temp2 = (Real)0.5 * pdmag / sigmasq;
    Real temp3 = temp2 /sigmasq;
    Real temp4 = (Real)3.0*temp2;
    Real temp5;
    int ind;
    int ngdh = ngd/2;

    for(int np = index; np < N; np += stride){
        xc = my_rint(Y[3*np + 0]/dx); // the index of the nearest grid point to the particle
        yc = my_rint(Y[3*np + 1]/dx);
        zc = my_rint(Y[3*np + 2]/dx);

        Fx = F[3*np + 0];
        Fy = F[3*np + 1];
        Fz = F[3*np + 2];
        g11 = + T[6*np + 0];
        g22 = + T[6*np + 1];
        g33 = + T[6*np + 2];
        g12 = + T[6*np + 3];
        g21 = - T[6*np + 3];
        g13 = + T[6*np + 4];
        g31 = - T[6*np + 4];
        g23 = + T[6*np + 5];
        g32 = - T[6*np + 5];
        for(k = 0; k < ngd; k++){
            zg = zc - ngdh + (k);
            kk = zg - nz * ((int) floor( ((Real) zg) / ((Real) nz)));
            zz = ((Real) zg)*dx - Y[3*np + 2];
            zz2 = zz*zz;
            gz = anorm*my_exp(-zz*zz/anorm2);
            zz = - zz / sigmadipsq;
            g13zz = g13*zz;
            g23zz = g23*zz;
            g33zz = g33*zz;
            for(j = 0; j < ngd; j++){
                yg = yc - ngdh + (j);
                jj = yg - ny * ((int) floor( ((Real) yg) / ((Real) ny)));
                yy = ((Real) yg)*dx - Y[3*np + 1];
                yy2 = yy*yy;
                gy = anorm*my_exp(-yy*yy/anorm2);
                yy = - yy / sigmadipsq;
                g12yy = g12*yy;
                g22yy = g22*yy;
                g32yy = g32*yy;
                for(i = 0; i < ngd; i++){
                    xg = xc - ngdh + (i);
                    ii = xg - nx * ((int) floor( ((Real) xg) / ((Real) nx)));
                    xx = ((Real) xg)*dx - Y[3*np + 0];
                    xx2 = xx*xx;
                    gx = anorm*my_exp(-xx*xx/anorm2);
                    xx = - xx / sigmadipsq;
                    g11xx = g11*xx;
                    g21xx = g21*xx;
                    g31xx = g31*xx;
                
                    ind = ii + jj*nx + kk*nx*ny;

                    r2 = xx2 + yy2 + zz2;
                    temp = gx*gy*gz;
                    temp5 = temp*( 1 + temp3*r2 - temp4);

                    atomicAdd(&fx[ind], Fx*temp5 + (g11xx + g12yy + g13zz)*temp);
                    atomicAdd(&fy[ind], Fy*temp5 + (g21xx + g22yy + g23zz)*temp);
                    atomicAdd(&fz[ind], Fz*temp5 + (g31xx + g32yy + g33zz)*temp);
                }
            }
        }
    }
}

__global__
void cufcm_mono_dipole_distribution_bpp_shared(myCufftReal *fx, myCufftReal *fy, myCufftReal *fz, Real *Y,
              Real *T, Real *F, int N, int ngd, 
              Real pdmag, Real sigmasq, Real sigmadipsq,
              Real anorm, Real anorm2,
              Real dx, Real nx, Real ny, Real nz){

    __shared__ int indx_shared[NGD];
    __shared__ int indy_shared[NGD];
    __shared__ int indz_shared[NGD];
    __shared__ Real xdis_shared[NGD];
    __shared__ Real ydis_shared[NGD];
    __shared__ Real zdis_shared[NGD];
    __shared__ Real gaussx_shared[NGD];
    __shared__ Real gaussy_shared[NGD];
    __shared__ Real gaussz_shared[NGD];
    __shared__ Real grad_gaussx_dip_shared[NGD];
    __shared__ Real grad_gaussy_dip_shared[NGD];
    __shared__ Real grad_gaussz_dip_shared[NGD];
    __shared__ Real Yx, Yy, Yz;
    __shared__ Real Fx, Fy, Fz;
    __shared__ Real g11, g22, g33, g12, g21, g13, g31, g23, g32;
    int ngdh = ngd/2;

    for(int np = blockIdx.x; np < N; np += gridDim.x){

        if(threadIdx.x == 0){
            Yx = Y[3*np + 0];
            Yy = Y[3*np + 1];
            Yz = Y[3*np + 2];

            Fx = F[3*np + 0];
            Fy = F[3*np + 1];
            Fz = F[3*np + 2];

            g11 = + Real(0.0);
            g22 = + Real(0.0);
            g33 = + Real(0.0);
            g12 = + Real(0.5)*T[3*np + 2];
            g21 = - Real(0.5)*T[3*np + 2];
            g13 = + Real(-0.5)*T[3*np + 1];
            g31 = - Real(-0.5)*T[3*np + 1];
            g23 = + Real(0.5)*T[3*np + 0];
            g32 = - Real(0.5)*T[3*np + 0];
        }
        __syncthreads();

        for(int i = threadIdx.x; i < 4*ngd; i += blockDim.x){
            
            Real xg = my_rint(Yx/dx) - ngdh + fmodf(i, ngd);
            Real yg = my_rint(Yy/dx) - ngdh + fmodf(i, ngd);
            Real zg = my_rint(Yz/dx) - ngdh + fmodf(i, ngd);

            Real xx = xg*dx - Yx;
            Real yy = yg*dx - Yy;
            Real zz = zg*dx - Yz;
            if(i<ngd){ 
                xdis_shared[i] = xx;
                ydis_shared[i] = yy;
                zdis_shared[i] = zz;
            }
            if(i>=ngd && i<2*ngd){
                gaussx_shared[i-ngd] = anorm*my_exp(-xx*xx/anorm2);
                gaussy_shared[i-ngd] = anorm*my_exp(-yy*yy/anorm2);
                gaussz_shared[i-ngd] = anorm*my_exp(-zz*zz/anorm2);
            }
            if(i>=2*ngd && i<3*ngd){
                grad_gaussx_dip_shared[i-2*ngd] = - xx / sigmadipsq;
                grad_gaussy_dip_shared[i-2*ngd] = - yy / sigmadipsq;
                grad_gaussz_dip_shared[i-2*ngd] = - zz / sigmadipsq;
            }
            if(i>=3*ngd){
                indx_shared[i-3*ngd] = xg - double(nx) * floor( xg / double(nx) );
                indy_shared[i-3*ngd] = yg - double(ny) * floor( yg / double(ny) );
                indz_shared[i-3*ngd] = zg - double(nz) * floor( zg / double(nz) );
            }
        }
        __syncthreads();
        
        for(int t = threadIdx.x; t < ngd*ngd*ngd; t += blockDim.x){
            const int k = t/(ngd*ngd);
            const int j = (t - k*ngd*ngd)/ngd;
            const int i = t - k*ngd*ngd - j*ngd;
  
            Real gx = gaussx_shared[i];
            Real gy = gaussy_shared[j];
            Real gz = gaussz_shared[k];

            Real gradx = grad_gaussx_dip_shared[i];
            Real grady = grad_gaussy_dip_shared[j];
            Real gradz = grad_gaussz_dip_shared[k];

            int ind = indx_shared[i] + indy_shared[j]*(double)nx + indz_shared[k]*(double)nx*(double)ny;
            Real r2 = xdis_shared[i]*xdis_shared[i] + ydis_shared[j]*ydis_shared[j] + zdis_shared[k]*zdis_shared[k];
            Real temp = gx*gy*gz;
            Real temp2 = (Real)0.5 * pdmag / sigmasq;
            Real temp3 = temp2 /sigmasq;
            Real temp4 = (Real)3.0*temp2;
            Real temp5 = temp*( (Real)1.0 + temp3*r2 - temp4);

            atomicAdd(&fx[ind], Fx*temp5 + (g11*gradx + g12*grady + g13*gradz)*temp);
            atomicAdd(&fy[ind], Fy*temp5 + (g21*gradx + g22*grady + g23*gradz)*temp);
            atomicAdd(&fz[ind], Fz*temp5 + (g31*gradx + g32*grady + g33*gradz)*temp);
        }
    }
}
*/

__global__
void cufcm_mono_dipole_distribution_bpp_recompute(myCufftReal *fx, myCufftReal *fy, myCufftReal *fz, Real *Y,
              Real *T, Real *F, int N, int ngd, 
              Real sigma, Real Sigma,
              Real dx, Real nx, Real ny, Real nz){
    
    int ngdh = ngd/2;

    Real Yx, Yy, Yz;
    Real Fx, Fy, Fz;
    Real g11, g22, g33, g12, g21, g13, g31, g23, g32;

    Real Sigmasq = Sigma*Sigma;
    Real Sigmadipsq = Sigmasq;
    Real Anorm = Real(1.0)/sqrt(Real(PI2)*Sigmasq);
    Real width2 = (Real(2.0)*Sigmasq);
    Real pdmag = sigma*sigma - Sigmasq;

    for(int np = blockIdx.x; np < N; np += gridDim.x){
        Yx = Y[3*np + 0];
        Yy = Y[3*np + 1];
        Yz = Y[3*np + 2];

        Fx = F[3*np + 0];
        Fy = F[3*np + 1];
        Fz = F[3*np + 2];

        g11 = + Real(0.0);
        g22 = + Real(0.0);
        g33 = + Real(0.0);
        g12 = + Real(0.5)*T[3*np + 2];
        g21 = - Real(0.5)*T[3*np + 2];
        g13 = + Real(-0.5)*T[3*np + 1];
        g31 = - Real(-0.5)*T[3*np + 1];
        g23 = + Real(0.5)*T[3*np + 0];
        g32 = - Real(0.5)*T[3*np + 0];
        
        for(int t = threadIdx.x; t < ngd*ngd*ngd; t += blockDim.x){
            const int k = t/(ngd*ngd);
            const int j = (t - k*ngd*ngd)/ngd;
            const int i = t - k*ngd*ngd - j*ngd;
            
            Real xg = my_rint(Yx/dx) - ngdh + (i);
            Real yg = my_rint(Yy/dx) - ngdh + (j);
            Real zg = my_rint(Yz/dx) - ngdh + (k);

            Real xx = xg*dx - Yx;
            Real yy = yg*dx - Yy;
            Real zz = zg*dx - Yz;

            Real gx = Anorm*my_exp(-xx*xx/width2);
            Real gy = Anorm*my_exp(-yy*yy/width2);
            Real gz = Anorm*my_exp(-zz*zz/width2);

            Real gradx = - xx / Sigmadipsq;
            Real grady = - yy / Sigmadipsq;
            Real gradz = - zz / Sigmadipsq;

            int ii = xg - nx * floor( xg / nx);
            int jj = yg - ny * floor( yg / ny);
            int kk = zg - nz * floor( zg / nz);

            int ind = ii + jj*int(nx) + kk*int(nx)*int(ny);
            Real r2 = xx*xx + yy*yy + zz*zz;
            Real temp = gx*gy*gz;
            Real temp2 = Real(0.5) * pdmag / Sigmasq;
            Real temp3 = temp2 / Sigmasq;
            Real temp4 = Real(3.0)*temp2;
            Real temp5 = temp*( Real(1.0) + temp3*r2 - temp4);

            atomicAdd(&fx[ind], Fx*temp5 + (g11*gradx + g12*grady + g13*gradz)*temp);
            atomicAdd(&fy[ind], Fy*temp5 + (g21*gradx + g22*grady + g23*gradz)*temp);
            atomicAdd(&fz[ind], Fz*temp5 + (g31*gradx + g32*grady + g33*gradz)*temp);
        }
    }
}

__global__
void cufcm_mono_dipole_distribution_bpp_shared_dynamic(myCufftReal *fx, myCufftReal *fy, myCufftReal *fz, Real *Y,
              Real *T, Real *F, int N, int ngd, 
              Real sigma, Real Sigma,
              Real dx, double nx, double ny, double nz){
    
    // TODO: GPU is more comfortable computing FP2 (double) than integer
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
    Real *grad_gaussx_dip_shared = (Real*)&gaussz_shared[ngd];
    Real *grad_gaussy_dip_shared = (Real*)&grad_gaussx_dip_shared[ngd];
    Real *grad_gaussz_dip_shared = (Real*)&grad_gaussy_dip_shared[ngd];
    Real *Y_shared = (Real*)&grad_gaussz_dip_shared[ngd];
    Real *F_shared = (Real*)&Y_shared[3];
    Real *g_shared = (Real*)&F_shared[3];

    Real Sigmasq = Sigma*Sigma;
    Real Sigmadipsq = Sigmasq;
    Real Anorm = Real(1.0)/sqrt(Real(PI2)*Sigmasq);
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
            Real xg = my_rint(Y_shared[0]/dx) - ngdh + fmodf(i, ngd);
            Real yg = my_rint(Y_shared[1]/dx) - ngdh + fmodf(i, ngd);
            Real zg = my_rint(Y_shared[2]/dx) - ngdh + fmodf(i, ngd);

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
            Real r2 = xdis_shared[i]*xdis_shared[i] + ydis_shared[j]*ydis_shared[j] + zdis_shared[k]*zdis_shared[k];
            Real temp = gaussx_shared[i]*gaussy_shared[j]*gaussz_shared[k];
            Real temp2 = Real(0.5) * pdmag / Sigmasq;
            Real temp3 = temp2 / Sigmasq;
            Real temp4 = Real(3.0)*temp2;
            Real temp5 = temp*( Real(1.0) + temp3*r2 - temp4);

            atomicAdd(&fx[ind], F_shared[0]*temp5 + (g_shared[0]*gradx + g_shared[3]*grady + g_shared[5]*gradz)*temp);
            atomicAdd(&fy[ind], F_shared[1]*temp5 + (g_shared[4]*gradx + g_shared[1]*grady + g_shared[7]*gradz)*temp);
            atomicAdd(&fz[ind], F_shared[2]*temp5 + (g_shared[6]*gradx + g_shared[8]*grady + g_shared[2]*gradz)*temp);

        }
    }
}


__global__
void cufcm_mono_dipole_distribution_selection(myCufftReal *fx, myCufftReal *fy, myCufftReal *fz, Real *Y,
              Real *T, Real *F, int N, int ngd, 
              Real sigma, Real Sigma,
              Real dx, double nx, double ny, double nz,
              int *particle_index, int start, int end){
    
    // TODO: GPU is more comfortable computing FP2 (double) than integer
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
    Real *grad_gaussx_dip_shared = (Real*)&gaussz_shared[ngd];
    Real *grad_gaussy_dip_shared = (Real*)&grad_gaussx_dip_shared[ngd];
    Real *grad_gaussz_dip_shared = (Real*)&grad_gaussy_dip_shared[ngd];
    Real *Y_shared = (Real*)&grad_gaussz_dip_shared[ngd];
    Real *F_shared = (Real*)&Y_shared[3];
    Real *g_shared = (Real*)&F_shared[3];

    Real Sigmasq = Sigma*Sigma;
    Real Sigmadipsq = Sigmasq;
    Real Anorm = Real(1.0)/sqrt(Real(PI2)*Sigmasq);
    Real width2 = (Real(2.0)*Sigmasq);
    Real pdmag = sigma*sigma - Sigmasq;

    for(int np = blockIdx.x; (np < N) && (particle_index[np] >= start && particle_index[np] < end); np += gridDim.x){

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
            Real xg = my_rint(Y_shared[0]/dx) - ngdh + fmodf(i, ngd);
            Real yg = my_rint(Y_shared[1]/dx) - ngdh + fmodf(i, ngd);
            Real zg = my_rint(Y_shared[2]/dx) - ngdh + fmodf(i, ngd);

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
            Real r2 = xdis_shared[i]*xdis_shared[i] + ydis_shared[j]*ydis_shared[j] + zdis_shared[k]*zdis_shared[k];
            Real temp = gaussx_shared[i]*gaussy_shared[j]*gaussz_shared[k];
            Real temp2 = Real(0.5) * pdmag / Sigmasq;
            Real temp3 = temp2 / Sigmasq;
            Real temp4 = Real(3.0)*temp2;
            Real temp5 = temp*( Real(1.0) + temp3*r2 - temp4);

            atomicAdd(&fx[ind], F_shared[0]*temp5 + (g_shared[0]*gradx + g_shared[3]*grady + g_shared[5]*gradz)*temp);
            atomicAdd(&fy[ind], F_shared[1]*temp5 + (g_shared[4]*gradx + g_shared[1]*grady + g_shared[7]*gradz)*temp);
            atomicAdd(&fz[ind], F_shared[2]*temp5 + (g_shared[6]*gradx + g_shared[8]*grady + g_shared[2]*gradz)*temp);

        }
    }
}

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
    Real *grad_gaussx_dip_shared = (Real*)&gaussz_shared[ngd];
    Real *grad_gaussy_dip_shared = (Real*)&grad_gaussx_dip_shared[ngd];
    Real *grad_gaussz_dip_shared = (Real*)&grad_gaussy_dip_shared[ngd];
    Real *Y_shared = (Real*)&grad_gaussz_dip_shared[ngd];
    Real *F_shared = (Real*)&Y_shared[3];

    Real Sigmasq = Sigma*Sigma;
    Real Anorm = Real(1.0)/sqrt(Real(PI2)*Sigmasq);
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
            Real xg = my_rint(Y_shared[0]/dx) - ngdh + fmodf(i, ngd);
            Real yg = my_rint(Y_shared[1]/dx) - ngdh + fmodf(i, ngd);
            Real zg = my_rint(Y_shared[2]/dx) - ngdh + fmodf(i, ngd);

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

__global__
void cufcm_mono_dipole_distribution_mono_selection(myCufftReal *fx, myCufftReal *fy, myCufftReal *fz,
              Real *Y, Real *F,
              int N, int ngd, 
              Real sigma, Real Sigma,
              Real dx, double nx, double ny, double nz,
              int *particle_index, int start, int end){
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
    Real Anorm = Real(1.0)/sqrt(Real(PI2)*Sigmasq);
    Real width2 = (Real(2.0)*Sigmasq);
    Real pdmag = sigma*sigma - Sigmasq;

    for(int np = blockIdx.x; (np < N) && (particle_index[np] >= start && particle_index[np] < end); np += gridDim.x){

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
            Real xg = my_rint(Y_shared[0]/dx) - ngdh + fmodf(i, ngd);
            Real yg = my_rint(Y_shared[1]/dx) - ngdh + fmodf(i, ngd);
            Real zg = my_rint(Y_shared[2]/dx) - ngdh + fmodf(i, ngd);

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

__global__
void cufcm_flow_solve(myCufftComplex* fk_x, myCufftComplex* fk_y, myCufftComplex* fk_z,
                      myCufftComplex* uk_x, myCufftComplex* uk_y, myCufftComplex* uk_z,
                      int nx, int ny, int nz, Real boxsize){
    const int i = threadIdx.x + blockIdx.x*blockDim.x;

    // TODO: removed all for loops!!!

    int fft_nx = nx/2 + 1;
    Real grid_size = nx*ny*nz;
    int fft_grid_size = fft_nx*ny*nz;

    if(i < fft_grid_size){
        const int indk = (i)/(ny*fft_nx);
        const int indj = (i - indk*(ny*fft_nx))/fft_nx;
        const int indi = i - (indk*ny + indj)*fft_nx;

        int nptsh = nx/2;
        Real q1 = ( (indi < nptsh || indi == nptsh)? Real(indi) : Real(indi - nx) ) * (Real(PI2)/boxsize);
        Real q2 = ( (indj < nptsh || indj == nptsh)? Real(indj) : Real(indj - ny) ) * (Real(PI2)/boxsize);
        Real q3 = ( (indk < nptsh || indk == nptsh)? Real(indk) : Real(indk - nz) ) * (Real(PI2)/boxsize);
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

/*
__global__
void cufcm_particle_velocities_tpp_recompute(myCufftReal *ux, myCufftReal *uy, myCufftReal *uz,
                                Real *Y,
                                Real *VTEMP, Real *WTEMP,
                                int N, int ngd, 
                                Real pdmag, Real sigmasq, Real sigmadipsq,
                                Real anorm, Real anorm2,
                                Real dx, Real nx, Real ny, Real nz){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    int xc, yc, zc;
    int xg, yg, zg;
    int i, j, k, ii, jj, kk;
    Real xx, yy, zz, r2;
    Real xx2, yy2, zz2;
    Real gx, gy, gz;
    Real norm, temp;
    Real ux_temp, uy_temp, uz_temp;
    Real temp2 = (Real)0.5 * pdmag / sigmasq;
    Real temp3 = temp2 / sigmasq;
    Real temp4 = (Real)3.0*temp2;
    Real temp5;
    int ind;
    int ngdh = ngd/2;

    norm = dx*dx*dx;

    for(int np = index; np < N; np += stride){
        xc = my_rint(Y[3*np + 0]/dx); // the index of the nearest grid point to the particle
        yc = my_rint(Y[3*np + 1]/dx);
        zc = my_rint(Y[3*np + 2]/dx);

        for(k = 0; k < ngd; k++){
            zg = zc - ngdh + (k);
            kk = zg - nz * ((int) floor( ((Real) zg) / ((Real) nz)));
            zz = ((Real) zg)*dx - Y[3*np + 2];
            zz2 = zz*zz;
            gz = anorm*my_exp(-zz*zz/anorm2);
            zz = - zz / sigmadipsq;
            for(j = 0; j < ngd; j++){
                yg = yc - ngdh + (j);
                jj = yg - ny * ((int) floor( ((Real) yg) / ((Real) ny)));
                yy = ((Real) yg)*dx - Y[3*np + 1];
                yy2 = yy*yy;
                gy = anorm*my_exp(-yy*yy/anorm2);
                yy = - yy / sigmadipsq;
                for(i = 0; i < ngd; i++){
                    xg = xc - ngdh + (i);
                    ii = xg - nx * ((int) floor( ((Real) xg) / ((Real) nx)));
                    xx = ((Real) xg)*dx - Y[3*np + 0];
                    xx2 = xx*xx;
                    gx = anorm*my_exp(-xx*xx/anorm2)*norm;
                    xx = - xx / sigmadipsq;
                    
                    ind = ii + jj*nx + kk*nx*ny;

                    r2 = xx2 + yy2 + zz2;
                    temp = gx*gy*gz;
                    temp5 = (1 + temp3*r2 - temp4);

                    ux_temp = ux[ind]*temp;
                    uy_temp = uy[ind]*temp;
                    uz_temp = uz[ind]*temp;

                    VTEMP[3*np + 0] += ux_temp*temp5;
                    VTEMP[3*np + 1] += uy_temp*temp5;
                    VTEMP[3*np + 2] += uz_temp*temp5;

                    WTEMP[3*np + 0] -= 0.5*(uz_temp*yy - uy_temp*zz);
                    WTEMP[3*np + 1] -= 0.5*(ux_temp*zz - uz_temp*xx);
                    WTEMP[3*np + 2] -= 0.5*(uy_temp*xx - ux_temp*yy);                 
                }
            }
        }
    }
}

__global__
void cufcm_particle_velocities_bpp_shared(myCufftReal *ux, myCufftReal *uy, myCufftReal *uz,
                                Real *Y,
                                Real *VTEMP, Real *WTEMP,
                                int N, int ngd, 
                                Real pdmag, Real sigmasq, Real sigmadipsq,
                                Real anorm, Real anorm2,
                                Real dx, Real nx, Real ny, Real nz){
    
    int ngdh = ngd/2;
    Real norm = dx*dx*dx;
    Real Vx = (Real) 0.0, Vy = (Real) 0.0, Vz = (Real) 0.0, Wx = (Real) 0.0, Wy = (Real) 0.0, Wz = (Real) 0.0;

    __shared__ int indx_shared[NGD];
    __shared__ int indy_shared[NGD];
    __shared__ int indz_shared[NGD];
    __shared__ Real xdis_shared[NGD];
    __shared__ Real ydis_shared[NGD];
    __shared__ Real zdis_shared[NGD];
    __shared__ Real gaussx_shared[NGD];
    __shared__ Real gaussy_shared[NGD];
    __shared__ Real gaussz_shared[NGD];
    __shared__ Real grad_gaussx_dip_shared[NGD];
    __shared__ Real grad_gaussy_dip_shared[NGD];
    __shared__ Real grad_gaussz_dip_shared[NGD];
    __shared__ Real Yx, Yy, Yz;
    // Specialize BlockReduce
    typedef cub::BlockReduce<Real, FCM_THREADS_PER_BLOCK> BlockReduce;
    // Allocate shared memory for BlockReduce
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    // TODO change to reduction
    for(int np = blockIdx.x; np < N; np += gridDim.x){
        if(threadIdx.x == 0){
            Yx = Y[3*np + 0];
            Yy = Y[3*np + 1];
            Yz = Y[3*np + 2];
        }
        __syncthreads();

        for(int i = threadIdx.x; i < 4*ngd; i += blockDim.x){
            Real xg = my_rint(Yx/dx) - ngdh + fmodf(i, ngd);
            Real yg = my_rint(Yy/dx) - ngdh + fmodf(i, ngd);
            Real zg = my_rint(Yz/dx) - ngdh + fmodf(i, ngd);

            Real xx = xg*dx - Yx;
            Real yy = yg*dx - Yy;
            Real zz = zg*dx - Yz;
            if(i<ngd){ 
                xdis_shared[i] = xx;
                ydis_shared[i] = yy;
                zdis_shared[i] = zz;
            }
            if(i>=ngd && i<2*ngd){
                gaussx_shared[i-ngd] = anorm*my_exp(-xx*xx/anorm2);
                gaussy_shared[i-ngd] = anorm*my_exp(-yy*yy/anorm2);
                gaussz_shared[i-ngd] = anorm*my_exp(-zz*zz/anorm2);
            }
            if(i>=2*ngd && i<3*ngd){
                grad_gaussx_dip_shared[i-2*ngd] = - xx / sigmadipsq;
                grad_gaussy_dip_shared[i-2*ngd] = - yy / sigmadipsq;
                grad_gaussz_dip_shared[i-2*ngd] = - zz / sigmadipsq;
            }
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

            Real gx = gaussx_shared[i];
            Real gy = gaussy_shared[j];
            Real gz = gaussz_shared[k];

            Real gradx = grad_gaussx_dip_shared[i];
            Real grady = grad_gaussy_dip_shared[j];
            Real gradz = grad_gaussz_dip_shared[k];
            
            int ind = indx_shared[i] + indy_shared[j]*nx + indz_shared[k]*nx*ny;
            Real r2 = xdis_shared[i]*xdis_shared[i] + ydis_shared[j]*ydis_shared[j] + zdis_shared[k]*zdis_shared[k];
            Real temp = gx*gy*gz*norm;
            Real temp2 = (Real)0.5 * pdmag / sigmasq;
            Real temp3 = temp2 / sigmasq;
            Real temp4 = (Real)3.0*temp2;
            Real temp5 = ((Real)1.0 + temp3*r2 - temp4);

            Real ux_temp = ux[ind]*temp;
            Real uy_temp = uy[ind]*temp;
            Real uz_temp = uz[ind]*temp;

            Vx += ux_temp*temp5;
            Vy += uy_temp*temp5;
            Vz += uz_temp*temp5;
            Wx += (Real)-0.5*(uz_temp*grady - uy_temp*gradz);
            Wy += (Real)-0.5*(ux_temp*gradz - uz_temp*gradx);
            Wz += (Real)-0.5*(uy_temp*gradx - ux_temp*grady);
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
*/

__global__
void cufcm_particle_velocities_bpp_recompute(myCufftReal *ux, myCufftReal *uy, myCufftReal *uz,
                                Real *Y, Real *VTEMP, Real *WTEMP,
                                int N, int ngd, 
                                Real sigma, Real Sigma,
                                Real dx, Real nx, Real ny, Real nz){

    int ngdh = ngd/2;

    Real norm = dx*dx*dx;
    Real Vx = (Real) 0.0, Vy = (Real) 0.0, Vz = (Real) 0.0, Wx = (Real) 0.0, Wy = (Real) 0.0, Wz = (Real) 0.0;
    Real Yx, Yy, Yz;

    Real Sigmasq = Sigma*Sigma;
    Real Sigmadipsq = Sigmasq;
    Real Anorm = Real(1.0)/sqrt(Real(PI2)*Sigmasq);
    Real width2 = (Real(2.0)*Sigmasq);
    Real pdmag = sigma*sigma - Sigmasq;

    // Specialize BlockReduce
    typedef cub::BlockReduce<Real, FCM_THREADS_PER_BLOCK> BlockReduce;
    // Allocate shared memory for BlockReduce
    __shared__ typename BlockReduce::TempStorage temp_storage;

    for(int np = blockIdx.x; np < N; np += gridDim.x){
        Yx = Y[3*np + 0];
        Yy = Y[3*np + 1];
        Yz = Y[3*np + 2];

        for(int t = threadIdx.x; t < ngd*ngd*ngd; t += blockDim.x){
            const int k = t/(ngd*ngd);
            const int j = (t - k*ngd*ngd)/ngd;
            const int i = t - k*ngd*ngd - j*ngd;
            
            Real xg = my_rint(Yx/dx) - ngdh + (i);
            Real yg = my_rint(Yy/dx) - ngdh + (j);
            Real zg = my_rint(Yz/dx) - ngdh + (k);

            Real xx = xg*dx - Yx;
            Real yy = yg*dx - Yy;
            Real zz = zg*dx - Yz;

            Real gx = Anorm*my_exp(-xx*xx/width2);
            Real gy = Anorm*my_exp(-yy*yy/width2);
            Real gz = Anorm*my_exp(-zz*zz/width2);

            Real gradx = - xx / Sigmadipsq;
            Real grady = - yy / Sigmadipsq;
            Real gradz = - zz / Sigmadipsq;

            int ii = xg - nx * floor( xg / nx);
            int jj = yg - ny * floor( yg / ny);
            int kk = zg - nz * floor( zg / nz);
            
            int ind = ii + jj*int(nx) + kk*int(nx)*int(ny);
            Real r2 = xx*xx + yy*yy + zz*zz;
            Real temp = gx*gy*gz*norm;
            Real temp2 = Real(0.5) * pdmag / Sigmasq;
            Real temp3 = temp2 / Sigmasq;
            Real temp4 = Real(3.0)*temp2;
            Real temp5 = ( Real(1.0) + temp3*r2 - temp4);

            Real ux_temp = ux[ind]*temp;
            Real uy_temp = uy[ind]*temp;
            Real uz_temp = uz[ind]*temp;

            Vx += ux_temp*temp5;
            Vy += uy_temp*temp5;
            Vz += uz_temp*temp5;
            Wx += Real(-0.5)*(uz_temp*grady - uy_temp*gradz);
            Wy += Real(-0.5)*(ux_temp*gradz - uz_temp*gradx);
            Wz += Real(-0.5)*(uy_temp*gradx - ux_temp*grady);

            // atomicAdd(&VTEMP[3*np + 0], ux_temp*temp5);
            // atomicAdd(&VTEMP[3*np + 1], uy_temp*temp5);
            // atomicAdd(&VTEMP[3*np + 2], uz_temp*temp5);

            // atomicAdd(&WTEMP[3*np + 0], -0.5*(uz_temp*grady - uy_temp*gradz));
            // atomicAdd(&WTEMP[3*np + 1], -0.5*(ux_temp*gradz - uz_temp*gradx));
            // atomicAdd(&WTEMP[3*np + 2], -0.5*(uy_temp*gradx - ux_temp*grady));                 
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

__global__
void cufcm_particle_velocities_bpp_shared_dynamic(myCufftReal *ux, myCufftReal *uy, myCufftReal *uz,
                                Real *Y, Real *VTEMP, Real *WTEMP,
                                int N, int ngd, 
                                Real sigma, Real Sigma,
                                Real dx, Real nx, Real ny, Real nz){
    
    int ngdh = ngd/2;
    Real norm = dx*dx*dx;
    Real Vx = (Real) 0.0, Vy = (Real) 0.0, Vz = (Real) 0.0, Wx = (Real) 0.0, Wy = (Real) 0.0, Wz = (Real) 0.0;

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
    Real *grad_gaussx_dip_shared = (Real*)&gaussz_shared[ngd];
    Real *grad_gaussy_dip_shared = (Real*)&grad_gaussx_dip_shared[ngd];
    Real *grad_gaussz_dip_shared = (Real*)&grad_gaussy_dip_shared[ngd];
    Real *Y_shared = (Real*)&grad_gaussz_dip_shared[ngd];

    Real Sigmasq = Sigma*Sigma;
    Real Sigmadipsq = Sigmasq;
    Real Anorm = Real(1.0)/sqrt(Real(PI2)*Sigmasq);
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
            Real xg = my_rint(Y_shared[0]/dx) - ngdh + fmodf(i, ngd);
            Real yg = my_rint(Y_shared[1]/dx) - ngdh + fmodf(i, ngd);
            Real zg = my_rint(Y_shared[2]/dx) - ngdh + fmodf(i, ngd);

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
            Wx += Real(-0.5)*(uz_temp*grady - uy_temp*gradz);
            Wy += Real(-0.5)*(ux_temp*gradz - uz_temp*gradx);
            Wz += Real(-0.5)*(uy_temp*gradx - ux_temp*grady);
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

__global__
void cufcm_particle_velocities_selection(myCufftReal *ux, myCufftReal *uy, myCufftReal *uz,
                                Real *Y, Real *VTEMP, Real *WTEMP,
                                int N, int ngd, 
                                Real sigma, Real Sigma,
                                Real dx, Real nx, Real ny, Real nz, 
                                int *particle_index, int start, int end){
    
    int ngdh = ngd/2;
    Real norm = dx*dx*dx;
    Real Vx = (Real) 0.0, Vy = (Real) 0.0, Vz = (Real) 0.0, Wx = (Real) 0.0, Wy = (Real) 0.0, Wz = (Real) 0.0;

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
    Real *grad_gaussx_dip_shared = (Real*)&gaussz_shared[ngd];
    Real *grad_gaussy_dip_shared = (Real*)&grad_gaussx_dip_shared[ngd];
    Real *grad_gaussz_dip_shared = (Real*)&grad_gaussy_dip_shared[ngd];
    Real *Y_shared = (Real*)&grad_gaussz_dip_shared[ngd];

    Real Sigmasq = Sigma*Sigma;
    Real Sigmadipsq = Sigmasq;
    Real Anorm = Real(1.0)/sqrt(Real(PI2)*Sigmasq);
    Real width2 = (Real(2.0)*Sigmasq);
    Real pdmag = sigma*sigma - Sigmasq;

    // Specialize BlockReduce
    typedef cub::BlockReduce<Real, FCM_THREADS_PER_BLOCK> BlockReduce;
    // Allocate shared memory for BlockReduce
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    for(int np = blockIdx.x; (np < N) && (particle_index[np] >= start && particle_index[np] < end); np += gridDim.x){
        if(threadIdx.x == 0){
            Y_shared[0] = Y[3*np + 0];
            Y_shared[1] = Y[3*np + 1];
            Y_shared[2] = Y[3*np + 2];
        }
        __syncthreads();

        for(int i = threadIdx.x; i < 4*ngd; i += blockDim.x){
            Real xg = my_rint(Y_shared[0]/dx) - ngdh + fmodf(i, ngd);
            Real yg = my_rint(Y_shared[1]/dx) - ngdh + fmodf(i, ngd);
            Real zg = my_rint(Y_shared[2]/dx) - ngdh + fmodf(i, ngd);

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
            Wx += Real(-0.5)*(uz_temp*grady - uy_temp*gradz);
            Wy += Real(-0.5)*(ux_temp*gradz - uz_temp*gradx);
            Wz += Real(-0.5)*(uy_temp*gradx - ux_temp*grady);
        }
        
        // Reduction
        Real total_Vx = BlockReduce(temp_storage).Sum(Vx);
        Real total_Vy = BlockReduce(temp_storage).Sum(Vy);
        Real total_Vz = BlockReduce(temp_storage).Sum(Vz);
        Real total_Wx = BlockReduce(temp_storage).Sum(Wx);
        Real total_Wy = BlockReduce(temp_storage).Sum(Wy);
        Real total_Wz = BlockReduce(temp_storage).Sum(Wz);
    
        if(threadIdx.x==0){
            VTEMP[3*np + 0] += total_Vx;  
            VTEMP[3*np + 1] += total_Vy;
            VTEMP[3*np + 2] += total_Vz;
            WTEMP[3*np + 0] += total_Wx;
            WTEMP[3*np + 1] += total_Wy;
            WTEMP[3*np + 2] += total_Wz;
        }
    }
}

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
    Real Anorm = Real(1.0)/sqrt(Real(PI2)*Sigmasq);
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
            Real xg = my_rint(Y_shared[0]/dx) - ngdh + fmodf(i, ngd);
            Real yg = my_rint(Y_shared[1]/dx) - ngdh + fmodf(i, ngd);
            Real zg = my_rint(Y_shared[2]/dx) - ngdh + fmodf(i, ngd);

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

__global__
void cufcm_particle_velocities_mono_selection(myCufftReal *ux, myCufftReal *uy, myCufftReal *uz,
                                Real *Y,
                                Real *VTEMP,
                                int N, int ngd, 
                                Real sigma, Real Sigma,
                                Real dx, Real nx, Real ny, Real nz,
                                int *particle_index, int start, int end){
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
    Real Anorm = Real(1.0)/sqrt(Real(PI2)*Sigmasq);
    Real width2 = (Real(2.0)*Sigmasq);
    Real pdmag = sigma*sigma - Sigmasq;

    // Specialize BlockReduce
    typedef cub::BlockReduce<Real, FCM_THREADS_PER_BLOCK> BlockReduce;
    // Allocate shared memory for BlockReduce
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    for(int np = blockIdx.x; (np < N) && (particle_index[np] >= start && particle_index[np] < end); np += gridDim.x){
        if(threadIdx.x == 0){
            Y_shared[0] = Y[3*np + 0];
            Y_shared[1] = Y[3*np + 1];
            Y_shared[2] = Y[3*np + 2];
        }
        __syncthreads();

        for(int i = threadIdx.x; i < 4*ngd; i += blockDim.x){
            Real xg = my_rint(Y_shared[0]/dx) - ngdh + fmodf(i, ngd);
            Real yg = my_rint(Y_shared[1]/dx) - ngdh + fmodf(i, ngd);
            Real zg = my_rint(Y_shared[2]/dx) - ngdh + fmodf(i, ngd);

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
            VTEMP[3*np + 0] += total_Vx;  
            VTEMP[3*np + 1] += total_Vy;
            VTEMP[3*np + 2] += total_Vz;
        }
    }                                  
}

///////////////////////////////////////////////////////////////////////////////
// Regular FCM
///////////////////////////////////////////////////////////////////////////////
__global__
void cufcm_mono_dipole_distribution_regular_fcm(myCufftReal *fx, myCufftReal *fy, myCufftReal *fz, Real *Y,
              Real *T, Real *F, int N, int ngd,
              Real sigma, Real sigmadip,
              Real dx, Real nx, Real ny, Real nz){

    int ngdh = ngd/2;

    extern __shared__ Integer ls[];
    Integer *indx_shared = (Integer*)ls;
    Integer *indy_shared = (Integer*)&indx_shared[ngd];
    Integer *indz_shared = (Integer*)&indy_shared[ngd];
    Real *gaussx_shared = (Real*)&indz_shared[ngd]; 
    Real *gaussy_shared = (Real*)&gaussx_shared[ngd];
    Real *gaussz_shared = (Real*)&gaussy_shared[ngd];
    Real *gaussx_dip_shared = (Real*)&gaussz_shared[ngd]; 
    Real *gaussy_dip_shared = (Real*)&gaussx_dip_shared[ngd];
    Real *gaussz_dip_shared = (Real*)&gaussy_dip_shared[ngd];
    Real *grad_gaussx_dip_shared = (Real*)&gaussz_dip_shared[ngd];
    Real *grad_gaussy_dip_shared = (Real*)&grad_gaussx_dip_shared[ngd];
    Real *grad_gaussz_dip_shared = (Real*)&grad_gaussy_dip_shared[ngd];
    Real *Y_shared = (Real*)&grad_gaussz_dip_shared[ngd];
    Real *F_shared = (Real*)&Y_shared[3];
    Real *g_shared = (Real*)&F_shared[3];

    Real sigmasq = sigma*sigma;
    Real sigmadipsq = sigmadip*sigmadip;
    Real anorm = Real(1.0)/sqrt(Real(2.0)*Real(PI)*sigmasq);
    Real anormdip = Real(1.0)/sqrt(Real(2.0)*Real(PI)*sigmadipsq);
    
    for(int np = blockIdx.x; np < N; np += gridDim.x){

        if(threadIdx.x == 0){
            Y_shared[0] = Y[3*np + 0];
            Y_shared[1] = Y[3*np + 1];
            Y_shared[2] = Y[3*np + 2];

            F_shared[0] = F[3*np + 0];
            F_shared[1] = F[3*np + 1];
            F_shared[2] = F[3*np + 2];

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
            Real xg = my_rint(Y_shared[0]/dx) - ngdh + fmodf(i, ngd);
            Real yg = my_rint(Y_shared[1]/dx) - ngdh + fmodf(i, ngd);
            Real zg = my_rint(Y_shared[2]/dx) - ngdh + fmodf(i, ngd);

            Real xx = xg*dx - Y_shared[0];
            Real yy = yg*dx - Y_shared[1];
            Real zz = zg*dx - Y_shared[2];
            
            // gauss
            if(i<ngd){ 
                gaussx_shared[i] = anorm*my_exp(-xx*xx/(Real(2.0)*sigmasq));
                gaussy_shared[i] = anorm*my_exp(-yy*yy/(Real(2.0)*sigmasq));
                gaussz_shared[i] = anorm*my_exp(-zz*zz/(Real(2.0)*sigmasq));
            }
            // gauss dip
            if(i>=ngd && i<2*ngd){
                gaussx_dip_shared[i-ngd] = anormdip*my_exp(-xx*xx/(Real(2.0)*sigmadipsq));
                gaussy_dip_shared[i-ngd] = anormdip*my_exp(-yy*yy/(Real(2.0)*sigmadipsq));
                gaussz_dip_shared[i-ngd] = anormdip*my_exp(-zz*zz/(Real(2.0)*sigmadipsq));
            }
            // grad_gauss
            if(i>=2*ngd && i<3*ngd){
                grad_gaussx_dip_shared[i-2*ngd] = - xx / sigmadipsq;
                grad_gaussy_dip_shared[i-2*ngd] = - yy / sigmadipsq;
                grad_gaussz_dip_shared[i-2*ngd] = - zz / sigmadipsq;
            }
            // ind
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
            Real temp = gaussx_shared[i]*gaussy_shared[j]*gaussz_shared[k];
            Real tempdip = gaussx_dip_shared[i]*gaussy_dip_shared[j]*gaussz_dip_shared[k];

            atomicAdd(&fx[ind], F_shared[0]*temp + (g_shared[0]*gradx + g_shared[3]*grady + g_shared[5]*gradz)*tempdip);
            atomicAdd(&fy[ind], F_shared[1]*temp + (g_shared[4]*gradx + g_shared[1]*grady + g_shared[7]*gradz)*tempdip);
            atomicAdd(&fz[ind], F_shared[2]*temp + (g_shared[6]*gradx + g_shared[8]*grady + g_shared[2]*gradz)*tempdip);
        }
    }
}

__global__
void cufcm_particle_velocities_regular_fcm(myCufftReal *ux, myCufftReal *uy, myCufftReal *uz,
                                Real *Y,
                                Real *VTEMP, Real *WTEMP,
                                int N, int ngd, 
                                Real sigma, Real sigmadip,
                                Real dx, Real nx, Real ny, Real nz){
    int ngdh = ngd/2;
    Real norm = dx*dx*dx;
    Real Vx = (Real) 0.0, Vy = (Real) 0.0, Vz = (Real) 0.0, Wx = (Real) 0.0, Wy = (Real) 0.0, Wz = (Real) 0.0;

    extern __shared__ Integer ls[];
    Integer *indx_shared = ls;
    Integer *indy_shared = (Integer*)&indx_shared[ngd];
    Integer *indz_shared = (Integer*)&indy_shared[ngd];
    Real *gaussx_shared = (Real*)&indz_shared[ngd]; 
    Real *gaussy_shared = (Real*)&gaussx_shared[ngd];
    Real *gaussz_shared = (Real*)&gaussy_shared[ngd];
    Real *gaussx_dip_shared = (Real*)&gaussz_shared[ngd]; 
    Real *gaussy_dip_shared = (Real*)&gaussx_dip_shared[ngd];
    Real *gaussz_dip_shared = (Real*)&gaussy_dip_shared[ngd];
    Real *grad_gaussx_dip_shared = (Real*)&gaussz_dip_shared[ngd];
    Real *grad_gaussy_dip_shared = (Real*)&grad_gaussx_dip_shared[ngd];
    Real *grad_gaussz_dip_shared = (Real*)&grad_gaussy_dip_shared[ngd];
    Real *Y_shared = (Real*)&grad_gaussz_dip_shared[ngd];

    Real sigmasq = sigma*sigma;
    Real sigmadipsq = sigmadip*sigmadip;
    Real anorm = Real(1.0)/sqrt(Real(2.0)*Real(PI)*sigmasq);
    Real anormdip = Real(1.0)/sqrt(Real(2.0)*Real(PI)*sigmadipsq);

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
            Real xg = my_rint(Y_shared[0]/dx) - ngdh + fmodf(i, ngd);
            Real yg = my_rint(Y_shared[1]/dx) - ngdh + fmodf(i, ngd);
            Real zg = my_rint(Y_shared[2]/dx) - ngdh + fmodf(i, ngd);

            Real xx = xg*dx - Y_shared[0];
            Real yy = yg*dx - Y_shared[1];
            Real zz = zg*dx - Y_shared[2];
            
            // gauss
            if(i<ngd){ 
                gaussx_shared[i] = anorm*my_exp(-xx*xx/(Real(2.0)*sigmasq));
                gaussy_shared[i] = anorm*my_exp(-yy*yy/(Real(2.0)*sigmasq));
                gaussz_shared[i] = anorm*my_exp(-zz*zz/(Real(2.0)*sigmasq));
            }
            // gauss dip
            if(i>=ngd && i<2*ngd){
                gaussx_dip_shared[i-ngd] = anormdip*my_exp(-xx*xx/(Real(2.0)*sigmadipsq));
                gaussy_dip_shared[i-ngd] = anormdip*my_exp(-yy*yy/(Real(2.0)*sigmadipsq));
                gaussz_dip_shared[i-ngd] = anormdip*my_exp(-zz*zz/(Real(2.0)*sigmadipsq));
            }
            // grad_gauss
            if(i>=2*ngd && i<3*ngd){
                grad_gaussx_dip_shared[i-2*ngd] = - xx / sigmadipsq;
                grad_gaussy_dip_shared[i-2*ngd] = - yy / sigmadipsq;
                grad_gaussz_dip_shared[i-2*ngd] = - zz / sigmadipsq;
            }
            // ind
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
            Real temp = gaussx_shared[i]*gaussy_shared[j]*gaussz_shared[k]*norm;
            Real tempdip = gaussx_dip_shared[i]*gaussy_dip_shared[j]*gaussz_dip_shared[k]*norm;

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
