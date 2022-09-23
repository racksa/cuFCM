#include <iostream>

#include "../config.hpp"
#include "../config_fcm.hpp"
#include "CUFCM_INCORPORATE.hpp"


__global__
void cufcm_mono_distribution_single_fx(Real *fx, Real *Y,
              Real *F, int N, int ngd, 
              Real sigmasq,
              Real anorm, Real anorm2,
              Real dx, int npts){
    
    __shared__ Real Yx, Yy, Yz;
    __shared__ Real Fx;
    __shared__ Real gaussx_shared[NGD_UAMMD];
    __shared__ Real gaussy_shared[NGD_UAMMD];
    __shared__ Real gaussz_shared[NGD_UAMMD];
    __shared__ int indx_shared[NGD_UAMMD];
    __shared__ int indy_shared[NGD_UAMMD];
    __shared__ int indz_shared[NGD_UAMMD];

    int ngdh = ngd/2;

    for(int np = blockIdx.x; np < N; np += gridDim.x){

        if(threadIdx.x == 0){
            Yx = Y[3*np + 0];
            Yy = Y[3*np + 1];
            Yz = Y[3*np + 2];

            Fx = F[np];
        }
        __syncthreads();


        for(int i = threadIdx.x; i < 4*ngd; i += blockDim.x){
            Real xg = round(Yx/dx) - ngdh + fmodf(i, ngd);
            Real yg = round(Yy/dx) - ngdh + fmodf(i, ngd);
            Real zg = round(Yz/dx) - ngdh + fmodf(i, ngd);

            Real xx = xg*dx - Yx;
            Real yy = yg*dx - Yy;
            Real zz = zg*dx - Yz;
            
            if(i<ngd){
                gaussx_shared[i] = anorm*exp(-xx*xx/anorm2);
            }
            if(i>=ngd && i<2*ngd){
                gaussy_shared[i-ngd] = anorm*exp(-yy*yy/anorm2);
            }
            if(i>=2*ngd && i<3*ngd){
                gaussz_shared[i-2*ngd] = anorm*exp(-zz*zz/anorm2);
            }
            if(i>=3*ngd && i<4*ngd){
                indx_shared[i-3*ngd] = xg - npts * floor( xg / ((Real) npts));
                indy_shared[i-3*ngd] = yg - npts * floor( yg / ((Real) npts));
                indz_shared[i-3*ngd] = zg - npts * floor( zg / ((Real) npts));
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

            int ind = indx_shared[i] + indy_shared[j]*npts + indz_shared[k]*npts*npts;
            Real temp = gx*gy*gz;

            atomicAdd(&fx[ind], Fx*temp);
        }
    }
}


__global__
void cufcm_mono_distribution_single_fx_recompute(Real *fx, Real *Y,
              Real *F, int N, int ngd, 
              Real sigmasq,
              Real anorm, Real anorm2,
              Real dx, int npts){

    int xc, yc, zc;
    int xg, yg, zg;
    Real xx, yy, zz;
    Real gx, gy, gz;
    Real temp;
    int ind;
    int ngdh = ngd/2;
    int ngd3 = ngd*ngd*ngd;

    Real Yx, Yy, Yz;
    Real Fx;

    for(int np = blockIdx.x; np < N; np += gridDim.x){

        Yx = Y[3*np + 0];
        Yy = Y[3*np + 1];
        Yz = Y[3*np + 2];

        Fx = F[np];

        xc = round(Yx/dx); // the index of the nearest grid point to the particle
        yc = round(Yy/dx);
        zc = round(Yz/dx);

        for(int t = threadIdx.x; t < ngd3; t += blockDim.x){
            const int k = t/(ngd*ngd);
            const int j = (t - k*ngd*ngd)/ngd;
            const int i = t - k*ngd*ngd - j*ngd;

            xg = xc - ngdh + (i);
            yg = yc - ngdh + (j);
            zg = zc - ngdh + (k);

            xx = ((Real) xg)*dx - Yx;
            yy = ((Real) yg)*dx - Yy;
            zz = ((Real) zg)*dx - Yz;
  
            gx = anorm*exp(-xx*xx/anorm2);
            gy = anorm*exp(-yy*yy/anorm2);
            gz = anorm*exp(-zz*zz/anorm2);

            int ii = xg - npts * ((int) floor( ((Real) xg) / ((Real) npts)));
            int jj = yg - npts * ((int) floor( ((Real) yg) / ((Real) npts)));
            int kk = zg - npts * ((int) floor( ((Real) zg) / ((Real) npts)));

            ind = ii + jj*npts + kk*npts*npts;

            temp = gx*gy*gz;

            atomicAdd(&fx[ind], Fx*temp);
        }
    }
}

__global__
void cufcm_mono_distribution_regular_fxfyfz(myCufftReal *fx, myCufftReal *fy, myCufftReal *fz,
              Real *Y,
              Real *F, int N, int ngd, 
              Real sigmasq,
              Real anorm, Real anorm2,
              Real dx, int npts){
    
    __shared__ Real Yx, Yy, Yz;
    __shared__ Real Fx, Fy, Fz;
    __shared__ Real gaussx_shared[NGD_UAMMD];
    __shared__ Real gaussy_shared[NGD_UAMMD];
    __shared__ Real gaussz_shared[NGD_UAMMD];
    __shared__ int indx_shared[NGD_UAMMD];
    __shared__ int indy_shared[NGD_UAMMD];
    __shared__ int indz_shared[NGD_UAMMD];

    int ngdh = ngd/2;

    for(int np = blockIdx.x; np < N; np += gridDim.x){

        if(threadIdx.x == 0){
            Yx = Y[3*np + 0];
            Yy = Y[3*np + 1];
            Yz = Y[3*np + 2];

            Fx = F[3*np + 0];
            Fy = F[3*np + 1];
            Fz = F[3*np + 2];
        }
        __syncthreads();

        for(int i = threadIdx.x; i < 4*ngd; i += blockDim.x){
            Real xg = round(Yx/dx) - ngdh + fmodf(i, ngd);
            Real yg = round(Yy/dx) - ngdh + fmodf(i, ngd);
            Real zg = round(Yz/dx) - ngdh + fmodf(i, ngd);

            Real xx = xg*dx - Yx;
            Real yy = yg*dx - Yy;
            Real zz = zg*dx - Yz;
            
            if(i<ngd){
                gaussx_shared[i] = anorm*exp(-xx*xx/anorm2);
            }
            if(i>=ngd && i<2*ngd){
                gaussy_shared[i-ngd] = anorm*exp(-yy*yy/anorm2);
            }
            if(i>=2*ngd && i<3*ngd){
                gaussz_shared[i-2*ngd] = anorm*exp(-zz*zz/anorm2);
            }
            if(i>=3*ngd && i<4*ngd){
                indx_shared[i-3*ngd] = xg - npts * floor( xg / ((Real) npts));
                indy_shared[i-3*ngd] = yg - npts * floor( yg / ((Real) npts));
                indz_shared[i-3*ngd] = zg - npts * floor( zg / ((Real) npts));
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

            int ind = indx_shared[i] + indy_shared[j]*npts + indz_shared[k]*npts*npts;
            Real temp = gx*gy*gz;

            atomicAdd(&fx[ind], Fx*temp);
            atomicAdd(&fy[ind], Fy*temp);
            atomicAdd(&fz[ind], Fz*temp);
        }
    }
}