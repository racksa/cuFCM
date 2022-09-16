#include <iostream>

#include "config.hpp"
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

    int xc, yc, zc;
    int xg, yg, zg;
    Real xx, yy, zz;
    Real gx, gy, gz;
    Real temp;
    int ind;
    int ngdh = ngd/2;
    int ngd3 = ngd*ngd*ngd;

    for(int np = blockIdx.x; np < N; np += gridDim.x){

        if(threadIdx.x == 0){
            Yx = Y[3*np + 0];
            Yy = Y[3*np + 1];
            Yz = Y[3*np + 2];

            Fx = F[np];
        }
        __syncthreads();

        xc = round(Yx/dx); // the index of the nearest grid point to the particle
        yc = round(Yy/dx);
        zc = round(Yz/dx);

        for(int i = threadIdx.x; i < ngd; i += blockDim.x){
            xg = xc - ngdh + (i);
            yg = yc - ngdh + (i);
            zg = zc - ngdh + (i);

            xx = ((Real) xg)*dx - Yx;
            yy = ((Real) yg)*dx - Yy;
            zz = ((Real) zg)*dx - Yz;
            
            // gauss
            gaussx_shared[i] = anorm*exp(-xx*xx/anorm2);
            gaussy_shared[i] = anorm*exp(-yy*yy/anorm2);
            gaussz_shared[i] = anorm*exp(-zz*zz/anorm2);
            // ind
            indx_shared[i] = xg - npts * floor( ((Real) xg) / ((Real) npts));
            indy_shared[i] = yg - npts * floor( ((Real) yg) / ((Real) npts));
            indz_shared[i] = zg - npts * floor( ((Real) zg) / ((Real) npts));
        }
        __syncthreads();
        
        for(int t = threadIdx.x; t < ngd3; t += blockDim.x){
            const int k = t/(ngd*ngd);
            const int j = (t - k*ngd*ngd)/ngd;
            const int i = t - k*ngd*ngd - j*ngd;
  
            gx = gaussx_shared[i];
            gy = gaussy_shared[j];
            gz = gaussz_shared[k];

            ind = indx_shared[i] + indy_shared[j]*npts + indz_shared[k]*npts*npts;
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

    int xc, yc, zc;
    int xg, yg, zg;
    Real xx, yy, zz;
    Real gx, gy, gz;
    Real temp;
    int ind;
    int ngdh = ngd/2;
    int ngd3 = ngd*ngd*ngd;

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

        xc = round(Yx/dx); // the index of the nearest grid point to the particle
        yc = round(Yy/dx);
        zc = round(Yz/dx);

        for(int i = threadIdx.x; i < ngd; i += blockDim.x){
            xg = xc - ngdh + (i);
            yg = yc - ngdh + (i);
            zg = zc - ngdh + (i);

            xx = ((Real) xg)*dx - Yx;
            yy = ((Real) yg)*dx - Yy;
            zz = ((Real) zg)*dx - Yz;
            
            // gauss
            gaussx_shared[i] = anorm*exp(-xx*xx/anorm2);
            gaussy_shared[i] = anorm*exp(-yy*yy/anorm2);
            gaussz_shared[i] = anorm*exp(-zz*zz/anorm2);

            // ind
            indx_shared[i] = xg - npts * floor( ((Real) xg) / ((Real) npts));
            indy_shared[i] = yg - npts * floor( ((Real) yg) / ((Real) npts));
            indz_shared[i] = zg - npts * floor( ((Real) zg) / ((Real) npts));
        }
        __syncthreads();
        
        for(int t = threadIdx.x; t < ngd3; t += blockDim.x){
            const int k = t/(ngd*ngd);
            const int j = (t - k*ngd*ngd)/ngd;
            const int i = t - k*ngd*ngd - j*ngd;
  
            gx = gaussx_shared[i];
            gy = gaussy_shared[j];
            gz = gaussz_shared[k];

            ind = indx_shared[i] + indy_shared[j]*npts + indz_shared[k]*npts*npts;
            temp = gx*gy*gz;

            atomicAdd(&fx[ind], Fx*temp);
            atomicAdd(&fy[ind], Fy*temp);
            atomicAdd(&fz[ind], Fz*temp);
        }
    }
}