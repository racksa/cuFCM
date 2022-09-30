
#include <iostream>
// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>

#include "config.hpp"
#include "config_fcm.hpp"
#include "CUFCM_FCM.cuh"
#include <cub/cub.cuh>


///////////////////////////////////////////////////////////////////////////////
// Fast FCM
///////////////////////////////////////////////////////////////////////////////
__global__
void cufcm_precompute_gauss(int N, int ngd, Real* Y,
                    Real* gaussx, Real* gaussy, Real* gaussz,
                    Real* grad_gaussx_dip, Real* grad_gaussy_dip, Real* grad_gaussz_dip,
                    Real* gaussgrid,
                    Real* xdis, Real* ydis, Real* zdis,
                    int* indx, int* indy, int* indz,
                    Real sigmadipsq, Real anorm, Real anorm2, Real dx, Real nx, Real ny, Real nz){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    int i, xc, yc, zc;
    int xg, yg, zg;
    int ngdh = ngd/2;

    Real xx;
    Real xxc, yyc, zzc;
    Real E2x, E2y, E2z, E3;
    Real anorm3, dxanorm2;

    anorm3 = anorm*anorm*anorm;
    dxanorm2 = dx/anorm2;

    // part1
    for(i = 0; i < ngd; i++){
        gaussgrid[i] = exp(-(i+1-ngdh)*(i+1-ngdh)*dx*dxanorm2);
    }

    for(int np = index; np < N; np += stride){
        xc = round(Y[3*np + 0]/dx); // the index of the nearest grid point to the particle
        yc = round(Y[3*np + 1]/dx);
        zc = round(Y[3*np + 2]/dx);

        xxc = (Real)xc*dx - Y[3*np + 0]; // distance to the nearest point (ksi-Y)
        yyc = (Real)yc*dx - Y[3*np + 1];
        zzc = (Real)zc*dx - Y[3*np + 2];

        // part2
        E2x = exp(-2*xxc*dxanorm2);
        E2y = exp(-2*yyc*dxanorm2);
        E2z = exp(-2*zzc*dxanorm2);

        // part3
        E3 = anorm3*exp(-(xxc*xxc + yyc*yyc + zzc*zzc)/anorm2);

        // old function
        for(i = 0; i < ngd; i++){
            xg = xc - ngdh + (i); 
            indx[ngd*np + i] = xg - NX * ((int) floor( ((Real) xg) / ((Real) NX)));
            xx = ((Real) xg)*dx-Y[3*np + 0];
            gaussx[ngd*np + i] = E3*int_pow(E2x,i+1-ngdh)*gaussgrid[i];
            grad_gaussx_dip[ngd*np + i] = - xx / sigmadipsq;
            xdis[ngd*np + i] = xx*xx;

            yg = yc - ngdh + (i);
            indy[ngd*np + i] = yg - NX * ((int) floor( ((Real) yg) / ((Real) NX)));
            xx = ((Real) yg)*dx - Y[3*np + 1];
            gaussy[ngd*np + i] = int_pow(E2y,i+1-ngdh)*gaussgrid[i];
            grad_gaussy_dip[ngd*np + i] = - xx / sigmadipsq;
            ydis[ngd*np + i] = xx*xx;

            zg = zc - ngdh + (i);
            indz[ngd*np + i] = zg - NX * ((int) floor( ((Real) zg) / ((Real) NX)));
            xx = ((Real) zg)*dx-Y[3*np + 2];
            gaussz[ngd*np + i] = int_pow(E2z,i+1-ngdh)*gaussgrid[i];
            grad_gaussz_dip[ngd*np + i] = - xx / sigmadipsq;
            zdis[ngd*np + i] = xx*xx;
        }
    }
    return;
}

__global__
void cufcm_mono_dipole_distribution_tpp_register(myCufftReal *fx, myCufftReal *fy, myCufftReal *fz, int N,
              Real *T, Real *F, Real pdmag, Real sigmasq, 
              Real *gaussx, Real *gaussy, Real *gaussz,
              Real *grad_gaussx_dip, Real *grad_gaussy_dip, Real *grad_gaussz_dip,
              Real *xdis, Real *ydis, Real *zdis,
              int *indx, int *indy, int *indz,
              int ngd, Real nx, Real ny, Real nz){

    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    int i, j, k, ii, jj, kk;
    Real xx, yy, zz, r2, temp;
    Real xx2, yy2, zz2;
    Real g11, g22, g33, g12, g21, g13, g31, g23, g32;
    Real gx, gy, gz, Fx, Fy, Fz;
    Real g11xx, g22yy, g33zz, g12yy, g21xx, g13zz, g31xx, g23zz, g32yy;
    Real temp2 = (Real)0.5 * pdmag / sigmasq;
    Real temp3 = temp2 /sigmasq;
    Real temp4 = (Real)3.0*temp2;
    Real temp5;
    int ind;

    for(int np = index; np < N; np += stride){
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
            kk = indz[ngd*np + k];
            zz = grad_gaussz_dip[ngd*np + k];
            zz2 = zdis[ngd*np + k];
            gz = gaussz[ngd*np + k];
            g13zz = g13*zz;
            g23zz = g23*zz;
            g33zz = g33*zz;
            for(j = 0; j < ngd; j++){
                jj = indy[ngd*np + j];
                yy = grad_gaussy_dip[ngd*np + j];
                yy2 = ydis[ngd*np + j];
                gy = gaussy[ngd*np + j];
                g12yy = g12*yy;
                g22yy = g22*yy;
                g32yy = g32*yy;
                for(i = 0; i < ngd; i++){
                    ii = indx[ngd*np + i];
                    xx = grad_gaussx_dip[ngd*np + i];
                    xx2 = xdis[ngd*np + i];
                    gx = gaussx[ngd*np + i];
                    g11xx = g11*xx;
                    g21xx = g21*xx;
                    g31xx = g31*xx;
                
                    ind = ii + jj*NX + kk*NX*NY;

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
        xc = round(Y[3*np + 0]/dx); // the index of the nearest grid point to the particle
        yc = round(Y[3*np + 1]/dx);
        zc = round(Y[3*np + 2]/dx);

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
            gz = anorm*exp(-zz*zz/anorm2);
            zz = - zz / sigmadipsq;
            g13zz = g13*zz;
            g23zz = g23*zz;
            g33zz = g33*zz;
            for(j = 0; j < ngd; j++){
                yg = yc - ngdh + (j);
                jj = yg - ny * ((int) floor( ((Real) yg) / ((Real) ny)));
                yy = ((Real) yg)*dx - Y[3*np + 1];
                yy2 = yy*yy;
                gy = anorm*exp(-yy*yy/anorm2);
                yy = - yy / sigmadipsq;
                g12yy = g12*yy;
                g22yy = g22*yy;
                g32yy = g32*yy;
                for(i = 0; i < ngd; i++){
                    xg = xc - ngdh + (i);
                    ii = xg - nx * ((int) floor( ((Real) xg) / ((Real) nx)));
                    xx = ((Real) xg)*dx - Y[3*np + 0];
                    xx2 = xx*xx;
                    gx = anorm*exp(-xx*xx/anorm2);
                    xx = - xx / sigmadipsq;
                    g11xx = g11*xx;
                    g21xx = g21*xx;
                    g31xx = g31*xx;
                
                    ind = ii + jj*NX + kk*NX*NY;

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
            
            Real xg = rintf(Yx/dx) - ngdh + fmodf(i, ngd);
            Real yg = rintf(Yy/dx) - ngdh + fmodf(i, ngd);
            Real zg = rintf(Yz/dx) - ngdh + fmodf(i, ngd);

            Real xx = xg*dx - Yx;
            Real yy = yg*dx - Yy;
            Real zz = zg*dx - Yz;
            /* dis */
            if(i<ngd){ 
                xdis_shared[i] = xx;
                ydis_shared[i] = yy;
                zdis_shared[i] = zz;
            }
            /* gauss */
            if(i>=ngd && i<2*ngd){
                gaussx_shared[i-ngd] = anorm*exp(-xx*xx/anorm2);
                gaussy_shared[i-ngd] = anorm*exp(-yy*yy/anorm2);
                gaussz_shared[i-ngd] = anorm*exp(-zz*zz/anorm2);
            }
            /* grad_gauss */
            if(i>=2*ngd && i<3*ngd){
                grad_gaussx_dip_shared[i-2*ngd] = - xx / sigmadipsq;
                grad_gaussy_dip_shared[i-2*ngd] = - yy / sigmadipsq;
                grad_gaussz_dip_shared[i-2*ngd] = - zz / sigmadipsq;
            }
            /* ind */
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

__global__
void cufcm_mono_dipole_distribution_bpp_recompute(myCufftReal *fx, myCufftReal *fy, myCufftReal *fz, Real *Y,
              Real *T, Real *F, int N, int ngd, 
              Real pdmag, Real sigmasq, Real sigmadipsq,
              Real anorm, Real anorm2,
              Real dx, Real nx, Real ny, Real nz){

    
    
    int ngdh = ngd/2;

    Real Yx, Yy, Yz;
    Real Fx, Fy, Fz;
    Real g11, g22, g33, g12, g21, g13, g31, g23, g32;

    for(int np = blockIdx.x; np < N; np += gridDim.x){
        Yx = Y[3*np + 0];
        Yy = Y[3*np + 1];
        Yz = Y[3*np + 2];

        Fx = F[3*np + 0];
        Fy = F[3*np + 1];
        Fz = F[3*np + 2];

        g11 = + 0.0;
        g22 = + 0.0;
        g33 = + 0.0;
        g12 = + 0.5*T[3*np + 2];
        g21 = - 0.5*T[3*np + 2];
        g13 = + (-0.5*T[3*np + 1]);
        g31 = - (-0.5*T[3*np + 1]);
        g23 = + 0.5*T[3*np + 0];
        g32 = - 0.5*T[3*np + 0];
        
        for(int t = threadIdx.x; t < ngd*ngd*ngd; t += blockDim.x){
            const int k = t/(ngd*ngd);
            const int j = (t - k*ngd*ngd)/ngd;
            const int i = t - k*ngd*ngd - j*ngd;
            
            Real xg = round(Yx/dx) - ngdh + (i);
            Real yg = round(Yy/dx) - ngdh + (j);
            Real zg = round(Yz/dx) - ngdh + (k);

            Real xx = xg*dx - Yx;
            Real yy = yg*dx - Yy;
            Real zz = zg*dx - Yz;

            Real gx = anorm*exp(-xx*xx/anorm2);
            Real gy = anorm*exp(-yy*yy/anorm2);
            Real gz = anorm*exp(-zz*zz/anorm2);

            Real gradx = - xx / sigmadipsq;
            Real grady = - yy / sigmadipsq;
            Real gradz = - zz / sigmadipsq;

            int ii = xg - NX * floor( xg / NX);
            int jj = yg - NY * floor( yg / NY);
            int kk = zg - NZ * floor( zg / NZ);

            int ind = ii + jj*NX + kk*NX*NY;
            Real r2 = xx*xx + yy*yy + zz*zz;
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

__global__
void cufcm_mono_dipole_distribution_bpp_shared_dynamic(myCufftReal *fx, myCufftReal *fy, myCufftReal *fz, Real *Y,
              Real *T, Real *F, int N, int ngd, 
              Real pdmag, Real sigmasq, Real sigmadipsq,
              Real anorm, Real anorm2,
              Real dx, double nx, double ny, double nz){

    // TODO: GPU is more comfortable computing FP2 (double) than integer
    int ngdh = ngd/2;

    extern __shared__ int s[];
    int *indx_shared = s;
    int *indy_shared = (int*)&indx_shared[ngd];
    int *indz_shared = (int*)&indy_shared[ngd];
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
            Real xg = rintf(Y_shared[0]/dx) - ngdh + fmodf(i, ngd);
            Real yg = rintf(Y_shared[1]/dx) - ngdh + fmodf(i, ngd);
            Real zg = rintf(Y_shared[2]/dx) - ngdh + fmodf(i, ngd);

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
                gaussx_shared[i-ngd] = anorm*expf(-xx*xx/anorm2);
                gaussy_shared[i-ngd] = anorm*expf(-yy*yy/anorm2);
                gaussz_shared[i-ngd] = anorm*expf(-zz*zz/anorm2);
            }
            /* grad_gauss */
            if(i>=2*ngd && i<3*ngd){
                grad_gaussx_dip_shared[i-2*ngd] = - xx / sigmadipsq;
                grad_gaussy_dip_shared[i-2*ngd] = - yy / sigmadipsq;
                grad_gaussz_dip_shared[i-2*ngd] = - zz / sigmadipsq;
            }
            /* ind */
            if(i>=3*ngd){
                indx_shared[i-3*ngd] = xg - nx * floorf( xg / nx );
                indy_shared[i-3*ngd] = yg - ny * floorf( yg / ny );
                indz_shared[i-3*ngd] = zg - nz * floorf( zg / nz );
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
            Real temp2 = Real(0.5) * pdmag / sigmasq;
            Real temp3 = temp2 /sigmasq;
            Real temp4 = Real(3.0)*temp2;
            Real temp5 = temp*( Real(1.0) + temp3*r2 - temp4);

            atomicAdd(&fx[ind], F_shared[0]*temp5 + (g_shared[0]*gradx + g_shared[3]*grady + g_shared[5]*gradz)*temp);
            atomicAdd(&fy[ind], F_shared[1]*temp5 + (g_shared[4]*gradx + g_shared[1]*grady + g_shared[7]*gradz)*temp);
            atomicAdd(&fz[ind], F_shared[2]*temp5 + (g_shared[6]*gradx + g_shared[8]*grady + g_shared[2]*gradz)*temp);
        }
    }
}

__global__
void cufcm_flow_solve(myCufftComplex* fk_x, myCufftComplex* fk_y, myCufftComplex* fk_z,
                      myCufftComplex* uk_x, myCufftComplex* uk_y, myCufftComplex* uk_z,
                      int nx, int ny, int nz){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    int fft_nx = nx/2 + 1;
    int grid_size = nx*ny*nz;
    int fft_grid_size = fft_nx*ny*nz;

    // Stay in the loop as long as any thread in the block still needs to compute velocities.
    for(int i = index; i < fft_grid_size; i += stride){
        const int indk = (i)/(ny*fft_nx);
        const int indj = (i - indk*(ny*fft_nx))/fft_nx;
        const int indi = i - indk*(ny*fft_nx) - indj*fft_nx;

        // Real q1 = q[indi];
        // Real q2 = q[indj];
        // Real q3 = q[indk];
        // Real qq = qsq[indi] + qsq[indj] + qsq[indk];
        // Real norm = (Real)1.0/(qq);

        int nptsh = nx/2;
        Real q1 = (indi < nptsh || indi == nptsh)? Real(indi) : Real(indi - nx);
        Real q2 = (indj < nptsh || indj == nptsh)? Real(indj) : Real(indj - ny);
        Real q3 = (indk < nptsh || indk == nptsh)? Real(indk) : Real(indk - nz);
        Real qq = q1*q1 + q2*q2 + q3*q3;
        Real norm = (Real)1.0/(qq);


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

        Real kdotf_re = (q1*f1_re+q2*f2_re+q3*f3_re)*norm;
        Real kdotf_im = (q1*f1_im+q2*f2_im+q3*f3_im)*norm;

        uk_x[i].x = norm*(f1_re-q1*(kdotf_re))/((Real)grid_size);
        uk_x[i].y = norm*(f1_im-q1*(kdotf_im))/((Real)grid_size);
        uk_y[i].x = norm*(f2_re-q2*(kdotf_re))/((Real)grid_size);
        uk_y[i].y = norm*(f2_im-q2*(kdotf_im))/((Real)grid_size);
        uk_z[i].x = norm*(f3_re-q3*(kdotf_re))/((Real)grid_size);
        uk_z[i].y = norm*(f3_im-q3*(kdotf_im))/((Real)grid_size);

        if(i==0){
            uk_x[0].x = (Real)0.0;
            uk_x[0].y = (Real)0.0;
            uk_y[0].x = (Real)0.0;
            uk_y[0].y = (Real)0.0;
            uk_z[0].x = (Real)0.0;
            uk_z[0].y = (Real)0.0;
        }
    }// End of striding loop over filament segment velocities.
    __syncthreads();
    return;
}

__global__
void cufcm_particle_velocities_tpp_register(myCufftReal *ux, myCufftReal *uy, myCufftReal *uz, int N,
                               Real *VTEMP, Real *WTEMP,
                               Real pdmag, Real sigmasq, 
                               Real *gaussx, Real *gaussy, Real *gaussz,
                               Real *grad_gaussx_dip, Real *grad_gaussy_dip, Real *grad_gaussz_dip,
                               Real *xdis, Real *ydis, Real *zdis,
                               int *indx, int *indy, int *indz,
                               int ngd, Real dx, Real nx, Real ny, Real nz){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    int i, j, k, ii, jj, kk;
    Real norm, temp;
    Real gx, gy, gz;
    Real ux_temp, uy_temp, uz_temp;
    Real xx, yy, zz;
    Real xx2, yy2, zz2;
    Real r2;
    Real temp2 = (Real)0.5 * pdmag / sigmasq;
    Real temp3 = temp2 / sigmasq;
    Real temp4 = (Real)3.0*temp2;
    Real temp5;
    int ind;

    norm = dx*dx*dx;

    for(int np = index; np < N; np += stride){
        for(k = 0; k < ngd; k++){
            kk = indz[ngd*np + k];
            gz = gaussz[ngd*np + k];
            zz = grad_gaussz_dip[ngd*np + k];
            zz2 = zdis[ngd*np + k];
            for(j = 0; j < ngd; j++){
                jj = indy[ngd*np + j];
                gy = gaussy[ngd*np + j];
                yy = grad_gaussy_dip[ngd*np + j];
                yy2 = ydis[ngd*np + j];
                for(i = 0; i < ngd; i++){
                    ii = indx[ngd*np + i];
                    gx = gaussx[ngd*np + i]*norm;
                    xx = grad_gaussx_dip[ngd*np + i];
                    xx2 = xdis[ngd*np + i];

                    ind = ii + jj*NX + kk*NX*NY;

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
        xc = round(Y[3*np + 0]/dx); // the index of the nearest grid point to the particle
        yc = round(Y[3*np + 1]/dx);
        zc = round(Y[3*np + 2]/dx);

        for(k = 0; k < ngd; k++){
            zg = zc - ngdh + (k);
            kk = zg - nz * ((int) floor( ((Real) zg) / ((Real) nz)));
            zz = ((Real) zg)*dx - Y[3*np + 2];
            zz2 = zz*zz;
            gz = anorm*exp(-zz*zz/anorm2);
            zz = - zz / sigmadipsq;
            for(j = 0; j < ngd; j++){
                yg = yc - ngdh + (j);
                jj = yg - ny * ((int) floor( ((Real) yg) / ((Real) ny)));
                yy = ((Real) yg)*dx - Y[3*np + 1];
                yy2 = yy*yy;
                gy = anorm*exp(-yy*yy/anorm2);
                yy = - yy / sigmadipsq;
                for(i = 0; i < ngd; i++){
                    xg = xc - ngdh + (i);
                    ii = xg - nx * ((int) floor( ((Real) xg) / ((Real) nx)));
                    xx = ((Real) xg)*dx - Y[3*np + 0];
                    xx2 = xx*xx;
                    gx = anorm*exp(-xx*xx/anorm2)*norm;
                    xx = - xx / sigmadipsq;
                    
                    ind = ii + jj*NX + kk*NX*NY;

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
    typedef cub::BlockReduce<Real, THREADS_PER_BLOCK> BlockReduce;
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
            Real xg = rintf(Yx/dx) - ngdh + fmodf(i, ngd);
            Real yg = rintf(Yy/dx) - ngdh + fmodf(i, ngd);
            Real zg = rintf(Yz/dx) - ngdh + fmodf(i, ngd);

            Real xx = xg*dx - Yx;
            Real yy = yg*dx - Yy;
            Real zz = zg*dx - Yz;
            /* dis */
            if(i<ngd){ 
                xdis_shared[i] = xx;
                ydis_shared[i] = yy;
                zdis_shared[i] = zz;
            }
            /* gauss */
            if(i>=ngd && i<2*ngd){
                gaussx_shared[i-ngd] = anorm*expf(-xx*xx/anorm2);
                gaussy_shared[i-ngd] = anorm*expf(-yy*yy/anorm2);
                gaussz_shared[i-ngd] = anorm*expf(-zz*zz/anorm2);
            }
            /* grad_gauss */
            if(i>=2*ngd && i<3*ngd){
                grad_gaussx_dip_shared[i-2*ngd] = - xx / sigmadipsq;
                grad_gaussy_dip_shared[i-2*ngd] = - yy / sigmadipsq;
                grad_gaussz_dip_shared[i-2*ngd] = - zz / sigmadipsq;
            }
            /* ind */
            if(i>=3*ngd){
                indx_shared[i-3*ngd] = xg - nx * floorf( xg / nx);
                indy_shared[i-3*ngd] = yg - ny * floorf( yg / ny);
                indz_shared[i-3*ngd] = zg - nz * floorf( zg / nz);
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

__global__
void cufcm_particle_velocities_bpp_recompute(myCufftReal *ux, myCufftReal *uy, myCufftReal *uz,
                                Real *Y,
                                Real *VTEMP, Real *WTEMP,
                                int N, int ngd, 
                                Real pdmag, Real sigmasq, Real sigmadipsq,
                                Real anorm, Real anorm2,
                                Real dx, Real nx, Real ny, Real nz){

    int xc, yc, zc;
    int xg, yg, zg;
    Real xx, yy, zz, r2;
    Real gradx, grady, gradz;
    Real gx, gy, gz;
    Real norm, temp;
    Real ux_temp, uy_temp, uz_temp;
    Real temp2 = (Real)0.5 * pdmag / sigmasq;
    Real temp3 = temp2 / sigmasq;
    Real temp4 = (Real)3.0*temp2;
    Real temp5;
    int ind;
    int ngdh = ngd/2;
    int ngd3 = ngd*ngd*ngd;

    norm = dx*dx*dx;

    Real Vx = (Real) 0.0, Vy = (Real) 0.0, Vz = (Real) 0.0, Wx = (Real) 0.0, Wy = (Real) 0.0, Wz = (Real) 0.0;

    // Specialize BlockReduce
    typedef cub::BlockReduce<Real, THREADS_PER_BLOCK> BlockReduce;
    // Allocate shared memory for BlockReduce
    __shared__ typename BlockReduce::TempStorage temp_storage;

    for(int np = blockIdx.x; np < N; np += gridDim.x){
        xc = round(Y[3*np + 0]/dx); // the index of the nearest grid point to the particle
        yc = round(Y[3*np + 1]/dx);
        zc = round(Y[3*np + 2]/dx);

        for(int t = threadIdx.x; t < ngd3; t += blockDim.x){
            const int k = t/(ngd*ngd);
            const int j = (t - k*ngd*ngd)/ngd;
            const int i = t - k*ngd*ngd - j*ngd;
            
            xg = xc - ngdh + (i);
            yg = yc - ngdh + (j);
            zg = zc - ngdh + (k);

            int ii = xg - nx * ((int) floor( ((Real) xg) / ((Real) nx)));
            int jj = yg - ny * ((int) floor( ((Real) yg) / ((Real) ny)));
            int kk = zg - nz * ((int) floor( ((Real) zg) / ((Real) nz)));

            xx = ((Real) xg)*dx - Y[3*np + 0];
            yy = ((Real) yg)*dx - Y[3*np + 1];
            zz = ((Real) zg)*dx - Y[3*np + 2];

            gx = anorm*exp(-xx*xx/anorm2);
            gy = anorm*exp(-yy*yy/anorm2);
            gz = anorm*exp(-zz*zz/anorm2);

            gradx = - xx / sigmadipsq;
            grady = - yy / sigmadipsq;
            gradz = - zz / sigmadipsq;
            
            ind = ii + jj*NX + kk*NX*NY;
            r2 = xx*xx + yy*yy + zz*zz;
            temp = gx*gy*gz*norm;
            temp5 = (1 + temp3*r2 - temp4);

            ux_temp = ux[ind]*temp;
            uy_temp = uy[ind]*temp;
            uz_temp = uz[ind]*temp;

            Vx += ux_temp*temp5;
            Vy += uy_temp*temp5;
            Vz += uz_temp*temp5;
            Wx += -0.5*(uz_temp*grady - uy_temp*gradz);
            Wy += -0.5*(ux_temp*gradz - uz_temp*gradx);
            Wz += -0.5*(uy_temp*gradx - ux_temp*grady);

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
                                Real *Y,
                                Real *VTEMP, Real *WTEMP,
                                int N, int ngd, 
                                Real pdmag, Real sigmasq, Real sigmadipsq,
                                Real anorm, Real anorm2,
                                Real dx, Real nx, Real ny, Real nz){
    
    int ngdh = ngd/2;
    Real norm = dx*dx*dx;
    Real Vx = (Real) 0.0, Vy = (Real) 0.0, Vz = (Real) 0.0, Wx = (Real) 0.0, Wy = (Real) 0.0, Wz = (Real) 0.0;

    extern __shared__ int s[];
    int *indx_shared = s;
    int *indy_shared = (int*)&indx_shared[ngd];
    int *indz_shared = (int*)&indy_shared[ngd];
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

    // Specialize BlockReduce
    typedef cub::BlockReduce<Real, THREADS_PER_BLOCK> BlockReduce;
    // Allocate shared memory for BlockReduce
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    // TODO change to reduction
    for(int np = blockIdx.x; np < N; np += gridDim.x){
        if(threadIdx.x == 0){
            Y_shared[0] = Y[3*np + 0];
            Y_shared[1] = Y[3*np + 1];
            Y_shared[2] = Y[3*np + 2];
        }
        __syncthreads();

        for(int i = threadIdx.x; i < 4*ngd; i += blockDim.x){
            Real xg = rintf(Y_shared[0]/dx) - ngdh + fmodf(i, ngd);
            Real yg = rintf(Y_shared[1]/dx) - ngdh + fmodf(i, ngd);
            Real zg = rintf(Y_shared[2]/dx) - ngdh + fmodf(i, ngd);

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
                gaussx_shared[i-ngd] = anorm*expf(-xx*xx/anorm2);
                gaussy_shared[i-ngd] = anorm*expf(-yy*yy/anorm2);
                gaussz_shared[i-ngd] = anorm*expf(-zz*zz/anorm2);
            }
            /* grad_gauss */
            if(i>=2*ngd && i<3*ngd){
                grad_gaussx_dip_shared[i-2*ngd] = - xx / sigmadipsq;
                grad_gaussy_dip_shared[i-2*ngd] = - yy / sigmadipsq;
                grad_gaussz_dip_shared[i-2*ngd] = - zz / sigmadipsq;
            }
            /* ind */
            if(i>=3*ngd){
                indx_shared[i-3*ngd] = xg - nx * floorf( xg / nx);
                indy_shared[i-3*ngd] = yg - ny * floorf( yg / ny);
                indz_shared[i-3*ngd] = zg - nz * floorf( zg / nz);
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
            Real temp2 = Real(0.5) * pdmag / sigmasq;
            Real temp3 = temp2 /sigmasq;
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

///////////////////////////////////////////////////////////////////////////////
// Regular FCM
///////////////////////////////////////////////////////////////////////////////
__global__
void cufcm_mono_dipole_distribution_regular_fcm(myCufftReal *fx, myCufftReal *fy, myCufftReal *fz, Real *Y,
              Real *T, Real *F, int N, int ngd, 
              Real sigmasq, Real sigmadipsq,
              Real anorm, Real anorm2,
              Real anormdip, Real anormdip2,
              Real dx, Real nx, Real ny, Real nz){

    int ngdh = ngd/2;

    extern __shared__ int s[];
    int *indx_shared = s;
    int *indy_shared = (int*)&indx_shared[ngd];
    int *indz_shared = (int*)&indy_shared[ngd];
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
            Real xg = rintf(Y_shared[0]/dx) - ngdh + fmodf(i, ngd);
            Real yg = rintf(Y_shared[1]/dx) - ngdh + fmodf(i, ngd);
            Real zg = rintf(Y_shared[2]/dx) - ngdh + fmodf(i, ngd);

            Real xx = xg*dx - Y_shared[0];
            Real yy = yg*dx - Y_shared[1];
            Real zz = zg*dx - Y_shared[2];
            
            // gauss
            if(i<ngd){ 
                gaussx_shared[i] = anorm*expf(-xx*xx/anorm2);
                gaussy_shared[i] = anorm*expf(-yy*yy/anorm2);
                gaussz_shared[i] = anorm*expf(-zz*zz/anorm2);
            }
            // gauss dip
            if(i>=ngd && i<2*ngd){
                gaussx_dip_shared[i-ngd] = anormdip*expf(-xx*xx/anormdip2);
                gaussy_dip_shared[i-ngd] = anormdip*expf(-yy*yy/anormdip2);
                gaussz_dip_shared[i-ngd] = anormdip*expf(-zz*zz/anormdip2);
            }
            // grad_gauss
            if(i>=2*ngd && i<3*ngd){
                grad_gaussx_dip_shared[i-2*ngd] = - xx / sigmadipsq;
                grad_gaussy_dip_shared[i-2*ngd] = - yy / sigmadipsq;
                grad_gaussz_dip_shared[i-2*ngd] = - zz / sigmadipsq;
            }
            // ind
            if(i>=3*ngd){
                indx_shared[i-3*ngd] = xg - nx * floorf( xg / nx);
                indy_shared[i-3*ngd] = yg - ny * floorf( yg / ny);
                indz_shared[i-3*ngd] = zg - nz * floorf( zg / nz);
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
                                Real sigmasq, Real sigmadipsq,
                                Real anorm, Real anorm2,
                                Real anormdip, Real anormdip2,
                                Real dx, Real nx, Real ny, Real nz){
    int ngdh = ngd/2;
    Real norm = dx*dx*dx;
    Real Vx = (Real) 0.0, Vy = (Real) 0.0, Vz = (Real) 0.0, Wx = (Real) 0.0, Wy = (Real) 0.0, Wz = (Real) 0.0;

    extern __shared__ int s[];
    int *indx_shared = s;
    int *indy_shared = (int*)&indx_shared[ngd];
    int *indz_shared = (int*)&indy_shared[ngd];
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

    // Specialize BlockReduce
    typedef cub::BlockReduce<Real, THREADS_PER_BLOCK> BlockReduce;
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
            Real xg = rintf(Y_shared[0]/dx) - ngdh + fmodf(i, ngd);
            Real yg = rintf(Y_shared[1]/dx) - ngdh + fmodf(i, ngd);
            Real zg = rintf(Y_shared[2]/dx) - ngdh + fmodf(i, ngd);

            Real xx = xg*dx - Y_shared[0];
            Real yy = yg*dx - Y_shared[1];
            Real zz = zg*dx - Y_shared[2];
            
            // gauss
            if(i<ngd){ 
                gaussx_shared[i] = anorm*expf(-xx*xx/anorm2);
                gaussy_shared[i] = anorm*expf(-yy*yy/anorm2);
                gaussz_shared[i] = anorm*expf(-zz*zz/anorm2);
            }
            // gauss dip
            if(i>=ngd && i<2*ngd){
                gaussx_dip_shared[i-ngd] = anormdip*expf(-xx*xx/anormdip2);
                gaussy_dip_shared[i-ngd] = anormdip*expf(-yy*yy/anormdip2);
                gaussz_dip_shared[i-ngd] = anormdip*expf(-zz*zz/anormdip2);
            }
            // grad_gauss
            if(i>=2*ngd && i<3*ngd){
                grad_gaussx_dip_shared[i-2*ngd] = - xx / sigmadipsq;
                grad_gaussy_dip_shared[i-2*ngd] = - yy / sigmadipsq;
                grad_gaussz_dip_shared[i-2*ngd] = - zz / sigmadipsq;
            }
            // ind
            if(i>=3*ngd){
                indx_shared[i-3*ngd] = xg - nx * floorf( xg / nx);
                indy_shared[i-3*ngd] = yg - ny * floorf( yg / ny);
                indz_shared[i-3*ngd] = zg - nz * floorf( zg / nz);
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

///////////////////////////////////////////////////////////////////////////////
// CPU code
///////////////////////////////////////////////////////////////////////////////
void cufcm_precompute_gauss_loop(int N, int ngd, Real* Y,
                    Real* gaussx, Real* gaussy, Real* gaussz,
                    Real* grad_gaussx_dip, Real* grad_gaussy_dip, Real* grad_gaussz_dip,
                    Real* gaussgrid,
                    Real* xdis, Real* ydis, Real* zdis,
                    int* indx, int* indy, int* indz,
                    Real sigmadipsq, Real anorm, Real anorm2, Real dx, Real nx, Real ny, Real nz){
    int np, i, xc, yc, zc;
    int xg, yg, zg;
    int ngdh = ngd/2;

    Real xx;
    Real xxc, yyc, zzc;
    Real E2x, E2y, E2z, E3;
    Real anorm3, dxanorm2;

    anorm3 = anorm*anorm*anorm;
    dxanorm2 = dx/anorm2;

    // part1
    for(i = 0; i < ngd; i++){
        gaussgrid[i] = exp(-(i+1-ngdh)*(i+1-ngdh)*dx*dxanorm2);
    }

    for(np = 0; np < N; np++){
        xc = round(Y[3*np + 0]/dx); // the index of the nearest grid point to the particle
        yc = round(Y[3*np + 1]/dx);
        zc = round(Y[3*np + 2]/dx);

        xxc = (Real)xc*dx - Y[3*np + 0]; // distance to the nearest point (ksi-Y)
        yyc = (Real)yc*dx - Y[3*np + 1];
        zzc = (Real)zc*dx - Y[3*np + 2];

        // part2
        E2x = exp(-2*xxc*dxanorm2);
        E2y = exp(-2*yyc*dxanorm2);
        E2z = exp(-2*zzc*dxanorm2);

        // part3
        E3 = anorm3*exp(-(xxc*xxc + yyc*yyc + zzc*zzc)/anorm2);

        // old function
        for(i = 0; i < ngd; i++){
            xg = xc - ngdh + (i); 
            indx[ngd*np + i] = xg - NX * ((int) floor( ((Real) xg) / ((Real) NX)));
            xx = ((Real) xg)*dx-Y[3*np + 0];
            gaussx[ngd*np + i] = E3*int_pow(E2x,i+1-ngdh)*gaussgrid[i];
            grad_gaussx_dip[ngd*np + i] = - xx / sigmadipsq;
            xdis[ngd*np + i] = xx*xx;

            yg = yc - ngdh + (i);
            indy[ngd*np + i] = yg - NX * ((int) floor( ((Real) yg) / ((Real) NX)));
            xx = ((Real) yg)*dx - Y[3*np + 1];
            gaussy[ngd*np + i] = int_pow(E2y,i+1-ngdh)*gaussgrid[i];
            grad_gaussy_dip[ngd*np + i] = - xx / sigmadipsq;
            ydis[ngd*np + i] = xx*xx;

            zg = zc - ngdh + (i);
            indz[ngd*np + i] = zg - NX * ((int) floor( ((Real) zg) / ((Real) NX)));
            xx = ((Real) zg)*dx-Y[3*np + 2];
            gaussz[ngd*np + i] = int_pow(E2z,i+1-ngdh)*gaussgrid[i];
            grad_gaussz_dip[ngd*np + i] = - xx / sigmadipsq;
            zdis[ngd*np + i] = xx*xx;
        }
    }
    return;
}

void cufcm_mono_dipole_distribution_tpp_loop(myCufftReal *fx, myCufftReal *fy, myCufftReal *fz, int N,
              Real *T, Real *F, Real pdmag, Real sigmasq, 
              Real *gaussx, Real *gaussy, Real *gaussz,
              Real *grad_gaussx_dip, Real *grad_gaussy_dip, Real *grad_gaussz_dip,
              Real *xdis, Real *ydis, Real *zdis,
              int *indx, int *indy, int *indz,
              int ngd, Real nx, Real ny, Real nz){
    int np, i, j, k, ii, jj, kk;
    Real xx, yy, zz, r2, temp;
    Real xx2, yy2, zz2;
    Real g11, g22, g33, g12, g21, g13, g31, g23, g32;
    Real gx, gy, gz, Fx, Fy, Fz;
    Real g11xx, g22yy, g33zz, g12yy, g21xx, g13zz, g31xx, g23zz, g32yy;
    Real smallx = 1e-18;
    int ind;
    Real temp2 = (Real)0.5 * pdmag / sigmasq;
    Real temp3 = temp2 /sigmasq;
    Real temp4 = (Real)3.0*temp2;
    Real temp5;

    for(np = 0; np < N; np++){
        Fx = F[3*np + 0];
        Fy = F[3*np + 1];
        Fz = F[3*np + 2];
        g11 = + 0.0;
        g22 = + 0.0;
        g33 = + 0.0;
        g12 = + 0.5*T[3*np + 2];
        g21 = - 0.5*T[3*np + 2];
        g13 = + (-0.5*T[3*np + 1]);
        g31 = - (-0.5*T[3*np + 1]);
        g23 = + 0.5*T[3*np + 0];
        g32 = - 0.5*T[3*np + 0];
        for(i = 0; i < ngd; i++){
            ii = indx[ngd*np + i];
            xx = grad_gaussx_dip[ngd*np + i];
            xx2 = xdis[ngd*np + i];
            gx = gaussx[ngd*np + i];
            g11xx = g11*xx;
            g21xx = g21*xx;
            g31xx = g31*xx;
            for(j = 0; j < ngd; j++){
                jj = indy[ngd*np + j];
                yy = grad_gaussy_dip[ngd*np + j];
                yy2 = ydis[ngd*np + j];
                gy = gaussy[ngd*np + j];
                g12yy = g12*yy;
                g22yy = g22*yy;
                g32yy = g32*yy;
                for(k = 0; k < ngd; k++){
                    kk = indz[ngd*np + k];
                    zz = grad_gaussz_dip[ngd*np + k];
                    zz2 = zdis[ngd*np + k];
                    gz = gaussz[ngd*np + k];
                    g13zz = g13*zz;
                    g23zz = g23*zz;
                    g33zz = g33*zz;

                    ind = ii + jj*NX + kk*NX*NY;

                    r2 = xx2 + yy2 + zz2;
                    temp = gx*gy*gz;
                    temp5 = temp*( 1 + temp3*r2 - temp4);
                    // printf("(%d %d %d) %lf\n", ii, jj, kk, temp);

                    fx[ind] += Fx*temp5 + (g11xx + g12yy + g13zz)*temp + smallx;
                    fy[ind] += Fy*temp5 + (g21xx + g22yy + g23zz)*temp + smallx;
                    fz[ind] += Fz*temp5 + (g31xx + g32yy + g33zz)*temp + smallx;
                }
            }
        }
    }
}

void cufcm_particle_velocities_loop(myCufftReal *ux, myCufftReal *uy, myCufftReal *uz, int N,
                               Real *VTEMP, Real *WTEMP,
                               Real pdmag, Real sigmasq, 
                               Real *gaussx, Real *gaussy, Real *gaussz,
                               Real *grad_gaussx_dip, Real *grad_gaussy_dip, Real *grad_gaussz_dip,
                               Real *xdis, Real *ydis, Real *zdis,
                               int *indx, int *indy, int *indz,
                               int ngd, Real dx, Real nx, Real ny, Real nz){
    int np, i, j, k, ii, jj, kk;
    Real norm, temp;
    Real gx, gy, gz;
    Real ux_temp, uy_temp, uz_temp;
    Real xx, yy, zz;
    Real xx2, yy2, zz2;
    Real r2;
    Real temp2 = 0.5 * pdmag / sigmasq;
    Real temp3 = temp2 / sigmasq;
    Real temp4 = 3.0*temp2;
    Real temp5;
    int ind;

    norm = dx*dx*dx;

    for(np = 0; np < N; np++){
        for(i = 0; i < ngd; i++){
            ii = indx[ngd*np + i];
            gx = gaussx[ngd*np + i]*norm;
            xx = grad_gaussx_dip[ngd*np + i];
            xx2 = xdis[ngd*np + i];
            for(j = 0; j < ngd; j++){
                jj = indy[ngd*np + j];
                gy = gaussy[ngd*np + j];
                yy = grad_gaussy_dip[ngd*np + j];
                yy2 = ydis[ngd*np + j];
                for(k = 0; k < ngd; k++){
                    kk = indz[ngd*np + k];
                    gz = gaussz[ngd*np + k];
                    zz = grad_gaussz_dip[ngd*np + k];
                    zz2 = zdis[ngd*np + k];

                    ind = ii + jj*NX + kk*NX*NY;

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

__device__ __host__
Real int_pow(Real base, int power){
    /* fast power function for integer powers */
    Real result = 1;
    if(power>=0){
        for(int i = 0; i < power; i++){
            result = result * base;
    }
    }
    if(power<0){
        for(int i = 0; i < -power; i++){
            result = result * base;
    }
        result = (Real) 1/result;
    }
    return result;
}

///////////////////////////////////////////////////////////////////////////////
// Test functions
///////////////////////////////////////////////////////////////////////////////
__global__
void cufcm_test_force(myCufftReal* fx, myCufftReal* fy, myCufftReal* fz, Real nx, Real ny, Real nz){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    // Stay in the loop as long as any thread in the block still needs to spread forces.
    for(int i = index; i < nx*ny*nz; i += stride){
        const int indk = (i)/(ny*nx);
        const int indj = (i - indk*(ny*nx))/(nx);
        const int indi = i - indk*(ny*nx) - indj*(nx);

        fx[i] = 1 + 3*(indi+indj*indk) + 7*(indj+2) + 3*(indk+2);
        fy[i] = 1 + 2*(indi+indj+indk) + 3*(indj*indj) + 2*(indk*indi);
        fz[i] = 1 + 3*(indi*indj) + 7*(indj*indi) + 4*(indk*indj);
    }// End of striding loop over filament segment velocities.

    __syncthreads();
    return;
}

void cufcm_test_force_loop(myCufftReal* fx, myCufftReal* fy, myCufftReal* fz, Real nx, Real ny, Real nz){
    for(int k=0; k<NZ; k++){
        for(int j=0; j<NY; j++){
            for(int i=0; k<NX; i++){
                const int index = i + j*NX + k*NX*NY;

                fx[index] = 1 + 3*(i+j*k) + 7*(j+2) + 3*(k+2);
                fy[index] = 1 + 2*(i+j+k) + 3*(j*j) + 2*(k*i);
                fz[index] = 1 + 3*(i*j) + 7*(j*i) + 4*(k*j);
            }
        }
    }
}

__global__
void normalise_array(myCufftReal* ux, myCufftReal* uy, myCufftReal* uz, Real nx, Real ny, Real nz){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;
    const Real temp = 1.0/((Real)GRID_SIZE);

    // Stay in the loop as long as any thread in the block still needs to compute velocities.
    for(int i = index; i < GRID_SIZE; i += stride){
        ux[i] *= temp;
        uy[i] *= temp;
        uz[i] *= temp;

    }// End of striding loop over filament segment velocities.

    __syncthreads();
}
