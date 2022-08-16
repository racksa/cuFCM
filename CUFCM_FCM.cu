#include <iostream>
// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>

#include "config.hpp"
#include "CUFCM_FCM.hpp"

__global__
void cufcm_test_force(cufftReal* fx, cufftReal* fy, cufftReal* fz){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    // Stay in the loop as long as any thread in the block still needs to spread forces.
    for(int i = index; i < GRID_SIZE; i += stride){
        const int indk = (i)/(NY*NX);
        const int indj = (i - indk*(NY*NX))/(NX);
        const int indi = i - indk*(NY*NX) - indj*(NX);

        fx[i] = 1 + 3*(indi+indj*indk) + 7*(indj+2) + 3*(indk+2);
        fy[i] = 1 + 2*(indi+indj+indk) + 3*(indj*indj) + 2*(indk*indi);
        fz[i] = 1 + 3*(indi*indj) + 7*(indj*indi) + 4*(indk*indj);
    }// End of striding loop over filament segment velocities.

    __syncthreads();
    return;
}

__global__
void cufcm_gaussian_setup(int N, int ngd, double* Y,
                    double* gaussx, double* gaussy, double* gaussz,
                    double* grad_gaussx_dip, double* grad_gaussy_dip, double* grad_gaussz_dip,
                    double* gaussgrid,
                    double* xdis, double* ydis, double* zdis,
                    int* indx, int* indy, int* indz,
                    double sigmadipsq, double anorm, double anorm2, double dx){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    int np, i, xc, yc, zc;
    int xg, yg, zg;
    int ngdh = ngd/2;

    double xx;
    double xxc, yyc, zzc;
    double E2x, E2y, E2z, E3;
    double anorm3, dxanorm2;

    anorm3 = anorm*anorm*anorm;
    dxanorm2 = dx/anorm2;

    // part1
    for(i = 0; i < ngd; i++){
        gaussgrid[i] = exp(-(i+1-ngdh)*(i+1-ngdh)*dx*dxanorm2);
    }

    for(int np = index; np < N; np += stride){
        xc = (int) (Y[3*np + 0]/dx); // the index of the nearest grid point to the particle
        yc = (int) (Y[3*np + 1]/dx);
        zc = (int) (Y[3*np + 2]/dx);

        xxc = (double)xc*dx - Y[3*np + 0]; // distance to the nearest point (ksi-Y)
        yyc = (double)yc*dx - Y[3*np + 1];
        zzc = (double)zc*dx - Y[3*np + 2];

        // part2
        E2x = exp(-2*xxc*dxanorm2);
        E2y = exp(-2*yyc*dxanorm2);
        E2z = exp(-2*zzc*dxanorm2);

        // part3
        E3 = anorm3*exp(-(xxc*xxc + yyc*yyc + zzc*zzc)/anorm2);

        // old function
        for(i = 0; i < ngd; i++){
            xg = xc - ngdh + (i+1); 
            indx[ngd*np + i] = xg - NPTS * ((int) floor( ((double) xg) / ((double) NPTS)));
            xx = ((double) xg)*dx-Y[3*np + 0];
            gaussx[ngd*np + i] = E3*int_pow(E2x,i+1-ngdh)*gaussgrid[i];
            grad_gaussx_dip[ngd*np + i] = - xx / sigmadipsq;
            xdis[ngd*np + i] = xx*xx;

            yg = yc - ngdh + (i+1);
            indy[ngd*np + i] = yg - NPTS * ((int) floor( ((double) yg) / ((double) NPTS)));
            xx = ((double) yg)*dx - Y[3*np + 1];
            gaussy[ngd*np + i] = int_pow(E2y,i+1-ngdh)*gaussgrid[i];
            grad_gaussy_dip[ngd*np + i] = - xx / sigmadipsq;
            ydis[ngd*np + i] = xx*xx;

            zg = zc - ngdh + (i+1);
            indz[ngd*np + i] = zg - NPTS * ((int) floor( ((double) zg) / ((double) NPTS)));
            xx = ((double) zg)*dx-Y[3*np + 2];
            gaussz[ngd*np + i] = int_pow(E2z,i+1-ngdh)*gaussgrid[i];
            grad_gaussz_dip[ngd*np + i] = - xx / sigmadipsq;
            zdis[ngd*np + i] = xx*xx;
        }
    }
    return;
}

__global__
void GA_setup(double *GA, double *T, int N){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    for(int np = index; np < N; np += stride){
        GA[6*np + 0] = 0.0;
        GA[6*np + 1] = 0.0;
        GA[6*np + 2] = 0.0;
        GA[6*np + 3] = 0.5*T[3*np + 2];
        GA[6*np + 4] = -0.5*T[3*np + 1];
        GA[6*np + 5] = 0.5*T[3*np + 0];
    }
    return;
}

__global__
void cufcm_mono_dipole_distribution(cufftReal *fx, cufftReal *fy, cufftReal *fz, int N,
              double *GA, double *F, double pdmag, double sigmasq, 
              double *gaussx, double *gaussy, double *gaussz,
              double *grad_gaussx_dip, double *grad_gaussy_dip, double *grad_gaussz_dip,
              double *xdis, double *ydis, double *zdis,
              int *indx, int *indy, int *indz,
              int ngd){

    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    

    int np, i, j, k, ii, jj, kk;
    double xx, yy, zz, r2, temp;
    double xx2, yy2, zz2;
    double g11, g22, g33, g12, g21, g13, g31, g23, g32;
    double gx, gy, gz, Fx, Fy, Fz;
    double g11xx, g22yy, g33zz, g12yy, g21xx, g13zz, g31xx, g23zz, g32yy;
    double smallx = 1e-18;
    double temp2 = 0.5 * pdmag / sigmasq;
    double temp3 = temp2 /sigmasq;
    double temp4 = 3.0*temp2;
    double temp5;
    int ind;

    for(int np = index; np < N; np += stride){
        Fx = F[3*np + 0];
        Fy = F[3*np + 1];
        Fz = F[3*np + 2];
        g11 = + GA[6*np + 0];
        g22 = + GA[6*np + 1];
        g33 = + GA[6*np + 2];
        g12 = + GA[6*np + 3];
        g21 = - GA[6*np + 3];
        g13 = + GA[6*np + 4];
        g31 = - GA[6*np + 4];
        g23 = + GA[6*np + 5];
        g32 = - GA[6*np + 5];
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

__global__
void cufcm_flow_solve(cufftComplex* fk_x, cufftComplex* fk_y, cufftComplex* fk_z,
                      cufftComplex* uk_x, cufftComplex* uk_y, cufftComplex* uk_z,
                      double* q, double* qpad, double* qsq, double* qpadsq){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    double norm, kdotf_re, kdotf_im;
    double q1, q2, q3, qq;
    double f1_re, f1_im, f2_re, f2_im, f3_re, f3_im;

    // Stay in the loop as long as any thread in the block still needs to compute velocities.
    for(int i = index; i < FFT_GRID_SIZE; i += stride){
        const int indk = (i)/(NY*(NX/2+1));
        const int indj = (i - indk*(NY*(NX/2+1)))/(NX/2+1);
        const int indi = i - indk*(NY*(NX/2+1)) - indj*(NX/2+1);

        q1 = q[indi];
        q2 = q[indj];
        q3 = q[indk];
        qq = qsq[indi] + qsq[indj] + qsq[indk];
        norm = 1.0/(qq);

        f1_re = fk_x[i].x;
        f1_im = fk_x[i].y;
        f2_re = fk_y[i].x;
        f2_im = fk_y[i].y;
        f3_re = fk_z[i].x;
        f3_im = fk_z[i].y;

        if(i==0){
            f1_re = 0;
            f1_im = 0;
            f2_re = 0;
            f2_im = 0;
            f3_re = 0;
            f3_im = 0;
        }

        kdotf_re = (q1*f1_re+q2*f2_re+q3*f3_re)*norm;
        kdotf_im = (q1*f1_im+q2*f2_im+q3*f3_im)*norm;

        uk_x[i].x = norm*(f1_re-q1*(kdotf_re))/((double)GRID_SIZE);
        uk_x[i].y = norm*(f1_im-q1*(kdotf_im))/((double)GRID_SIZE);
        uk_y[i].x = norm*(f2_re-q2*(kdotf_re))/((double)GRID_SIZE);
        uk_y[i].y = norm*(f2_im-q2*(kdotf_im))/((double)GRID_SIZE);
        uk_z[i].x = norm*(f3_re-q3*(kdotf_re))/((double)GRID_SIZE);
        uk_z[i].y = norm*(f3_im-q3*(kdotf_im))/((double)GRID_SIZE);

        // // uk_x[i].x = f1_re;
        // // uk_x[i].y = f1_im;
        // // uk_y[i].x = f2_re;
        // // uk_y[i].y = f2_im;
        // // uk_z[i].x = f3_re;
        // // uk_z[i].y = f3_im;

        if(i==0){
            uk_x[0].x = 0;
            uk_x[0].y = 0;
            uk_y[0].x = 0;
            uk_y[0].y = 0;
            uk_z[0].x = 0;
            uk_z[0].y = 0;
        }
    }// End of striding loop over filament segment velocities.
    __syncthreads();
    return;
}

__global__
void normalise_array(cufftReal* ux, cufftReal* uy, cufftReal* uz){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;
    const double temp = 1.0/((double)GRID_SIZE);

    // Stay in the loop as long as any thread in the block still needs to compute velocities.
    for(int i = index; i < GRID_SIZE; i += stride){
        ux[i] *= temp;
        uy[i] *= temp;
        uz[i] *= temp;

    }// End of striding loop over filament segment velocities.

    __syncthreads();
}

__global__
void cufcm_particle_velocities(cufftReal *ux, cufftReal *uy, cufftReal *uz, int N,
                               double *VTEMP, double *WTEMP,
                               double pdmag, double sigmasq, 
                               double *gaussx, double *gaussy, double *gaussz,
                               double *grad_gaussx_dip, double *grad_gaussy_dip, double *grad_gaussz_dip,
                               double *xdis, double *ydis, double *zdis,
                               int *indx, int *indy, int *indz,
                               int ngd, double dx){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    int np, i, j, k, ii, jj, kk;
    double norm, temp;
    double gx, gy, gz;
    double ux_temp, uy_temp, uz_temp;
    double xx, yy, zz;
    double xx2, yy2, zz2;
    double r2;
    double temp2 = 0.5 * pdmag / sigmasq;
    double temp3 = temp2 / sigmasq;
    double temp4 = 3.0*temp2;
    double temp5;
    int ind;

    norm = dx*dx*dx;

    for(int np = index; np < N; np += stride){
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

void cufcm_test_force_loop(cufftReal* fx, cufftReal* fy, cufftReal* fz){
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

void cufcm_gaussian_setup_loop(int N, int ngd, double* Y,
                    double* gaussx, double* gaussy, double* gaussz,
                    double* grad_gaussx_dip, double* grad_gaussy_dip, double* grad_gaussz_dip,
                    double* gaussgrid,
                    double* xdis, double* ydis, double* zdis,
                    int* indx, int* indy, int* indz,
                    double sigmadipsq, double anorm, double anorm2, double dx){
    int np, i, xc, yc, zc;
    int xg, yg, zg;
    int ngdh = ngd/2;

    double xx;
    double xxc, yyc, zzc;
    double E2x, E2y, E2z, E3;
    double anorm3, dxanorm2;

    anorm3 = anorm*anorm*anorm;
    dxanorm2 = dx/anorm2;

    // part1
    for(i = 0; i < ngd; i++){
        gaussgrid[i] = exp(-(i+1-ngdh)*(i+1-ngdh)*dx*dxanorm2);
    }

    for(np = 0; np < N; np++){
        xc = (int) (Y[3*np + 0]/dx); // the index of the nearest grid point to the particle
        yc = (int) (Y[3*np + 1]/dx);
        zc = (int) (Y[3*np + 2]/dx);

        xxc = (double)xc*dx - Y[3*np + 0]; // distance to the nearest point (ksi-Y)
        yyc = (double)yc*dx - Y[3*np + 1];
        zzc = (double)zc*dx - Y[3*np + 2];

        // part2
        E2x = exp(-2*xxc*dxanorm2);
        E2y = exp(-2*yyc*dxanorm2);
        E2z = exp(-2*zzc*dxanorm2);

        // part3
        E3 = anorm3*exp(-(xxc*xxc + yyc*yyc + zzc*zzc)/anorm2);

        // old function
        for(i = 0; i < ngd; i++){
            xg = xc - ngdh + (i+1); 
            indx[ngd*np + i] = xg - NPTS * ((int) floor( ((double) xg) / ((double) NPTS)));
            xx = ((double) xg)*dx-Y[3*np + 0];
            gaussx[ngd*np + i] = E3*int_pow(E2x,i+1-ngdh)*gaussgrid[i];
            grad_gaussx_dip[ngd*np + i] = - xx / sigmadipsq;
            xdis[ngd*np + i] = xx*xx;

            yg = yc - ngdh + (i+1);
            indy[ngd*np + i] = yg - NPTS * ((int) floor( ((double) yg) / ((double) NPTS)));
            xx = ((double) yg)*dx - Y[3*np + 1];
            gaussy[ngd*np + i] = int_pow(E2y,i+1-ngdh)*gaussgrid[i];
            grad_gaussy_dip[ngd*np + i] = - xx / sigmadipsq;
            ydis[ngd*np + i] = xx*xx;

            zg = zc - ngdh + (i+1);
            indz[ngd*np + i] = zg - NPTS * ((int) floor( ((double) zg) / ((double) NPTS)));
            xx = ((double) zg)*dx-Y[3*np + 2];
            gaussz[ngd*np + i] = int_pow(E2z,i+1-ngdh)*gaussgrid[i];
            grad_gaussz_dip[ngd*np + i] = - xx / sigmadipsq;
            zdis[ngd*np + i] = xx*xx;
        }
    }
    return;
}

void GA_setup_loop(double *GA, double *T, int N){
    for(int i = 0; i < N; i++){
        GA[6*i + 0] = 0.0;
        GA[6*i + 1] = 0.0;
        GA[6*i + 2] = 0.0;
        GA[6*i + 3] = 0.5*T[3*i + 2];
        GA[6*i + 4] = -0.5*T[3*i + 1];
        GA[6*i + 5] = 0.5*T[3*i + 0];
    }
    return;
}

void cufcm_mono_dipole_distribution_loop(cufftReal *fx, cufftReal *fy, cufftReal *fz, int N,
              double *GA, double *F, double pdmag, double sigmasq, 
              double *gaussx, double *gaussy, double *gaussz,
              double *grad_gaussx_dip, double *grad_gaussy_dip, double *grad_gaussz_dip,
              double *xdis, double *ydis, double *zdis,
              int *indx, int *indy, int *indz,
              int ngd){
    int np, i, j, k, ii, jj, kk;
    double xx, yy, zz, r2, temp;
    double xx2, yy2, zz2;
    double g11, g22, g33, g12, g21, g13, g31, g23, g32;
    double gx, gy, gz, Fx, Fy, Fz;
    double g11xx, g22yy, g33zz, g12yy, g21xx, g13zz, g31xx, g23zz, g32yy;
    double smallx = 1e-18;
    int ind;
    double temp2 = 0.5 * pdmag / sigmasq;
    double temp3 = temp2 /sigmasq;
    double temp4 = 3.0*temp2;
    double temp5;

    for(np = 0; np < N; np++){
        Fx = F[3*np + 0];
        Fy = F[3*np + 1];
        Fz = F[3*np + 2];
        g11 = + GA[6*np + 0];
        g22 = + GA[6*np + 1];
        g33 = + GA[6*np + 2];
        g12 = + GA[6*np + 3];
        g21 = - GA[6*np + 3];
        g13 = + GA[6*np + 4];
        g31 = - GA[6*np + 4];
        g23 = + GA[6*np + 5];
        g32 = - GA[6*np + 5];
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

void cufcm_particle_velocities_loop(cufftReal *ux, cufftReal *uy, cufftReal *uz, int N,
                               double *VTEMP, double *WTEMP,
                               double pdmag, double sigmasq, 
                               double *gaussx, double *gaussy, double *gaussz,
                               double *grad_gaussx_dip, double *grad_gaussy_dip, double *grad_gaussz_dip,
                               double *xdis, double *ydis, double *zdis,
                               int *indx, int *indy, int *indz,
                               int ngd, double dx){
    int np, i, j, k, ii, jj, kk;
    double norm, temp;
    double gx, gy, gz;
    double ux_temp, uy_temp, uz_temp;
    double xx, yy, zz;
    double xx2, yy2, zz2;
    double r2;
    double temp2 = 0.5 * pdmag / sigmasq;
    double temp3 = temp2 / sigmasq;
    double temp4 = 3.0*temp2;
    double temp5;
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
double int_pow(double base, int power){
    /* fast power function for integer powers */
    double result = 1;
    if(power>=0){
        for(int i = 0; i < power; i++){
            result = result * base;
    }
    }
    if(power<0){
        for(int i = 0; i < -power; i++){
            result = result * base;
    }
        result = (double) 1/result;
    }
    return result;
}

