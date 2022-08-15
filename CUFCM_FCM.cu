#include <iostream>
// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>

#include "config.hpp"
#include "CUFCM_FCM.hpp"

void cufcm_force_distribution_loop(cufftReal* fx, cufftReal* fy, cufftReal* fz){
    for(int i=0; i<NX; i++){
        for(int j=0; j<NY; j++){
            for(int k=0; k<NZ; k++){
                const int index = index_3to1(i, j, k);

                fx[index] = 1 + 3*(i+j*k) + 7*(j+2) + 3*(k+2);
                fy[index] = 1 + 2*(i+j+k) + 3*(j*j) + 2*(k*i);
                fz[index] = 1 + 3*(i*j) + 7*(j*i) + 4*(k*j);
            }
        }
    }
}

__global__
void cufcm_force_distribution(cufftReal* fx, cufftReal* fy, cufftReal* fz){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    // Stay in the loop as long as any thread in the block still needs to spread forces.
    for(int i = index; i < GRID_SIZE; i += stride){
        const int indk = (i)/(NY*NX);
        const int indj = (i - indk*(NY*NX))/(NX);
        const int indi = i - indk*(NY*NX) - indj*(NX);

        // const int indi = (i)/(NY*NZ);
        // const int indj = (i - indi*(NY*NZ))/(NZ);
        // const int indk = i - indi*(NY*NZ) - indj*(NZ);

        fx[i] = 1 + 3*(indi+indj*indk) + 7*(indj+2) + 3*(indk+2);
        fy[i] = 1 + 2*(indi+indj+indk) + 3*(indj*indj) + 2*(indk*indi);
        fz[i] = 1 + 3*(indi*indj) + 7*(indj*indi) + 4*(indk*indj);
    }// End of striding loop over filament segment velocities.

    __syncthreads();
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
    for(int i = index; i < GRID_SIZE; i += stride){
        const int indk = (i)/(NY*(NX/2+1));
        const int indj = (i - indk*(NY*(NX/2+1)))/(NX/2+1);
        const int indi = i - indk*(NY*(NX/2+1)) - indj*(NX/2+1);

        printf("(%d %d %d) %d\n", indi, indj, indk, i);

        // const int indi = (i)/(NY*((NZ/2+1)));
        // const int indj = (i - indi*(NY*(NZ/2+1)))/(NZ/2+1);
        // const int indk = i - indi*(NY*(NZ/2+1)) - indj*(NZ/2+1);
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

        // uk_x[i].x = f1_re;
        // uk_x[i].y = f1_im;
        // uk_y[i].x = f2_re;
        // uk_y[i].y = f2_im;
        // uk_z[i].x = f3_re;
        // uk_z[i].y = f3_im;

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
}

__global__
void normalise_array(cufftReal* ux, cufftReal* uy, cufftReal* uz){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;
    const double temp = 1.0/((double)GRID_SIZE);

    // Stay in the loop as long as any thread in the block still needs to compute velocities.
    for(int i = index; i < GRID_SIZE; i += stride){
        ux[index] *= temp;
        uy[index] *= temp;
        uz[index] *= temp;

    }// End of striding loop over filament segment velocities.

    __syncthreads();
}

int index_3to1(int i, int j, int k){
	return i*NY*NZ + j*NZ + k;
}
