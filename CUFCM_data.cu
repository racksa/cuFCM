#include "CUFCM_data.hpp"
#include <cstdio>
#include "config.hpp"
#include <curand_kernel.h>
#include <curand.h>

void read_init_data(Real *Y, int N, const char *file_name){
    FILE *ifile;
    ifile = fopen(file_name, "r");
    for(int np = 0; np < N; np++){
        #if USE_DOUBLE_PRECISION == true
        if(fscanf(ifile, "%lf %lf %lf", &Y[3*np + 0], &Y[3*np + 1], &Y[3*np + 2]) == 0){
            printf("fscanf error: Unable to read data");
        }
        #else
        if(fscanf(ifile, "%f %f %f", &Y[3*np + 0], &Y[3*np + 1], &Y[3*np + 2]) == 0){
            printf("fscanf error: Unable to read data");
        }
        #endif
    }
    fclose(ifile);

    return;
}

void init_pos(Real *Y, Real rad, int N){
    int check = 0;
    Real rsq = 0.0, xi = 0.0, yi = 0.0, zi = 0.0, xij = 0.0, yij = 0.0, zij = 0.0, rsqcheck = 4.0*rad*rad;

    for(int j = 0; j < N; j++){

        if(fmodf(j, 100000) == 0){
            printf("init particle %d\n", j);
        }
        check = 0;
        while(check == 0){
            Y[3*j + 0] = PI2*((float)rand() / (float)RAND_MAX);
            Y[3*j + 1] = PI2*((float)rand() / (float)RAND_MAX);
            Y[3*j + 2] = PI2*((float)rand() / (float)RAND_MAX);

            check = 1;
            if(j > 0){
                for(int i = 0; i < j; i++){
                    xi = Y[3*i + 0];
                    yi = Y[3*i + 1];
                    zi = Y[3*i + 2];
                    xij = xi - Y[3*j + 0];
                    yij = yi - Y[3*j + 1];
                    zij = zi - Y[3*j + 2];
                    xij = xij - PI2 * ((Real) ((int) (xij/PI)));
                    yij = yij - PI2 * ((Real) ((int) (yij/PI)));
                    zij = zij - PI2 * ((Real) ((int) (zij/PI)));
                    rsq = xij*xij+yij*yij+zij*zij;
                    if(rsq < rsqcheck){
                        check = 0;
                    }
                }
            }
        }
    }
    return;
}

void init_pos_gpu(Real *Y, Real rad, int N){

    Real x, y, z;
    Real rsqcheck = 4.0*rad*rad;

    int *check_host = (int*)(malloc(sizeof(int)));
    int *check_device;
    cudaMalloc(&check_device, sizeof(int)); 

    for(int j = 0; j < N; j++){

        if(fmodf(j, 10000) == 0){
            printf("init particle %d\n", j);
        }

        check_host[0] = 0;
        while(check_host[0] == 0){
            x = PI2*((float)rand() / (float)RAND_MAX);
            y = PI2*((float)rand() / (float)RAND_MAX);
            z = PI2*((float)rand() / (float)RAND_MAX);

            int num_thread_blocks_j = (j + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
            (num_thread_blocks_j<1? num_thread_blocks_j=1:true);

            check_overlap<<<num_thread_blocks_j, THREADS_PER_BLOCK>>>(x, y, z, Y, rsqcheck, j, check_device);
            
            cudaMemcpy(check_host, check_device, sizeof(int), cudaMemcpyDeviceToHost);

            if(check_host[0] == 1){
                append<<<1, THREADS_PER_BLOCK>>>(x, y, z, Y, j);
            }
            
        }
    }
    return;
}

__global__
void append(Real x, Real y, Real z, Real *Y, int np){
    // printf("successful np %d\n", np);
    Y[3*np + 0] = x;
    Y[3*np + 1] = y;
    Y[3*np + 2] = z;
}

__global__
void check_overlap(Real x, Real y, Real z, Real *Y, Real rsqcheck, int np, int *check){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    Real xij, yij, zij, rsq;

    check[0] = 1;

    if(np == 0){
        check[0] = 1;
    }

    for(int i = index; i < np; i += stride){
        if(check[0] == 1){
            xij = x - Y[3*i + 0];
            yij = y - Y[3*i + 1];
            zij = z - Y[3*i + 2];
            xij = xij - PI2 * ((Real) ((int) (xij/PI)));
            yij = yij - PI2 * ((Real) ((int) (yij/PI)));
            zij = zij - PI2 * ((Real) ((int) (zij/PI)));
            rsq = xij*xij+yij*yij+zij*zij;

            if(rsq <= rsqcheck){
                // printf("reject (%d %d)\t(%.2f %.2f %.2f) (%.2f %.2f %.2f) rsq %.8f/%.8f\n", np, i, x, y, z, Y[3*i + 0], Y[3*i + 1], Y[3*i + 2], rsq, rsqcheck);
                check[0] = 0;
            }
        }
        
    }

    __syncthreads();

    return;
}

void init_force(Real *F, Real rad, int N){
  
    for(int j = 0; j < N; j++){
        F[3*j + 0] = 12.0*PI*rad*(((float)rand() / (float)RAND_MAX) - 0.5);
        F[3*j + 1] = 12.0*PI*rad*(((float)rand() / (float)RAND_MAX) - 0.5);
        F[3*j + 2] = 12.0*PI*rad*(((float)rand() / (float)RAND_MAX) - 0.5);
    }
    return;
}

__global__
void init_force_kernel(Real *F, Real rad, int N, curandState *states){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    int seed = index; // different seed per thread
    curand_init(seed, index, 0, &states[index]);

    Real rnd1, rnd2, rnd3;

    for(int j = 0; j < N; j += stride){
        rnd1 = curand_uniform (&states[index]);
        rnd2 = curand_uniform (&states[index]);
        rnd3 = curand_uniform (&states[index]);
        F[3*j + 0] = 12.0*PI*rad*(rnd1 - 0.5);
        F[3*j + 1] = 12.0*PI*rad*(rnd2 - 0.5);
        F[3*j + 2] = 12.0*PI*rad*(rnd3 - 0.5);
    }
    return;
}


__global__
void init_wave_vector(Real *q, Real *qsq, Real *qpad, Real *qpadsq, int nptsh, int pad){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    for(int i = index; i < NX; i += stride){
        if(i < nptsh || i == nptsh){
			q[i] = (Real) i;
		}
		if(i > nptsh){
			q[i] = (Real) (i - NX);
		}
		qsq[i] = q[i]*q[i];
    }

	for(int i = index; i < pad; i += stride){
		qpad[i] = (Real) i;
		qpadsq[i] = qpad[i]*qpad[i];
	}
    return;
}