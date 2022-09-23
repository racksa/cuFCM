#include "CUFCM_data.hpp"
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
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

void read_validate_data(Real *Y, Real *F, Real *V, Real *W, int N, const char *file_name){
    FILE *ifile;
    ifile = fopen(file_name, "r");
    for(int np = 0; np < N; np++){
        #if USE_DOUBLE_PRECISION == true
            if(fscanf(ifile, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
            &Y[3*np + 0],
            &Y[3*np + 0], &Y[3*np + 1], &Y[3*np + 2],
            &F[3*np + 0], &F[3*np + 1], &F[3*np + 2],
            &V[3*np + 0], &V[3*np + 1], &V[3*np + 2],
            &W[3*np + 0], &W[3*np + 1], &W[3*np + 2]) == 0){
                printf("fscanf error: Unable to read data");
            }
        #else
            if(fscanf(ifile, "%f %f %f %f %f %f %f %f %f %f %f %f %f",
            &Y[3*np + 0],
            &Y[3*np + 0], &Y[3*np + 1], &Y[3*np + 2],
            &F[3*np + 0], &F[3*np + 1], &F[3*np + 2],
            &V[3*np + 0], &V[3*np + 1], &V[3*np + 2],
            &W[3*np + 0], &W[3*np + 1], &W[3*np + 2]) == 0){
                printf("fscanf error: Unable to read data");
            }
        #endif
    }
    fclose(ifile);

    return;
}

void write_data(Real *Y, Real *F, Real *V, Real *W, int N, const char *file_name){
    FILE *pfile;
    pfile = fopen(file_name, "w");
    for(int i = 0; i < N; i++){
        fprintf(pfile, "%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\n", 
        Y[3*i + 0], Y[3*i + 1], Y[3*i + 2], 
        F[3*i + 0], F[3*i + 1], F[3*i + 2], 
        V[3*i + 0], V[3*i + 1], V[3*i + 2], 
        W[3*i + 0], W[3*i + 1], W[3*i + 2]);
        }
    fprintf(pfile, "\n");
    fclose(pfile);
}

void write_init_data(Real *Y, Real *F, Real *T, int N){
    FILE *pfile;
    printf("Writing position data...\n");
    pfile = fopen("./init_data/new/pos_data.dat", "w");
    for(int i = 0; i < N; i++){
        fprintf(pfile, "%.8f %.8f %.8f\n", Y[3*i + 0], Y[3*i + 1], Y[3*i + 2]);
    }
    fclose(pfile);
    printf("Writing force data...\n");
    pfile = fopen("./init_data/new/force_data.dat", "w");
    for(int i = 0; i < N; i++){
        fprintf(pfile, "%.8f %.8f %.8f\n", F[3*i + 0], F[3*i + 1], F[3*i + 2]);
    }
    fclose(pfile);
    printf("Writing torque data...\n");
    pfile = fopen("./init_data/new/torque_data.dat", "w");
    for(int i = 0; i < N; i++){
        fprintf(pfile, "%.8f %.8f %.8f\n", T[3*i + 0], T[3*i + 1], T[3*i + 2]);
    }
    fclose(pfile);
    printf("Finished writing...\n");
}


void write_time(Real time_cuda_initialisation, 
                Real time_readfile,
                Real time_hashing, 
                Real time_linklist,
                Real time_precompute_gauss,
                Real time_spreading,
                Real time_FFT,
                Real time_gathering,
                Real time_correction,
                Real time_compute,
                const char *file_name){
    FILE *pfile;
    pfile = fopen(file_name, "w");
    fprintf(pfile, "time_cuda_initialisation=%.8f\n", time_cuda_initialisation);
    fprintf(pfile, "time_readfile=%.8f\n", time_readfile);
    fprintf(pfile, "time_hashing=%.8f\n", time_hashing);
    fprintf(pfile, "time_linklist=%.8f\n", time_linklist);
    fprintf(pfile, "time_precompute_gauss=%.8f\n", time_precompute_gauss);
    fprintf(pfile, "time_spreading=%.8f\n", time_spreading);
    fprintf(pfile, "time_FFT=%.8f\n", time_FFT);
    fprintf(pfile, "time_gathering=%.8f\n", time_gathering);
    fprintf(pfile, "time_correction=%.8f\n", time_correction);
    fprintf(pfile, "time_compute=%.8f\n", time_compute);
    fclose(pfile);
}

void write_error(Real Verror,
                 Real Werror,
                 const char *file_name){
    FILE *pfile;
    pfile = fopen(file_name, "a");
    fprintf(pfile, "Verror=%.8f\n", Verror);
    fprintf(pfile, "Werror=%.8f\n", Werror);
    fclose(pfile);
}

void read_config(Real *values, const char *file_name){
    std::ifstream cFile (file_name);
    if (cFile.is_open()){
        std::string line;
        int line_count = 0;
        while(getline(cFile, line)){
            line.erase(std::remove_if(line.begin(), line.end(), isspace),
                                line.end());
            if( line.empty() || line[0] == '#' ){
                continue;
            }
            auto delimiterPos = line.find("=");
            Real value = (Real) std::stod(line.substr(delimiterPos + 1));
            values[line_count] = value;
            line_count += 1;
        }
    }
    else{
        std::cerr << "Couldn't open config file for reading.\n";
    }
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
void init_pos_random_overlapping(Real *Y, int N, curandState *states){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    int seed = index; // different seed per thread
    curand_init(seed, index, 0, &states[index]);
    Real rnd1, rnd2, rnd3;

    for(int np = index; np < N; np += stride){
        rnd1 = curand_uniform (&states[index]);
        rnd2 = curand_uniform (&states[index]);
        rnd3 = curand_uniform (&states[index]);
        Y[3*np + 0] = PI2*rnd1;
        Y[3*np + 1] = PI2*rnd2;
        Y[3*np + 2] = PI2*rnd3;
    }
    return;
}

__global__
void init_pos_lattice_random(Real *Y, Real rad, int N, curandState *states){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    int seed = index; // different seed per thread
    curand_init(seed, index, 0, &states[index]);

    Real rnd1, rnd2, rnd3;
    int NP = ceil(cbrtf(N));
    Real dpx = PI2/(Real)NP;
    Real gaph = (dpx - (Real)2.0*rad)/(Real)2.0;

    if(dpx < rad && index == 0){
        printf("Particle overlapping\n");
    }

    for(int np = index; np < N; np += stride){
        const int k = np/(NP*NP);
        const int j = (np - k*NP*NP)/NP;
        const int i = np - k*NP*NP - j*NP;

        rnd1 = curand_uniform (&states[index]);
        rnd2 = curand_uniform (&states[index]);
        rnd3 = curand_uniform (&states[index]);
        Y[3*np + 0] = 0.5*dpx + i*dpx + rnd1*gaph;
        Y[3*np + 1] = 0.5*dpx + j*dpx + rnd2*gaph;
        Y[3*np + 2] = 0.5*dpx + k*dpx + rnd3*gaph;
    }
    return;
}

__global__
void append(Real x, Real y, Real z, Real *Y, int np){
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

    for(int j = index; j < N; j += stride){
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
void init_wave_vector(Real *q, Real *qsq, Real *qpad, Real *qpadsq, int nptsh, int pad, Real nx, Real ny, Real nz){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    for(int i = index; i < nx; i += stride){
        if(i < nptsh || i == nptsh){
			q[i] = (Real) i;
		}
		if(i > nptsh){
			q[i] = (Real) (i - nx);
		}
		qsq[i] = q[i]*q[i];
    }

	for(int i = index; i < pad; i += stride){
		qpad[i] = (Real) i;
		qpadsq[i] = qpad[i]*qpad[i];
	}
    return;
}