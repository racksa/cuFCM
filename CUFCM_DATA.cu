#include "CUFCM_DATA.cuh"
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
        #if USE_DOUBLE_PRECISION
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

void write_data(Real *Y, Real *F, Real *V, Real *W, int N, const char *file_name, const char *mode){
    FILE *pfile;
    pfile = fopen(file_name, mode);
    for(int i = 0; i < N; i++){
        fprintf(pfile, "%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\n", 
        Y[3*i + 0], Y[3*i + 1], Y[3*i + 2], 
        F[3*i + 0], F[3*i + 1], F[3*i + 2], 
        V[3*i + 0], V[3*i + 1], V[3*i + 2], 
        W[3*i + 0], W[3*i + 1], W[3*i + 2]);
        }
    fprintf(pfile, "\n#");
    fclose(pfile);
}

void write_pos(Real *Y, Real rh, int N, const char *file_name){
    FILE *pfile;
    pfile = fopen(file_name, "a");
    fprintf(pfile, "#\n");
    for(int i = 0; i < N; i++){
        fprintf(pfile, "%.6f %.6f %.6f %.6f %.6f \n", 
        Y[3*i + 0], Y[3*i + 1], Y[3*i + 2], rh, 0.0);
        }
    fprintf(pfile, "\n");
    fclose(pfile);
}

void write_init_data(Real *Y, Real *F, Real *T, int N){
    FILE *pfile;
    // printf("Writing position data...\n");
    pfile = fopen("./data/init_data/new/pos_data.dat", "w");
    for(int i = 0; i < N; i++){
        fprintf(pfile, "%.8f %.8f %.8f\n", Y[3*i + 0], Y[3*i + 1], Y[3*i + 2]);
    }
    fclose(pfile);
    // printf("Writing force data...\n");
    pfile = fopen("./data/init_data/new/force_data.dat", "w");
    for(int i = 0; i < N; i++){
        fprintf(pfile, "%.8f %.8f %.8f\n", F[3*i + 0], F[3*i + 1], F[3*i + 2]);
    }
    fclose(pfile);
    // printf("Writing torque data...\n");
    pfile = fopen("./data/init_data/new/torque_data.dat", "w");
    for(int i = 0; i < N; i++){
        fprintf(pfile, "%.8f %.8f %.8f\n", T[3*i + 0], T[3*i + 1], T[3*i + 2]);
    }
    fclose(pfile);
    // printf("Finished writing...\n");
}

void write_time(Real time_cuda_initialisation, 
                Real time_readfile,
                Real time_hashing,
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

void read_config(Real *values, std::vector<std::string>& datafile_names, const char *file_name){
    std::ifstream cFile (file_name);
    if (cFile.is_open()){
        std::string line;
        int line_count = 0;
        int file_count = 0;
        while(getline(cFile, line)){
            line.erase(std::remove_if(line.begin(), line.end(), isspace),
                                line.end());
            if( line.empty() || line[0] == '#' ){
                continue;
            }
            else if (line[0] == '$') {
                auto delimiterPos = line.find("=");
                datafile_names[file_count] = line.substr(delimiterPos + 1);
                file_count += 1;
            }
            else{
                auto delimiterPos = line.find("=");
                Real value = (Real) std::stod(line.substr(delimiterPos + 1));
                values[line_count] = value;
            }

            line_count += 1;
        }
    }
    else{
        std::cerr << "Couldn't open config file for reading.\n";
    }
}

__global__
void init_pos_random_overlapping(Real *Y, int N, Real boxsize, curandState *states){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    int seed = index; // different seed per thread
    curand_init(seed, index, 0, &states[index]);
    Real rnd1, rnd2, rnd3;

    for(int np = index; np < N; np += stride){
        rnd1 = curand_uniform (&states[index]);
        rnd2 = curand_uniform (&states[index]);
        rnd3 = curand_uniform (&states[index]);
        Y[3*np + 0] = boxsize*rnd1;
        Y[3*np + 1] = boxsize*rnd2;
        Y[3*np + 2] = boxsize*rnd3;
    }
    return;
}


__global__
void init_pos_lattice(Real *Y, int N, Real boxsize){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    int NP = ceil(cbrtf(N));
    Real dpx = boxsize/(Real)NP;

    for(int np = index; np < N; np += stride){
        const int k = np/(NP*NP);
        const int j = (np - k*NP*NP)/NP;
        const int i = np - k*NP*NP - j*NP;

        Y[3*np + 0] = 0.5*dpx + i*dpx;
        Y[3*np + 1] = 0.5*dpx + j*dpx;
        Y[3*np + 2] = 0.5*dpx + k*dpx;
    }
    return;
}

__global__
void interleaved2separate(Real *F_device_interleave,
                          Real *F_device, Real *T_device, int N){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    for(int i = index; i < N; i += stride){
        F_device[3*i + 0] = F_device_interleave[6*i + 0];
        F_device[3*i + 1] = F_device_interleave[6*i + 1];
        F_device[3*i + 2] = F_device_interleave[6*i + 2];

        T_device[3*i + 0] = F_device_interleave[6*i + 3];
        T_device[3*i + 1] = F_device_interleave[6*i + 4];
        T_device[3*i + 2] = F_device_interleave[6*i + 5];
    }

}

__global__
void separate2interleaved(Real *F_device_interleave,
                          Real *F_device, Real *T_device, int N){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    for(int i = index; i < N; i += stride){
        F_device_interleave[6*i + 0] = F_device[3*i + 0];
        F_device_interleave[6*i + 1] = F_device[3*i + 1];
        F_device_interleave[6*i + 2] = F_device[3*i + 2];

        F_device_interleave[6*i + 3] = T_device[3*i + 0];
        F_device_interleave[6*i + 4] = T_device[3*i + 1];
        F_device_interleave[6*i + 5] = T_device[3*i + 2];
    }

}




void init_random_force(Real *F, Real rad, int N){

    int num_thread_blocks_N;
    curandState *dev_random;
	num_thread_blocks_N = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
	cudaMalloc((void**)&dev_random, num_thread_blocks_N*THREADS_PER_BLOCK*sizeof(curandState));

    init_force_kernel<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(F, rad, N, dev_random);
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
void box(Real *Y, int N, Real box_size){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    for(int i = index; i < N; i += stride){
        if(Y[3*i + 0]>box_size){
            Y[3*i + 0] -= box_size;
        }
        if(Y[3*i + 0]<0){
            Y[3*i + 0] += box_size;
        }

        if(Y[3*i + 1]>box_size){
            Y[3*i + 1] -= box_size;
        }
        if(Y[3*i + 1]<0){
            Y[3*i + 1] += box_size;
        }

        if(Y[3*i + 2]>box_size){
            Y[3*i + 2] -= box_size;
        }
        if(Y[3*i + 2]<0){
            Y[3*i + 2] += box_size;
        }
    }

}