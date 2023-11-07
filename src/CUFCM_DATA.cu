#include "CUFCM_DATA.cuh"
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
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

void read_init_data_thrust(thrust::host_vector<Real>& Y, const char *initpos_file_name){
    std::ifstream input_file(initpos_file_name);
    if (!input_file.is_open()) {
        std::cerr << "Failed to open input file." << std::endl;
        return ;
    }

    // Read the input file line by line
    std::string line;
    int index = 0;
    while (std::getline(input_file, line)) {
        // Split the line into three space-separated values
        Real a, b, c;
        std::istringstream iss(line);
        if(iss >> a >> b >> c){
            // Assign the values to the host vector
            Y[index++] = a;
            Y[index++] = b;
            Y[index++] = c;
        }        
    }

    // Close the input file
    input_file.close();
}

void read_validate_data(Real *Y, Real *F, Real *T, Real *V, Real *W, int N, const char *file_name){
    FILE *ifile;
    ifile = fopen(file_name, "r");
    for(int np = 0; np < N; np++){
        #if USE_DOUBLE_PRECISION == true
            if(fscanf(ifile, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
            &Y[3*np + 0], &Y[3*np + 1], &Y[3*np + 2],
            &F[3*np + 0], &F[3*np + 1], &F[3*np + 2],
            &T[3*np + 0], &T[3*np + 1], &T[3*np + 2],
            &V[3*np + 0], &V[3*np + 1], &V[3*np + 2],
            &W[3*np + 0], &W[3*np + 1], &W[3*np + 2]) == 0){
                printf("fscanf error: Unable to read data");
            }
        #else
            if(fscanf(ifile, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f",
            &Y[3*np + 0], &Y[3*np + 1], &Y[3*np + 2],
            &F[3*np + 0], &F[3*np + 1], &F[3*np + 2],
            &T[3*np + 0], &T[3*np + 1], &T[3*np + 2],
            &V[3*np + 0], &V[3*np + 1], &V[3*np + 2],
            &W[3*np + 0], &W[3*np + 1], &W[3*np + 2]) == 0){
                printf("fscanf error: Unable to read data");
            }
        #endif
    }
    fclose(ifile);

    return;
}

void read_validate_data_thrust(thrust::host_vector<Real>& Y,
                            thrust::host_vector<Real>& F,
                            thrust::host_vector<Real>& T, 
                            thrust::host_vector<Real>& V, 
                            thrust::host_vector<Real>& W, const char *file_name){
    std::ifstream input_file(file_name);
    if (!input_file.is_open()) {
        std::cerr << "Failed to open validation file." << std::endl;
        return ;
    }

    // Read the input file line by line
    std::string line;
    int index = 0;
    while (std::getline(input_file, line)) {
        // Split the line into three space-separated values
        Real y1, y2, y3, f1, f2, f3, t1, t2, t3, v1, v2, v3, w1, w2, w3;
        Real i;
        std::istringstream iss(line);
        if(iss >> i >> y1 >> y2 >> y3 >> f1 >> f2 >> f3 >> t1 >> t2 >> t3 >> v1 >> v2 >> v3 >> w1 >> w2 >> w3){
            // Assign the values to the host vector
            Y[3*index] = y1;
            Y[3*index+1] = y2;
            Y[3*index+2] = y3;
            F[3*index] = f1;
            F[3*index+1] = f2;
            F[3*index+2] = f3;
            T[3*index] = t1;
            T[3*index+1] = t2;
            T[3*index+2] = t3;
            V[3*index] = v1;
            V[3*index+1] = v2;
            V[3*index+2] = v3;
            W[3*index] = w1;
            W[3*index+1] = w2;
            W[3*index+2] = w3;
            index++;
        }
    }

    // Close the input file
    input_file.close();
}

Real percentage_error_magnitude_thrust(thrust::host_vector<Real> data, thrust::host_vector<Real> ref_data, int N){
    Real ret = Real(0.0);

    for(int np = 0; np < N; np++){
        Real x = data[3*np];
        Real y = data[3*np + 1];
        Real z = data[3*np + 2];
        Real x_ref = ref_data[3*np];
        Real y_ref = ref_data[3*np + 1];
        Real z_ref = ref_data[3*np + 2];

        ret += sqrt( (x - x_ref)*(x - x_ref) + 
                     (y - y_ref)*(y - y_ref) + 
                     (z - z_ref)*(z - z_ref) ) / 
               sqrt( x_ref*x_ref + 
                     y_ref*y_ref + 
                     z_ref*z_ref );
    }
    ret = ret / (Real) N;

    return ret;
}

void write_data(Real *Y, Real *F, Real *T, Real *V, Real *W, int N, const char *file_name, const char *mode){
    FILE *pfile;
    pfile = fopen(file_name, mode);
    for(int i = 0; i < N; i++){
        fprintf(pfile, "%d %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f\n", 
        i, Y[3*i + 0], Y[3*i + 1], Y[3*i + 2], 
        F[3*i + 0], F[3*i + 1], F[3*i + 2], 
        T[3*i + 0], T[3*i + 1], T[3*i + 2],
        V[3*i + 0], V[3*i + 1], V[3*i + 2], 
        W[3*i + 0], W[3*i + 1], W[3*i + 2]);
        }
    fprintf(pfile, "\n#");
    fclose(pfile);
}

void write_data_thrust(thrust::host_vector<Real> Y,
                        thrust::host_vector<Real> F,
                        thrust::host_vector<Real> T,
                        thrust::host_vector<Real> V,
                        thrust::host_vector<Real> W, 
                        int N, const char *file_name, const char *mode){
    FILE *pfile;
    pfile = fopen(file_name, mode);
    for(int i = 0; i < N; i++){
        fprintf(pfile, "%d %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f\n", 
        i, Y[3*i + 0], Y[3*i + 1], Y[3*i + 2], 
        F[3*i + 0], F[3*i + 1], F[3*i + 2], 
        T[3*i + 0], T[3*i + 1], T[3*i + 2],
        V[3*i + 0], V[3*i + 1], V[3*i + 2], 
        W[3*i + 0], W[3*i + 1], W[3*i + 2]);
        }
    fprintf(pfile, "\n#");
    fclose(pfile);
}

void write_pos(Real *Y, Real rh, int N, const char *file_name){
    FILE *pfile;
    pfile = fopen(file_name, "w");
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
        fprintf(pfile, "%.16f %.16f %.16f\n", Y[3*i + 0], Y[3*i + 1], Y[3*i + 2]);
    }
    fclose(pfile);
    // printf("Writing force data...\n");
    pfile = fopen("./data/init_data/new/force_data.dat", "w");
    for(int i = 0; i < N; i++){
        fprintf(pfile, "%.16f %.16f %.16f\n", F[3*i + 0], F[3*i + 1], F[3*i + 2]);
    }
    fclose(pfile);
    // printf("Writing torque data...\n");
    pfile = fopen("./data/init_data/new/torque_data.dat", "w");
    for(int i = 0; i < N; i++){
        fprintf(pfile, "%.16f %.16f %.16f\n", T[3*i + 0], T[3*i + 1], T[3*i + 2]);
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

void write_celllist(int *cell_start_list, int *cell_end_list, int *map_list, int ncell, int Mx, int My, int Mz, const char *file_name){
    FILE *pfile;
    pfile = fopen(file_name, "w");
    fprintf(pfile, "#\n");
    for(int i = 0; i < ncell; i++){

        int nk = i/(Mx*My);
        int nj = (i - nk*Mx*My)/Mx;
        int ni = i - nk*Mx*My - nj*Mx;

        fprintf(pfile, "celli %d/%d (%d %d %d) particle index [%d %d] with neighbors [%d %d %d %d %d %d %d %d %d %d %d %d %d]\n", 
        i, ncell, ni, nj, nk, cell_start_list[i], cell_end_list[i],
        map_list[13*i + 0], map_list[13*i + 1], map_list[13*i + 2],
        map_list[13*i + 3], map_list[13*i + 4], map_list[13*i + 5],
        map_list[13*i + 6], map_list[13*i + 7], map_list[13*i + 8],
        map_list[13*i + 9], map_list[13*i + 10], map_list[13*i + 11],
        map_list[13*i + 12]);
        }
    fprintf(pfile, "\n");
    fclose(pfile);
}

void write_flow_field(Real *h, int grid_size, const char *file_name){
    FILE *pfile;
    pfile = fopen(file_name, "w");
    for(int i = 0; i < grid_size; i++){
        fprintf(pfile, "%.6f ", h[i]);
        }
    fprintf(pfile, "\n");
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

void parser_config(Real *values, Pars& pars){
    pars.N = values[0];
	pars.rh = values[1];
	pars.alpha = values[2];
	pars.beta = values[3];
	pars.eta = values[4];
	pars.nx = values[5];
	pars.ny = values[6];
	pars.nz = values[7];
	pars.repeat = values[8];
	pars.prompt = values[9];
	pars.boxsize = values[13];
	pars.checkerror = values[14];
}

__global__
void init_pos_lattice(Real *Y, int N, Real Lx, Real Ly, Real Lz){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    int b = round(Lx/Ly);
    int c = round(Lx/Lz);
    int Nx = ceil(cbrtf(N*b*c));
    int Ny = Nx/b;
    // int Nz = Nx/c;

    Real dpx = Lx/(Real)Nx;

    for(int np = index; np < N; np += stride){
        int k = np/(Nx*Ny);
        int j = (np - k*Nx*Ny)/Nx;
        int i = np - k*Nx*Ny - j*Nx;

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


void init_random_force(Real *F, Real Fref, Real rad, int N){

    int num_thread_blocks_N;
    curandState *dev_random;
	num_thread_blocks_N = (N + FCM_THREADS_PER_BLOCK - 1)/FCM_THREADS_PER_BLOCK;
	cudaMalloc((void**)&dev_random, num_thread_blocks_N*FCM_THREADS_PER_BLOCK*sizeof(curandState));

    init_force_kernel<<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(F, Fref*rad, N, dev_random);
}

__global__
void init_force_kernel(Real *F, Real Fref, int N, curandState *states){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    int seed = index + clock64(); // different seed per thread
    curand_init(seed, index, 0, &states[index]);

    Real rnd1, rnd2, rnd3;

    for(int j = index; j < N; j += stride){
        rnd1 = curand_uniform (&states[index]);
        rnd2 = curand_uniform (&states[index]);
        rnd3 = curand_uniform (&states[index]);

        F[3*j + 0] = Fref*(rnd1 - 0.5);
        F[3*j + 1] = Fref*(rnd2 - 0.5);
        F[3*j + 2] = Fref*(rnd3 - 0.5);
    }
    return;
}

__global__
void box(Real *Y, int N, Real Lx, Real Ly, Real Lz){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    for(int i = index; i < N; i += stride){

        images(Y[3*i + 0], Lx);
        images(Y[3*i + 1], Ly);
        images(Y[3*i + 2], Lz);

        if(Y[3*i + 0]<0.0f || Y[3*i + 1]<0.0f || Y[3*i + 2]<0.0f || 
           Y[3*i + 0]>=Lx || Y[3*i + 1]>=Ly || Y[3*i + 2]>=Lz){
			printf("-------- boxing,\
                    particle %d (%.8f %.8f %.8f) not in box\n", 
			i, Y[3*i + 0], Y[3*i + 1], Y[3*i + 2]);
		}
    }

}

__device__
void images(Real& x, Real boxsize){
    // Real x_old = Real(10.0);
    // x_old = x;

    x -= my_floor(x/boxsize)*boxsize;

    if(x == boxsize){
        x = Real(0.0);
    }

    // if(x>=boxsize){
    //     printf("-------- images,\
    //             boxing %lf into [0 %lf] get %lf\n", 
    //     x_old, boxsize, x);        
    // }

}


__global__
void check_nan_in(Real* arr, int L, bool* result){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    for(int i = index; i < L; i += stride){
        if (isnan(arr[i])){
            *result = false;
        }
    }
}
