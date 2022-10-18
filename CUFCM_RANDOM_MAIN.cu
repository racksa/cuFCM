#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <curand.h>
#include <cudaProfiler.h>
#include <cuda_profiler_api.h>

#include <cub/device/device_radix_sort.cuh>

#include "config.hpp"

#include "CUFCM_DATA.cuh"
#include "CUFCM_RANDOMPACKER.cuh"

#include "util/cuda_util.hpp"


int main(int argc, char** argv) {
	///////////////////////////////////////////////////////////////////////////////
	// Initialise parameters
	///////////////////////////////////////////////////////////////////////////////
	Real values[100];
	read_config(values, "simulation_info_long");
	int N = values[0];
	Real rh = values[1];
	int repeat = values[8];
	int prompt = values[9];
	int packrep = values[12];
	Real boxsize = values[13];

	int num_thread_blocks_N;
    curandState *dev_random;
	num_thread_blocks_N = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
	cudaMalloc((void**)&dev_random, num_thread_blocks_N*THREADS_PER_BLOCK*sizeof(curandState));

	Real* Y_host = malloc_host<Real>(3*N);						Real* Y_device = malloc_device<Real>(3*N);
	Real* F_host = malloc_host<Real>(3*N);						Real* F_device = malloc_device<Real>(3*N);
	Real* T_host = malloc_host<Real>(3*N);						Real* T_device = malloc_device<Real>(3*N);

	///////////////////////////////////////////////////////////////////////////////
	// Start repeat
	///////////////////////////////////////////////////////////////////////////////	

	/* Create random genertaor solver */
	init_force_kernel<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(F_device, rh, N, dev_random);
	init_force_kernel<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(T_device, rh, N, dev_random);

	{
		random_packer *packer = new random_packer(Y_host, Y_device, N, boxsize);
		for(int t = 0; t < packrep; t++){
			if(prompt > 5){
				std::cout << "\rGenerating random spheres iteration: " << t+1 << "/" << packrep;
			}
			packer->update();
		}
		packer->finish();
		if(prompt > 5){
			printf("\nFinished packing");
		}
	}

	if(prompt > 5){
		printf("\nCopying to host...\n");
	}
	copy_to_host<Real>(Y_device, Y_host, 3*N);
	copy_to_host<Real>(F_device, F_host, 3*N);
	copy_to_host<Real>(T_device, T_host, 3*N);

	write_init_data(Y_host, F_host, T_host, N);

	if(prompt > 5){
		printf("\nFinished packing");
	}

	return 0;
}

