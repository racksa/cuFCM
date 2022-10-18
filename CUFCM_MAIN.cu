#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>
#include <curand_kernel.h>
#include <curand.h>
#include <cudaProfiler.h>
#include <cuda_profiler_api.h>

#include <cub/device/device_radix_sort.cuh>


#include "config.hpp"

#include "CUFCM_data.cuh"
#include "CUFCM_SOLVER.cuh"
#include "CUFCM_RANDOMPACKER.cuh"

#include "util/cuda_util.hpp"


int main(int argc, char** argv) {
	///////////////////////////////////////////////////////////////////////////////
	// Initialise parameters
	///////////////////////////////////////////////////////////////////////////////
	Real values[100];
	read_config(values, "simulation_info_long");
	int N = values[0];
	int repeat = values[8];
	int prompt = values[9];
	Real Fref = values[11];
	int packrep = values[12];
	Real boxsize = values[13];
	Real Ffac = values[14];
	Real Tfac = values[15];

	int num_thread_blocks_N;
    curandState *dev_random;
	num_thread_blocks_N = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
	cudaMalloc((void**)&dev_random, num_thread_blocks_N*THREADS_PER_BLOCK*sizeof(curandState));

	Real* Y_host = malloc_host<Real>(3*N);						Real* Y_device = malloc_device<Real>(3*N);
	Real* F_host = malloc_host<Real>(3*N);						Real* F_device = malloc_device<Real>(3*N);
	Real* T_host = malloc_host<Real>(3*N);						Real* T_device = malloc_device<Real>(3*N);

	///////////////////////////////////////////////////////////////////////////////
	// Physical system initialisation
	///////////////////////////////////////////////////////////////////////////////

	#if INIT_FROM_FILE == 1

		// read_init_data(Y_host, N, "./data/init_data/new/pos_data.dat");
		// read_init_data(F_host, N, "./data/init_data/new/force_data.dat");
		// read_init_data(T_host, N, "./data/init_data/new/torque_data.dat");

		read_init_data(Y_host, N, "./data/init_data/N500000/pos-N500000-rh02609300-2.dat");
		read_init_data(F_host, N, "./data/init_data/N500000/force-N500000-rh02609300.dat");
		read_init_data(T_host, N, "./data/init_data/N500000/force-N500000-rh02609300-2.dat");

		// read_init_data(Y_host, N, "./data/init_data/N500000/pos-N500000-rh02609300-2-artificial.dat");
		// read_init_data(F_host, N, "./data/init_data/N500000/force-N500000-rh02609300-artificial.dat");
		// read_init_data(T_host, N, "./data/init_data/N500000/force-N500000-rh02609300-2.dat");

		// read_init_data(Y_host, N, "./data/init_data/N16777216/pos-N16777216-rh008089855.dat");
		// read_init_data(F_host, N, "./data/init_data/N16777216/force-N16777216-rh008089855.dat");
		// read_init_data(T_host, N, "./data/init_data/N16777216/force-N16777216-rh008089855-2.dat");

		for(int i = 0; i<3*N; i++){
			Y_host[i] = Y_host[i] * boxsize/PI2;
			F_host[i] = F_host[i] * Ffac;
			T_host[i] = T_host[i] * Tfac;
		}

		copy_to_device<Real>(Y_host, Y_device, 3*N);
		copy_to_device<Real>(F_host, F_device, 3*N);
		copy_to_device<Real>(T_host, T_device, 3*N);


	#elif INIT_FROM_FILE == 0

		Real rh = values[1];
		
		init_force_kernel<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(F_device, rh, N, dev_random);
		init_force_kernel<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(T_device, rh, N, dev_random);

		{
			FILE *pfile;
			pfile = fopen("data/simulation/spherepacking.dat", "w");
			fclose(pfile);
			
			random_packer *packer = new random_packer(Y_host, Y_device, N);
			for(int t = 0; t < packrep; t++){
				std::cout << "\rGenerating random spheres iteration: " << t+1 << "/" << packrep;
				packer->update();
				if(t>packrep-11){
					packer->write();
				}
			}
			packer->finish();
			printf("\nFinished packing");
		}

		printf("\nCopying to host...\n");
		copy_to_host<Real>(Y_device, Y_host, 3*N);
		copy_to_host<Real>(F_device, F_host, 3*N);
		copy_to_host<Real>(T_device, T_host, 3*N);

		write_init_data(Y_host, F_host, T_host, N);

	#endif

	///////////////////////////////////////////////////////////////////////////////
	// Start repeat
	///////////////////////////////////////////////////////////////////////////////	

	/* Create FCM solver */
	cudaDeviceSynchronize();
	FCM_solver *solver = new FCM_solver;;
	for(int t = 0; t < repeat; t++){
		if(prompt > 5){
			std::cout << "\r====Computing repeat " << t+1 << "/" << repeat;
		}
		solver->hydrodynamic_solver(Y_host, F_host, T_host,
								    Y_device, F_device, T_device);
	}
	if(prompt > 5){
		printf("\nFinished loop:)\n");
	}
	

	solver->finish();

	return 0;
}

