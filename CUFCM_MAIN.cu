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
#if CONFIG_TYPE == 0
	#include "config_fcm.hpp"
#endif
#include "CUFCM_FCM.cuh"
#include "CUFCM_CORRECTION.cuh"
#include "CUFCM_data.cuh"
#include "CUFCM_SOLVER.cuh"

#include "util/cuda_util.hpp"
#include "util/CUFCM_linklist.hpp"
#include "util/CUFCM_print.hpp"
// #include "util/CUFCM_hashing.hpp"
#include "util/maths_util.hpp"


int main(int argc, char** argv) {
	///////////////////////////////////////////////////////////////////////////////
	// Initialise parameters
	///////////////////////////////////////////////////////////////////////////////

	Real values[100];
	read_config(values, "simulation_info");
	int N = values[0];
	int repeat = values[8];

	Real* Y_host = malloc_host<Real>(3*N);						Real* Y_device = malloc_device<Real>(3*N);
	Real* F_host = malloc_host<Real>(3*N);						Real* F_device = malloc_device<Real>(3*N);
	Real* T_host = malloc_host<Real>(3*N);						Real* T_device = malloc_device<Real>(3*N);
	///////////////////////////////////////////////////////////////////////////////
	// Physical system initialisation
	///////////////////////////////////////////////////////////////////////////////

	#if INIT_FROM_FILE == 1

		read_init_data(Y_host, N, "./data/init_data/N500000/pos-N500000-rh02609300-2.dat");
		read_init_data(F_host, N, "./data/init_data/N500000/force-N500000-rh02609300.dat");
		read_init_data(T_host, N, "./data/init_data/N500000/force-N500000-rh02609300-2.dat");	

		// read_init_data(Y_host, N, "./data/init_data/N16777216/pos-N16777216-rh008089855.dat");
		// read_init_data(F_host, N, "./data/init_data/N16777216/force-N16777216-rh008089855.dat");
		// read_init_data(T_host, N, "./data/init_data/N16777216/force-N16777216-rh008089855-2.dat");

	#elif INIT_FROM_FILE == 0

		// init_pos_random_check_gpu(Y_device, rh, N);
		// init_pos_random_overlapping<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(Y_device, N, dev_random);
		init_pos_lattice_random<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(Y_device, rh, N, dev_random);
		init_force_kernel<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(F_device, rh, N, dev_random);
		init_force_kernel<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(T_device, rh, N, dev_random);

		printf("Copying to host...\n");
		copy_to_host<Real>(Y_device, Y_host, 3*N);
		copy_to_host<Real>(F_device, F_host, 3*N);
		copy_to_host<Real>(T_device, T_host, 3*N);

		write_init_data(Y_host, F_host, T_host, N);

	#endif

	///////////////////////////////////////////////////////////////////////////////
	// Start repeat
	///////////////////////////////////////////////////////////////////////////////
	FCM_solver solver;

	for(int t = 0; t < repeat; t++){
		solver.hydrodynamic_solver(Y_host, F_host, T_host);
	}
	printf("finished loop:)\n");

	solver.finish();

	return 0;
}

