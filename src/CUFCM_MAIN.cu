#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>
#include <string>
#include <vector>

#include "config.hpp"
#include "CUFCM_DATA.cuh"
#include "CUFCM_SOLVER.cuh"
#include "CUFCM_RANDOMPACKER.cuh"
#include "util/cuda_util.hpp"


int main(int argc, char** argv) {
	///////////////////////////////////////////////////////////////////////////////
	// Initialise parameters
	///////////////////////////////////////////////////////////////////////////////
	Pars pars;
	Real values[100];
	std::vector<std::string> datafile_names{3};
	read_config(values, datafile_names, "simulation_info");
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

	Real* Yf_host = malloc_host<Real>(3*pars.N);					Real* Yf_device = malloc_device<Real>(3*pars.N);
	Real* Yv_host = malloc_host<Real>(3*pars.N);					Real* Yv_device = malloc_device<Real>(3*pars.N);
	Real* F_host = malloc_host<Real>(3*pars.N);						Real* F_device = malloc_device<Real>(3*pars.N);
	Real* T_host = malloc_host<Real>(3*pars.N);						Real* T_device = malloc_device<Real>(3*pars.N);
	Real* V_host = malloc_host<Real>(3*pars.N);						Real* V_device = malloc_device<Real>(3*pars.N);
	Real* W_host = malloc_host<Real>(3*pars.N);						Real* W_device = malloc_device<Real>(3*pars.N);

	///////////////////////////////////////////////////////////////////////////////
	// Physical system initialisation
	///////////////////////////////////////////////////////////////////////////////

	read_init_data(Yf_host, pars.N, datafile_names[0].c_str());
	read_init_data(F_host, pars.N, datafile_names[1].c_str());
	read_init_data(T_host, pars.N, datafile_names[2].c_str());

	for(int i = 0; i<3*pars.N; i++){
		Yf_host[i] = Yf_host[i];
		Yv_host[i] = Yf_host[i];
		F_host[i] = F_host[i];
		T_host[i] = T_host[i];
	}

	copy_to_device<Real>(Yf_host, Yf_device, 3*pars.N);
	copy_to_device<Real>(Yv_host, Yv_device, 3*pars.N);
	copy_to_device<Real>(F_host, F_device, 3*pars.N);
	copy_to_device<Real>(T_host, T_device, 3*pars.N);

	///////////////////////////////////////////////////////////////////////////////
	// Start repeat
	///////////////////////////////////////////////////////////////////////////////	

	/* Create FCM solver */
	cudaDeviceSynchronize();
	FCM_solver solver(pars);
	solver.assign_host_array_pointers(Yf_host, Yv_host, F_host, T_host, V_host, W_host);

	for(int t = 0; t < pars.repeat; t++){
		if(pars.prompt > 5){
			std::cout << "\r====Computing repeat " << t+1 << "/" << pars.repeat;
		}
		solver.hydrodynamic_solver(Yf_device, Yv_device, F_device, T_device, V_device, W_device);
	}
	if(pars.prompt > 5){
		printf("\nFinished loop:)\n");
	}
	solver.finish();

	return 0;
}

