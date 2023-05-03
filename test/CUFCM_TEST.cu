#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>
#include <string>
#include <vector>

#include "../src/config.hpp"
#include "../src/CUFCM_DATA.cuh"
#include "../src/CUFCM_SOLVER.cuh"
#include "../src/util/cuda_util.hpp"
#include "../src/util/maths_util.hpp"


int main(int argc, char** argv) {
	///////////////////////////////////////////////////////////////////////////////
	// Initialise parameters
	///////////////////////////////////////////////////////////////////////////////
	Pars pars;
	Real values[100];
	std::vector<std::string> datafile_names{3};
	read_config(values, datafile_names, "./test/test_info/test_fastfcm_info");
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

	///////////////////////////////////////////////////////////////////////////////
	// Check error
	///////////////////////////////////////////////////////////////////////////////
    Real Yerror = -1;
    Real Verror = -1;
    Real Werror = -1;

	if (pars.checkerror == 1){
        Real* Y_validation = malloc_host<Real>(3*pars.N);
		Real* F_validation = malloc_host<Real>(3*pars.N);
        Real* T_validation = malloc_host<Real>(3*pars.N);
		Real* V_validation = malloc_host<Real>(3*pars.N);
		Real* W_validation = malloc_host<Real>(3*pars.N);

		read_validate_data(Y_validation,
						   F_validation,
                           T_validation,
						   V_validation,
						   W_validation, pars.N, "./data/refdata/ref_data_N500000");

		Yerror = percentage_error_magnitude(Yf_host, Y_validation, pars.N);
		Verror = percentage_error_magnitude(V_host, V_validation, pars.N);
		Werror = percentage_error_magnitude(W_host, W_validation, pars.N);

		if(pars.prompt > 1){
			std::cout << "-------\nError\n-------\n";
			std::cout << "%Y error:\t" << Yerror << "\n";
			std::cout << "%V error:\t" << Verror << "\n";
			std::cout << "%W error:\t" << Werror << "\n";
		}

		return 0;
	}
}

