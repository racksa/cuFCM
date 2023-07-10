#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

#include "config.hpp"
#include "CUFCM_DATA.cuh"
#include "CUFCM_RANDOMPACKER.cuh"
#include "util/cuda_util.hpp"


int main(int argc, char** argv) {
	///////////////////////////////////////////////////////////////////////////////
	// Initialise parameters
	///////////////////////////////////////////////////////////////////////////////
	Random_Pars pars;
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
	pars.dt = values[10];
	pars.Fref = values[11];
	Real packrep = values[12];
	pars.boxsize = values[13];


	Real* Y_host = malloc_host<Real>(3*pars.N);						Real* Y_device = malloc_device<Real>(3*pars.N);
	Real* F_host = malloc_host<Real>(3*pars.N);						Real* F_device = malloc_device<Real>(3*pars.N);
	Real* T_host = malloc_host<Real>(3*pars.N);						Real* T_device = malloc_device<Real>(3*pars.N);

	///////////////////////////////////////////////////////////////////////////////
	// Start repeat
	///////////////////////////////////////////////////////////////////////////////	

	/* Create random genertaor solver */
	init_random_force(F_device, pars.Fref, pars.rh, pars.N);
	init_random_force(T_device, pars.Fref, pars.rh, pars.N);

	copy_to_host<Real>(F_device, F_host, 3*pars.N);
	copy_to_host<Real>(T_device, T_host, 3*pars.N);

	random_packer packer(Y_host, Y_device, pars);
	for(int t = 0; t < packrep; t++){
		if(pars.prompt > 5){
			std::cout << "\rGenerating random spheres iteration: " << t+1 << "/" << packrep;
		}
		packer.update();
	}
	packer.finish();
	if(pars.prompt > 5){
		printf("\nFinished packing & Copying to host...");
	}

	copy_to_host<Real>(Y_device, Y_host, 3*pars.N);

	if(pars.prompt > 5){
		printf("\nFinished copying...\n");
	}

	write_init_data(Y_host, F_host, T_host, pars.N);

	if(pars.prompt > 5){
		printf("\nFinished packing");
	}

	return 0;
}

