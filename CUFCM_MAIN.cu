#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>

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
	read_config(values, "simulation_info_long");
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

	Real* Y_host = malloc_host<Real>(3*pars.N);						Real* Y_device = malloc_device<Real>(3*pars.N);
	Real* F_host = malloc_host<Real>(3*pars.N);						Real* F_device = malloc_device<Real>(3*pars.N);
	Real* T_host = malloc_host<Real>(3*pars.N);						Real* T_device = malloc_device<Real>(3*pars.N);
	Real* V_host = malloc_host<Real>(3*pars.N);						Real* V_device = malloc_device<Real>(3*pars.N);
	Real* W_host = malloc_host<Real>(3*pars.N);						Real* W_device = malloc_device<Real>(3*pars.N);

	///////////////////////////////////////////////////////////////////////////////
	// Physical system initialisation
	///////////////////////////////////////////////////////////////////////////////

	#if INIT_FROM_FILE == 1

		Real Ffac = values[14];
		Real Tfac = values[15];

		// read_init_data(Y_host, pars.N, "./data/init_data/new/pos_data.dat");
		// read_init_data(F_host, pars.N, "./data/init_data/new/force_data.dat");
		// read_init_data(T_host, pars.N, "./data/init_data/new/torque_data.dat");

		read_init_data(Y_host, pars.N, "./data/init_data/N500000/pos-N500000-rh02609300-2.dat");
		read_init_data(F_host, pars.N, "./data/init_data/N500000/force-N500000-rh02609300.dat");
		read_init_data(T_host, pars.N, "./data/init_data/N500000/force-N500000-rh02609300-2.dat");

		// read_init_data(Y_host, pars.N, "./data/init_data/artificial/artificial_pos.dat");
		// read_init_data(F_host, pars.N, "./data/init_data/artificial/artificial_force.dat");
		// read_init_data(T_host, pars.N, "./data/init_data/artificial/artificial_torque.dat");

		for(int i = 0; i<3*pars.N; i++){
			Y_host[i] = Y_host[i] * pars.boxsize/PI2;
			F_host[i] = F_host[i] * Ffac;
			T_host[i] = T_host[i] * Tfac;
		}

		copy_to_device<Real>(Y_host, Y_device, 3*pars.N);
		copy_to_device<Real>(F_host, F_device, 3*pars.N);
		copy_to_device<Real>(T_host, T_device, 3*pars.N);


	#elif INIT_FROM_FILE == 0

		{
			Random_Pars rpars;
			Real rvalues[100];
			read_config(rvalues, "simulation_info_long");
			rpars.N = rvalues[0];
			rpars.rh = rvalues[1];
			rpars.alpha = rvalues[2];
			rpars.beta = rvalues[3];
			rpars.eta = rvalues[4];
			rpars.nx = rvalues[5];
			rpars.ny = rvalues[6];
			rpars.nz = rvalues[7];
			rpars.repeat = rvalues[8];
			rpars.prompt = rvalues[9];
			rpars.dt = rvalues[10];
			rpars.Fref = rvalues[11];
			rpars.boxsize = rvalues[13];

			Real packrep = values[12];

			init_random_force(F_device, rpars.rh, rpars.N);
			init_random_force(T_device, rpars.rh, rpars.N);

			FILE *pfile;
			pfile = fopen("data/simulation/spherepacking.dat", "w");
			fclose(pfile);
			
			random_packer *packer = new random_packer(Y_host, Y_device, rpars);
			for(int t = 0; t < packrep; t++){
				if(pars.prompt > 5){
					std::cout << "\rGenerating random spheres iteration: " << t+1 << "/" << packrep;
				}
				packer->update();
				if(t>packrep-11){
					packer->write();
				}
			}
			packer->finish();
			if(pars.prompt > 5){
				printf("\nFinished packing");
			}
		}
		if(pars.prompt > 5){
			printf("\nCopying to host...\n");
		}
		copy_to_host<Real>(Y_device, Y_host, 3*pars.N);
		copy_to_host<Real>(F_device, F_host, 3*pars.N);
		copy_to_host<Real>(T_device, T_host, 3*pars.N);

		write_init_data(Y_host, F_host, T_host, pars.N);

	#endif

	///////////////////////////////////////////////////////////////////////////////
	// Start repeat
	///////////////////////////////////////////////////////////////////////////////	

	/* Create FCM solver */
	cudaDeviceSynchronize();
	FCM_solver *solver = new FCM_solver(pars);
	solver->assign_host_array_pointers(Y_host, F_host, T_host, V_host, W_host);

	for(int t = 0; t < pars.repeat; t++){
		if(pars.prompt > 5){
			std::cout << "\r====Computing repeat " << t+1 << "/" << pars.repeat;
		}
		solver->hydrodynamic_solver(Y_device, F_device, T_device, V_device, W_device);
	}
	if(pars.prompt > 5){
		printf("\nFinished loop:)\n");
	}
	solver->finish();

	return 0;
}

