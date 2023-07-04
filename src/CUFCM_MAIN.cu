// #include <cstdlib>
// #include <iostream>
// #include <fstream>
// #include <algorithm>
// #include <cmath>
// #include <cuda_runtime.h>
// #include <cufft.h>
// #include <string>
// #include <vector>

// #include "../src/config.hpp"
// #include "../src/CUFCM_DATA.cuh"
// #include "../src/CUFCM_SOLVER.cuh"
// #include "../src/util/cuda_util.hpp"
// #include "../src/util/maths_util.hpp"


// int main(int argc, char** argv) {
// 	#ifdef USE_REGULAR_FCM
// 		std::string info_name = "./test/test_info/test_fcm_info";
// 	#else
// 		std::string info_name = "./test/test_info/test_fastfcm_info";
// 	#endif

// 	#if ROTATION == 0
// 		std::string ref_name = "./data/refdata/ref_data_N500000_translate_1e-4.dat";
// 	#elif ROTATION == 1
// 		std::string ref_name = "./data/refdata/ref_data_N500000_rotation_1e-4.dat";
// 	#endif

// 	///////////////////////////////////////////////////////////////////////////////
// 	// Initialise parameters
// 	///////////////////////////////////////////////////////////////////////////////
// 	Pars pars;
// 	Real values[100];
// 	std::vector<std::string> datafile_names{3};
// 	read_config(values, datafile_names, "simulation_info");
// 	parser_config(values, pars);

// 	thrust::host_vector<Real> Yf_host(3*pars.N);						thrust::device_vector<Real> Yf_device(3*pars.N);
// 	thrust::host_vector<Real> Yv_host(3*pars.N);						thrust::device_vector<Real> Yv_device(3*pars.N);
// 	thrust::host_vector<Real> F_host(3*pars.N);							thrust::device_vector<Real> F_device(3*pars.N);
// 	thrust::host_vector<Real> T_host(3*pars.N);							thrust::device_vector<Real> T_device(3*pars.N);
// 	thrust::host_vector<Real> V_host(3*pars.N);							thrust::device_vector<Real> V_device(3*pars.N);
// 	thrust::host_vector<Real> W_host(3*pars.N);							thrust::device_vector<Real> W_device(3*pars.N);

// 	// Real* Yf_host = malloc_host<Real>(3*pars.N);					Real* Yf_device = malloc_device<Real>(3*pars.N);
// 	// Real* Yv_host = malloc_host<Real>(3*pars.N);					Real* Yv_device = malloc_device<Real>(3*pars.N);
// 	// Real* F_host = malloc_host<Real>(3*pars.N);						Real* F_device = malloc_device<Real>(3*pars.N);
// 	// Real* T_host = malloc_host<Real>(3*pars.N);						Real* T_device = malloc_device<Real>(3*pars.N);
// 	// Real* V_host = malloc_host<Real>(3*pars.N);						Real* V_device = malloc_device<Real>(3*pars.N);
// 	// Real* W_host = malloc_host<Real>(3*pars.N);						Real* W_device = malloc_device<Real>(3*pars.N);

// 	///////////////////////////////////////////////////////////////////////////////
// 	// Physical system initialisation
// 	///////////////////////////////////////////////////////////////////////////////
// 	read_init_data_thrust(Yf_host, pars.N, datafile_names[0].c_str());
// 	read_init_data_thrust(F_host, pars.N, datafile_names[1].c_str());
// 	read_init_data_thrust(T_host, pars.N, datafile_names[2].c_str());

// 	// read_init_data(Yf_host, pars.N, datafile_names[0].c_str());
// 	// read_init_data(F_host, pars.N, datafile_names[1].c_str());
// 	// read_init_data(T_host, pars.N, datafile_names[2].c_str());

// 	Yf_device = Yf_host;
// 	Yv_device = Yf_host; // This is on purpose
// 	F_device = F_host;
// 	T_device = T_host;

// 	// for(int i = 0; i<3*pars.N; i++){
// 	// 	Yf_host[i] = Yf_host[i];
// 	// 	Yv_host[i] = Yf_host[i];
// 	// 	F_host[i] = F_host[i];
// 	// 	T_host[i] = T_host[i];
// 	// }

// 	// copy_to_device<Real>(Yf_host, Yf_device, 3*pars.N);
// 	// copy_to_device<Real>(Yv_host, Yv_device, 3*pars.N);
// 	// copy_to_device<Real>(F_host, F_device, 3*pars.N);
// 	// copy_to_device<Real>(T_host, T_device, 3*pars.N);

// 	///////////////////////////////////////////////////////////////////////////////
// 	// Start repeat
// 	///////////////////////////////////////////////////////////////////////////////	

// 	/* Create FCM solver */
// 	cudaDeviceSynchronize();
// 	FCM_solver solver(pars);
// 	solver.assign_host_array_pointers(thrust::raw_pointer_cast(Yf_host.data()), 
// 									  thrust::raw_pointer_cast(Yv_host.data()), 
// 									  thrust::raw_pointer_cast(F_host.data()), 
// 									  thrust::raw_pointer_cast(T_host.data()), 
// 									  thrust::raw_pointer_cast(V_host.data()), 
// 									  thrust::raw_pointer_cast(W_host.data()));
// 	// solver.assign_host_array_pointers(Yf_host, Yv_host, F_host, T_host, V_host, W_host);

// 	for(int t = 0; t < pars.repeat; t++){
// 		if(pars.prompt > 5){
// 			std::cout << "\r====Computing repeat " << t+1 << "/" << pars.repeat;
// 		}
// 		solver.hydrodynamic_solver(thrust::raw_pointer_cast(Yf_device.data()), 
// 								   thrust::raw_pointer_cast(Yv_device.data()), 
// 								   thrust::raw_pointer_cast(F_device.data()), 
// 								   thrust::raw_pointer_cast(T_device.data()), 
// 								   thrust::raw_pointer_cast(V_device.data()), 
// 								   thrust::raw_pointer_cast(W_device.data()));
// 		// solver.hydrodynamic_solver(Yf_device, Yv_device, F_device, T_device, V_device, W_device);
// 	}
// 	if(pars.prompt > 5){
// 		printf("\nFinished loop:)\n");
// 	}
// 	solver.finish();

// 	///////////////////////////////////////////////////////////////////////////////
// 	// Check error
// 	///////////////////////////////////////////////////////////////////////////////
//     Real Yerror = -1, Verror = -1, Werror = -1;

// 	Yf_host = Yf_device;
// 	Yv_host = Yv_device;
// 	V_host = V_device;
// 	W_host = W_device;

// 	if (pars.checkerror == 1){
// 		thrust::host_vector<Real> Y_validation(3*pars.N);
// 		thrust::host_vector<Real> F_validation(3*pars.N);
// 		thrust::host_vector<Real> T_validation(3*pars.N); 
// 		thrust::host_vector<Real> V_validation(3*pars.N); 
// 		thrust::host_vector<Real> W_validation(3*pars.N);
//         // Real* Y_validation = malloc_host<Real>(3*pars.N);
// 		// Real* F_validation = malloc_host<Real>(3*pars.N);
//         // Real* T_validation = malloc_host<Real>(3*pars.N);
// 		// Real* V_validation = malloc_host<Real>(3*pars.N);
// 		// Real* W_validation = malloc_host<Real>(3*pars.N);

// 		read_validate_data_thrust(Y_validation,
// 								  F_validation,
// 								  T_validation, 
// 								  V_validation, 
// 								  W_validation, pars.N, "./data/refdata/ref_data_N500000_rotation_1e-4.dat");

// 		// read_validate_data(Y_validation,
// 		// 				   F_validation,
//         //                 T_validation,
// 		// 				   V_validation,
// 		// 				   W_validation, pars.N, "./data/refdata/ref_data_N500000_rotation_1e-4.dat");

// 		Yerror = percentage_error_magnitude_thrust(Yf_host, Y_validation, pars.N);
// 		Verror = percentage_error_magnitude_thrust(V_host, V_validation, pars.N);
// 		Werror = percentage_error_magnitude_thrust(W_host, W_validation, pars.N);

// 		// Yerror = percentage_error_magnitude(Yf_host, Y_validation, pars.N);
// 		// Verror = percentage_error_magnitude(V_host, V_validation, pars.N);
// 		// Werror = percentage_error_magnitude(W_host, W_validation, pars.N);

// 		if(pars.prompt > 1){
// 			std::cout << "-------\nError\n-------\n";
// 			std::cout << "%Y error:\t" << Yerror << "\n";
// 			std::cout << "%V error:\t" << Verror << "\n";
// 			std::cout << "%W error:\t" << Werror << "\n";
// 		}
// 	}

// 	///////////////////////////////////////////////////////////////////////////////
// 	// Write to file
// 	///////////////////////////////////////////////////////////////////////////////

// 	// write_data(Yf_host, F_host, T_host, V_host, W_host, solver.N, 
// 	// 			"./data/simulation/simulation_data.dat", "w");

// 	// write_data_thrust(Yf_host, F_host, T_host, V_host, W_host, solver.N, 
// 	// 			"./data/simulation/simulation_data.dat", "w");

// 	// write_time(solver.time_cuda_initialisation, 
// 	// 		solver.time_readfile,
// 	// 		solver.time_hashing,
// 	// 		solver.time_spreading,
// 	// 		solver.time_FFT,
// 	// 		solver.time_gathering,
// 	// 		solver.time_correction,
// 	// 		solver.time_compute,
// 	// 		"./data/simulation/simulation_scalar.dat");

// 	// write_error(
// 	// 	Verror,
// 	// 	Werror,
// 	// 	"./data/simulation/simulation_scalar.dat");

// }

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>
#include <string>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "../src/config.hpp"
#include "../src/CUFCM_DATA.cuh"
#include "../src/CUFCM_SOLVER.cuh"
#include "../src/util/cuda_util.hpp"
#include "../src/util/maths_util.hpp"


int main(int argc, char** argv) {
	#ifdef USE_REGULAR_FCM
		std::string info_name = "./test/test_info/test_fcm_info";
	#else
		std::string info_name = "./simulation_info";
	#endif

	#if ROTATION == 0
		std::string ref_name = "./data/refdata/ref_data_N500000_translate_1e-4.dat";
	#elif ROTATION == 1
		std::string ref_name = "./data/refdata/ref_data_N500000_rotation_1e-4.dat";
	#endif

	///////////////////////////////////////////////////////////////////////////////
	// Initialise parameters
	///////////////////////////////////////////////////////////////////////////////
	Pars pars;
	Real values[100];
	std::vector<std::string> datafile_names{3};
	read_config(values, datafile_names, info_name.c_str());
	parser_config(values, pars);

	thrust::host_vector<Real> Yf_host(3*pars.N);						thrust::device_vector<Real> Yf_device(3*pars.N);
	thrust::host_vector<Real> Yv_host(3*pars.N);						thrust::device_vector<Real> Yv_device(3*pars.N);
	thrust::host_vector<Real> F_host(3*pars.N);							thrust::device_vector<Real> F_device(3*pars.N);
	thrust::host_vector<Real> T_host(3*pars.N);							thrust::device_vector<Real> T_device(3*pars.N);
	thrust::host_vector<Real> V_host(3*pars.N);							thrust::device_vector<Real> V_device(3*pars.N);
	thrust::host_vector<Real> W_host(3*pars.N);							thrust::device_vector<Real> W_device(3*pars.N);

	///////////////////////////////////////////////////////////////////////////////
	// Physical system initialisation
	///////////////////////////////////////////////////////////////////////////////
	read_init_data_thrust(Yf_host, pars.N, datafile_names[0].c_str());
	read_init_data_thrust(F_host, pars.N, datafile_names[1].c_str());
	read_init_data_thrust(T_host, pars.N, datafile_names[2].c_str());

	Yf_device = Yf_host;
	Yv_device = Yf_host; // This is on purpose
	F_device = F_host;
	T_device = T_host;

	///////////////////////////////////////////////////////////////////////////////
	// Start repeat
	///////////////////////////////////////////////////////////////////////////////	

	/* Create FCM solver */
	cudaDeviceSynchronize();
	FCM_solver solver(pars);
	solver.assign_host_array_pointers(thrust::raw_pointer_cast(Yf_host.data()), 
									  thrust::raw_pointer_cast(Yv_host.data()), 
									  thrust::raw_pointer_cast(F_host.data()), 
									  thrust::raw_pointer_cast(T_host.data()), 
									  thrust::raw_pointer_cast(V_host.data()), 
									  thrust::raw_pointer_cast(W_host.data()));

	for(int t = 0; t < pars.repeat; t++){
		if(pars.prompt > 5){
			std::cout << "\r====Computing repeat " << t+1 << "/" << pars.repeat;
		}
		solver.hydrodynamic_solver(thrust::raw_pointer_cast(Yf_device.data()), 
								   thrust::raw_pointer_cast(Yv_device.data()), 
								   thrust::raw_pointer_cast(F_device.data()), 
								   thrust::raw_pointer_cast(T_device.data()), 
								   thrust::raw_pointer_cast(V_device.data()), 
								   thrust::raw_pointer_cast(W_device.data()));
	}
	if(pars.prompt > 5){
		printf("\nFinished loop:)\n");
	}
	solver.finish();

	///////////////////////////////////////////////////////////////////////////////
	// Check error
	///////////////////////////////////////////////////////////////////////////////
    Real Yerror = -1, Verror = -1, Werror = -1;

	Yf_host = Yf_device;
	Yv_host = Yv_device;
	V_host = V_device;
	W_host = W_device;

	if (pars.checkerror == 1){
		thrust::host_vector<Real> Y_validation(3*pars.N);
		thrust::host_vector<Real> F_validation(3*pars.N);
		thrust::host_vector<Real> T_validation(3*pars.N); 
		thrust::host_vector<Real> V_validation(3*pars.N); 
		thrust::host_vector<Real> W_validation(3*pars.N);

		read_validate_data_thrust(Y_validation,
								  F_validation,
								  T_validation, 
								  V_validation, 
								  W_validation, pars.N, ref_name.c_str());

		Yerror = percentage_error_magnitude_thrust(Yf_host, Y_validation, pars.N);
		Verror = percentage_error_magnitude_thrust(V_host, V_validation, pars.N);
		Werror = percentage_error_magnitude_thrust(W_host, W_validation, pars.N);

		if(pars.prompt > 1){
			std::cout << "-------\nError\n-------\n";
			std::cout << "%Y error:\t" << Yerror << "\n";
			std::cout << "%V error:\t" << Verror << "\n";
			std::cout << "%W error:\t" << Werror << "\n";
		}
	}

	///////////////////////////////////////////////////////////////////////////////
	// Write to file
	///////////////////////////////////////////////////////////////////////////////

	write_data_thrust(Yf_host, F_host, T_host, V_host, W_host, solver.N, 
				"./data/simulation/simulation_data.dat", "w");

	write_time(solver.time_cuda_initialisation, 
			solver.time_readfile,
			solver.time_hashing,
			solver.time_spreading,
			solver.time_FFT,
			solver.time_gathering,
			solver.time_correction,
			solver.time_compute,
			"./data/simulation/simulation_scalar.dat");

	write_error(
		Verror,
		Werror,
		"./data/simulation/simulation_scalar.dat");

	return 0;
}