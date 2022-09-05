#include <iostream>
#include <cmath>
// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>


#include "cuda_util.hpp"
#include "config.hpp"
#include "CUFCM_FCM.hpp"
#include "CUFCM_CORRECTION.hpp"
#include "CUFCM_util.hpp"
#include "CUFCM_data.hpp"



int main(int argc, char** argv) {

	///////////////////////////////////////////////////////////////////////////////
	// Initialise parameters
	///////////////////////////////////////////////////////////////////////////////

	auto time_start = get_time();

	int N = 500000;

	int ngd = NGD;

	double sigma_fac = 1.55917641;
	double dx = (PI2)/(NX);

	/* Link list */
	double Rref_fac = 5.21186960;
	double Rref = Rref_fac*dx;
	int M = (int) (PI2/Rref);
	double Rrefsq = Rref*Rref;
	if(M < 3){
		M = 3;
	}
	int ncell = M*M*M;
	int mapsize = 13*ncell;

	/* Monopole */
	const double rh = 0.02609300415934458;
	const double sigmaFCM = rh/sqrt(PI); // Real particle size sigmaFCM
	const double sigmaFCMsq = sigmaFCM*sigmaFCM;
	const double anormFCM = 1.0/sqrt(2.0*PI*sigmaFCMsq);
	const double anormFCM2 = 2.0*sigmaFCMsq;

	const double sigmaGRID = sigmaFCM * sigma_fac;
	const double sigmaGRIDsq = sigmaGRID * sigmaGRID;
	const double anormGRID = 1.0/sqrt(2.0*PI*sigmaGRIDsq);
	const double anormGRID2 = 2.0*sigmaGRIDsq;

	const double gammaGRID = sqrt(2.0)*sigmaGRID;
	const double pdmag = sigmaFCMsq - sigmaGRIDsq;

	/* Dipole */
	const double sigmaFCMdip = rh/pow(6.0*sqrt(PI), 1.0/3.0);
	const double sigmaFCMdipsq = sigmaFCMdip*sigmaFCMdip;
	const double anormFCMdip = 1.0/sqrt(2.0*PI*sigmaFCMdipsq);
	const double anormFCMdip2 = 2.0*sigmaFCMdipsq;

	const double sigma_dip_fac = sigmaGRID/sigmaFCMdip;
	// sigma_dip_fac = 1;

	const double sigmaGRIDdip = sigmaFCMdip * sigma_dip_fac;
	const double sigmaGRIDdipsq = sigmaGRIDdip * sigmaGRIDdip;
	const double anormGRIDdip = 1.0/sqrt(2.0*PI*sigmaGRIDdipsq);
	const double anormGRIDdip2 = 2.0*sigmaGRIDdipsq;

	/* Self corrections */
	const double StokesMob = 1.0/(6.0*PI*rh);
	const double ModStokesMob = 1.0/(6.0*PI*sigmaGRID*sqrt(PI));

	double PDStokesMob = 2.0/pow(2.0*PI, 1.5);
	PDStokesMob = PDStokesMob/pow(gammaGRID, 3.0);
	PDStokesMob = PDStokesMob*pdmag/3.0;

	double BiLapMob = 1.0/pow(4.0*PI*sigmaGRIDsq, 1.5);
	BiLapMob = BiLapMob/(4.0*sigmaGRIDsq)*pdmag*pdmag;

	const double WT1Mob = 1.0/(8.0*PI)/pow(rh, 3) ;
	const double WT2Mob = 1.0/(8.0*PI)/pow(sigmaGRIDdip*pow(6.0*sqrt(PI), 1.0/3.0), 3) ;

	


	///////////////////////////////////////////////////////////////////////////////
	// CUDA initialisation
	///////////////////////////////////////////////////////////////////////////////
	cudaDeviceSynchronize();
	time_start = get_time();
	
    cufftHandle plan, iplan;

	cufftDoubleReal* fx_host = malloc_host<cufftDoubleReal>(GRID_SIZE);					cufftDoubleReal* fx_device = malloc_device<cufftDoubleReal>(GRID_SIZE);
	cufftDoubleReal* fy_host = malloc_host<cufftDoubleReal>(GRID_SIZE);					cufftDoubleReal* fy_device = malloc_device<cufftDoubleReal>(GRID_SIZE);
	cufftDoubleReal* fz_host = malloc_host<cufftDoubleReal>(GRID_SIZE);					cufftDoubleReal* fz_device = malloc_device<cufftDoubleReal>(GRID_SIZE);
    cufftDoubleComplex* fk_x_host = malloc_host<cufftDoubleComplex>(FFT_GRID_SIZE);		cufftDoubleComplex* fk_x_device = malloc_device<cufftDoubleComplex>(FFT_GRID_SIZE);
    cufftDoubleComplex* fk_y_host = malloc_host<cufftDoubleComplex>(FFT_GRID_SIZE);		cufftDoubleComplex* fk_y_device = malloc_device<cufftDoubleComplex>(FFT_GRID_SIZE);
    cufftDoubleComplex* fk_z_host = malloc_host<cufftDoubleComplex>(FFT_GRID_SIZE);		cufftDoubleComplex* fk_z_device = malloc_device<cufftDoubleComplex>(FFT_GRID_SIZE);

	cufftDoubleReal* ux_host = malloc_host<cufftDoubleReal>(GRID_SIZE);					cufftDoubleReal* ux_device = malloc_device<cufftDoubleReal>(GRID_SIZE);
	cufftDoubleReal* uy_host = malloc_host<cufftDoubleReal>(GRID_SIZE);					cufftDoubleReal* uy_device = malloc_device<cufftDoubleReal>(GRID_SIZE);
	cufftDoubleReal* uz_host = malloc_host<cufftDoubleReal>(GRID_SIZE);					cufftDoubleReal* uz_device = malloc_device<cufftDoubleReal>(GRID_SIZE);
    cufftDoubleComplex* uk_x_host = malloc_host<cufftDoubleComplex>(FFT_GRID_SIZE);		cufftDoubleComplex* uk_x_device = malloc_device<cufftDoubleComplex>(FFT_GRID_SIZE);
    cufftDoubleComplex* uk_y_host = malloc_host<cufftDoubleComplex>(FFT_GRID_SIZE);		cufftDoubleComplex* uk_y_device = malloc_device<cufftDoubleComplex>(FFT_GRID_SIZE);
    cufftDoubleComplex* uk_z_host = malloc_host<cufftDoubleComplex>(FFT_GRID_SIZE);		cufftDoubleComplex* uk_z_device = malloc_device<cufftDoubleComplex>(FFT_GRID_SIZE);

	double* Y_host = malloc_host<double>(3*N);					double* Y_device = malloc_device<double>(3*N);
	double* F_host = malloc_host<double>(3*N);					double* F_device = malloc_device<double>(3*N);
	double* T_host = malloc_host<double>(3*N);					double* T_device = malloc_device<double>(3*N);
	double* V_host = malloc_host<double>(3*N);					double* V_device = malloc_device<double>(3*N);
	double* W_host = malloc_host<double>(3*N);					double* W_device = malloc_device<double>(3*N);
	double* GA_host = malloc_host<double>(6*N);					double* GA_device = malloc_device<double>(6*N);

	double* gaussx_host = malloc_host<double>(ngd*N);			double* gaussx_device = malloc_device<double>(ngd*N);
	double* gaussy_host = malloc_host<double>(ngd*N);			double* gaussy_device = malloc_device<double>(ngd*N);
	double* gaussz_host = malloc_host<double>(ngd*N);			double* gaussz_device = malloc_device<double>(ngd*N);
	double* grad_gaussx_dip_host = malloc_host<double>(ngd*N);	double* grad_gaussx_dip_device = malloc_device<double>(ngd*N);
	double* grad_gaussy_dip_host = malloc_host<double>(ngd*N);	double* grad_gaussy_dip_device = malloc_device<double>(ngd*N);
	double* grad_gaussz_dip_host = malloc_host<double>(ngd*N);	double* grad_gaussz_dip_device = malloc_device<double>(ngd*N);
	double* gaussgrid_host = malloc_host<double>(ngd);			double* gaussgrid_device = malloc_device<double>(ngd);
	double* xdis_host = malloc_host<double>(ngd*N);				double* xdis_device = malloc_device<double>(ngd*N);
	double* ydis_host = malloc_host<double>(ngd*N);				double* ydis_device = malloc_device<double>(ngd*N);
	double* zdis_host = malloc_host<double>(ngd*N);				double* zdis_device = malloc_device<double>(ngd*N);
	int* indx_host = malloc_host<int>(ngd*N);					int* indx_device = malloc_device<int>(ngd*N);
	int* indy_host = malloc_host<int>(ngd*N);					int* indy_device = malloc_device<int>(ngd*N);
	int* indz_host = malloc_host<int>(ngd*N);					int* indz_device = malloc_device<int>(ngd*N);

	int* map_host = malloc_host<int>(mapsize);					int* map_device = malloc_device<int>(mapsize);
	int* head_host = malloc_host<int>(ncell);					int* head_device = malloc_device<int>(ncell);
	int* list_host = malloc_host<int>(N);						int* list_device = malloc_device<int>(N);

	int* Y_hash_host = malloc_host<int>(N);
	int* F_hash_host = malloc_host<int>(N);
	int* T_hash_host = malloc_host<int>(N);
	int* data_hash_host = malloc_host<int>(N);
	int* original_index_host = malloc_host<int>(N);

	bulkmap_loop(map_host, M, HASH_ENCODE_FUNC);
	copy_to_device<int>(map_host, map_device, mapsize);

	/* Create 3D FFT plans */
	if (cufftPlan3d(&plan, NX, NY, NZ, CUFFT_D2Z) != CUFFT_SUCCESS){
		printf("CUFFT error: Plan creation failed");
		return 0;	
	}

	if (cufftPlan3d(&iplan, NX, NY, NZ, CUFFT_Z2D) != CUFFT_SUCCESS){
		printf("CUFFT error: Plan creation failed");
		return 0;	
	}

	const int num_thread_blocks_GRID = (GRID_SIZE + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
	const int num_thread_blocks_N = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;

	auto time_cuda_initialisation = get_time() - time_start;
	///////////////////////////////////////////////////////////////////////////////
	// Wave vector initialisation
	///////////////////////////////////////////////////////////////////////////////
	int pad = (NX/2 + 1);
	int nptsh = (NX/2);
	double* q_host = malloc_host<double>(NX);			double* q_device = malloc_device<double>(NX);
	double* qpad_host = malloc_host<double>(pad);		double* qpad_device = malloc_device<double>(pad);
	double* qsq_host = malloc_host<double>(NX);			double* qsq_device = malloc_device<double>(NX);
	double* qpadsq_host = malloc_host<double>(pad);		double* qpadsq_device = malloc_device<double>(pad);

	for(int i=0; i<NX; i++){
		if(i < nptsh || i == nptsh){
			q_host[i] = (double) i;
		}
		if(i > nptsh){
			q_host[i] = (double) (i - NX);
		}
		qsq_host[i] = q_host[i]*q_host[i];
	}
	
	for(int i=0; i<pad; i++){
		qpad_host[i] = (double) i;
		qpadsq_host[i] = qpad_host[i]*qpad_host[i];
	}
	copy_to_device<double>(q_host, q_device, NX);
	copy_to_device<double>(qpad_host, qpad_device, pad);
	copy_to_device<double>(qsq_host, qsq_device, NX);
	copy_to_device<double>(qpadsq_host, qpadsq_device, pad);

	///////////////////////////////////////////////////////////////////////////////
	// Physical system initialisation
	///////////////////////////////////////////////////////////////////////////////
	cudaDeviceSynchronize();	time_start = get_time();
	read_init_data(Y_host, N, "./init_data/pos-N500000-rh02609300-2.dat");
	read_init_data(F_host, N, "./init_data/force-N500000-rh02609300.dat");
	read_init_data(T_host, N, "./init_data/force-N500000-rh02609300-2.dat");

	/* Sorting */
	#if SPATIAL_HASHING

		for(int i = 0; i < N; i++){
			original_index_host[i] = i;
		}
		create_hash(Y_hash_host, Y_host, N, dx, HASH_ENCODE_FUNC);
		create_hash(F_hash_host, Y_host, N, dx, HASH_ENCODE_FUNC);
		create_hash(T_hash_host, Y_host, N, dx, HASH_ENCODE_FUNC);
		create_hash(data_hash_host, Y_host, N, dx, HASH_ENCODE_FUNC);
		quicksort(Y_hash_host, Y_host, 0, N - 1);
		quicksort(F_hash_host, F_host, 0, N - 1);
		quicksort(T_hash_host, T_host, 0, N - 1);
		quicksort_1D(data_hash_host, original_index_host, 0, N - 1);

	#endif

	copy_to_device<double>(Y_host, Y_device, 3*N);
	copy_to_device<double>(F_host, F_device, 3*N);
	copy_to_device<double>(T_host, T_device, 3*N);

	cudaDeviceSynchronize();	auto time_readfile = get_time() - time_start;
	///////////////////////////////////////////////////////////////////////////////
	// Link
	///////////////////////////////////////////////////////////////////////////////
	cudaDeviceSynchronize();	time_start = get_time();
	// link<<<num_thread_blocks, THREADS_PER_BLOCK>>>(list_device, head_device, Y_device, M, ncell, N);
	link_loop(list_host, head_host, Y_host, M, N, HASH_ENCODE_FUNC);

	copy_to_device<int>(list_host, list_device, N);
	copy_to_device<int>(head_host, head_device, ncell);

	cudaDeviceSynchronize();	auto time_linklist = get_time() - time_start;
	///////////////////////////////////////////////////////////////////////////////
	// Gaussian initialisation
	///////////////////////////////////////////////////////////////////////////////
	cudaDeviceSynchronize();	time_start = get_time();

	GA_setup<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(GA_device, T_device, N);

	// #if PARALLELISATION_TYPE == 0

		cufcm_precompute_gauss<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(N, ngd, Y_device,
					gaussx_device, gaussy_device, gaussz_device,
					grad_gaussx_dip_device, grad_gaussy_dip_device, grad_gaussz_dip_device,
					gaussgrid_device,
					xdis_device, ydis_device, zdis_device,
					indx_device, indy_device, indz_device,
					sigmaGRIDdipsq, anormGRID, anormGRID2, dx);

	// #endif
	
	cudaDeviceSynchronize();	auto time_precompute_gauss = get_time() - time_start;
	///////////////////////////////////////////////////////////////////////////////
	// Spreading
	///////////////////////////////////////////////////////////////////////////////
	cudaDeviceSynchronize();	time_start = get_time();

	#if PARALLELISATION_TYPE == 0

		cufcm_mono_dipole_distribution_tpp_register<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(fx_device, fy_device, fz_device, N,
											GA_device, F_device, pdmag, sigmaGRIDsq,
											gaussx_device, gaussy_device, gaussz_device,
											grad_gaussx_dip_device, grad_gaussy_dip_device, grad_gaussz_dip_device,
											xdis_device, ydis_device, zdis_device,
											indx_device, indy_device, indz_device,
											ngd);

	#elif PARALLELISATION_TYPE == 1

		cufcm_mono_dipole_distribution_tpp_recompute<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(fx_device, fy_device, fz_device,
											Y_device, GA_device, F_device,
											N, ngd,
											pdmag, sigmaGRIDsq, sigmaGRIDdipsq,
											anormGRID, anormGRID2,
											dx);
	#elif PARALLELISATION_TYPE == 2

		cufcm_mono_dipole_distribution_bpp<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(fx_device, fy_device, fz_device, 
											Y_device, GA_device, F_device,
											N, ngd,
											pdmag, sigmaGRIDsq, sigmaGRIDdipsq,
											anormGRID, anormGRID2,
											dx);

	#endif

	cudaDeviceSynchronize();	auto time_spreading = get_time() - time_start;
	///////////////////////////////////////////////////////////////////////////////
	// FFT
	///////////////////////////////////////////////////////////////////////////////
	cudaDeviceSynchronize();	time_start = get_time();
	if (cufftExecD2Z(plan, fx_device, fk_x_device) != CUFFT_SUCCESS){
		printf("CUFFT error: ExecD2Z Forward failed (fx)\n");
		return 0;	
	}
	if (cufftExecD2Z(plan, fy_device, fk_y_device) != CUFFT_SUCCESS){
		printf("CUFFT error: ExecD2Z Forward failed (fy)\n");
		return 0;	
	}
	if (cufftExecD2Z(plan, fz_device, fk_z_device) != CUFFT_SUCCESS){
		printf("CUFFT error: ExecD2Z Forward failed (fz)\n");
		return 0;	
	}

	///////////////////////////////////////////////////////////////////////////////
	// Solve for the flow
	///////////////////////////////////////////////////////////////////////////////
	cufcm_flow_solve<<<num_thread_blocks_GRID, THREADS_PER_BLOCK>>>(fk_x_device, fk_y_device, fk_z_device,
															   uk_x_device, uk_y_device, uk_z_device,
															   q_device, qpad_device, qsq_device, qpadsq_device);

	///////////////////////////////////////////////////////////////////////////////
	// IFFT
	///////////////////////////////////////////////////////////////////////////////
	if (cufftExecZ2D(iplan, uk_x_device, ux_device) != CUFFT_SUCCESS){
		printf("CUFFT error: ExecD2Z Backward failed (fx)\n");
		return 0;	
	}
	if (cufftExecZ2D(iplan, uk_y_device, uy_device) != CUFFT_SUCCESS){
		printf("CUFFT error: ExecD2Z Backward failed (fy)\n");
		return 0;	
	}
	if (cufftExecZ2D(iplan, uk_z_device, uz_device) != CUFFT_SUCCESS){
		printf("CUFFT error: ExecZ2D Backward failed (fz)\n");
		return 0;	
	}

	cudaDeviceSynchronize();	auto time_FFT = get_time() - time_start;
	///////////////////////////////////////////////////////////////////////////////
	// Gathering
	///////////////////////////////////////////////////////////////////////////////
	cudaDeviceSynchronize();	time_start = get_time();

	#if PARALLELISATION_TYPE == 0

		cufcm_particle_velocities_tpp_register<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(ux_device, uy_device, uz_device, N,
									V_device, W_device,
									pdmag, sigmaGRIDsq,
									gaussx_device, gaussy_device, gaussz_device,
									grad_gaussx_dip_device, grad_gaussy_dip_device, grad_gaussz_dip_device,
									xdis_device, ydis_device, zdis_device,
									indx_device, indy_device, indz_device,
									ngd, dx);

	#elif PARALLELISATION_TYPE == 1

		cufcm_particle_velocities_tpp_recompute<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(ux_device, uy_device, uz_device,
									Y_device,
									V_device, W_device,
									N, ngd,
									pdmag, sigmaGRIDsq, sigmaGRIDdipsq,
									anormGRID, anormGRID2,
									dx);

	#elif PARALLELISATION_TYPE == 2

		cufcm_particle_velocities_bpp<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(ux_device, uy_device, uz_device,
									Y_device,
									V_device, W_device,
									N, ngd,
									pdmag, sigmaGRIDsq, sigmaGRIDdipsq,
									anormGRID, anormGRID2,
									dx);

	#endif

	cudaDeviceSynchronize();	auto time_gathering = get_time() - time_start;
	///////////////////////////////////////////////////////////////////////////////
	// Correction
	///////////////////////////////////////////////////////////////////////////////
	cudaDeviceSynchronize();	time_start = get_time();

	#if CORRECTION_TYPE == 0

		cufcm_pair_correction<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(Y_device, V_device, W_device, F_device, T_device, N,
							map_device, head_device, list_device,
							ncell, Rrefsq,
							pdmag,
							sigmaGRID, sigmaGRIDsq,
							sigmaFCM, sigmaFCMsq,
							sigmaFCMdip, sigmaFCMdipsq);
	
	#elif CORRECTION_TYPE == 1

		cufcm_pair_correction_spatial_hashing<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(Y_device, V_device, W_device, F_device, T_device, N,
							map_device, head_device, list_device,
							ncell, Rrefsq,
							pdmag,
							sigmaGRID, sigmaGRIDsq,
							sigmaFCM, sigmaFCMsq,
							sigmaFCMdip, sigmaFCMdipsq);

	#endif

	cufcm_self_correction<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(V_device, W_device, F_device, T_device, N,
							   StokesMob, ModStokesMob,
							   PDStokesMob, BiLapMob,
							   WT1Mob, WT2Mob);

	cudaDeviceSynchronize();	auto time_correction = get_time() - time_start;


	/* Print */
	copy_to_host<double>(V_device, V_host, 3*N);
	copy_to_host<double>(W_device, W_host, 3*N);
	// print_host_data_real_3D_flat<double>(V_host, N, 3);


	#if SPATIAL_HASHING

		for(int i = 0; i < N; i++){
			F_hash_host[i] = original_index_host[i];
			T_hash_host[i] = original_index_host[i];
		}
		quicksort(F_hash_host, V_host, 0, N - 1);
		quicksort(T_hash_host, W_host, 0, N - 1);

	#endif


	for(int i = N-10; i < N; i++){
		printf("%d V ( ", i);
		for(int n = 0; n < 3; n++){
			printf("%.8f ", V_host[3*i + n]);
		}
		printf(")     \t");
		printf("W ( ");
		for(int n = 0; n < 3; n++){
			printf("%.8f ", W_host[3*i + n]);
		}
		printf(")\n");
	}

	///////////////////////////////////////////////////////////////////////////////
	// Time
	///////////////////////////////////////////////////////////////////////////////
	auto time_compute = time_linklist + time_precompute_gauss + time_spreading + time_FFT + time_gathering + time_correction;
	auto PTPS = N/time_compute;
	std::cout << "-------\nTimings\n-------\n";
	std::cout << "Init CUDA:\t" << time_cuda_initialisation << " s\n";
	std::cout << "Readfile:\t" << time_readfile << " s\n";
	std::cout << "Linklist:\t" << time_linklist << " s\n";
    std::cout << "Precomputing:\t" << time_precompute_gauss << " s\n";
    std::cout << "Spreading:\t" << time_spreading << " s\n";
    std::cout << "FFT+flow:\t" << time_FFT << " s\n";
	std::cout << "Gathering:\t" << time_gathering << " s\n";
	std::cout << "Correction:\t" << time_correction << " s\n";
	std::cout << "Compute total:\t" << time_compute << " s\n";
	std::cout << "PTPS:\t" << PTPS << "\n";
    std::cout << std::endl;

	std::cout << "--------------\nFreeing memory\n--------------\n";

	///////////////////////////////////////////////////////////////////////////////
	// Finish
	///////////////////////////////////////////////////////////////////////////////
	cufftDestroy(plan);
	cufftDestroy(iplan);
	cudaFree(fx_device); cudaFree(fy_device); cudaFree(fz_device); 
	cudaFree(fk_x_device); cudaFree(fk_y_device); cudaFree(fk_z_device);
	cudaFree(ux_device); cudaFree(uy_device); cudaFree(uz_device); 
	cudaFree(uk_x_device); cudaFree(uk_y_device); cudaFree(uk_z_device);
	cudaFree(Y_device);
	cudaFree(F_device);
	cudaFree(T_device);
	cudaFree(V_device);
	cudaFree(W_device);
	cudaFree(GA_device);

	cudaFree(gaussx_device);
	cudaFree(gaussy_device);
	cudaFree(gaussz_device);
	cudaFree(grad_gaussx_dip_device);
	cudaFree(grad_gaussy_dip_device);
	cudaFree(grad_gaussz_dip_device);
	cudaFree(gaussgrid_device);
	cudaFree(xdis_device);
	cudaFree(ydis_device);
	cudaFree(zdis_device);
	cudaFree(indx_device);
	cudaFree(indy_device);
	cudaFree(indz_device);

	cudaFree(map_device);
	cudaFree(head_device);
	cudaFree(list_device);

	cudaFree(q_device);
	cudaFree(qpad_device);
	cudaFree(qsq_device);
	cudaFree(qpadsq_device);

	return 0;
}

