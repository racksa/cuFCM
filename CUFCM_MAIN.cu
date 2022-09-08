#include <iostream>
#include <cmath>
// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>

#include <cub/device/device_radix_sort.cuh>


#include "config.hpp"
#include "CUFCM_FCM.hpp"
#include "CUFCM_CORRECTION.hpp"
#include "CUFCM_data.hpp"

#include "util/cuda_util.hpp"
#include "util/CUFCM_linklist.hpp"
#include "util/CUFCM_print.hpp"
#include "util/CUFCM_hashing.hpp"


int main(int argc, char** argv) {
	///////////////////////////////////////////////////////////////////////////////
	// Initialise parameters
	///////////////////////////////////////////////////////////////////////////////

	// int n = 6;
	// int key_host[n] = {4, 5, 3, 6, 2, 1};
	// int* key_sorted_host = malloc_host<int>(n);
	// int* key_device = malloc_device<int>(n);
	// int* key_sorted_device = malloc_device<int>(n);
	// int value_host[n] = {40, 50, 30, 60, 20, 10};
	// int* value_sorted_host = malloc_host<int>(n);
	// int* value_device = malloc_device<int>(n);
	// int* value_sorted_device = malloc_device<int>(n);

	// for(int i = 0; i < n; i++){
	// 	printf("init (%d %d)\n", key_host[i], value_host[i]);
	// }

	// copy_to_device<int>(key_host, key_device, n);
	// copy_to_device<int>(value_host, value_device, n);

	// sort_index_by_key(key_device, value_device, n);

	// copy_to_host<int>(key_device, key_host, n);
	// copy_to_host<int>(value_device, value_host, n);

	// for(int i = 0; i < n; i++){
	// 	printf("sorted (%d %d)\n", key_host[i], value_host[i]);
	// }



	auto time_start = get_time();

	int N = 500000;

	int ngd = NGD;

	Real sigma_fac = 1.55917641;
	Real dx = (PI2)/(NX);

	/* Link list */
	Real Rref_fac = 5.21186960;
	Real Rref = Rref_fac*dx;
	int M = (int) (PI2/Rref);
	Real cellL = PI2 / (Real)M;
	Real Rrefsq = Rref*Rref;
	if(M < 3){
		M = 3;
	}
	int ncell = M*M*M;
	int mapsize = 13*ncell;

	/* Monopole */
	const Real rh = 0.02609300415934458;
	const Real sigmaFCM = rh/sqrt(PI); // Real particle size sigmaFCM
	const Real sigmaFCMsq = sigmaFCM*sigmaFCM;
	const Real anormFCM = 1.0/sqrt(2.0*PI*sigmaFCMsq);
	const Real anormFCM2 = 2.0*sigmaFCMsq;

	const Real sigmaGRID = sigmaFCM * sigma_fac;
	const Real sigmaGRIDsq = sigmaGRID * sigmaGRID;
	const Real anormGRID = 1.0/sqrt(2.0*PI*sigmaGRIDsq);
	const Real anormGRID2 = 2.0*sigmaGRIDsq;

	const Real gammaGRID = sqrt(2.0)*sigmaGRID;
	const Real pdmag = sigmaFCMsq - sigmaGRIDsq;

	/* Dipole */
	const Real sigmaFCMdip = rh/pow(6.0*sqrt(PI), 1.0/3.0);
	const Real sigmaFCMdipsq = sigmaFCMdip*sigmaFCMdip;
	const Real anormFCMdip = 1.0/sqrt(2.0*PI*sigmaFCMdipsq);
	const Real anormFCMdip2 = 2.0*sigmaFCMdipsq;

	const Real sigma_dip_fac = sigmaGRID/sigmaFCMdip;
	// sigma_dip_fac = 1;

	const Real sigmaGRIDdip = sigmaFCMdip * sigma_dip_fac;
	const Real sigmaGRIDdipsq = sigmaGRIDdip * sigmaGRIDdip;
	const Real anormGRIDdip = 1.0/sqrt(2.0*PI*sigmaGRIDdipsq);
	const Real anormGRIDdip2 = 2.0*sigmaGRIDdipsq;

	/* Self corrections */
	const Real StokesMob = 1.0/(6.0*PI*rh);
	const Real ModStokesMob = 1.0/(6.0*PI*sigmaGRID*sqrt(PI));

	Real PDStokesMob = 2.0/pow(2.0*PI, 1.5);
	PDStokesMob = PDStokesMob/pow(gammaGRID, 3.0);
	PDStokesMob = PDStokesMob*pdmag/3.0;

	Real BiLapMob = 1.0/pow(4.0*PI*sigmaGRIDsq, 1.5);
	BiLapMob = BiLapMob/(4.0*sigmaGRIDsq)*pdmag*pdmag;

	const Real WT1Mob = 1.0/(8.0*PI)/pow(rh, 3) ;
	const Real WT2Mob = 1.0/(8.0*PI)/pow(sigmaGRIDdip*pow(6.0*sqrt(PI), 1.0/3.0), 3) ;

	


	///////////////////////////////////////////////////////////////////////////////
	// CUDA initialisation
	///////////////////////////////////////////////////////////////////////////////
	cudaDeviceSynchronize();
	time_start = get_time();
	
    cufftHandle plan, iplan;

	myCufftReal* fx_host = malloc_host<myCufftReal>(GRID_SIZE);					myCufftReal* fx_device = malloc_device<myCufftReal>(GRID_SIZE);
	myCufftReal* fy_host = malloc_host<myCufftReal>(GRID_SIZE);					myCufftReal* fy_device = malloc_device<myCufftReal>(GRID_SIZE);
	myCufftReal* fz_host = malloc_host<myCufftReal>(GRID_SIZE);					myCufftReal* fz_device = malloc_device<myCufftReal>(GRID_SIZE);
    myCufftComplex* fk_x_host = malloc_host<myCufftComplex>(FFT_GRID_SIZE);		myCufftComplex* fk_x_device = malloc_device<myCufftComplex>(FFT_GRID_SIZE);
    myCufftComplex* fk_y_host = malloc_host<myCufftComplex>(FFT_GRID_SIZE);		myCufftComplex* fk_y_device = malloc_device<myCufftComplex>(FFT_GRID_SIZE);
    myCufftComplex* fk_z_host = malloc_host<myCufftComplex>(FFT_GRID_SIZE);		myCufftComplex* fk_z_device = malloc_device<myCufftComplex>(FFT_GRID_SIZE);

	myCufftReal* ux_host = malloc_host<myCufftReal>(GRID_SIZE);					myCufftReal* ux_device = malloc_device<myCufftReal>(GRID_SIZE);
	myCufftReal* uy_host = malloc_host<myCufftReal>(GRID_SIZE);					myCufftReal* uy_device = malloc_device<myCufftReal>(GRID_SIZE);
	myCufftReal* uz_host = malloc_host<myCufftReal>(GRID_SIZE);					myCufftReal* uz_device = malloc_device<myCufftReal>(GRID_SIZE);
    myCufftComplex* uk_x_host = malloc_host<myCufftComplex>(FFT_GRID_SIZE);		myCufftComplex* uk_x_device = malloc_device<myCufftComplex>(FFT_GRID_SIZE);
    myCufftComplex* uk_y_host = malloc_host<myCufftComplex>(FFT_GRID_SIZE);		myCufftComplex* uk_y_device = malloc_device<myCufftComplex>(FFT_GRID_SIZE);
    myCufftComplex* uk_z_host = malloc_host<myCufftComplex>(FFT_GRID_SIZE);		myCufftComplex* uk_z_device = malloc_device<myCufftComplex>(FFT_GRID_SIZE);

	Real *aux_host = malloc_host<Real>(3*N);						Real *aux_device = malloc_device<Real>(3*N);
	Real* Y_host = malloc_host<Real>(3*N);						Real* Y_device = malloc_device<Real>(3*N);
	Real* F_host = malloc_host<Real>(3*N);						Real* F_device = malloc_device<Real>(3*N);
	Real* T_host = malloc_host<Real>(3*N);						Real* T_device = malloc_device<Real>(3*N);
	Real* V_host = malloc_host<Real>(3*N);						Real* V_device = malloc_device<Real>(3*N);
	Real* W_host = malloc_host<Real>(3*N);						Real* W_device = malloc_device<Real>(3*N);
	Real* GA_host = malloc_host<Real>(6*N);						Real* GA_device = malloc_device<Real>(6*N);

	Real* gaussx_host = malloc_host<Real>(ngd*N);				Real* gaussx_device = malloc_device<Real>(ngd*N);
	Real* gaussy_host = malloc_host<Real>(ngd*N);				Real* gaussy_device = malloc_device<Real>(ngd*N);
	Real* gaussz_host = malloc_host<Real>(ngd*N);				Real* gaussz_device = malloc_device<Real>(ngd*N);
	Real* grad_gaussx_dip_host = malloc_host<Real>(ngd*N);		Real* grad_gaussx_dip_device = malloc_device<Real>(ngd*N);
	Real* grad_gaussy_dip_host = malloc_host<Real>(ngd*N);		Real* grad_gaussy_dip_device = malloc_device<Real>(ngd*N);
	Real* grad_gaussz_dip_host = malloc_host<Real>(ngd*N);		Real* grad_gaussz_dip_device = malloc_device<Real>(ngd*N);
	Real* gaussgrid_host = malloc_host<Real>(ngd);				Real* gaussgrid_device = malloc_device<Real>(ngd);
	Real* xdis_host = malloc_host<Real>(ngd*N);					Real* xdis_device = malloc_device<Real>(ngd*N);
	Real* ydis_host = malloc_host<Real>(ngd*N);					Real* ydis_device = malloc_device<Real>(ngd*N);
	Real* zdis_host = malloc_host<Real>(ngd*N);					Real* zdis_device = malloc_device<Real>(ngd*N);
	int* indx_host = malloc_host<int>(ngd*N);					int* indx_device = malloc_device<int>(ngd*N);
	int* indy_host = malloc_host<int>(ngd*N);					int* indy_device = malloc_device<int>(ngd*N);
	int* indz_host = malloc_host<int>(ngd*N);					int* indz_device = malloc_device<int>(ngd*N);

	int* map_host = malloc_host<int>(mapsize);					int* map_device = malloc_device<int>(mapsize);
	int* head_host = malloc_host<int>(ncell);					int* head_device = malloc_device<int>(ncell);
	int* list_host = malloc_host<int>(N);						int* list_device = malloc_device<int>(N);

	int* Y_hash_host = malloc_host<int>(N);								int* Y_hash_device = malloc_device<int>(N);	
	int* F_hash_host = malloc_host<int>(N);								int* F_hash_device = malloc_device<int>(N);
	int* T_hash_host = malloc_host<int>(N);								int* T_hash_device = malloc_device<int>(N);
	int* particle_cellindex_host = malloc_host<int>(N);					int* particle_cellindex_device = malloc_device<int>(N);
	int* particle_cellhash_host = malloc_host<int>(N);					int* particle_cellhash_device = malloc_device<int>(N);
	int* Y_index_host = malloc_host<int>(N);							int* Y_index_device = malloc_device<int>(N);	
	int* F_index_host = malloc_host<int>(N);							int* F_index_device = malloc_device<int>(N);
	int* T_index_host = malloc_host<int>(N);							int* T_index_device = malloc_device<int>(N);
	int* particle_index_host = malloc_host<int>(N);						int* particle_index_device = malloc_device<int>(N);
	int* sortback_index_host = malloc_host<int>(N);						int* sortback_index_device = malloc_device<int>(N);
	
	int* cell_start_host = malloc_host<int>(ncell);						int* cell_start_device = malloc_device<int>(ncell);
	int* cell_end_host = malloc_host<int>(ncell);						int* cell_end_device = malloc_device<int>(ncell);

	bulkmap_loop(map_host, M, HASH_ENCODE_FUNC);
	copy_to_device<int>(map_host, map_device, mapsize);

	/* Create 3D FFT plans */
	if (cufftPlan3d(&plan, NX, NY, NZ, cufftReal2Complex) != CUFFT_SUCCESS){
		printf("CUFFT error: Plan creation failed");
		return 0;	
	}

	if (cufftPlan3d(&iplan, NX, NY, NZ, cufftComplex2Real) != CUFFT_SUCCESS){
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
	Real* q_host = malloc_host<Real>(NX);			Real* q_device = malloc_device<Real>(NX);
	Real* qpad_host = malloc_host<Real>(pad);		Real* qpad_device = malloc_device<Real>(pad);
	Real* qsq_host = malloc_host<Real>(NX);			Real* qsq_device = malloc_device<Real>(NX);
	Real* qpadsq_host = malloc_host<Real>(pad);		Real* qpadsq_device = malloc_device<Real>(pad);

	for(int i=0; i<NX; i++){
		if(i < nptsh || i == nptsh){
			q_host[i] = (Real) i;
		}
		if(i > nptsh){
			q_host[i] = (Real) (i - NX);
		}
		qsq_host[i] = q_host[i]*q_host[i];
	}
	
	for(int i=0; i<pad; i++){
		qpad_host[i] = (Real) i;
		qpadsq_host[i] = qpad_host[i]*qpad_host[i];
	}
	copy_to_device<Real>(q_host, q_device, NX);
	copy_to_device<Real>(qpad_host, qpad_device, pad);
	copy_to_device<Real>(qsq_host, qsq_device, NX);
	copy_to_device<Real>(qpadsq_host, qpadsq_device, pad);

	///////////////////////////////////////////////////////////////////////////////
	// Physical system initialisation
	///////////////////////////////////////////////////////////////////////////////
	cudaDeviceSynchronize();	time_start = get_time();

	read_init_data(Y_host, N, "./init_data/pos-N500000-rh02609300-2.dat");
	read_init_data(F_host, N, "./init_data/force-N500000-rh02609300.dat");
	read_init_data(T_host, N, "./init_data/force-N500000-rh02609300-2.dat");
	
	cudaDeviceSynchronize();	auto time_readfile = get_time() - time_start;
	///////////////////////////////////////////////////////////////////////////////
	// Spatial hashing
	///////////////////////////////////////////////////////////////////////////////
	cudaDeviceSynchronize();	time_start = get_time();

	/* CPU Hashing */
	#if SPATIAL_HASHING == 0 or SPATIAL_HASHING == 1

		for(int i = 0; i < N; i++){
			particle_index_host[i] = i;
		}
		create_hash(Y_hash_host, Y_host, N, dx, M, HASH_ENCODE_FUNC);
		create_hash(F_hash_host, Y_host, N, dx, M, HASH_ENCODE_FUNC);
		create_hash(T_hash_host, Y_host, N, dx, M, HASH_ENCODE_FUNC);
		create_hash(particle_cellhash_host, Y_host, N, dx, M, HASH_ENCODE_FUNC);

	#endif
	
	/* Sorting */
	#if SPATIAL_HASHING == 1

		quicksortIterative(Y_hash_host, Y_host, 0, N - 1);
		quicksortIterative(F_hash_host, F_host, 0, N - 1);
		quicksortIterative(T_hash_host, T_host, 0, N - 1);
		quicksort_1D(particle_cellhash_host, particle_index_host, 0, N - 1);	

	#endif

	copy_to_device<Real>(Y_host, Y_device, 3*N);
	copy_to_device<Real>(F_host, F_device, 3*N);
	copy_to_device<Real>(T_host, T_device, 3*N);

	/* GPU Hashing */
	#if SPATIAL_HASHING == 2

		// Create Hash (i, j, k) -> Hash
		particle_index_range<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(particle_index_device, N);
		create_hash_gpu<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(particle_cellhash_device, Y_device, N, cellL, M, HASH_ENCODE_FUNC);

		// Sort particle index by hash
		sort_index_by_key(particle_cellhash_device, particle_index_device, N);
		
		// Sort pos/force/torque by particle index
		copy_device<Real><<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(Y_device, aux_device, 3*N);
		sort_3d_by_index<Real><<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(particle_index_device, Y_device, aux_device, N);
		copy_device<Real><<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(F_device, aux_device, 3*N);
		sort_3d_by_index<Real><<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(particle_index_device, F_device, aux_device, N);
		copy_device<Real><<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(T_device, aux_device, 3*N);
		sort_3d_by_index<Real><<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(particle_index_device, T_device, aux_device, N);

		// Find cell starting/ending points
		create_cell_list<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(particle_cellhash_device, cell_start_device, cell_end_device, N);
		
	#endif

	cudaDeviceSynchronize();	auto time_hashing = get_time() - time_start;

	///////////////////////////////////////////////////////////////////////////////
	// Link
	///////////////////////////////////////////////////////////////////////////////
	cudaDeviceSynchronize();	time_start = get_time();

	#if CORRECTION_TYPE == 0

		copy_to_host<Real>(Y_device, Y_host, 3*N);
		link_loop(list_host, head_host, Y_host, M, N, linear_encode);
		copy_to_device<int>(list_host, list_device, N);
		copy_to_device<int>(head_host, head_device, ncell);

	#endif

	cudaDeviceSynchronize();	auto time_linklist = get_time() - time_start;
	///////////////////////////////////////////////////////////////////////////////
	// Gaussian initialisation
	///////////////////////////////////////////////////////////////////////////////
	cudaDeviceSynchronize();	time_start = get_time();

	GA_setup<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(GA_device, T_device, N);

	#if PARALLELISATION_TYPE == 0

		cufcm_precompute_gauss<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(N, ngd, Y_device,
					gaussx_device, gaussy_device, gaussz_device,
					grad_gaussx_dip_device, grad_gaussy_dip_device, grad_gaussz_dip_device,
					gaussgrid_device,
					xdis_device, ydis_device, zdis_device,
					indx_device, indy_device, indz_device,
					sigmaGRIDdipsq, anormGRID, anormGRID2, dx);

	#endif
	
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

		cufcm_mono_dipole_distribution_bpp_shared<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(fx_device, fy_device, fz_device, 
											Y_device, GA_device, F_device,
											N, ngd,
											pdmag, sigmaGRIDsq, sigmaGRIDdipsq,
											anormGRID, anormGRID2,
											dx);
	
	#elif PARALLELISATION_TYPE == 3

		cufcm_mono_dipole_distribution_bpp_recompute<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(fx_device, fy_device, fz_device, 
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
	if (cufftExecReal2Complex(plan, fx_device, fk_x_device) != CUFFT_SUCCESS){
		printf("CUFFT error: ExecD2Z Forward failed (fx)\n");
		return 0;	
	}
	if (cufftExecReal2Complex(plan, fy_device, fk_y_device) != CUFFT_SUCCESS){
		printf("CUFFT error: ExecD2Z Forward failed (fy)\n");
		return 0;	
	}
	if (cufftExecReal2Complex(plan, fz_device, fk_z_device) != CUFFT_SUCCESS){
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
	if (cufftExecComplex2Real(iplan, uk_x_device, ux_device) != CUFFT_SUCCESS){
		printf("CUFFT error: ExecD2Z Backward failed (fx)\n");
		return 0;	
	}
	if (cufftExecComplex2Real(iplan, uk_y_device, uy_device) != CUFFT_SUCCESS){
		printf("CUFFT error: ExecD2Z Backward failed (fy)\n");
		return 0;	
	}
	if (cufftExecComplex2Real(iplan, uk_z_device, uz_device) != CUFFT_SUCCESS){
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

		cufcm_particle_velocities_bpp_shared<<<N, THREADS_PER_BLOCK>>>(ux_device, uy_device, uz_device,
									Y_device,
									V_device, W_device,
									N, ngd,
									pdmag, sigmaGRIDsq, sigmaGRIDdipsq,
									anormGRID, anormGRID2,
									dx);

	#elif PARALLELISATION_TYPE == 3

		cufcm_particle_velocities_bpp_recompute<<<N, THREADS_PER_BLOCK>>>(ux_device, uy_device, uz_device,
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

		cufcm_pair_correction_linklist<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(Y_device, V_device, W_device, F_device, T_device, N,
							map_device, head_device, list_device,
							ncell, Rrefsq,
							pdmag,
							sigmaGRID, sigmaGRIDsq,
							sigmaFCM, sigmaFCMsq,
							sigmaFCMdip, sigmaFCMdipsq);
	
	#elif CORRECTION_TYPE == 1

		cufcm_pair_correction_spatial_hashing<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(Y_device, V_device, W_device, F_device, T_device, N,
							particle_cellhash_device, cell_start_device, cell_end_device,
							map_device,
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

	/* Sort back */
	#if SPATIAL_HASHING == 2 and SORT_BACK == 1

		particle_index_range<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(sortback_index_device, N);
		sort_index_by_key(particle_index_device, sortback_index_device, N);

		copy_device<Real><<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(V_device, aux_device, 3*N);
		sort_3d_by_index<Real><<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(sortback_index_device, V_device, aux_device, N);

		copy_device<Real><<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(W_device, aux_device, 3*N);
		sort_3d_by_index<Real><<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(sortback_index_device, W_device, aux_device, N);

		copy_to_host<Real>(V_device, V_host, 3*N);
		copy_to_host<Real>(W_device, W_host, 3*N);

	#endif

	copy_to_host<Real>(V_device, V_host, 3*N);
	copy_to_host<Real>(W_device, W_host, 3*N);

	#if SPATIAL_HASHING == 1 and SORT_BACK == 1

		copy_to_host<Real>(V_device, V_host, 3*N);
		copy_to_host<Real>(W_device, W_host, 3*N);

		for(int i = 0; i < N; i++){
			F_hash_host[i] = particle_index_host[i];
			T_hash_host[i] = particle_index_host[i];
		}
		quicksort(F_hash_host, V_host, 0, N - 1);
		quicksort(T_hash_host, W_host, 0, N - 1);

	#endif
	


	/* Print */
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
	std::cout << "Hashing:\t" << time_hashing << " s\n";
	std::cout << "Linklist:\t" << time_linklist << " s\n";
    std::cout << "Precomputing:\t" << time_precompute_gauss << " s\n";
    std::cout << "Spreading:\t" << time_spreading << " s\n";
    std::cout << "FFT+flow:\t" << time_FFT << " s\n";
	std::cout << "Gathering:\t" << time_gathering << " s\n";
	std::cout << "Correction:\t" << time_correction << " s\n";
	std::cout << "Compute total:\t" << time_compute << " s\n";
	std::cout << "PTPS:\t" << PTPS << " /s\n";
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

