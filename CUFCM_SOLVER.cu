#include <cstdlib>
#include <iostream>
#include <fstream>
// #include <algorithm>
// #include <cmath>
// #include <cuda_runtime.h>
// #include <cufft.h>
// #include <curand_kernel.h>
// #include <curand.h>
// #include <cudaProfiler.h>
// #include <cuda_profiler_api.h>

// #include <cub/device/device_radix_sort.cuh>


#include "config.hpp"
#include "CUFCM_FCM.cuh"
#include "CUFCM_CORRECTION.cuh"
#include "CUFCM_DATA.cuh"
#include "CUFCM_SOLVER.cuh"
#include "CUFCM_RANDOMPACKER.cuh"
#include "CUFCM_CELLLIST.cuh"

#include "util/cuda_util.hpp"
#include "util/CUFCM_linklist.hpp"
#include "util/maths_util.hpp"

__host__
FCM_solver::FCM_solver(Pars pars_input){
    
    pars = pars_input;

    init_config();

    init_fcm_var();

    init_time_array();

    prompt_info();

    init_cuda();

}

__host__
void FCM_solver::init_config(){
        N = pars.N;
        rh = pars.rh;
        alpha = pars.alpha;
        beta = pars.beta;
        eta = pars.eta;
        nx = pars.nx;
        ny = pars.ny;
        nz = pars.nz;
        repeat = pars.repeat;
        prompt = pars.prompt;
        boxsize = pars.boxsize;


    /* Deduced FCM parameters */
        grid_size = nx*ny*nz;
        fft_grid_size = (nx/2+1)*ny*nz;
        dx = boxsize/nx;
        ngd = round(alpha*beta);
        Rc_fac = Real(eta*alpha);

        /* Neighbour list */
        Rc = Rc_fac*dx;
        Rcsq = Rc*Rc;
        M = (int) (boxsize/Rc);
        if(M < 3){
            M = 3;
        }
        cellL = boxsize / (Real)M;
        ncell = M*M*M;
        mapsize = 13*ncell;

        Volume_frac = (N*4.0/3.0*PI*rh*rh*rh) / (boxsize*boxsize*boxsize);

        /* Repeat number */
        warmup = 0.2*repeat;
}


__host__
void FCM_solver::init_fcm_var(){
    #if SOLVER_MODE == 1

        /* Monopole */
        sigmaFCM = rh/sqrt(PI); // Real particle size sigmaFCM
        sigmaFCMsq = sigmaFCM*sigmaFCM;
        anormFCM = 1.0/sqrt(2.0*PI*sigmaFCMsq);
        anormFCM2 = 2.0*sigmaFCMsq;

        sigmaGRID = dx * alpha;
        sigmaGRIDsq = sigmaGRID * sigmaGRID;
        anormGRID = 1.0/sqrt(2.0*PI*sigmaGRIDsq);
        anormGRID2 = 2.0*sigmaGRIDsq;

        sigma_fac = sigmaGRID/sigmaFCM;

        gammaGRID = sqrt(2.0)*sigmaGRID;
        pdmag = sigmaFCMsq - sigmaGRIDsq;
        /* Dipole */
        sigmaFCMdip = rh/pow(6.0*sqrt(PI), 1.0/3.0);
        sigmaFCMdipsq = sigmaFCMdip*sigmaFCMdip;
        anormFCMdip = 1.0/sqrt(2.0*PI*sigmaFCMdipsq);
        anormFCMdip2 = 2.0*sigmaFCMdipsq;

        sigma_dip_fac = sigmaGRID/sigmaFCMdip;

        sigmaGRIDdip = sigmaFCMdip * sigma_dip_fac;
        sigmaGRIDdipsq = sigmaGRIDdip * sigmaGRIDdip;
        anormGRIDdip = 1.0/sqrt(2.0*PI*sigmaGRIDdipsq);
        anormGRIDdip2 = 2.0*sigmaGRIDdipsq;

        /* Self corrections */
        StokesMob = 1.0/(6.0*PI*rh);
        ModStokesMob = 1.0/(6.0*PI*sigmaGRID*sqrt(PI));

        PDStokesMob = 2.0/pow(2.0*PI, 1.5);
        PDStokesMob = PDStokesMob/pow(gammaGRID, 3.0);
        PDStokesMob = PDStokesMob*pdmag/3.0;

        BiLapMob = 1.0/pow(4.0*PI*sigmaGRIDsq, 1.5);
        BiLapMob = BiLapMob/(4.0*sigmaGRIDsq)*pdmag*pdmag;

        WT1Mob = 1.0/(8.0*PI)/pow(rh, 3) ;
        WT2Mob = 1.0/(8.0*PI)/pow(sigmaGRIDdip*pow(6.0*sqrt(PI), 1.0/3.0), 3) ;

    #elif SOLVER_MODE == 0

        /* Monopole */
        sigmaFCM = rh/sqrt(PI); // Real particle size sigmaFCM
        sigmaFCMsq = sigmaFCM*sigmaFCM;
        anormFCM = 1.0/sqrt(2.0*PI*sigmaFCMsq);
        anormFCM2 = 2.0*sigmaFCMsq;

        /* Dipole */
        sigmaFCMdip = rh/pow(6.0*sqrt(PI), 1.0/3.0);
        sigmaFCMdipsq = sigmaFCMdip*sigmaFCMdip;
        anormFCMdip = 1.0/sqrt(2.0*PI*sigmaFCMdipsq);
        anormFCMdip2 = 2.0*sigmaFCMdipsq;

        StokesMob = 1.0/(6.0*PI*rh);
        WT1Mob = 1.0/(8.0*PI)/pow(rh, 3) ;

    #endif
}

__host__
void FCM_solver::init_time_array(){
    /* Timing variables */
    time_start = (Real)0.0;
    time_cuda_initialisation = (Real)0.0;
    time_readfile = (Real)0.0;

    time_hashing_array = malloc_host<Real>(repeat);
    time_linklist_array = malloc_host<Real>(repeat);
    time_precompute_array = malloc_host<Real>(repeat);
    time_spreading_array = malloc_host<Real>(repeat);
    time_FFT_array = malloc_host<Real>(repeat);
    time_gathering_array = malloc_host<Real>(repeat);
    time_correction_array = malloc_host<Real>(repeat);
}

__host__
void FCM_solver::prompt_info() {
    if(prompt > -1){
		std::cout << "-------\nSimulation\n-------\n";
		std::cout << "Particle number:\t" << N << "\n";
		std::cout << "Particle radius:\t" << rh << "\n";
		#if SOLVER_MODE == 1
			std::cout << "Solver:\t\t\t" << "<Fast FCM>" << "\n";
		#elif SOLVER_MODE == 0
			std::cout << "Solver:\t\t\t" << "<Regular FCM>" << "\n";
		#endif
		std::cout << "Grid points:\t\t" << nx << "\n";
		std::cout << "Grid support:\t\t" << ngd << "\n";
		#if SOLVER_MODE == 1
			std::cout << "Sigma/sigma:\t\t" << sigma_fac << "\n";
			std::cout << "Alpha:\t\t\t" << alpha << "\n";
			std::cout << "Beta:\t\t\t" << beta << "\n";
			std::cout << "Eta:\t\t\t" << eta << "\n";
		#endif
		#if SOLVER_MODE == 1
			std::cout << "Sigma:\t\t\t" << sigmaGRID << "\n";
		#elif SOLVER_MODE == 0
			std::cout << "sigma:\t\t\t" << sigmaFCM << "\n";
		#endif
		std::cout << "dx:\t\t\t" << dx<< "\n";
		std::cout << "Cell number:\t\t" << M << "\n";
		#if ENABLE_REPEAT == 1
			std::cout << "Repeat number:\t\t" << repeat << "\n";
		#endif
		std::cout << "Volume fraction:\t" << Volume_frac << "\n";
		
		std::cout << std::endl;
	}
}

__host__
void FCM_solver::init_cuda(){
    ///////////////////////////////////////////////////////////////////////////////
	// CUDA initialisation
	///////////////////////////////////////////////////////////////////////////////
	cudaDeviceSynchronize();
	time_start = get_time();

	aux_host = malloc_host<Real>(3*N);					    aux_device = malloc_device<Real>(3*N);

    V_host = malloc_host<Real>(3*N);						V_device = malloc_device<Real>(3*N);
    W_host = malloc_host<Real>(3*N);						W_device = malloc_device<Real>(3*N);

	hx_host = malloc_host<myCufftReal>(grid_size);
	hy_host = malloc_host<myCufftReal>(grid_size);
	hz_host = malloc_host<myCufftReal>(grid_size);
	hx_device = malloc_device<myCufftReal>(grid_size);
	hy_device = malloc_device<myCufftReal>(grid_size);
	hz_device = malloc_device<myCufftReal>(grid_size);

	fk_x_host = malloc_host<myCufftComplex>(fft_grid_size);		fk_x_device = malloc_device<myCufftComplex>(fft_grid_size);
	fk_y_host = malloc_host<myCufftComplex>(fft_grid_size);		fk_y_device = malloc_device<myCufftComplex>(fft_grid_size);
	fk_z_host = malloc_host<myCufftComplex>(fft_grid_size);		fk_z_device = malloc_device<myCufftComplex>(fft_grid_size);
	uk_x_host = malloc_host<myCufftComplex>(fft_grid_size);		uk_x_device = malloc_device<myCufftComplex>(fft_grid_size);
	uk_y_host = malloc_host<myCufftComplex>(fft_grid_size);		uk_y_device = malloc_device<myCufftComplex>(fft_grid_size);
	uk_z_host = malloc_host<myCufftComplex>(fft_grid_size);		uk_z_device = malloc_device<myCufftComplex>(fft_grid_size);

    particle_cellhash_host = malloc_host<int>(N);					 particle_cellhash_device = malloc_device<int>(N);
    particle_index_host = malloc_host<int>(N);						 particle_index_device = malloc_device<int>(N);
    sortback_index_host = malloc_host<int>(N);						 sortback_index_device = malloc_device<int>(N);

    cell_start_host = malloc_host<int>(ncell);						 cell_start_device = malloc_device<int>(ncell);
    cell_end_host = malloc_host<int>(ncell);						 cell_end_device = malloc_device<int>(ncell);

	#if CORRECTION_TYPE == 0

		 head_host = malloc_host<int>(ncell);					 head_device = malloc_device<int>(ncell);
		 list_host = malloc_host<int>(N);						 list_device = malloc_device<int>(N);

	#endif

	map_host = malloc_host<int>(mapsize);							map_device = malloc_device<int>(mapsize);

	bulkmap_loop(map_host, M, linear_encode);
	copy_to_device<int>(map_host, map_device, mapsize);

	/* Create 3D FFT plans */
	if (cufftPlan3d(&plan, nx, ny, nz, cufftReal2Complex) != CUFFT_SUCCESS){
		printf("CUFFT error: Plan creation failed");
		return ;	
	}

	if (cufftPlan3d(&iplan, nx, ny, nz, cufftComplex2Real) != CUFFT_SUCCESS){
		printf("CUFFT error: Plan creation failed");
		return ;	
	}

	num_thread_blocks_GRID = (grid_size + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
	num_thread_blocks_N = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
	num_thread_blocks_NX = (nx + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
	
	cudaMalloc((void**)&dev_random, num_thread_blocks_N*THREADS_PER_BLOCK*sizeof(curandState));

	time_cuda_initialisation += get_time() - time_start;
}

__host__ 
void FCM_solver::hydrodynamic_solver(Real *Y_host_input, Real *F_host_input, Real *T_host_input,
                                     Real *Y_device_input, Real * F_device_input, Real *T_device_input){

    Y_host = Y_host_input;
    F_host = F_host_input;
    T_host = T_host_input;

    Y_device = Y_device_input;
    F_device = F_device_input;
    T_device = T_device_input;

    reset_grid();

    spatial_hashing();

    spread();

    fft_solve();

    gather();

    correction();

    sortback();

    rept += 1;

}

__host__ 
void FCM_solver::reset_grid(){
    reset_device(V_device, 3*N);
    reset_device(W_device, 3*N);
    reset_device(hx_device, grid_size);
    reset_device(hy_device, grid_size);
    reset_device(hz_device, grid_size);
}

__host__
void FCM_solver::spatial_hashing(){
    ///////////////////////////////////////////////////////////////////////////////
    // Spatial hashing
    ///////////////////////////////////////////////////////////////////////////////
    cudaDeviceSynchronize();	time_start = get_time();

    // Create Hash (i, j, k) -> Hash
    create_hash_gpu<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(particle_cellhash_device, Y_device, N, cellL, M, linear_encode);

    // Sort particle index by hash
    particle_index_range<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(particle_index_device, N);
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

    cudaDeviceSynchronize();	time_hashing_array[rept] = get_time() - time_start;
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

    cudaDeviceSynchronize();	time_linklist_array[rept] = get_time() - time_start;
}


__host__
void FCM_solver::sort_particle(){
    
}



__host__
void FCM_solver::spread(){
    ///////////////////////////////////////////////////////////////////////////////
    // Spreading
    ///////////////////////////////////////////////////////////////////////////////
    cudaDeviceSynchronize();	time_start = get_time();

    #if SOLVER_MODE == 1

        #if SPREAD_TYPE == 1

            cufcm_mono_dipole_distribution_tpp_recompute<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(hx_device, hy_device, hz_device,
                                                Y_device, T_device, F_device,
                                                N, ngd,
                                                pdmag, sigmaGRIDsq, sigmaGRIDdipsq,
                                                anormGRID, anormGRID2,
                                                dx, nx, ny, nz);

        #elif SPREAD_TYPE == 2

            cufcm_mono_dipole_distribution_bpp_shared<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(hx_device, hy_device, hz_device, 
                                                Y_device, T_device, F_device,
                                                N, ngd,
                                                pdmag, sigmaGRIDsq, sigmaGRIDdipsq,
                                                anormGRID, anormGRID2,
                                                dx, nx, ny, nz);
            
        #elif SPREAD_TYPE == 3

            cufcm_mono_dipole_distribution_bpp_recompute<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(hx_device, hy_device, hz_device, 
                                                Y_device, T_device, F_device,
                                                N, ngd,
                                                pdmag, sigmaGRIDsq, sigmaGRIDdipsq,
                                                anormGRID, anormGRID2,
                                                dx, nx, ny, nz);
                                            
        #elif SPREAD_TYPE == 4

            cufcm_mono_dipole_distribution_bpp_shared_dynamic<<<num_thread_blocks_N, THREADS_PER_BLOCK, 3*ngd*sizeof(int)+(9*ngd+15)*sizeof(Real)>>>
                                                    (hx_device, hy_device, hz_device, 
                                                    Y_device, T_device, F_device,
                                                    N, ngd,
                                                    pdmag, sigmaGRIDsq, sigmaGRIDdipsq,
                                                    anormGRID, anormGRID2,
                                                    dx, nx, ny, nz);

        #endif
    
    #elif SOLVER_MODE == 0
        
        cufcm_mono_dipole_distribution_regular_fcm<<<num_thread_blocks_N, THREADS_PER_BLOCK, 3*ngd*sizeof(int)+(9*ngd+15)*sizeof(Real)>>>
                                                (hx_device, hy_device, hz_device, 
                                                Y_device, T_device, F_device,
                                                N, ngd,
                                                sigmaFCMsq, sigmaFCMdipsq,
                                                anormFCM, anormFCM2,
                                                anormFCMdip, anormFCMdip2,
                                                dx, nx, ny, nz);

    #endif
    cudaDeviceSynchronize();	time_spreading_array[rept] = get_time() - time_start;
}

__host__
void FCM_solver::fft_solve(){
    ///////////////////////////////////////////////////////////////////////////////
    // FFT
    ///////////////////////////////////////////////////////////////////////////////
    cudaDeviceSynchronize();	time_start = get_time();
    if (cufftExecReal2Complex(plan, hx_device, fk_x_device) != CUFFT_SUCCESS){
        printf("CUFFT error: ExecD2Z Forward failed (fx)\n");
        return ;	
    }
    if (cufftExecReal2Complex(plan, hy_device, fk_y_device) != CUFFT_SUCCESS){
        printf("CUFFT error: ExecD2Z Forward failed (fy)\n");
        return ;	
    }
    if (cufftExecReal2Complex(plan, hz_device, fk_z_device) != CUFFT_SUCCESS){
        printf("CUFFT error: ExecD2Z Forward failed (fz)\n");
        return ;	
    }
    ///////////////////////////////////////////////////////////////////////////////
    // Solve for the flow
    ///////////////////////////////////////////////////////////////////////////////
    cufcm_flow_solve<<<num_thread_blocks_GRID, THREADS_PER_BLOCK>>>(fk_x_device, fk_y_device, fk_z_device,
                                                            uk_x_device, uk_y_device, uk_z_device,
                                                            nx, ny, nz, boxsize);
    ///////////////////////////////////////////////////////////////////////////////
    // IFFT
    ///////////////////////////////////////////////////////////////////////////////
    if (cufftExecComplex2Real(iplan, uk_x_device, hx_device) != CUFFT_SUCCESS){
        printf("CUFFT error: ExecD2Z Backward failed (fx)\n");
        return ;	
    }
    if (cufftExecComplex2Real(iplan, uk_y_device, hy_device) != CUFFT_SUCCESS){
        printf("CUFFT error: ExecD2Z Backward failed (fy)\n");
        return ;	
    }
    if (cufftExecComplex2Real(iplan, uk_z_device, hz_device) != CUFFT_SUCCESS){
        printf("CUFFT error: ExecZ2D Backward failed (fz)\n");
        return ;	
    }		

    cudaDeviceSynchronize();	time_FFT_array[rept] = get_time() - time_start;
}

__host__
void FCM_solver::gather(){
    ///////////////////////////////////////////////////////////////////////////////
    // Gathering
    ///////////////////////////////////////////////////////////////////////////////
    cudaDeviceSynchronize();	time_start = get_time();

    #if SOLVER_MODE == 1

        #if GATHER_TYPE == 1

            cufcm_particle_velocities_tpp_recompute<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(hx_device, hy_device, hz_device,
                                        Y_device,
                                        V_device, W_device,
                                        N, ngd,
                                        pdmag, sigmaGRIDsq, sigmaGRIDdipsq,
                                        anormGRID, anormGRID2,
                                        dx, nx, ny, nz);

        #elif GATHER_TYPE == 2

            cufcm_particle_velocities_bpp_shared<<<N, THREADS_PER_BLOCK>>>(hx_device, hy_device, hz_device,
                                        Y_device,
                                        V_device, W_device,
                                        N, ngd,
                                        pdmag, sigmaGRIDsq, sigmaGRIDdipsq,
                                        anormGRID, anormGRID2,
                                        dx, nx, ny, nz);

        #elif GATHER_TYPE == 3

            cufcm_particle_velocities_bpp_recompute<<<N, THREADS_PER_BLOCK>>>(hx_device, hy_device, hz_device,
                                        Y_device,
                                        V_device, W_device,
                                        N, ngd,
                                        pdmag, sigmaGRIDsq, sigmaGRIDdipsq,
                                        anormGRID, anormGRID2,
                                        dx, nx, ny, nz);

        #elif GATHER_TYPE == 4

            cufcm_particle_velocities_bpp_shared_dynamic<<<N, THREADS_PER_BLOCK, 3*ngd*sizeof(int)+(9*ngd+3)*sizeof(Real)>>>
                                        (hx_device, hy_device, hz_device,
                                        Y_device,
                                        V_device, W_device,
                                        N, ngd,
                                        pdmag, sigmaGRIDsq, sigmaGRIDdipsq,
                                        anormGRID, anormGRID2,
                                        dx, nx, ny, nz);

        #endif

    #elif SOLVER_MODE == 0

        cufcm_particle_velocities_regular_fcm<<<N, THREADS_PER_BLOCK, 3*ngd*sizeof(int)+(9*ngd+3)*sizeof(Real)>>>
                                        (hx_device, hy_device, hz_device,
                                        Y_device,
                                        V_device, W_device,
                                        N, ngd,
                                        sigmaFCMsq, sigmaFCMdipsq,
                                        anormFCM, anormFCM2,
                                        anormFCMdip, anormFCMdip2,
                                        dx, nx, ny, nz);

    #endif

    cudaDeviceSynchronize();	time_gathering_array[rept] = get_time() - time_start;
}

__host__
void FCM_solver::correction(){
    ///////////////////////////////////////////////////////////////////////////////
    // Correction
    ///////////////////////////////////////////////////////////////////////////////
    cudaDeviceSynchronize();	time_start = get_time();

    #if SOLVER_MODE == 1

        #if CORRECTION_TYPE == 0

            cufcm_pair_correction_linklist<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(Y_device, V_device, W_device, F_device, T_device, N,
                                map_device, head_device, list_device,
                                ncell, Rcsq,
                                pdmag,
                                sigmaGRID, sigmaGRIDsq,
                                sigmaFCM, sigmaFCMsq,
                                sigmaFCMdip, sigmaFCMdipsq);
        
        #elif CORRECTION_TYPE == 1

            cufcm_pair_correction_spatial_hashing_tpp<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(Y_device, V_device, W_device, F_device, T_device, N, boxsize,
                                particle_cellhash_device, cell_start_device, cell_end_device,
                                map_device,
                                ncell, Rcsq,
                                pdmag,
                                sigmaGRID, sigmaGRIDsq,
                                sigmaFCM, sigmaFCMsq,
                                sigmaFCMdip, sigmaFCMdipsq);

        #endif

        cufcm_self_correction<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(V_device, W_device, F_device, T_device, N, boxsize,
                                StokesMob, ModStokesMob,
                                PDStokesMob, BiLapMob,
                                WT1Mob, WT2Mob);

    #endif

    cudaDeviceSynchronize();	time_correction_array[rept] = get_time() - time_start;
}

__host__
void FCM_solver::sortback(){
     /* Sort back */
    #if SORT_BACK == 1
        
        particle_index_range<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(sortback_index_device, N);
        sort_index_by_key(particle_index_device, sortback_index_device, N);

        copy_device<Real><<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(V_device, aux_device, 3*N);
        sort_3d_by_index<Real><<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(sortback_index_device, V_device, aux_device, N);

        copy_device<Real><<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(W_device, aux_device, 3*N);
        sort_3d_by_index<Real><<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(sortback_index_device, W_device, aux_device, N);

        copy_device<Real><<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(Y_device, aux_device, 3*N);
        sort_3d_by_index<Real><<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(sortback_index_device, Y_device, aux_device, N);

        copy_device<Real><<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(F_device, aux_device, 3*N);
        sort_3d_by_index<Real><<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(sortback_index_device, F_device, aux_device, N);

        copy_device<Real><<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(T_device, aux_device, 3*N);
        sort_3d_by_index<Real><<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(sortback_index_device, T_device, aux_device, N);

    #endif
}

__host__
void FCM_solver::finish(){
   
	copy_to_host<Real>(Y_device, Y_host, 3*N);
	copy_to_host<Real>(F_device, F_host, 3*N);
	copy_to_host<Real>(T_device, T_host, 3*N);
	copy_to_host<Real>(V_device, V_host, 3*N);
	copy_to_host<Real>(W_device, W_host, 3*N);

	///////////////////////////////////////////////////////////////////////////////
	// Time
	///////////////////////////////////////////////////////////////////////////////
	auto time_hashing = mean(&time_hashing_array[warmup], repeat-warmup);
	auto time_linklist = mean(&time_linklist_array[warmup], repeat-warmup);
	auto time_precompute = mean(&time_precompute_array[warmup], repeat-warmup);
	auto time_spreading = mean(&time_spreading_array[warmup], repeat-warmup);
	auto time_FFT = mean(&time_FFT_array[warmup], repeat-warmup);
	auto time_gathering = mean(&time_gathering_array[warmup], repeat-warmup);
	auto time_correction = mean(&time_correction_array[warmup], repeat-warmup);

	auto time_hashing_stdv = stdv(&time_hashing_array[warmup], repeat-warmup);
	auto time_linklist_stdv = stdv(&time_linklist_array[warmup], repeat-warmup);
	auto time_precompute_stdv = stdv(&time_precompute_array[warmup], repeat-warmup);
	auto time_spreading_stdv = stdv(&time_spreading_array[warmup], repeat-warmup);
	auto time_FFT_stdv = stdv(&time_FFT_array[warmup], repeat-warmup);
	auto time_gathering_stdv = stdv(&time_gathering_array[warmup], repeat-warmup);
	auto time_correction_stdv = stdv(&time_correction_array[warmup], repeat-warmup);

	auto time_compute = time_linklist + time_precompute + time_spreading + time_FFT + time_gathering + time_correction;
	auto PTPS = N/time_compute;

	if(prompt > 1){
		std::cout.precision(5);
		std::cout << std::endl;
		std::cout << "-------\nTimings\n-------\n";
		std::cout << "Init CUDA:\t" << time_cuda_initialisation << "s\n";
		std::cout << "Readfile:\t" << time_readfile << " s\n";
		std::cout << "Hashing:\t" << time_hashing << " \t+/-\t " << time_hashing_stdv << " s\n";
		std::cout << "Linklist:\t" << time_linklist << " \t+/-\t " << time_linklist_stdv <<" s\n";
		std::cout << "Precomputing:\t" << time_precompute << " \t+/-\t " << time_precompute_stdv <<" s\n";
		std::cout << "Spreading:\t" << time_spreading << " \t+/-\t " << time_spreading_stdv <<" s\n";
		std::cout << "FFT+flow:\t" << time_FFT << " \t+/-\t " << time_FFT_stdv <<" s\n";
		std::cout << "Gathering:\t" << time_gathering << " \t+/-\t " << time_gathering_stdv <<" s\n";
		std::cout << "Correction:\t" << time_correction << " \t+/-\t " << time_correction_stdv <<" s\n";
		std::cout << "Compute total:\t" << time_compute <<" s\n";
		std::cout << "PTPS:\t" << PTPS << " /s\n";
		std::cout << std::endl;
	}
	///////////////////////////////////////////////////////////////////////////////
	// Check error
	///////////////////////////////////////////////////////////////////////////////
	#if CHECK_ERROR == 1 and INIT_FROM_FILE == 1

		Real* Y_validation = malloc_host<Real>(3*N);
		Real* F_validation = malloc_host<Real>(3*N);
		Real* V_validation = malloc_host<Real>(3*N);
		Real* W_validation = malloc_host<Real>(3*N);

		read_validate_data(Y_validation,
						   F_validation,
						   V_validation,
						   W_validation, N, "./data/refdata/ref_data_N500000");

		Real Yerror = percentage_error_magnitude(Y_host, Y_validation, N);
		Real Verror = percentage_error_magnitude(V_host, V_validation, N);
		Real Werror = percentage_error_magnitude(W_host, W_validation, N);

		if(prompt > 1){
			std::cout << "-------\nError\n-------\n";
			std::cout << "%Y error:\t" << Yerror << "\n";
			std::cout << "%V error:\t" << Verror << "\n";
			std::cout << "%W error:\t" << Werror << "\n";
		}

	#elif CHECK_ERROR == 2
		int N_truncate;
		if(N>1000){
			N_truncate = int(N*0.001);
		}
		else{
			N_truncate = int(N);
		}
		
		Real* V_validation = malloc_host<Real>(3*N);
		Real* W_validation = malloc_host<Real>(3*N);
		Real* V_validation_device = malloc_device<Real>(3*N_truncate);
		Real* W_validation_device = malloc_device<Real>(3*N_truncate);

		Real hasimoto = Real(1.0) - Real(1.7601)*pow(Volume_frac, 1.0/3.0) - Real(1.5593)*pow(Volume_frac, 2.0);

		const int num_thread_blocks_N_trunc = (N_truncate + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
		cufcm_compute_formula<<<num_thread_blocks_N_trunc, THREADS_PER_BLOCK>>>
							(Y_device, V_validation_device, W_validation_device,
							F_device, T_device, N, N_truncate,
							sigmaFCM, sigmaFCMdip, StokesMob, WT1Mob, hasimoto);

		copy_to_host<Real>(V_validation_device, V_validation, 3*N_truncate);
		copy_to_host<Real>(W_validation_device, W_validation, 3*N_truncate);
		Real Verror = percentage_error_magnitude(V_host, V_validation, N_truncate);
		Real Werror = percentage_error_magnitude(W_host, W_validation, N_truncate);
		
		if(prompt > 1){
			std::cout << "-------\nError\n-------\n";
			std::cout << "%Y error:\t" << 0 << "\n";
			std::cout << "%V error:\t" << Verror << "\n";
			std::cout << "%W error:\t" << Werror << "\n";
		}
		
	#endif

	///////////////////////////////////////////////////////////////////////////////
	// Write to file
	///////////////////////////////////////////////////////////////////////////////
	#if OUTPUT_TO_FILE == 1
		write_data(Y_host, F_host, V_host, W_host, N, "./data/simulation/simulation_data.dat");
		
		write_time(time_cuda_initialisation, 
				time_readfile,
				time_hashing, 
				time_linklist,
				time_precompute,
				time_spreading,
				time_FFT,
				time_gathering,
				time_correction,
				time_compute,
				"./data/simulation/simulation_scalar.dat");

		#if CHECK_ERROR > 0 and INIT_FROM_FILE == 1

			write_error(
				Verror,
				Werror,
				"./data/simulation/simulation_scalar.dat");

		#else

			write_error(
				-1,
				-1,
				"./data/simulation/simulation_scalar.dat");

		#endif
		
	#endif
}