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
        checkerror = pars.checkerror;


        /* Deduced FCM parameters */
        grid_size = nx*ny*nz;
        fft_grid_size = (nx/2+1)*ny*nz;
        dx = boxsize/nx;
        ngd = round(alpha*beta);
        

        /* Repeat number */
        warmup = 0.2*repeat;
}

__host__
void FCM_solver::init_fcm_var(){
    #if SOLVER_MODE == 1

        /* Monopole */
        sigmaFCM = rh/sqrt(PI); // Real particle size sigmaFCM
        SigmaGRID = dx * alpha;

        /* Dipole */
        sigmaFCMdip = rh/pow(6.0*sqrt(PI), 1.0/3.0);
        sigmaGRIDdip = SigmaGRID;

        /* Self corrections */
        gammaGRID = sqrt(2.0)*SigmaGRID;
        pdmag = sigmaFCM*sigmaFCM - SigmaGRID*SigmaGRID;

        StokesMob = 1.0/(6.0*PI*rh);
        ModStokesMob = 1.0/(6.0*PI*SigmaGRID*sqrt(PI));

        PDStokesMob = 2.0/pow(2.0*PI, 1.5);
        PDStokesMob = PDStokesMob/pow(gammaGRID, 3.0);
        PDStokesMob = PDStokesMob*pdmag/3.0;

        BiLapMob = 1.0/pow(4.0*PI*SigmaGRID*SigmaGRID, 1.5);
        BiLapMob = BiLapMob/(4.0*SigmaGRID*SigmaGRID)*pdmag*pdmag;

        WT1Mob = 1.0/(8.0*PI)/pow(rh, 3) ;
        WT2Mob = 1.0/(8.0*PI)/pow(sigmaGRIDdip*pow(6.0*sqrt(PI), 1.0/3.0), 3) ;

        // Rc = Real(-eta*pdmag);

    #elif SOLVER_MODE == 0

        sigmaFCM = rh/sqrt(PI);
        sigmaFCMdip = rh/pow(6.0*sqrt(PI), 1.0/3.0);

    #endif

    /* Neighbour list */
    Rc_fac = Real(eta*alpha);
    Rc = Rc_fac*dx;
    
    Rcsq = Rc*Rc;
    M = (int) (boxsize/Rc);
    if(M < 3){
        M = 3;
    }
    cellL = boxsize / Real(M);
    ncell = M*M*M;
    mapsize = 13*ncell;

    Volume_frac = (N*4.0/3.0*PI*rh*rh*rh) / (boxsize*boxsize*boxsize);
}

__host__
void FCM_solver::init_time_array(){
    /* Timing variables */
    time_start = (Real)0.0;
    time_cuda_initialisation = (Real)0.0;
    time_readfile = (Real)0.0;

    time_hashing_array = malloc_host<Real>(repeat);
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
			std::cout << "Sigma/sigma:\t\t" << SigmaGRID/sigmaFCM << "\n";
			std::cout << "Alpha:\t\t\t" << alpha << "\n";
			std::cout << "Beta:\t\t\t" << beta << "\n";
			std::cout << "Eta:\t\t\t" << eta << "\n";
		#endif
		#if SOLVER_MODE == 1
			std::cout << "Sigma:\t\t\t" << SigmaGRID << "\n";
		#elif SOLVER_MODE == 0
			std::cout << "sigma:\t\t\t" << sigmaFCM << "\n";
		#endif
		std::cout << "dx:\t\t\t" << dx<< "\n";
        std::cout << "boxsize:\t\t" << boxsize<< "\n";
		std::cout << "Cell number:\t\t" << M << "\n";
        std::cout << "Rc/a:\t\t\t" << Rc/rh << "\n";
		std::cout << "Repeat number:\t\t" << repeat << "\n";
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

    cudaMalloc((void **)&nan_check_device, sizeof (bool));

	aux_host = malloc_host<Real>(3*N);					    aux_device = malloc_device<Real>(3*N);
    

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
    
    key_buf = malloc_device<int>(N);
    index_buf = malloc_device<int>(N);

    cell_start_host = malloc_host<int>(ncell);						 cell_start_device = malloc_device<int>(ncell);
    cell_end_host = malloc_host<int>(ncell);						 cell_end_device = malloc_device<int>(ncell);

	#if CORRECTION_TYPE == 0

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

    num_thread_blocks_FFTGRID = (fft_grid_size + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
	num_thread_blocks_GRID = (grid_size + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
	num_thread_blocks_N = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
	num_thread_blocks_NX = (nx + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
	
	cudaMalloc((void**)&dev_random, num_thread_blocks_N*THREADS_PER_BLOCK*sizeof(curandState));

	time_cuda_initialisation += get_time() - time_start;
}

__host__
void FCM_solver::init_aux_for_filament(){

    Y_host = malloc_host<Real>(3*pars.N);						Y_device = malloc_device<Real>(3*pars.N);
	F_host = malloc_host<Real>(3*pars.N);						F_device = malloc_device<Real>(3*pars.N);
	T_host = malloc_host<Real>(3*pars.N);						T_device = malloc_device<Real>(3*pars.N);
	V_host = malloc_host<Real>(3*pars.N);						V_device = malloc_device<Real>(3*pars.N);
	W_host = malloc_host<Real>(3*pars.N);						W_device = malloc_device<Real>(3*pars.N);

}

__host__
void FCM_solver::reform_data(Real *x_seg, Real *f_seg, Real *v_seg,
                             Real *x_blob, Real *f_blob, Real *v_blob,
                             int num_seg, int num_blob){
    
    int num_thread_blocks_Nseg = (num_seg + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
    int num_thread_blocks_Nblob = (num_blob + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;

    copy_device<Real> <<<num_thread_blocks_Nseg, THREADS_PER_BLOCK>>>(x_seg, Y_device, 3*num_seg);
    interleaved2separate<<<num_thread_blocks_Nseg, THREADS_PER_BLOCK>>>(f_seg, F_device, T_device, num_seg);

    copy_device<Real> <<<num_thread_blocks_Nblob, THREADS_PER_BLOCK>>>(x_blob, &Y_device[3*num_seg], 3*num_blob);
    copy_device<Real> <<<num_thread_blocks_Nblob, THREADS_PER_BLOCK>>>(f_blob, &F_device[3*num_seg], 3*num_blob);

    return;
}

__host__
void FCM_solver::Mss(){

    box_particle();
    if(prompt>10){printf("pass1\n");}
    
    reset_grid();
    if(prompt>10){printf("pass2\n");}

    spatial_hashing();
    
    if(prompt>10){printf("pass3\n");}

    // check_overlap();

    spread();
    if(prompt>10){printf("pass4\n");}

    fft_solve();
    if(prompt>10){printf("pass5\n");}

    gather();
    if(prompt>10){printf("pass6\n");}

    correction();
    if(prompt>10){printf("pass7\n");}

    sortback();
    if(prompt>10){printf("pass8\n");}

    if(prompt>9){prompt_time();}

}

__host__
void FCM_solver::apply_repulsion_for_timcode(){
    box_particle();

    // spatial_hashing();

    apply_repulsion();

    // sortback();
}

__host__
void FCM_solver::reform_data_back(Real *x_seg, Real *f_seg, Real *v_seg,
                             Real *x_blob, Real *f_blob, Real *v_blob,
                             int num_seg, int num_blob){

    int num_thread_blocks_Nseg = (num_seg + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
    int num_thread_blocks_Nblob = (num_blob + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;

    // copy_device<Real> <<<num_thread_blocks_Nseg, THREADS_PER_BLOCK>>>(Y_device, x_seg, 3*num_seg);
    separate2interleaved<<<num_thread_blocks_Nseg, THREADS_PER_BLOCK>>>(f_seg, F_device, T_device, num_seg);
    separate2interleaved<<<num_thread_blocks_Nseg, THREADS_PER_BLOCK>>>(v_seg, V_device, W_device, num_seg);

    // copy_device<Real> <<<num_thread_blocks_Nblob, THREADS_PER_BLOCK>>>(&Y_device[3*num_seg], x_blob, 3*num_blob);
    // copy_device<Real> <<<num_thread_blocks_Nblob, THREADS_PER_BLOCK>>>(&F_device[3*num_seg], f_blob, 3*num_blob);
    copy_device<Real> <<<num_thread_blocks_Nblob, THREADS_PER_BLOCK>>>(&V_device[3*num_seg], v_blob, 3*num_blob);

    return;
}

__host__ 
void FCM_solver::hydrodynamic_solver(Real *Y_device_input, Real *F_device_input, Real *T_device_input, Real *V_device_input, Real *W_device_input){

    Y_device = Y_device_input;
    F_device = F_device_input;
    T_device = T_device_input;
    V_device = V_device_input;
    W_device = W_device_input;

    box_particle();

    if(prompt>10){printf("pass1\n");}
    
    reset_grid();

    if(prompt>10){printf("pass2\n");}

    spatial_hashing();

    if(prompt>10){printf("pass3\n");}

    spread();

    if(prompt>10){printf("pass4\n");}

    fft_solve();

    if(prompt>10){printf("pass5\n");}

    gather();

    if(prompt>10){printf("pass6\n");}

    correction();

    if(prompt>10){printf("pass7\n");}

    // check_nan();

    if(prompt>10){printf("pass8\n");}

    sortback();

    if(prompt>10){printf("pass9\n");}

    rept += 1;

}

__host__
void FCM_solver::box_particle(){
    box<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(Y_device, N, boxsize);
}

__host__ 
void FCM_solver::reset_grid(){
    reset_device<Real> (V_device, 3*N);
    reset_device<Real> (W_device, 3*N);
    reset_device<Real> (hx_device, grid_size);
    reset_device<Real> (hy_device, grid_size);
    reset_device<Real> (hz_device, grid_size);
}

__host__
void FCM_solver::spatial_hashing(){
    ///////////////////////////////////////////////////////////////////////////////
    // Spatial hashing
    ///////////////////////////////////////////////////////////////////////////////
    cudaDeviceSynchronize();	time_start = get_time();

    // Create Hash (i, j, k) -> Hash
    create_hash_gpu<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(particle_cellhash_device, Y_device, N, cellL, M, boxsize);

    // DEBUG
    copy_to_host<int>(particle_cellhash_device, particle_cellhash_host, N);
    copy_to_host<Real>(Y_device, Y_host, 3*N);
    for(int i=0; i < N; i++){
        if(particle_cellhash_host[i] > (ncell-1) || particle_cellhash_host[i] < 0){
            printf("cellL: %.6f cellL*M: %.6f boxsize: %.6f\n", cellL, cellL*Real(M), boxsize);
            printf("particle %d with cellindex:%d at (%.6f %.6f %.6f)\n", i, particle_cellhash_host[i], Y_host[3*i], Y_host[3*i+1], Y_host[3*i+2]);
            printf("xc yc zc (%d %d %d)\n", int(Y_host[3*i]/cellL), int(Y_host[3*i+1]/cellL), int(Y_host[3*i+2]/cellL));
            printf("compute index %d\n", int(Y_host[3*i]/cellL) + (int(Y_host[3*i+1]/cellL) + int(Y_host[3*i+2]/cellL)*M)*M );
        }
    }

    // Sort particle index by hash
    particle_index_range<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(particle_index_device, N);
    sort_index_by_key(particle_cellhash_device, particle_index_device, key_buf, index_buf, N);

    // Sort pos/force/torque by particle index
    copy_device<Real> <<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(Y_device, aux_device, 3*N);
    sort_3d_by_index<Real> <<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(particle_index_device, Y_device, aux_device, N);
    copy_device<Real> <<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(F_device, aux_device, 3*N);
    sort_3d_by_index<Real> <<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(particle_index_device, F_device, aux_device, N);
    copy_device<Real> <<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(T_device, aux_device, 3*N);
    sort_3d_by_index<Real> <<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(particle_index_device, T_device, aux_device, N);

    // Find cell starting/ending points
    reset_device<int>(cell_start_device, ncell);
    reset_device<int>(cell_end_device, ncell);
    create_cell_list<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(particle_cellhash_device, cell_start_device, cell_end_device, N);


    cudaDeviceSynchronize();	time_hashing_array[rept] = get_time() - time_start;
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

            cufcm_mono_dipole_distribution_bpp_recompute<<<N, THREADS_PER_BLOCK>>>(hx_device, hy_device, hz_device, 
                                                Y_device, T_device, F_device,
                                                N, ngd,
                                                sigmaFCM, SigmaGRID,
                                                dx, nx, ny, nz);
                                            
        #elif SPREAD_TYPE == 4

            #if ROTATION

                cufcm_mono_dipole_distribution_bpp_shared_dynamic<<<N, THREADS_PER_BLOCK, 3*ngd*sizeof(Integer)+(9*ngd+15)*sizeof(Real)>>>
                                                    (hx_device, hy_device, hz_device, 
                                                    Y_device, T_device, F_device,
                                                    N, ngd,
                                                    sigmaFCM, SigmaGRID,
                                                    dx, nx, ny, nz);
            #else 


            #endif
                                                    

        #endif
    
    #elif SOLVER_MODE == 0
        
        cufcm_mono_dipole_distribution_regular_fcm<<<N, THREADS_PER_BLOCK, 3*ngd*sizeof(Integer)+(9*ngd+15)*sizeof(Real)>>>
                                                (hx_device, hy_device, hz_device, 
                                                Y_device, T_device, F_device,
                                                N, ngd,
                                                sigmaFCM, sigmaFCMdip,
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
    cufcm_flow_solve<<<num_thread_blocks_FFTGRID, THREADS_PER_BLOCK>>>(fk_x_device, fk_y_device, fk_z_device,
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
                                        Y_device, V_device, W_device,
                                        N, ngd,
                                        pdmag, sigmaGRIDsq, sigmaGRIDdipsq,
                                        anormGRID, anormGRID2,
                                        dx, nx, ny, nz);

        #elif GATHER_TYPE == 2

            cufcm_particle_velocities_bpp_shared<<<N, THREADS_PER_BLOCK>>>(hx_device, hy_device, hz_device,
                                        Y_device, V_device, W_device,
                                        N, ngd,
                                        pdmag, sigmaGRIDsq, sigmaGRIDdipsq,
                                        anormGRID, anormGRID2,
                                        dx, nx, ny, nz);

        #elif GATHER_TYPE == 3

            cufcm_particle_velocities_bpp_recompute<<<N, THREADS_PER_BLOCK>>>
                                        (hx_device, hy_device, hz_device,
                                        Y_device, V_device, W_device,
                                        N, ngd,
                                        sigmaFCM, SigmaGRID,
                                        dx, nx, ny, nz);

        #elif GATHER_TYPE == 4

            #if ROTATION

            cufcm_particle_velocities_bpp_shared_dynamic<<<N, THREADS_PER_BLOCK, 3*ngd*sizeof(Integer)+(9*ngd+3)*sizeof(Real)>>>
                                        (hx_device, hy_device, hz_device,
                                        Y_device, V_device, W_device,
                                        N, ngd,
                                        sigmaFCM, SigmaGRID,
                                        dx, nx, ny, nz);

            #else
            #endif

        #endif

    #elif SOLVER_MODE == 0

        cufcm_particle_velocities_regular_fcm<<<N, THREADS_PER_BLOCK, 3*ngd*sizeof(Integer)+(9*ngd+3)*sizeof(Real)>>>
                                        (hx_device, hy_device, hz_device,
                                        Y_device, V_device, W_device,
                                        N, ngd,
                                        sigmaFCM, sigmaFCMdip,
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
                                SigmaGRID, sigmaGRIDsq,
                                sigmaFCM, sigmaFCMsq,
                                sigmaFCMdip, sigmaFCMdipsq);
        
        #elif CORRECTION_TYPE == 1

            #if ROTATION

                cufcm_pair_correction<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(Y_device, V_device, W_device, F_device, T_device, N, boxsize,
                                    particle_cellhash_device, cell_start_device, cell_end_device,
                                    map_device,
                                    ncell, Rcsq,
                                    SigmaGRID,
                                    sigmaFCM,
                                    sigmaFCMdip);
            #else

            

            #endif

        #endif

        #if ROTATION

            cufcm_self_correction<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(V_device, W_device, F_device, T_device, N, boxsize,
                                    StokesMob, ModStokesMob,
                                    PDStokesMob, BiLapMob,
                                    WT1Mob, WT2Mob);
        #else


        #endif

    #endif

    cudaDeviceSynchronize();	time_correction_array[rept] = get_time() - time_start;
}

__host__
void FCM_solver::sortback(){
     /* Sort back */
    #if SORT_BACK == 1

        if(prompt>10){printf("sortback pass0\n");}
        particle_index_range<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(sortback_index_device, N);
        if(prompt>10){printf("sortback pass1\n");}
        sort_index_by_key(particle_index_device, sortback_index_device, key_buf, index_buf, N);
        if(prompt>10){printf("sortback pass2\n");}

        copy_device<Real> <<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(Y_device, aux_device, 3*N);
        if(prompt>10){printf("sortback pass3\n");}
        sort_3d_by_index<Real> <<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(sortback_index_device, Y_device, aux_device, N);
        if(prompt>10){printf("sortback pass4\n");}

        copy_device<Real> <<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(F_device, aux_device, 3*N);
        sort_3d_by_index<Real> <<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(sortback_index_device, F_device, aux_device, N);

        copy_device<Real> <<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(T_device, aux_device, 3*N);
        sort_3d_by_index<Real> <<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(sortback_index_device, T_device, aux_device, N);

        copy_device<Real> <<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(V_device, aux_device, 3*N);
        sort_3d_by_index<Real> <<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(sortback_index_device, V_device, aux_device, N);

        copy_device<Real> <<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(W_device, aux_device, 3*N);
        sort_3d_by_index<Real> <<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(sortback_index_device, W_device, aux_device, N);
        
    #endif
}

__host__
void FCM_solver::check_nan(){
    nan_check_host = true;
    cudaMemcpy(nan_check_device, &nan_check_host, sizeof(bool), cudaMemcpyHostToDevice);
    check_nan_in<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(V_device, 3*N, nan_check_device);
    check_nan_in<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(W_device, 3*N, nan_check_device);
    cudaMemcpy(&nan_check_host, nan_check_device, sizeof(bool), cudaMemcpyDeviceToHost);
    if(!nan_check_host){
        printf("\n\n********FATAL ERROR: NAN encountered********\n\n");
    }
}

__host__
void FCM_solver::assign_host_array_pointers(Real *Y_host_o,
                                            Real *F_host_o, Real *T_host_o,
                                            Real *V_host_o, Real *W_host_o){
    Y_host = Y_host_o;
    F_host = F_host_o;
    T_host = T_host_o;
    V_host = V_host_o;
    W_host = W_host_o;
}

__host__
void FCM_solver::prompt_time(){
    auto time_hashing = mean(&time_hashing_array[warmup], repeat-warmup);
	auto time_spreading = mean(&time_spreading_array[warmup], repeat-warmup);
	auto time_FFT = mean(&time_FFT_array[warmup], repeat-warmup);
	auto time_gathering = mean(&time_gathering_array[warmup], repeat-warmup);
	auto time_correction = mean(&time_correction_array[warmup], repeat-warmup);

	auto time_hashing_stdv = stdv(&time_hashing_array[warmup], repeat-warmup);
	auto time_spreading_stdv = stdv(&time_spreading_array[warmup], repeat-warmup);
	auto time_FFT_stdv = stdv(&time_FFT_array[warmup], repeat-warmup);
	auto time_gathering_stdv = stdv(&time_gathering_array[warmup], repeat-warmup);
	auto time_correction_stdv = stdv(&time_correction_array[warmup], repeat-warmup);

	auto time_compute = time_spreading + time_FFT + time_gathering + time_correction;
	auto PTPS = N/time_compute;

	if(prompt > 1){
		std::cout.precision(5);
		std::cout << std::endl;
		std::cout << "-------\nTimings\n-------\n";
		std::cout << "Init CUDA:\t" << time_cuda_initialisation << "s\n";
		std::cout << "Readfile:\t" << time_readfile << " s\n";
		std::cout << "Hashing:\t" << time_hashing << " \t+/-\t " << time_hashing_stdv << " s\n";
        std::cout << "Spreading:\t" << time_spreading << " \t+/-\t " << time_spreading_stdv <<" s\n";
		std::cout << "FFT+flow:\t" << time_FFT << " \t+/-\t " << time_FFT_stdv <<" s\n";
		std::cout << "Gathering:\t" << time_gathering << " \t+/-\t " << time_gathering_stdv <<" s\n";
		std::cout << "Correction:\t" << time_correction << " \t+/-\t " << time_correction_stdv <<" s\n";
		std::cout << "Compute total:\t" << time_compute <<" s\n";
		std::cout << "PTPS:\t" << PTPS << " /s\n";
		std::cout << std::endl;
	}
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
	auto time_spreading = mean(&time_spreading_array[warmup], repeat-warmup);
	auto time_FFT = mean(&time_FFT_array[warmup], repeat-warmup);
	auto time_gathering = mean(&time_gathering_array[warmup], repeat-warmup);
	auto time_correction = mean(&time_correction_array[warmup], repeat-warmup);

	auto time_hashing_stdv = stdv(&time_hashing_array[warmup], repeat-warmup);
	auto time_spreading_stdv = stdv(&time_spreading_array[warmup], repeat-warmup);
	auto time_FFT_stdv = stdv(&time_FFT_array[warmup], repeat-warmup);
	auto time_gathering_stdv = stdv(&time_gathering_array[warmup], repeat-warmup);
	auto time_correction_stdv = stdv(&time_correction_array[warmup], repeat-warmup);

	auto time_compute = time_spreading + time_FFT + time_gathering + time_correction;
	auto PTPS = N/time_compute;

	if(prompt > 1){
		std::cout.precision(5);
		std::cout << std::endl;
		std::cout << "-------\nTimings\n-------\n";
		std::cout << "Init CUDA:\t" << time_cuda_initialisation << "s\n";
		std::cout << "Readfile:\t" << time_readfile << " s\n";
		std::cout << "Hashing:\t" << time_hashing << " \t+/-\t " << time_hashing_stdv << " s\n";
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
    Real Yerror = -1;
    Real Verror = -1;
    Real Werror = -1;

	if (checkerror == 1){
        Real* Y_validation = malloc_host<Real>(3*N);
		Real* F_validation = malloc_host<Real>(3*N);
        Real* T_validation = malloc_host<Real>(3*N);
		Real* V_validation = malloc_host<Real>(3*N);
		Real* W_validation = malloc_host<Real>(3*N);

		read_validate_data(Y_validation,
						   F_validation,
                           T_validation,
						   V_validation,
						   W_validation, N, "./data/refdata/ref_data_N500000");

		Yerror = percentage_error_magnitude(Y_host, Y_validation, N);
		Verror = percentage_error_magnitude(V_host, V_validation, N);
		Werror = percentage_error_magnitude(W_host, W_validation, N);

		if(prompt > 1){
			std::cout << "-------\nError\n-------\n";
			std::cout << "%Y error:\t" << Yerror << "\n";
			std::cout << "%V error:\t" << Verror << "\n";
			std::cout << "%W error:\t" << Werror << "\n";
		}
    }

	///////////////////////////////////////////////////////////////////////////////
	// Write to file
	///////////////////////////////////////////////////////////////////////////////
	#if OUTPUT_TO_FILE == 1
		write_data(Y_host, F_host, T_host, V_host, W_host, N, 
                   "./data/simulation/simulation_data.dat", "w");
		
		write_time(time_cuda_initialisation, 
				time_readfile,
				time_hashing,
				time_spreading,
				time_FFT,
				time_gathering,
				time_correction,
				time_compute,
				"./data/simulation/simulation_scalar.dat");

        write_error(
            Verror,
            Werror,
            "./data/simulation/simulation_scalar.dat");

	#endif
}

void FCM_solver::write_cell_list(){
    copy_to_host<int>(cell_start_device, cell_start_host, ncell);
    copy_to_host<int>(cell_end_device, cell_end_host, ncell);
    write_celllist(cell_start_host, cell_end_host, ncell, "./data/simulation/celllist_data.dat");
}

__host__
void FCM_solver::apply_repulsion(){
    contact_force<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(Y_device, F_device, rh, N, boxsize,
                    particle_cellhash_device, cell_start_device, cell_end_device,
                    map_device,
                    ncell, 1.21*(2*rh)*(2*rh),
                    1.0);
}

__host__
void FCM_solver::check_overlap(){
    check_overlap_gpu<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(Y_device, rh, N, boxsize,
                      particle_cellhash_device, cell_start_device, cell_end_device, 
                      map_device, ncell, Rcsq);
}

__global__
void contact_force(Real* Y, Real *F, Real rad, int N, Real box_size,
                    int *particle_cellindex, int *cell_start, int *cell_end,
                    int *map,
                    int ncell, Real Rcsq,
                    Real Fref){

    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;
    const Real a_sum = 2.0*rad;

    for(int i = index; i < N; i += stride){
        Real fxi = (Real)0.0, fyi = (Real)0.0, fzi = (Real)0.0;
        int icell = particle_cellindex[i];
        Real xi = Y[3*i + 0], yi = Y[3*i + 1], zi = Y[3*i + 2];

        /* intra-cell interactions */
        /* corrections only apply to particle i */
        for(int j = cell_start[icell]; j < cell_end[icell]; j++){
            if(i != j){
                Real xij = xi - Y[3*j + 0];
                Real yij = yi - Y[3*j + 1];
                Real zij = zi - Y[3*j + 2];

                xij -= box_size * Real(int(xij/(Real(0.5)*box_size)));
                yij -= box_size * Real(int(yij/(Real(0.5)*box_size)));
                zij -= box_size * Real(int(zij/(Real(0.5)*box_size)));

                Real rijsq=xij*xij+yij*yij+zij*zij;
                if(rijsq < 1.21*a_sum*a_sum){

                    Real rij = sqrtf(rijsq);
                    Real chi_fac = Real(10.0)/a_sum;

                    Real fac = fmin(1.0, 1.0 - chi_fac*(rij - a_sum));
                    fac *= Fref*Real(2.0)*(Real(220.0)*Real(1800.0)/(Real(2.2)*Real(2.2)*Real(40.0)*Real(40.0)))*fac*fac*fac;

                    const double dm1 = 1.0/rij;

                    fxi += fac*xij*dm1;
                    fyi += fac*yij*dm1;
                    fzi += fac*zij*dm1;

                    // Real fxij = Fref*xij/rijsq;
                    // Real fyij = Fref*yij/rijsq;
                    // Real fzij = Fref*zij/rijsq;

                    // fxi += fxij;
                    // fyi += fyij;
                    // fzi += fzij;

                }
            }
            
        }
        int jcello = 13*icell;
        /* inter-cell interactions */
        /* corrections apply to both parties in different cells */
        for(int nabor = 0; nabor < 13; nabor++){
            int jcell = map[jcello + nabor];
            for(int j = cell_start[jcell]; j < cell_end[jcell]; j++){
                Real xij = xi - Y[3*j + 0];
                Real yij = yi - Y[3*j + 1];
                Real zij = zi - Y[3*j + 2];

                xij = xij - box_size * Real(int(xij/(Real(0.5)*box_size)));
                yij = yij - box_size * Real(int(yij/(Real(0.5)*box_size)));
                zij = zij - box_size * Real(int(zij/(Real(0.5)*box_size)));

                Real rijsq=xij*xij+yij*yij+zij*zij;
                if(rijsq < 1.21*a_sum*a_sum){

                    Real rij = sqrtf(rijsq);
                    Real chi_fac = Real(10.0)/a_sum;

                    Real fac = fmin(1.0, 1.0 - chi_fac*(rij - a_sum));
                    fac *= Fref*Real(2.0)*(Real(220.0)*Real(1800.0)/(Real(2.2)*Real(2.2)*Real(40.0)*Real(40.0)))*fac*fac*fac;

                    const double dm1 = 1.0/rij;

                    Real fxij = fac*xij*dm1;
                    Real fyij = fac*yij*dm1;
                    Real fzij = fac*zij*dm1;
                    
                    // Real fxij = Fref*xij/rijsq;
                    // Real fyij = Fref*yij/rijsq;
                    // Real fzij = Fref*zij/rijsq;

                    fxi += fxij;
                    fyi += fyij;
                    fzi += fzij;

                    atomicAdd(&F[3*j + 0], -fxij);
                    atomicAdd(&F[3*j + 1], -fyij);
                    atomicAdd(&F[3*j + 2], -fzij);
                }
            }
        }

        if(i==0){
            printf("fx=%.4f x = (%.4f %.4f %.4f)\n", fxi, xi, yi, zi);
        }
        atomicAdd(&F[3*i + 0], fxi);
        atomicAdd(&F[3*i + 1], fyi);
        atomicAdd(&F[3*i + 2], fzi);

        return;
    }
}

__global__
void check_overlap_gpu(Real *Y, Real rad, int N, Real box_size,
                    int *particle_cellindex, int *cell_start, int *cell_end,
                    int *map,
                    int ncell, Real Rcsq){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    for(int i = index; i < N; i += stride){
        int icell = particle_cellindex[i];
        
        Real xi = Y[3*i + 0], yi = Y[3*i + 1], zi = Y[3*i + 2];
        for(int j = cell_start[icell]; j < cell_end[icell]; j++){
            if(i != j){
                Real xij = xi - Y[3*j + 0];
                Real yij = yi - Y[3*j + 1];
                Real zij = zi - Y[3*j + 2];

                xij = xij - box_size * Real(int(xij/(Real(0.5)*box_size)));
                yij = yij - box_size * Real(int(yij/(Real(0.5)*box_size)));
                zij = zij - box_size * Real(int(zij/(Real(0.5)*box_size)));

                Real rijsq=xij*xij+yij*yij+zij*zij;
                if(rijsq < Rcsq){
                    if (rijsq < 3.98*rad*rad){
                        printf("ERROR: Overlap between %d (%.4f %.4f %.4f) and %d (%.4f %.4f %.4f) within same cell. sep = %.6f 2rad = %.6f \n",
                        i, xi, yi, zi, j, Y[3*j + 0], Y[3*j + 1], Y[3*j + 2], sqrt(rijsq), (2*rad));
                    }
                }
            }
        }
        int jcello = 13*icell;
        /* inter-cell interactions */
        /* corrections apply to both parties in different cells */
        for(int nabor = 0; nabor < 13; nabor++){
            int jcell = map[jcello + nabor];
            for(int j = cell_start[jcell]; j < cell_end[jcell]; j++){                
                Real xij = xi - Y[3*j + 0];
                Real yij = yi - Y[3*j + 1];
                Real zij = zi - Y[3*j + 2];

                xij = xij - box_size * Real(int(xij/(Real(0.5)*box_size)));
                yij = yij - box_size * Real(int(yij/(Real(0.5)*box_size)));
                zij = zij - box_size * Real(int(zij/(Real(0.5)*box_size)));
                Real rijsq=xij*xij+yij*yij+zij*zij;
                if(rijsq < Rcsq){
                    if (rijsq < 3.98*rad*rad){
                        printf("ERROR: Overlap between %d (%.4f %.4f %.4f) in %d and %d (%.4f %.4f %.4f) in %d. sep = %.6f 2rad = %.6f  \n", 
                        i, xi, yi, zi, icell, j, Y[3*j + 0], Y[3*j + 1], Y[3*j + 2], jcell, sqrt(rijsq), (2*rad));
                    }
                }
            }
        }

        return;
    }
}