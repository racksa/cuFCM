#include <cstdlib>
#include <iostream>
#include <fstream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


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
        Nf = N;
        Nv = N;
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
    #if USE_REGULARFCM

        sigmaFCM = rh/sqrt(PI);
        sigmaFCMdip = rh/pow(6.0*sqrt(PI), 1.0/3.0);

        SigmaGRID = sigmaFCM;
        
    #else

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

        // PDStokesMob = 2.0/pow(2.0*PI, 1.5);
        // PDStokesMob = PDStokesMob/pow(gammaGRID, 3.0);
        // PDStokesMob = PDStokesMob*pdmag/3.0;
        PDStokesMob = pdmag/(12.0*pow(sqrt(PI)*SigmaGRID, 3.0));

        // BiLapMob = 1.0/pow(4.0*PI*SigmaGRID*SigmaGRID, 1.5);
        // BiLapMob = BiLapMob/(4.0*SigmaGRID*SigmaGRID)*pdmag*pdmag;
        BiLapMob = pdmag*pdmag/(32.0*pow(PI, 1.5)*pow(SigmaGRID,5));

        WT1Mob = 1.0/(8.0*PI)/pow(rh, 3) ;
        // WT2Mob = 1.0/(8.0*PI)/pow(sigmaGRIDdip*pow(6.0*sqrt(PI), 1.0/3.0), 3);
        WT2Mob = 1.0/(48.0*PI*sqrt(PI)*pow(sigmaGRIDdip,3));

        // Rc = Real(-eta*pdmag);

    #endif

    /* Neighbour list */
    Rc_fac = Real(eta*alpha);
    Rc = Rc_fac*dx;
    Rcsq = Rc*Rc;

    Lx = boxsize;
    Ly = boxsize/Real(nx)*Real(ny);
    Lz = boxsize/Real(nx)*Real(nz);
    int Lmin = std::min(std::min(Lx, Ly), Lz);
    int Lmax = std::max(std::max(Lx, Ly), Lz);
    Mx = std::max((Lx/Rc), Real(3.0));
    My = std::max((Ly/Rc), Real(3.0));
    Mz = std::max((Lz/Rc), Real(3.0));

    ncell = Mx*My*Mz;
    mapsize = 13*ncell;

    Volume_frac = (N*4.0/3.0*PI*rh*rh*rh) / (Lx*Ly*Lz);
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
        #if USE_DOUBLE_PRECISION
            std::cout << "Data:\t\t\t<Double precision>" << "\n";
        #else
            std::cout << "Data:\t\t\t<Single precision>" << "\n";
        #endif

        #ifndef USE_REGULARFCM
			std::cout << "Solver:\t\t\t" << "<Fast FCM>" << "\n";
		#elif USE_REGULARFCM
			std::cout << "Solver:\t\t\t" << "<Regular FCM>" << "\n";
		#endif
        #if ROTATION == 0
			std::cout << "Order:\t\t\t" << "<Translation>" << "\n";
		#elif ROTATION == 1
			std::cout << "Order:\t\t\t" << "<Rotation>" << "\n";
		#endif
		std::cout << "Particle number:\t" << N << "\n";
		std::cout << "Particle radius:\t" << rh << "\n";
		std::cout << "Grid support:\t\t" << ngd << "\n";
		#ifndef USE_REGULARFCM
			std::cout << "Sigma/sigma:\t\t" << SigmaGRID/sigmaFCM << "\n";
			std::cout << "Alpha:\t\t\t" << alpha << "\n";
			std::cout << "Beta:\t\t\t" << beta << "\n";
			std::cout << "Eta:\t\t\t" << eta << "\n";
		#endif
		#ifndef USE_REGULARFCM
			std::cout << "Sigma:\t\t\t" << SigmaGRID << "\n";
		#elif USE_REGULARFCM
			std::cout << "sigma:\t\t\t" << sigmaFCM << "\n";
		#endif
		std::cout << "dx:\t\t\t" << dx<< "\n";
        std::cout << "boxsize:\t\t" << "(" << Lx << " " << Ly << " " << Lz << ")\n";
        std::cout << "Grid points:\t\t" << "(" << nx << " " << ny << " " << nz << ")\n";
		std::cout << "Cell number:\t\t" << "(" << Mx << " " << My << " " << Mz << ")\n";
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

	map_host = malloc_host<int>(mapsize);							 map_device = malloc_device<int>(mapsize);

	bulkmap_loop(map_host, Mx, My, Mz, linear_encode);
	copy_to_device<int>(map_host, map_device, mapsize);

	/* Create 3D FFT plans */
	if (cufftPlan3d(&plan, nz, ny, nx, cufftReal2Complex) != CUFFT_SUCCESS){
		printf("CUFFT error: Plan creation failed");
		return ;	
	}

	if (cufftPlan3d(&iplan, nz, ny, nx, cufftComplex2Real) != CUFFT_SUCCESS){
		printf("CUFFT error: Plan creation failed");
		return ;	
	}

    num_thread_blocks_FFTGRID = (fft_grid_size + FCM_THREADS_PER_BLOCK - 1)/FCM_THREADS_PER_BLOCK;
	num_thread_blocks_GRID = (grid_size + FCM_THREADS_PER_BLOCK - 1)/FCM_THREADS_PER_BLOCK;
	num_thread_blocks_N = (N + FCM_THREADS_PER_BLOCK - 1)/FCM_THREADS_PER_BLOCK;
	num_thread_blocks_NX = (nx + FCM_THREADS_PER_BLOCK - 1)/FCM_THREADS_PER_BLOCK;
	
	cudaMalloc((void**)&dev_random, num_thread_blocks_N*FCM_THREADS_PER_BLOCK*sizeof(curandState));

	time_cuda_initialisation += get_time() - time_start;
}

/*  Cilia code start */

void FCM_solver::init_aux_for_filament(){

    Yf_host = malloc_host<Real>(3*pars.N);						Yf_device = malloc_device<Real>(3*pars.N);
	F_host = malloc_host<Real>(3*pars.N);						F_device = malloc_device<Real>(3*pars.N);
	T_host = malloc_host<Real>(3*pars.N);						T_device = malloc_device<Real>(3*pars.N);
	V_host = malloc_host<Real>(3*pars.N);						V_device = malloc_device<Real>(3*pars.N);
	W_host = malloc_host<Real>(3*pars.N);						W_device = malloc_device<Real>(3*pars.N);

}

__host__
void FCM_solver::reform_xsegblob(Real *x_seg, Real *x_blob, bool to_solver){
    int num_thread_blocks_Nseg = std::max(1,(num_seg + FCM_THREADS_PER_BLOCK - 1)/FCM_THREADS_PER_BLOCK);
    int num_thread_blocks_Nblob = std::max(1, (num_blob + FCM_THREADS_PER_BLOCK - 1)/FCM_THREADS_PER_BLOCK);
    if(to_solver){copy_device<Real> <<<num_thread_blocks_Nseg, FCM_THREADS_PER_BLOCK>>>(x_seg, Yf_device, 3*num_seg);
    copy_device<Real> <<<num_thread_blocks_Nblob, FCM_THREADS_PER_BLOCK>>>(x_blob, &Yf_device[3*num_seg], 3*num_blob);
    }
    else{copy_device<Real> <<<num_thread_blocks_Nseg, FCM_THREADS_PER_BLOCK>>>(Yf_device, x_seg, 3*num_seg);
    copy_device<Real> <<<num_thread_blocks_Nblob, FCM_THREADS_PER_BLOCK>>>(&Yf_device[3*num_seg], x_blob, 3*num_blob);}
}   

__host__
void FCM_solver::reform_fseg(Real *f_seg, bool to_solver){
    int num_thread_blocks_Nseg = std::max(1, (num_seg + FCM_THREADS_PER_BLOCK - 1)/FCM_THREADS_PER_BLOCK);
    if(to_solver){interleaved2separate<<<num_thread_blocks_Nseg, FCM_THREADS_PER_BLOCK>>>(f_seg, F_device, T_device, num_seg);}
    else{separate2interleaved<<<num_thread_blocks_Nseg, FCM_THREADS_PER_BLOCK>>>(f_seg, F_device, T_device, num_seg);}
}

__host__
void FCM_solver::reform_vseg(Real *v_seg, bool to_solver){
    int num_thread_blocks_Nseg = std::max(1, (num_seg + FCM_THREADS_PER_BLOCK - 1)/FCM_THREADS_PER_BLOCK);
    if(to_solver){interleaved2separate<<<num_thread_blocks_Nseg, FCM_THREADS_PER_BLOCK>>>(v_seg, V_device, W_device, num_seg);}
    else{separate2interleaved<<<num_thread_blocks_Nseg, FCM_THREADS_PER_BLOCK>>>(v_seg, V_device, W_device, num_seg);}
}

__host__
void FCM_solver::reform_fblob(Real *f_blob, bool to_solver){
    int num_thread_blocks_Nblob = std::max(1, (num_blob + FCM_THREADS_PER_BLOCK - 1)/FCM_THREADS_PER_BLOCK);
    if(to_solver){copy_device<Real> <<<num_thread_blocks_Nblob, FCM_THREADS_PER_BLOCK>>>(f_blob, &F_device[3*num_seg], 3*num_blob);
    reset_device<Real> (&T_device[3*num_seg], 3*num_blob);}
    else{copy_device<Real> <<<num_thread_blocks_Nblob, FCM_THREADS_PER_BLOCK>>>(&F_device[3*num_seg], f_blob, 3*num_blob);}
}

__host__
void FCM_solver::reform_vblob(Real *v_blob, bool to_solver){
    int num_thread_blocks_Nblob = std::max(1, (num_blob + FCM_THREADS_PER_BLOCK - 1)/FCM_THREADS_PER_BLOCK);
    if(to_solver){copy_device<Real> <<<num_thread_blocks_Nblob, FCM_THREADS_PER_BLOCK>>>(v_blob, &V_device[3*num_seg], 3*num_blob);}
    else{copy_device<Real> <<<num_thread_blocks_Nblob, FCM_THREADS_PER_BLOCK>>>(&V_device[3*num_seg], v_blob, 3*num_blob);}
}

__host__ 
void FCM_solver::evaluate_mobility_cilia(){

    cudaError_t err = cudaGetLastError();

    reset_grid();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error reset_grid: %s\n", cudaGetErrorString(err));
    }

    box_particle();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error box_particle: %s\n", cudaGetErrorString(err));
    }

    spatial_hashing();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error spatial_hashing: %s\n", cudaGetErrorString(err));
    }

    sort_particle();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error sort_particle: %s\n", cudaGetErrorString(err));
    }

    spread();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error spread: %s\n", cudaGetErrorString(err));
    }
    
    fft_solve();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error fft_solve: %s\n", cudaGetErrorString(err));
    }

    gather();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error gather: %s\n", cudaGetErrorString(err));
    }

    correction();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error correction: %s\n", cudaGetErrorString(err));
    }

    sortback();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error sortback: %s\n", cudaGetErrorString(err));
    }

}


/*  Cilia code end */

__host__ 
void FCM_solver::hydrodynamic_solver(Real *Yf_device_input,
                                     Real *F_device_input, Real *T_device_input, 
                                     Real *V_device_input, Real *W_device_input){

    Yf_device = Yf_device_input;
    F_device = F_device_input;
    T_device = T_device_input;
    V_device = V_device_input;
    W_device = W_device_input;

    cudaError_t err = cudaGetLastError();

    reset_grid();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error reset_grid: %s\n", cudaGetErrorString(err));
    }

    box_particle();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error box_particle: %s\n", cudaGetErrorString(err));
    }

    spatial_hashing();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error spatial_hashing: %s\n", cudaGetErrorString(err));
    }

    sort_particle();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error sort_particle: %s\n", cudaGetErrorString(err));
    }

    spread();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error spread: %s\n", cudaGetErrorString(err));
    }
    
    fft_solve();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error fft_solve: %s\n", cudaGetErrorString(err));
    }

    gather();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error gather: %s\n", cudaGetErrorString(err));
    }

    correction();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error correction: %s\n", cudaGetErrorString(err));
    }

    sortback();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error sortback: %s\n", cudaGetErrorString(err));
    }

    rept += 1;

}

__host__
void FCM_solver::box_particle(){
    box<<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(Yf_device, N, Lx, Ly, Lz);
}

__host__ 
void FCM_solver::reset_grid(){
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
    create_hash_gpu<<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(particle_cellhash_device, Yf_device, N, Mx, My, Mz, Lx, Ly, Lz);
    particle_index_range<<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(particle_index_device, N);

    // verify_hash_gpu<<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(particle_cellhash_device, Yf_device, N, Mx, My, Mz, Lx, Ly, Lz);
}

__host__
void FCM_solver::sort_particle(){
    // Sort particle index by hash
    sort_index_by_key(particle_cellhash_device, particle_index_device, key_buf, index_buf, N);

    // Sort pos/force/torque by particle index
    copy_device<Real> <<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(Yf_device, aux_device, 3*N);
    sort_3d_by_index<Real> <<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(particle_index_device, Yf_device, aux_device, N);
    copy_device<Real> <<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(F_device, aux_device, 3*N);
    sort_3d_by_index<Real> <<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(particle_index_device, F_device, aux_device, N);
    copy_device<Real> <<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(V_device, aux_device, 3*N);
    sort_3d_by_index<Real> <<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(particle_index_device, V_device, aux_device, N);
    copy_device<Real> <<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(T_device, aux_device, 3*N);
    sort_3d_by_index<Real> <<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(particle_index_device, T_device, aux_device, N);
    copy_device<Real> <<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(W_device, aux_device, 3*N);
    sort_3d_by_index<Real> <<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(particle_index_device, W_device, aux_device, N);

    // Find cell starting/ending points
    reset_device<int>(cell_start_device, ncell);
    reset_device<int>(cell_end_device, ncell);
    create_cell_list<<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(particle_cellhash_device, cell_start_device, cell_end_device, N);

    cudaDeviceSynchronize();	time_hashing_array[rept] = get_time() - time_start;

    // verify_cell_list<<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(particle_cellhash_device, cell_start_device, cell_end_device, Yf_device, 
    // N, Mx, My, Mz, Lx, Ly, Lz);

}

__host__
void FCM_solver::sortback(){
     /* Sort back */
    #if SORT_BACK == 1

        particle_index_range<<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(sortback_index_device, N);
        sort_index_by_key(particle_index_device, sortback_index_device, key_buf, index_buf, N);
        // particle cellhash is not sorted back!!!
        // so need to re-compute.

        copy_device<Real> <<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(Yf_device, aux_device, 3*N);
        sort_3d_by_index<Real> <<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(sortback_index_device, Yf_device, aux_device, N);
        copy_device<Real> <<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(F_device, aux_device, 3*N);
        sort_3d_by_index<Real> <<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(sortback_index_device, F_device, aux_device, N);
        copy_device<Real> <<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(V_device, aux_device, 3*N);
        sort_3d_by_index<Real> <<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(sortback_index_device, V_device, aux_device, N);
        copy_device<Real> <<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(T_device, aux_device, 3*N);
        sort_3d_by_index<Real> <<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(sortback_index_device, T_device, aux_device, N);
        copy_device<Real> <<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(W_device, aux_device, 3*N);
        sort_3d_by_index<Real> <<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(sortback_index_device, W_device, aux_device, N);

    #endif
}

__host__
void FCM_solver::spread(){
    ///////////////////////////////////////////////////////////////////////////////
    // Spreading
    ///////////////////////////////////////////////////////////////////////////////
    cudaDeviceSynchronize();	time_start = get_time();

    long shared_size = 3*ngd*sizeof(Integer)+(9*ngd+15)*sizeof(Real);

    cufcm_mono_dipole_distribution_bpp_shared_dynamic<<<N, FCM_THREADS_PER_BLOCK, shared_size>>>
                                        (hx_device, hy_device, hz_device, 
                                        Yf_device, T_device, F_device,
                                        N, ngd,
                                        sigmaFCM, sigmaFCMdip, SigmaGRID,
                                        dx, nx, ny, nz,
                                        particle_index_device, 0, N,
                                        ROTATION);

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
    cufcm_flow_solve<<<num_thread_blocks_FFTGRID, FCM_THREADS_PER_BLOCK>>>(fk_x_device, fk_y_device, fk_z_device,
                                                            uk_x_device, uk_y_device, uk_z_device,
                                                            nx, ny, nz, Lx, Ly, Lz);
    ///////////////////////////////////////////////////////////////////////////////
    // IFFT
    ///////////////////////////////////////////////////////////////////////////////
    if (cufftExecComplex2Real(iplan, uk_x_device, hx_device) != CUFFT_SUCCESS){
        printf("CUFFT error: ExecZ2D Backward failed (fx)\n");
        return ;	
    }
    if (cufftExecComplex2Real(iplan, uk_y_device, hy_device) != CUFFT_SUCCESS){
        printf("CUFFT error: ExecZ2D Backward failed (fy)\n");
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

    long shared_size = 3*ngd*sizeof(Integer)+(9*ngd+3)*sizeof(Real);

    cufcm_particle_velocities_bpp_shared_dynamic<<<N, FCM_THREADS_PER_BLOCK, shared_size>>>
                                        (hx_device, hy_device, hz_device,
                                        Yf_device, V_device, W_device,
                                        N, ngd,
                                        sigmaFCM, sigmaFCMdip, SigmaGRID,
                                        dx, nx, ny, nz,
                                        particle_index_device, 0, N,
                                        ROTATION);

    cudaDeviceSynchronize();	time_gathering_array[rept] = get_time() - time_start;
}

__host__
void FCM_solver::correction(){
    ///////////////////////////////////////////////////////////////////////////////
    // Correction
    ///////////////////////////////////////////////////////////////////////////////
    cudaDeviceSynchronize();	time_start = get_time();

    #ifndef USE_REGULARFCM
        
        cufcm_pair_correction<<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(Yf_device, V_device, W_device, F_device, T_device, N, Lx, Ly, Lz,
                            particle_cellhash_device, cell_start_device, cell_end_device,
                            map_device,
                            ncell, Rcsq,
                            SigmaGRID,
                            sigmaFCM,
                            sigmaFCMdip);

        cufcm_self_correction<<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(V_device, W_device, F_device, T_device, N,
                                StokesMob, ModStokesMob,
                                PDStokesMob, BiLapMob,
                                WT1Mob, WT2Mob);

    #endif

    cudaDeviceSynchronize();	time_correction_array[rept] = get_time() - time_start;
}


__host__
void FCM_solver::assign_host_array_pointers(Real *Yf_host_o, 
                                            Real *F_host_o, Real *T_host_o,
                                            Real *V_host_o, Real *W_host_o){
    Yf_host = Yf_host_o;
    F_host = F_host_o;
    T_host = T_host_o;
    V_host = V_host_o;
    W_host = W_host_o;
}

__host__
void FCM_solver::prompt_time(){
    time_hashing = mean(&time_hashing_array[warmup], repeat-warmup);
	time_spreading = mean(&time_spreading_array[warmup], repeat-warmup);
	time_FFT = mean(&time_FFT_array[warmup], repeat-warmup);
	time_gathering = mean(&time_gathering_array[warmup], repeat-warmup);
	time_correction = mean(&time_correction_array[warmup], repeat-warmup);

	time_hashing_stdv = stdv(&time_hashing_array[warmup], repeat-warmup);
	time_spreading_stdv = stdv(&time_spreading_array[warmup], repeat-warmup);
	time_FFT_stdv = stdv(&time_FFT_array[warmup], repeat-warmup);
	time_gathering_stdv = stdv(&time_gathering_array[warmup], repeat-warmup);
	time_correction_stdv = stdv(&time_correction_array[warmup], repeat-warmup);

	time_compute = time_spreading + time_FFT + time_gathering + time_correction;
	PTPS = N/time_compute;

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

	///////////////////////////////////////////////////////////////////////////////
	// Time
	///////////////////////////////////////////////////////////////////////////////
    prompt_time();
}

__host__
void FCM_solver::write_data_call(){    
    copy_to_host<Real>(Yf_device, Yf_host, 3*N);
	copy_to_host<Real>(F_device, F_host, 3*N);
	copy_to_host<Real>(T_device, T_host, 3*N);
	copy_to_host<Real>(V_device, V_host, 3*N);
	copy_to_host<Real>(W_device, W_host, 3*N);
    write_data(Yf_host, F_host, T_host, V_host, W_host, N, 
                   "simulation_data.dat", "a");
}

__host__
void FCM_solver::write_flowfield_call(){
    copy_to_host<Real>(hx_device, hx_host, grid_size);
    copy_to_host<Real>(hy_device, hy_host, grid_size);
    copy_to_host<Real>(hz_device, hz_host, grid_size);
    write_flow_field(hx_host, grid_size, "./data/simulation/flow_x.dat");
    write_flow_field(hy_host, grid_size, "./data/simulation/flow_y.dat");
    write_flow_field(hz_host, grid_size, "./data/simulation/flow_z.dat");
}

__host__
void FCM_solver::write_cell_list(){
    copy_to_host<int>(cell_start_device, cell_start_host, ncell);
    copy_to_host<int>(cell_end_device, cell_end_host, ncell);
    write_celllist(cell_start_host, cell_end_host, map_host, ncell, Mx, My, Mz, "./data/simulation/celllist_data.dat");
}

__host__
void FCM_solver::apply_repulsion(){
    reset_device<Real> (F_device, 3*N);
    contact_force<<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(Yf_device, F_device, rh, N, Lx, Ly, Lz,
                    particle_cellhash_device, cell_start_device, cell_end_device,
                    map_device,
                    ncell, 1.21*(2*rh)*(2*rh),
                    (Real(2.0)*Real(44.0)));
}

__host__
void FCM_solver::check_overlap(){
    check_overlap_gpu<<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(Yf_device, rh, N, Lx, Ly, Lz,
                      particle_cellhash_device, cell_start_device, cell_end_device, 
                      map_device, ncell, 0.5*Rcsq);
}

