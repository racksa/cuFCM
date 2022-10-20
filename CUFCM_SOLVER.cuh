#pragma once
#include "config.hpp"

class FCM_solver{

public:
    Pars pars;

    int N, nx, ny, nz, repeat, prompt, warmup;
    Real rh, alpha, beta, eta;
    Real boxsize;
    Real values[100];
    int grid_size, fft_grid_size, ngd;
    Real dx, Rc_fac, Rc, Rcsq,  Volume_frac;

    /* Timing */
    Real time_start, time_cuda_initialisation, time_readfile;
    Real *time_hashing_array, *time_linklist_array,*time_precompute_array,
         *time_spreading_array, *time_FFT_array, *time_gathering_array, *time_correction_array;
    int rept = 0;

    cufftHandle plan, iplan;
    /* source */
    Real *aux_host, *aux_device, *Y_host, *Y_device, *F_host, *F_device,
	     *T_host, *T_device, *V_host, *V_device, *W_host, *W_device;

    /* grid */
    myCufftReal *hx_host, *hy_host, *hz_host, *hx_device, *hy_device, *hz_device;
    myCufftComplex *fk_x_host, *fk_x_device, *fk_y_host, *fk_y_device,
	               *fk_z_host, *fk_z_device, *uk_x_host, *uk_x_device,
	               *uk_y_host, *uk_y_device, *uk_z_host, *uk_z_device;
    
    /* sorting & cell list */
    int *particle_cellhash_host, *particle_cellhash_device,
        *particle_index_host, *particle_index_device,
        *sortback_index_host, *sortback_index_device;

    int M, ncell, mapsize;
    
	int *cell_start_host, *cell_start_device,
        *cell_end_host, *cell_end_device;

    Real cellL;

	#if CORRECTION_TYPE == 0

		int *head_host, *head_device, *list_host, *list_device;

	#endif

	#if SPATIAL_HASHING == 0 or SPATIAL_HASHING == 1

		int  *Y_hash_host, *Y_hash_device, 
             *F_hash_host, *F_hash_device,
             *T_hash_host, *T_hash_device;

	#endif

	int *map_host, *map_device;
    /* cuda thread */
    int num_thread_blocks_GRID;
	int num_thread_blocks_N;
	int num_thread_blocks_NX;
	curandState *dev_random;

    #if SOLVER_MODE == 1
        Real sigmaFCM, sigmaFCMsq, anormFCM, anormFCM2,
            sigmaGRID, sigmaGRIDsq, anormGRID, anormGRID2,
            sigma_fac, gammaGRID, pdmag,
            sigmaFCMdip, sigmaFCMdipsq, anormFCMdip, anormFCMdip2,
            sigma_dip_fac,
            sigmaGRIDdip, sigmaGRIDdipsq, anormGRIDdip, anormGRIDdip2,
            StokesMob, ModStokesMob, PDStokesMob, BiLapMob, WT1Mob, WT2Mob;
    #elif SOLVER_MODE == 0
        Real sigmaFCM, sigmaFCMsq, anormFCM, anormFCM2,
            sigmaFCMdip, sigmaFCMdipsq, anormFCMdip, anormFCMdip2,
            StokesMob, WT1Mob;
    #endif

    __host__
    FCM_solver();

    __host__
    FCM_solver(Pars);

    __host__
    void init();

    __host__
    void prompt_info();
    
    __host__
    void init_config();

    __host__
    void init_fcm_var();

    __host__
    void init_time_array();

    __host__
    void init_cuda();

    __host__
    void hydrodynamic_solver(Real *Y_host_input, Real *F_host_input, Real *T_host_input,
                             Real *Y_device_input, Real * F_device_input, Real *T_device_input);

    __host__
    void reset_grid();

    __host__
    void spatial_hashing();

    __host__
    void sort_particle();

    __host__
    void spread();

     __host__
    void fft_solve();

     __host__
    void gather();

    __host__
    void correction();

    __host__
    void sortback();

    __host__
    void finish();

};