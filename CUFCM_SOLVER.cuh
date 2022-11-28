#pragma once
#include "config.hpp"

class FCM_solver{

public:
    Pars pars;

    int N, nx, ny, nz, repeat, prompt, warmup, checkerror;
    Real rh, alpha, beta, eta;
    Real boxsize;
    Real values[100];
    int grid_size, fft_grid_size, ngd;
    Real dx, Rc_fac, Rc, Rcsq,  Volume_frac;

    bool *nan_check_device, nan_check_host=true;

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
    int *key_buf, *index_buf;

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
    int num_thread_blocks_FFTGRID;
    int num_thread_blocks_GRID;
	int num_thread_blocks_N;
	int num_thread_blocks_NX;
	curandState *dev_random;

    #if SOLVER_MODE == 1
        Real sigmaFCM, SigmaGRID, 
            gammaGRID, pdmag,
            sigmaFCMdip, sigmaGRIDdip,
            StokesMob, ModStokesMob, PDStokesMob, BiLapMob, WT1Mob, WT2Mob;
    #elif SOLVER_MODE == 0
        Real sigmaFCM,
            sigmaFCMdip;
    #endif

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
    void hydrodynamic_solver(Real *Y_device_input, Real * F_device_input, Real *T_device_input,
                             Real * V_device_input, Real *W_device_input);

    __host__
    void box_particle();

    /* Filament code start*/
    __host__
    void init_aux_for_filament();

    __host__
    void reform_data(Real *x_seg, Real *f_seg, Real *v_seg,
                     Real *x_blob, Real *f_blob, Real *v_blob,
                     int num_seg, int num_blob);
                
    void reform_data_back(Real *x_seg, Real *f_seg, Real *v_seg,
                     Real *x_blob, Real *f_blob, Real *v_blob,
                     int num_seg, int num_blob);

    __host__
    void Mss();
            
    __host__
    void Msb();

    __host__
    void Mbs();

    __host__
    void Mbb();
    /* Filament code end*/

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
    void check_nan();

    __host__
    void prompt_time();

    __host__
    void finish();

    __host__
    void assign_host_array_pointers(Real *Y_host_o, 
                                    Real *F_host_o, Real *T_host_o, 
                                    Real *V_host_o, Real *W_host_o);

};