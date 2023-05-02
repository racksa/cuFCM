#pragma once
#include "config.hpp"

__global__
void init_drag(Real *F, Real rad, int N, Real Fref, curandState *states);

__global__
void decrease_drag(Real *F, Real fac, int N);

__global__
void apply_repulsion(Real* Y, Real *F, Real rad, int N, Real Lx, Real Ly, Real Lz,
                    int *particle_cellindex, int *cell_start, int *cell_end,
                    int *map,
                    int ncell, Real Rrefsq,
                    Real Fref);

__global__
void move_forward(Real *Y, Real *V, Real dt, int N);

class random_packer{

public:
    
    Random_Pars pars;

    int N, nx, ny, nz, Mx, My, Mz, prompt;
    Real rh;
    Real boxsize;
    Real Lx, Ly, Lz;
    Real values[100];
    int *overlap_counter;

    Real *aux_host, *aux_device,
         *Y_host, *Y_device,
         *F_host, *F_device,
         *init_F_host, *init_F_device,
	     *V_host, *V_device;
    
    int *key_buf, *index_buf;

    int M, ncell, mapsize;
    int *map_host, *map_device;
    
	int *cell_start_host, *cell_start_device,
        *cell_end_host, *cell_end_device;

    Real dt, Rc, Rcsq, Fref;

    int num_thread_blocks_N;
    curandState *dev_random;


    __host__
    random_packer(Real *Y_host_input, Real *Y_device_input, Random_Pars pars_input);

    __host__
    void init_cuda();
    
    __host__
    void spatial_hashing();

    __host__
    void sort_back();

    __host__
    void update();

    __host__
    void finish();

    __host__
    void write();
    

private:
    /* sorting & cell list */
    int *particle_cellhash_host, *particle_cellhash_device,
        *particle_index_host, *particle_index_device,
        *sortback_index_host, *sortback_index_device;


};