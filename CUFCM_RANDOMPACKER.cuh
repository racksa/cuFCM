#pragma once
#include "config.hpp"

__global__
void check_overlap_gpu(Real *Y, Real rad, int N, 
                    int *particle_cellindex, int *cell_start, int *cell_end,
                    int *map,
                    int ncell, Real Rrefsq);

__global__
void box(Real *Y, int N);

__global__
void apply_drag(Real *Y, Real *F, Real rad, int N, Real Fref, curandState *states);

__global__
void apply_repulsion(Real* Y, Real *F, Real rad, int N,
                    int *particle_cellindex, int *cell_start, int *cell_end,
                    int *map,
                    int ncell, Real Rrefsq,
                    Real Fref);

__global__
void move_forward(Real *Y, Real *V, Real dt, int N);

class random_packer{

public:
    int N;
    Real rh;
    Real values[100];
    int *overlap_counter;

    Real *aux_host, *aux_device,
         *Y_host, *Y_device,
         *F_host, *F_device,
         *init_F_host, *init_F_device,
	     *V_host, *V_device;

    int M, ncell, mapsize;
    int *map_host, *map_device;
    
	int *cell_start_host, *cell_start_device,
        *cell_end_host, *cell_end_device;

    Real dt, Rc, Rcsq, Fref;

    Real cellL;

    int num_thread_blocks_N;
    curandState *dev_random;


    __host__
    random_packer(Real *Y_host_input, Real *Y_device_input, int N_input);

    __host__
    void init_cuda();
    
    __host__
    void spatial_hashing();

    __host__
    void repulsive_force();

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