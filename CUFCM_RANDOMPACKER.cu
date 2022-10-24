#include <asm-generic/errno.h>
#include <cstdlib>
#include <iostream>
#include <fstream>

#include "config.hpp"
#include "CUFCM_DATA.cuh"
#include "CUFCM_RANDOMPACKER.cuh"
#include "CUFCM_CELLLIST.cuh"

#include "util/cuda_util.hpp"
#include "util/maths_util.hpp"

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
        Real xij = (Real)0.0, yij = (Real)0.0, zij = (Real)0.0;
        /* intra-cell interactions */
        /* corrections only apply to particle i */
        for(int j = cell_start[icell]; j < cell_end[icell]; j++){
            if(i != j){
                Real xij = xi - Y[3*j + 0];
                Real yij = yi - Y[3*j + 1];
                Real zij = zi - Y[3*j + 2];

                xij = xij - box_size * (Real) ((int) (xij/(box_size/Real(2.0))));
                yij = yij - box_size * (Real) ((int) (yij/(box_size/Real(2.0))));
                zij = zij - box_size * (Real) ((int) (zij/(box_size/Real(2.0))));

                Real rijsq=xij*xij+yij*yij+zij*zij;
                if(rijsq < Rcsq){
                    if (rijsq < rad*rad){
                        printf("ERROR: Overlap between %d and %d within same cell\n", i, j);
                    }
                }
            }
        }
        int jcello = 13*icell;
        /* inter-cell interactions */
        /* corrections apply to both parties in different cells */
        for(int nabor = 0; nabor < 13; nabor++){
            int jcell = map[jcello + nabor];
            // if(i == 79){
            //     printf("icell %d jcell(%d %d)\n",cell_start[icell], cell_end[icell], cell_start[jcell], cell_end[jcell]);
            // }
            for(int j = cell_start[jcell]; j < cell_end[jcell]; j++){                
                xij = xi - Y[3*j + 0];
                yij = yi - Y[3*j + 1];
                zij = zi - Y[3*j + 2];

                xij = xij - box_size * ((Real) ((int) (xij/(box_size/Real(2.0)))));
                yij = yij - box_size * ((Real) ((int) (yij/(box_size/Real(2.0)))));
                zij = zij - box_size * ((Real) ((int) (zij/(box_size/Real(2.0)))));
                Real rijsq=xij*xij+yij*yij+zij*zij;
                if(rijsq < Rcsq){
                    if (rijsq < rad*rad){
                        printf("ERROR: Overlap between %d in %d and %d in %d \n", i, icell, j, jcell);
                    }
                }
            }
        }

        return;
    }
}

__global__
void init_drag(Real *F, Real rad, int N, Real Fref, curandState *states){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    int seed = index; // different seed per thread
    curand_init(seed, index, 0, &states[index]);
    Real rnd1, rnd2, rnd3;

    for(int i = index; i < N; i += stride){
        rnd1 = curand_uniform (&states[index]);
        rnd2 = curand_uniform (&states[index]);
        rnd3 = curand_uniform (&states[index]);

        F[3*i + 0] = Fref*(rnd1 - Real(0.5));
        F[3*i + 1] = Fref*(rnd2 - Real(0.5));
        F[3*i + 2] = Fref*(rnd3 - Real(0.5));

    }
    return ;
}


__global__
void apply_repulsion(Real* Y, Real *F, Real rad, int N, Real box_size,
                    int *particle_cellindex, int *cell_start, int *cell_end,
                    int *map,
                    int ncell, Real Rcsq,
                    Real Fref){

    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;


    for(int i = index; i < N; i += stride){
        Real fxi = (Real)0.0, fyi = (Real)0.0, fzi = (Real)0.0;
        int icell = particle_cellindex[i];
        
        Real xi = Y[3*i + 0], yi = Y[3*i + 1], zi = Y[3*i + 2];
        Real xij = (Real)0.0, yij = (Real)0.0, zij = (Real)0.0;
        /* intra-cell interactions */
        /* corrections only apply to particle i */
        for(int j = cell_start[icell]; j < cell_end[icell]; j++){
            if(i != j){
                Real xij = xi - Y[3*j + 0];
                Real yij = yi - Y[3*j + 1];
                Real zij = zi - Y[3*j + 2];

                xij = xij - box_size * Real(int(xij/(box_size/Real(2.0))));
                yij = yij - box_size * Real(int(yij/(box_size/Real(2.0))));
                zij = zij - box_size * Real(int(zij/(box_size/Real(2.0))));

                Real rijsq=xij*xij+yij*yij+zij*zij;
                if(rijsq < Rcsq){

                    Real rij = sqrtf(rijsq);
                    // Real temp = Rcsq - Real(4.0) * rad * rad;
                    // Real temp2 = (Rcsq - rijsq)/temp;
                    // temp2 = temp2*temp2;

                    // Real fxij = Real(4.0)*Fref*temp2*temp2*xij/(Real(2.0)*rad);
                    // Real fyij = Real(4.0)*Fref*temp2*temp2*yij/(Real(2.0)*rad);
                    // Real fzij = Real(4.0)*Fref*temp2*temp2*zij/(Real(2.0)*rad);

                    Real fxij = Fref*xij/rijsq;
                    Real fyij = Fref*yij/rijsq;
                    Real fzij = Fref*zij/rijsq;

                    fxi += fxij;
                    fyi += fyij;
                    fzi += fzij;

                }
            }
            
        }
        int jcello = 13*icell;
        /* inter-cell interactions */
        /* corrections apply to both parties in different cells */
        for(int nabor = 0; nabor < 13; nabor++){
            int jcell = map[jcello + nabor];
            for(int j = cell_start[jcell]; j < cell_end[jcell]; j++){
                xij = xi - Y[3*j + 0];
                yij = yi - Y[3*j + 1];
                zij = zi - Y[3*j + 2];

                xij = xij - box_size * Real(int(xij/(box_size/Real(2.0))));
                yij = yij - box_size * Real(int(yij/(box_size/Real(2.0))));
                zij = zij - box_size * Real(int(zij/(box_size/Real(2.0))));

                Real rijsq=xij*xij+yij*yij+zij*zij;
                if(rijsq < Rcsq){

                    Real rij = sqrtf(rijsq);
                    // Real temp = Rcsq - Real(4.0) * rad * rad;
                    // Real temp2 = (Rcsq - rijsq)/temp;
                    // temp2 = temp2*temp2;

                    // Real fxij = Real(4.0)*Fref*temp2*temp2*xij/(Real(2.0)*rad);
                    // Real fyij = Real(4.0)*Fref*temp2*temp2*yij/(Real(2.0)*rad);
                    // Real fzij = Real(4.0)*Fref*temp2*temp2*zij/(Real(2.0)*rad);

                    Real fxij = Fref*xij/rijsq;
                    Real fyij = Fref*yij/rijsq;
                    Real fzij = Fref*zij/rijsq;

                    fxi += fxij;
                    fyi += fyij;
                    fzi += fzij;

                    atomicAdd(&F[3*j + 0], -fxij);
                    atomicAdd(&F[3*j + 1], -fyij);
                    atomicAdd(&F[3*j + 2], -fzij);
                }
            }
        }
        atomicAdd(&F[3*i + 0], fxi);
        atomicAdd(&F[3*i + 1], fyi);
        atomicAdd(&F[3*i + 2], fzi);

        return;
    }
}

__global__
void compute_stokes(Real *F, Real *V, Real rad, int N){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    for(int i = index; i < N; i += stride){
        // V[3*i + 0] = Real(1.0)/(Real(6.0)*PI*rad) * F[3*i + 0];
        // V[3*i + 1] = Real(1.0)/(Real(6.0)*PI*rad) * F[3*i + 1];
        // V[3*i + 2] = Real(1.0)/(Real(6.0)*PI*rad) * F[3*i + 2];
        V[3*i + 0] = F[3*i + 0];
        V[3*i + 1] = F[3*i + 1];
        V[3*i + 2] = F[3*i + 2];
    }
}

__global__
void move_forward(Real *Y, Real *V, Real dt, int N){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    for(int i = index; i < N; i += stride){
        Y[3*i + 0] += V[3*i + 0] * dt;
        Y[3*i + 1] += V[3*i + 1] * dt;
        Y[3*i + 2] += V[3*i + 2] * dt;
    }

}


__host__
random_packer::random_packer(Real *Y_host_input, Real *Y_device_input, Random_Pars pars_input){

    Y_host = Y_host_input;
    Y_device = Y_device_input;
    pars = pars_input;

    init_cuda();

    init_pos_lattice<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(Y_device, N, boxsize);

    spatial_hashing();

    init_drag<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(init_F_device, rh, N, Fref, dev_random);

    write();

}


__host__
void random_packer::init_cuda(){
    N = pars.N;
    rh = pars.rh;
    dt = pars.dt;
    Fref = pars.Fref;
    boxsize = pars.boxsize;

    /* Neighbour list */
    Rc = 4*rh;
    Rcsq = Rc*Rc;
    M = (int) (boxsize/Rc);
    if(M < 3){
        M = 3;
    }
    cellL = boxsize / (Real)M;
    ncell = M*M*M;
    mapsize = 13*ncell;

    aux_host = malloc_host<Real>(3*N);					    aux_device = malloc_device<Real>(3*N);
    F_host = malloc_host<Real>(3*N);						F_device = malloc_device<Real>(3*N);
    init_F_host = malloc_host<Real>(3*N);					init_F_device = malloc_device<Real>(3*N);
    V_host = malloc_host<Real>(3*N);						V_device = malloc_device<Real>(3*N);

    particle_cellhash_host = malloc_host<int>(N);					 particle_cellhash_device = malloc_device<int>(N);
    particle_index_host = malloc_host<int>(N);						 particle_index_device = malloc_device<int>(N);
    sortback_index_host = malloc_host<int>(N);						 sortback_index_device = malloc_device<int>(N);

    cell_start_host = malloc_host<int>(ncell);						 cell_start_device = malloc_device<int>(ncell);
    cell_end_host = malloc_host<int>(ncell);						 cell_end_device = malloc_device<int>(ncell);


	map_host = malloc_host<int>(mapsize);							 map_device = malloc_device<int>(mapsize);

	bulkmap_loop(map_host, M, linear_encode);
	copy_to_device<int>(map_host, map_device, mapsize);

	num_thread_blocks_N = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
	
	cudaMalloc((void**)&dev_random, num_thread_blocks_N*THREADS_PER_BLOCK*sizeof(curandState));

}


__host__
void random_packer::spatial_hashing(){
    ///////////////////////////////////////////////////////////////////////////////
    // Spatial hashing
    ///////////////////////////////////////////////////////////////////////////////

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
    // Find cell starting/ending points
    create_cell_list<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(particle_cellhash_device, cell_start_device, cell_end_device, N);
    
}


__host__
void random_packer::update(){

    copy_device<Real><<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(init_F_device, F_device, 3*N);

    apply_repulsion<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(Y_device, F_device, rh, N, boxsize,
                    particle_cellhash_device, cell_start_device, cell_end_device,
                    map_device,
                    ncell, Rcsq,
                    Fref);

    compute_stokes<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(F_device, V_device, rh, N);

    move_forward<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(Y_device, V_device, dt, N);

    box<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(Y_device, N, boxsize);
}

__host__
void random_packer::finish(){

    box<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(Y_device, N, boxsize);

    check_overlap_gpu<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(Y_device, rh, N, boxsize,
                      particle_cellhash_device, cell_start_device, cell_end_device, 
                      map_device, ncell, Rcsq);

    return;
}

__host__
void random_packer::write(){

    copy_to_host<Real>(Y_device, Y_host, 3*N);

    write_pos(Y_host, rh, N, "data/simulation/spherepacking.dat");
}
