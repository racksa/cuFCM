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
void init_drag(Real *F, Real rad, int N, Real Fref, curandState *states){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    int seed = index + clock64(); // different seed per thread
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
void decrease_drag(Real *F, Real fac, int N){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    for(int i = index; i < N; i += stride){
        F[3*i + 0] = F[3*i + 0]*fac;
        F[3*i + 1] = F[3*i + 0]*fac;
        F[3*i + 2] = F[3*i + 0]*fac;
    }
    return ;
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

    init_pos_lattice<<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(Y_device, N, Lx, Ly, Lz);

    init_drag<<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(F_device, rh, N, Fref, dev_random);

    write();

}


__host__
void random_packer::init_cuda(){
    N = pars.N;
    rh = pars.rh;
    dt = pars.dt;
    prompt = pars.prompt;
    Fref = pars.Fref;
    boxsize = pars.boxsize;
    nx = pars.nx;
    ny = pars.ny;
    nz = pars.nz;

    /* Neighbour list */
    Rc = 6*rh;
    Rcsq = Rc*Rc;

    Lx = boxsize;
    Ly = boxsize/Real(nx)*Real(ny);
    Lz = boxsize/Real(nx)*Real(nz);
    int Lmin = std::min(std::min(Lx, Ly), Lz);
    int Lmax = std::max(std::max(Lx, Ly), Lz);
    Mx = std::max((Lx/Rc), Real(3.0));
    My = std::max((Ly/Rc), Real(3.0));
    Mz = std::max((Lz/Rc), Real(3.0));
    
    
    // int Mmin = Lmin/Rc;
    // if(Mmin < 3){
    //     Mmin = 3;
    // }
    // Mx = Mmin*int(Lx/Lmin);
    // My = Mmin*int(Ly/Lmin);
    // Mz = Mmin*int(Lz/Lmin);
    // if(Mx%Mmin!=0 || My%Mmin!=0 || Mz%Mmin!=0){
    //     std::cout<< "Fatal ERROR : box dimension not divisible"<<std::endl;
    // }

    ncell = Mx*My*Mz;
    mapsize = 13*ncell;

    if(prompt > -1){
		std::cout << "-------\nSimulation\n-------\n";
		std::cout << "Particle number:\t" << N << "\n";
		std::cout << "Particle radius:\t" << rh << "\n";
        std::cout << "boxsize:\t\t" << "(" << Lx << " " << Ly << " " << Lz << ")\n";
        std::cout << "Volume fraction:\t" << (4./3*PI*rh*rh*rh*N/(boxsize*boxsize*boxsize)) << "\n";
        std::cout << "Grid points:\t\t" << "(" << nx << " " << ny << " " << nz << ")\n";
		std::cout << "Cell number:\t\t" << "(" << Mx << " " << My << " " << Mz << ")\n";
        std::cout << "Rc/a:\t\t\t" << Rc/rh << "\n";
		
		std::cout << std::endl;
	}


    aux_host = malloc_host<Real>(3*N);					    aux_device = malloc_device<Real>(3*N);
    F_host = malloc_host<Real>(3*N);						F_device = malloc_device<Real>(3*N);
    // init_F_host = malloc_host<Real>(3*N);					init_F_device = malloc_device<Real>(3*N);
    V_host = malloc_host<Real>(3*N);						V_device = malloc_device<Real>(3*N);

    key_buf = malloc_device<int>(N);
    index_buf = malloc_device<int>(N);

    particle_cellhash_host = malloc_host<int>(N);					 particle_cellhash_device = malloc_device<int>(N);
    particle_index_host = malloc_host<int>(N);						 particle_index_device = malloc_device<int>(N);
    sortback_index_host = malloc_host<int>(N);						 sortback_index_device = malloc_device<int>(N);

    cell_start_host = malloc_host<int>(ncell);						 cell_start_device = malloc_device<int>(ncell);
    cell_end_host = malloc_host<int>(ncell);						 cell_end_device = malloc_device<int>(ncell);


	map_host = malloc_host<int>(mapsize);							 map_device = malloc_device<int>(mapsize);

	bulkmap_loop(map_host, Mx, My, Mz, linear_encode);
	copy_to_device<int>(map_host, map_device, mapsize);

	num_thread_blocks_N = (N + FCM_THREADS_PER_BLOCK - 1)/FCM_THREADS_PER_BLOCK;
	
	cudaMalloc((void**)&dev_random, num_thread_blocks_N*FCM_THREADS_PER_BLOCK*sizeof(curandState));

}


__host__
void random_packer::spatial_hashing(){
    ///////////////////////////////////////////////////////////////////////////////
    // Spatial hashing
    ///////////////////////////////////////////////////////////////////////////////

    // Create Hash (i, j, k) -> Hash
    create_hash_gpu<<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(particle_cellhash_device, Y_device, N, Mx, My, Mz, Lx, Ly, Lz);
    particle_index_range<<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(particle_index_device, N);
    
    sort_index_by_key(particle_cellhash_device, particle_index_device, key_buf, index_buf, N);

    // Sort pos/force/torque by particle index
    copy_device<Real><<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(Y_device, aux_device, 3*N);
    sort_3d_by_index<Real><<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(particle_index_device, Y_device, aux_device, N);
    copy_device<Real><<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(F_device, aux_device, 3*N);
    sort_3d_by_index<Real><<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(particle_index_device, F_device, aux_device, N);
    // Find cell starting/ending points
    reset_device<int>(cell_start_device, ncell);
    reset_device<int>(cell_end_device, ncell);
    create_cell_list<<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(particle_cellhash_device, cell_start_device, cell_end_device, N);
    
}

__host__
void random_packer::sort_back(){
    particle_index_range<<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(sortback_index_device, N);
    sort_index_by_key(particle_index_device, sortback_index_device, key_buf, index_buf, N);
    // particle cellhash is not sorted back!!!
    // so need to re-compute.

    copy_device<Real> <<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(Y_device, aux_device, 3*N);
    sort_3d_by_index<Real> <<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(sortback_index_device, Y_device, aux_device, N);
    // copy_device<Real> <<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(F_device, aux_device, 3*N);
    // sort_3d_by_index<Real> <<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(sortback_index_device, F_device, aux_device, N);
}

__host__
void random_packer::update(){

    // copy_device<Real><<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(init_F_device, F_device, 3*N);
    // reset_device<Real> (F_device, 3*N);

    spatial_hashing();

    // copy_to_host<int>(cell_start_device, cell_start_host, ncell);
    // copy_to_host<int>(cell_end_device, cell_end_host, ncell);
    // write_celllist(cell_start_host, cell_end_host, map_host, ncell, Mx, My, Mz, "./data/simulation/celllist_data.dat");

    contact_force<<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(Y_device, F_device, rh, N, Lx, Ly, Lz,
                    particle_cellhash_device, cell_start_device, cell_end_device,
                    map_device,
                    ncell, Rcsq,
                    0.1*Fref);

    compute_stokes<<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(F_device, V_device, rh, N);

    move_forward<<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(Y_device, V_device, dt, N);

    box<<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(Y_device, N, Lx, Ly, Lz);

    sort_back();

    decrease_drag<<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(F_device, 0.98, N);

    
}

__host__
void random_packer::finish(){

    box<<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(Y_device, N, Lx, Ly, Lz);

    spatial_hashing();

    check_overlap_gpu<<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(Y_device, rh, N, Lx, Ly, Lz,
                      particle_cellhash_device, cell_start_device, cell_end_device, 
                      map_device, ncell, Rcsq);

    return;
}

__host__
void random_packer::write(){

    copy_to_host<Real>(Y_device, Y_host, 3*N);

    write_pos(Y_host, rh, N, "data/simulation/spherepacking.dat");
}
