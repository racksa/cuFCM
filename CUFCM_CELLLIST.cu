#include <chrono>
#include <cmath>
#include <iostream>
#include <limits.h>
#include <stdint.h>

#include <cub/device/device_radix_sort.cuh>

#include "CUFCM_CELLLIST.cuh"
#include <cufft.h>
#include "util/cuda_util.hpp"
#include "config.hpp"

__device__ __host__
int icell(int Mx, int My, int Mz, int x, int y, int z, uint64_t (*f)(unsigned int, unsigned int, unsigned int, int, int, int)){
	int xi, yi, zi;
	xi = fmodf((x+Mx), Mx);
	yi = fmodf((y+My), My);
	zi = fmodf((z+Mz), Mz);

	return f(xi, yi, zi, Mx, My, Mz);
}
__host__ __device__
uint64_t linear_encode(unsigned int xi, unsigned int yi, unsigned int zi, int Mx, int My, int Mz){
	return xi + (yi + zi*My)*Mx;
}

__device__ __host__
void bulkmap_loop(int* map, int Mx, int My, int Mz, uint64_t (*f)(unsigned int, unsigned int, unsigned int, int, int, int)){
	int imap=0, tempmap=0;
	unsigned int iz = 0, iy = 0, ix = 0;
	for(iz = 0; iz < Mz; iz++){
		for(iy = 0; iy < My; iy++){
			for(ix = 0; ix < Mx; ix++){
				tempmap=icell(Mx, My, Mz, ix, iy, iz, linear_encode);
				imap=tempmap*13;
				map[imap]=icell(Mx, My, Mz, ix+1, iy, iz, f);
				map[imap+1]=icell(Mx, My, Mz, ix+1, iy+1, iz, f);
				map[imap+2]=icell(Mx, My, Mz, ix, iy+1, iz, f);
				map[imap+3]=icell(Mx, My, Mz, ix-1, iy+1, iz, f);
				map[imap+4]=icell(Mx, My, Mz, ix+1, iy, iz-1, f);
				map[imap+5]=icell(Mx, My, Mz, ix+1, iy+1, iz-1, f);
				map[imap+6]=icell(Mx, My, Mz, ix, iy+1, iz-1, f);
				map[imap+7]=icell(Mx, My, Mz, ix-1, iy+1, iz-1, f);
				map[imap+8]=icell(Mx, My, Mz, ix+1, iy, iz+1, f);
				map[imap+9]=icell(Mx, My, Mz, ix+1, iy+1, iz+1, f);
				map[imap+10]=icell(Mx, My, Mz, ix, iy+1, iz+1, f);
				map[imap+11]=icell(Mx, My, Mz, ix-1, iy+1, iz+1, f);
				map[imap+12]=icell(Mx, My, Mz, ix, iy, iz+1, f);
			}
		}
	}
	return;
}

__global__
void create_hash_gpu(int *hash, Real *Y, int N, int Mx, int My, int Mz,
					Real Lx, Real Ly, Real Lz){
	const int index = threadIdx.x + blockIdx.x*blockDim.x;

	if(index < N){
		if(Y[3*index + 0]<0 || Y[3*index + 1]<0 || Y[3*index + 2]<0){
			printf("ERROR particle %d (%.4f %.4f %.4f) not in box\n", 
			index, Y[3*index + 0], Y[3*index + 1], Y[3*index + 2]);
		}

		int xc = int(Y[3*index + 0]/Lx * Mx);
		int yc = int(Y[3*index + 1]/Ly * My);
		int zc = int(Y[3*index + 2]/Lz * Mz);

		hash[index] = xc + (yc + zc*My)*Mx;
	}
	return;
}

__global__
void particle_index_range(int *particle_index, int N){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;

	if(index < N){
		particle_index[index] = index;
	}
    
    return;
}

// template <typename T>
// __global__
// void copy_device(T *from, T *to, int L){
//     const int index = threadIdx.x + blockIdx.x*blockDim.x;
//     const int stride = blockDim.x*gridDim.x;

//     for(int np = index; np < L; np += stride){
//         to[np] = from[np];
//     }
    
//     return;
// }

// template <typename T>
// __global__
// void sort_3d_by_index(int *pindex, T *in, T *aux, int L){
//     const int index = threadIdx.x + blockIdx.x*blockDim.x;
//     const int stride = blockDim.x*gridDim.x;

//     for(int np = index; np < L; np+=stride){

//         in[3*np + 0] = aux[3*pindex[np] + 0];
//         in[3*np + 1] = aux[3*pindex[np] + 1];
//         in[3*np + 2] = aux[3*pindex[np] + 2];
//     }
    
//     return;
// }


void sort_index_by_key(int *key, int *index, int *key_buf_, int *index_buf_, int N){
	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;
    int *key_buf = malloc_device<int>(N);
    int *index_buf = malloc_device<int>(N);

    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, key, key_buf, index, index_buf, N);
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, key, key_buf, index, index_buf, N);

    const int num_thread_blocks_N = (N + FCM_THREADS_PER_BLOCK - 1)/FCM_THREADS_PER_BLOCK;
    copy_device<int><<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(key_buf, key, N);
    copy_device<int><<<num_thread_blocks_N, FCM_THREADS_PER_BLOCK>>>(index_buf, index, N);

    cudaFree(d_temp_storage);
    cudaFree(key_buf);
    cudaFree(index_buf);
}

__global__
void create_cell_list(const int *particle_cellindex, int *cell_start, int *cell_end, int N){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    unsigned int c1, c2;

    for(int np = index; np < N; np+=stride){
        c2 = particle_cellindex[np];
        c1 = particle_cellindex[np-1];

        if(c1 != c2 || np == 0){
            cell_start[c2] = np;
            if(np > 0){
                cell_end[c1] = np;
            }
        }
        if(np == N - 1){
            cell_end[c2] = N;
        }
    }
    
    return;
    
}

__global__
void contact_force(Real* Y, Real *F, Real rad, int N, Real Lx, Real Ly, Real Lz,
                    int *particle_cellindex, int *cell_start, int *cell_end,
                    int *map,
                    int ncell, Real Rcsq,
                    Real Fref){

    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;
    const Real a_sum = 2.0*rad;

    for(int i = index; i < N; i += stride){
        Real fxi = (Real)0.0, fyi = (Real)0.0, fzi = (Real)0.0;
        Real xi = Y[3*i + 0], yi = Y[3*i + 1], zi = Y[3*i + 2];
        int icell = particle_cellindex[i];
        
        /* intra-cell interactions */
        /* corrections only apply to particle i */
        for(int j = cell_start[icell]; j < cell_end[icell]; j++){
            if(i != j){
                Real xij = xi - Y[3*j + 0];
                Real yij = yi - Y[3*j + 1];
                Real zij = zi - Y[3*j + 2];

                xij = xij - Lx * Real(int(xij/(Real(0.5)*Lx)));
                yij = yij - Ly * Real(int(yij/(Real(0.5)*Ly)));
                zij = zij - Lz * Real(int(zij/(Real(0.5)*Lz)));

                Real rijsq=xij*xij+yij*yij+zij*zij;
                if(rijsq < 1.21*a_sum*a_sum){
                    Real rij = sqrt(rijsq);
                    Real chi_fac = Real(10.0)/a_sum;

                    Real fac = fmin(1.0, 1.0 - chi_fac*(rij - a_sum));
                    fac *= Fref*Real(1.0)*(Real(220.0)*Real(1800.0)/(Real(2.2)*Real(2.2)*Real(40.0)*Real(40.0)))*fac*fac*fac;

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

                xij = xij - Lx * Real(int(xij/(Real(0.5)*Lx)));
                yij = yij - Ly * Real(int(yij/(Real(0.5)*Ly)));
                zij = zij - Lz * Real(int(zij/(Real(0.5)*Lz)));

                Real rijsq=xij*xij+yij*yij+zij*zij;
                if(rijsq < 1.21*a_sum*a_sum){
                    
                    Real rij = sqrt(rijsq);
                    Real chi_fac = Real(10.0)/a_sum;

                    Real fac = fmin(1.0, 1.0 - chi_fac*(rij - a_sum));
                    fac *= Fref*Real(1.0)*(Real(220.0)*Real(1800.0)/(Real(2.2)*Real(2.2)*Real(40.0)*Real(40.0)))*fac*fac*fac;

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

        atomicAdd(&F[3*i + 0], fxi);
        atomicAdd(&F[3*i + 1], fyi);
        atomicAdd(&F[3*i + 2], fzi);

        return;
    }
}

__global__
void check_overlap_gpu(Real *Y, Real rad, int N, Real Lx, Real Ly, Real Lz,
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

                xij = xij - Lx * Real(int(xij/(Real(0.5)*Lx)));
                yij = yij - Ly * Real(int(yij/(Real(0.5)*Ly)));
                zij = zij - Lz * Real(int(zij/(Real(0.5)*Lz)));

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
                if(i != j){          
                    Real xij = xi - Y[3*j + 0];
                    Real yij = yi - Y[3*j + 1];
                    Real zij = zi - Y[3*j + 2];

                    xij = xij - Lx * Real(int(xij/(Real(0.5)*Lx)));
                    yij = yij - Ly * Real(int(yij/(Real(0.5)*Ly)));
                    zij = zij - Lz * Real(int(zij/(Real(0.5)*Lz)));
                    Real rijsq=xij*xij+yij*yij+zij*zij;
                    if(rijsq < Rcsq){
                        if (rijsq < 3.98*rad*rad){
                            printf("ERROR: Overlap between %d (%.4f %.4f %.4f) in %d and %d (%.4f %.4f %.4f) in %d. sep = %.6f 2rad = %.6f  \n", 
                            i, xi, yi, zi, icell, j, Y[3*j + 0], Y[3*j + 1], Y[3*j + 2], jcell, sqrt(rijsq), (2*rad));
                        }
                    }
                }
            }
        }

        return;
    }
}