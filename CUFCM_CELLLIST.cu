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
int icell(int M, int x, int y, int z, uint64_t (*f)(unsigned int, unsigned int, unsigned int, int)){
	int xi, yi, zi;
	xi = fmodf((x+M), M);
	yi = fmodf((y+M), M);
	zi = fmodf((z+M), M);

	return f(xi, yi, zi, M);
}
__host__ __device__
uint64_t linear_encode(unsigned int xi, unsigned int yi, unsigned int zi, int M){
	return xi + (yi + zi*M)*M;
}

__device__ __host__
void bulkmap_loop(int* map, int M, uint64_t (*f)(unsigned int, unsigned int, unsigned int, int)){
	int imap=0, tempmap=0;
	unsigned int iz = 0, iy = 0, ix = 0;
	for(iz = 0; iz < M; iz++){
		for(iy = 0; iy < M; iy++){
			for(ix = 0; ix < M; ix++){
				// printf("\t---------bulkmap(%d %d %d)---------\n", ix, iy, iz);
				tempmap=icell(M, ix, iy, iz, linear_encode);
				imap=tempmap*13;
				map[imap]=icell(M, ix+1, iy, iz, f);
				map[imap+1]=icell(M, ix+1, iy+1, iz, f);
				map[imap+2]=icell(M, ix, iy+1, iz, f);
				map[imap+3]=icell(M, ix-1, iy+1, iz, f);
				map[imap+4]=icell(M, ix+1, iy, iz-1, f);
				map[imap+5]=icell(M, ix+1, iy+1, iz-1, f);
				map[imap+6]=icell(M, ix, iy+1, iz-1, f);
				map[imap+7]=icell(M, ix-1, iy+1, iz-1, f);
				map[imap+8]=icell(M, ix+1, iy, iz+1, f);
				map[imap+9]=icell(M, ix+1, iy+1, iz+1, f);
				map[imap+10]=icell(M, ix, iy+1, iz+1, f);
				map[imap+11]=icell(M, ix-1, iy+1, iz+1, f);
				map[imap+12]=icell(M, ix, iy, iz+1, f);
			}
		}
	}
	return;
}

__global__
void create_hash_gpu(int *hash, Real *Y, int N, Real dx, int M){
	const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

	for(int np = index; np < N; np += stride){
		if(Y[3*np + 0]<0 || Y[3*np + 1]<0 || Y[3*np + 2]<0){
			printf("\nERROR position\n\n");
		}
		int xc = (int) (Y[3*np + 0]/dx);
		int yc = (int) (Y[3*np + 1]/dx);
		int zc = (int) (Y[3*np + 2]/dx);

		hash[np] = xc + (yc + zc*M)*M;
	}
	return;
}

__global__
void particle_index_range(int *particle_index, int N){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    for(int np = index; np < N; np+=stride){
        particle_index[np] = np;
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


void sort_index_by_key(int *key, int *index, int *key_buf, int *index_buf, int N){
	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;
    // int *key_buf = malloc_device<int>(N);
    // int *index_buf = malloc_device<int>(N);

    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, key, key_buf, index, index_buf, N);
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, key, key_buf, index, index_buf, N);

    const int num_thread_blocks_N = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
    copy_device<int><<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(key_buf, key, N);
    copy_device<int><<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(index_buf, index, N);

    cudaFree(d_temp_storage);
    // cudaFree(key_buf);
    // cudaFree(index_buf);
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
            cell_end[c2] = np;
        }
    }
    
    return;
    
}