#pragma once

#include <chrono>
#include <cmath>
#include <iostream>
#include <limits.h>
#include <stdint.h>

#include <cub/device/device_radix_sort.cuh>

#include <cufft.h>
#include "../config.hpp"

__global__
void create_hash_gpu(int *hash, Real *Y, int N, Real dx, uint64_t (*f)(unsigned int, unsigned int, unsigned int, int)){
	const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

	int xc, yc, zc;

	for(int np = index; np < N; np += stride){
		xc = (int) (Y[3*np + 0]/dx);
		yc = (int) (Y[3*np + 1]/dx);
		zc = (int) (Y[3*np + 2]/dx);

		hash[np] = xc + (yc + zc*NX)*NX;
	}
	return;
}

void create_hash(int *hash, Real *Y, int N, Real dx, uint64_t (*f)(unsigned int, unsigned int, unsigned int, int)){
	int xc, yc, zc;

	for(int np = 0; np < N; np++){
		xc = (int) (Y[3*np + 0]/dx);
		yc = (int) (Y[3*np + 1]/dx);
		zc = (int) (Y[3*np + 2]/dx);

		hash[np] = xc + (yc + zc*NX)*NX;
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

template <typename T>
__global__
void copy_device(T *from, T *to, int L){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    for(int np = index; np < L; np += stride){
        to[np] = from[np];
    }
    
    return;
}

template <typename T>
__global__
void sort_3d_by_index(int *pindex, T *in, T *aux, int L){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    for(int np = index; np < L; np+=stride){

        in[3*np + 0] = aux[3*pindex[np] + 0];
        in[3*np + 1] = aux[3*pindex[np] + 1];
        in[3*np + 2] = aux[3*pindex[np] + 2];
    }
    
    return;
}


void sort_index_by_key(int *key, int *index, int N){
	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;

    int* key_buf = malloc_device<int>(N);
    int* index_buf = malloc_device<int>(N);

    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, key, key_buf, index, index_buf, N);
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, key, key_buf, index, index_buf, N);

    const int num_thread_blocks_N = (N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
    copy_device<int><<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(key_buf, key, N);
    copy_device<int><<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(index_buf, index, N);

}

__global__
void create_cell_list(int *particle_hash, int *cell_start, int *cell_end, int N, int ncell){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    for(int np = index; np < N; np+=stride){

    }
    
    return;
    
}

	
