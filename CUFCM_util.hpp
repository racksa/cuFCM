#pragma once

#include <chrono>
#include <cmath>
#include <iostream>
#include <limits.h>
#include <stdint.h>

#include <cufft.h>
#include "config.hpp"


///////////////////////////////////////////////////////////////////////////////
// Print
///////////////////////////////////////////////////////////////////////////////
template <typename T>
void print_host_data_real(T* host_data){
    printf("NOT IMPLEMENTED ERROR");
    printf("[");
	for(int i = 0; i < NX; i++){
		printf("[");
		for(int j = 0; j < NY; j++){
			printf("[");
			for(int k = 0; k < NZ; k++){
				const int index = i*NY*NZ + j*NZ + k;
                if(std::is_same<T, myCufftReal>::value){
                    printf("%.8f,\t", host_data[index]);
                }
			}
            if(!(j == NY-1)){
                printf("],\n");
            }
            else
			    printf("]");
		}
		if(!(i == NX-1))
            printf("],\n\n");
        else
            printf("]");
	}
	printf("]\n");
}

template <typename T>
void print_host_data_real_3D_indexstyle(T* host_data1, T* host_data2, T* host_data3){
	for(int k = 0; k < NZ; k++){
		for(int j = 0; j < NY; j++){
			for(int i = 0; i < NX; i++){
				const int index = i + j*NX + k*NY*NX;
                if(std::is_same<T, myCufftReal>::value){
                    printf("(%d %d %d) %d (%.8f %.8f %.8f) \n", i, j, k, index, host_data1[index], host_data2[index], host_data3[index]);
                }
			}
		}
	}
}

template <typename T>
void print_host_data_complex(T* host_data){
    printf("NOT IMPLEMENTED ERROR");
    printf("[");
	for(int i = 0; i < NX; i++){
		printf("[");
		for(int j = 0; j < NY; j++){
			printf("[");
			for(int k = 0; k < (NZ/2+1); k++){
				const int index = i*NY*NZ + j*NZ + k;
				if(std::is_same<T, myCufftComplex>::value){
					printf("%.8f + i%.8f,\t", host_data[index].x, host_data[index].y);
				}
			}
            if(!(j == NY-1)){
                printf("],\n");
            }
            else
			    printf("]");
		}
		if(!(i == NX-1))
            printf("],\n\n");
        else
            printf("]");
	}
	printf("]\n");
}

template <typename T>
void print_host_data_complex_3D_indexstyle(T* host_data1, T* host_data2, T* host_data3){
	for(int k = 0; k < NZ; k++){
		for(int j = 0; j < NY; j++){
			for(int i = 0; i < (NX/2+1); i++){
				const int index = i + j*(NX/2+1) + k*NY*(NX/2+1);
				if(std::is_same<T, myCufftComplex>::value){
					printf("(%d %d %d) %d (%.2f %.2f) (%.2f %.2f) (%.2f %.2f) \n", i, j, k, index, host_data1[index].x, host_data1[index].y, 
																						host_data2[index].x, host_data2[index].y, 
																						host_data3[index].x, host_data3[index].y);
				}
			}
		}
	}
}

template <typename T>
void print_host_data_real_3D_flat(T* host_data, int N, int L){
	for(int np = 0; np < N; np++){
		printf("%d ( ", np);
		for(int l = 0; l < L; l++){
			printf("%.8f ", host_data[L*np + l]);
		}
		printf(")\n");
	}
}

template <typename T>
void print_host_data_int_3D_flat(T* host_data, int N, int L){
	for(int np = 0; np < N; np++){
		printf("%d ( ", np);
		for(int l = 0; l < L; l++){
			printf("%d ", host_data[L*np + l]);
		}
		printf(")\n");
	}
}

template <typename T>
void print_host_data_complex_3D_flat(T* host_data, int L1, int L2){
	if(std::is_same<T, myCufftComplex>::value){
		for(int l1 = 0; l1 < L1; l1++){
			printf("%d ( ", l1);
			for(int l2 = 0; l2 < L2; l2++){
				printf("( %.8f %.8f ) ", host_data[L2*l1 + l2].x, host_data[L2*l1 + l2].y );
			}
			printf(")\n");
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
// Linklist
///////////////////////////////////////////////////////////////////////////////
/* Hash functions */
uint64_t linear_encode(unsigned int xi, unsigned int yi, unsigned int zi, int M){
	return xi + (yi + zi*M)*M;
}

uint64_t morton_encode_for(unsigned int x, unsigned int y, unsigned int z, int M){
	uint64_t answer = 0;
	for (uint64_t i = 0; i < (sizeof(uint64_t)* CHAR_BIT)/3; ++i) {
		answer |= ((x & ((uint64_t)1 << i)) << 2*i) | ((y & ((uint64_t)1 << i)) << (2*i + 1)) | ((z & ((uint64_t)1 << i)) << (2*i + 2));
	}
	return answer;
}

inline uint64_t splitBy3(unsigned int a){
	uint64_t x = a & 0x1fffff; // we only look at the first 21 bits
	x = (x | x << 32) & 0x1f00000000ffff; // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
	x = (x | x << 16) & 0x1f0000ff0000ff; // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
	x = (x | x << 8) & 0x100f00f00f00f00f; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
	x = (x | x << 4) & 0x10c30c30c30c30c3; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
	x = (x | x << 2) & 0x1249249249249249;
	return x;
}

inline uint64_t mortonEncode_magicbits(unsigned int x, unsigned int y, unsigned int z, int M){
	uint64_t answer = 0;
	answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
	return answer;
}


int icell(int M, int x, int y, int z, uint64_t (*f)(unsigned int, unsigned int, unsigned int, int)){
	int xi, yi, zi;
	xi = fmodf((x+M), M);
	yi = fmodf((y+M), M);
	zi = fmodf((z+M), M);

	return f(xi, yi, zi, M);

	// Real beta;
	// int q;
	// Real celli = (Real) M;
	// xi = fmodf((x+M), M);
	// yi = fmodf((y+M), M);
	// zi = fmodf((z+M), M);
	// beta = xi + (yi + zi*celli)*celli;
	// q = (int)(beta);
	// return q;
}

/* Linklist functions */
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

void link_loop(int *list, int *head, Real *Y, int M, int N, uint64_t (*f)(unsigned int, unsigned int, unsigned int, int)){
	uint64_t index;
	int ncell = M*M*M;
	int i = 0, j = 0;
	unsigned int xi, yi, zi;
	Real xr=0.0, yr=0.0, zr=0.0;
	Real LX = PI2, celli;
	for(i = 0; i<ncell; i++){
		head[i]=-1;
	}
	celli = (Real) (M);
	for(j = 0; j<N; j++){
		xr = Y[3*j + 0];
		yr = Y[3*j + 1];
		zr = Y[3*j + 2];

		xi = (int)((xr/LX)*celli);
		yi = (int)((yr/LX)*celli);
		zi = (int)((zr/LX)*celli);

		// printf("\t---------link(%d %d %d)---------\n", xi, yi, zi);

		// index = index of cell
		// head = head particle of each cell
		// list[j] = returns the particle ahead that is within the same cell as j

		index = f(xi, yi, zi, M);
		list[j] = head[index];
		head[index] = j;
	}
	return;
}

// __global__
// void bulkmap(int* map, int M){
// 	int imap=0, tempmap=0;
// 	int iz = 0, iy = 0, ix = 0;
// 	for(iz = 0; iz < M; iz++){
// 		for(iy = 0; iy < M; iy++){
// 			for(ix = 0; ix < M; ix++){
// 				tempmap=icell(M, ix, iy, iz);
// 				imap=tempmap*13;
// 				map[imap]=icell(M, ix+1, iy, iz);
// 				map[imap+1]=icell(M, ix+1, iy+1, iz);
// 				map[imap+2]=icell(M, ix, iy+1, iz);
// 				map[imap+3]=icell(M, ix-1, iy+1, iz);
// 				map[imap+4]=icell(M, ix+1, iy, iz-1);
// 				map[imap+5]=icell(M, ix+1, iy+1, iz-1);
// 				map[imap+6]=icell(M, ix, iy+1, iz-1);
// 				map[imap+7]=icell(M, ix-1, iy+1, iz-1);
// 				map[imap+8]=icell(M, ix+1, iy, iz+1);
// 				map[imap+9]=icell(M, ix+1, iy+1, iz+1);
// 				map[imap+10]=icell(M, ix, iy+1, iz+1);
// 				map[imap+11]=icell(M, ix-1, iy+1, iz+1);
// 				map[imap+12]=icell(M, ix, iy, iz+1);
// 			}
// 		}
// 	}
// 	return;
// }

// __global__
// void link(int *list, int *head, Real *Y, int M, int ncell, int N){
// 	const int index = threadIdx.x + blockIdx.x*blockDim.x;
//     const int stride = blockDim.x*gridDim.x;

// 	int ind, i;
// 	Real xr=0.0, yr=0.0, zr=0.0;
// 	Real LX = PI2, celli;

// 	for(i = index; i < ncell; i += stride){
// 		head[i] = -1;
// 	}
// 	celli = (Real) (M);

// 	for(int j = index; j < N; j += stride){
// 		xr=Y[3*j + 0];
// 		yr=Y[3*j + 1];
// 		zr=Y[3*j + 2];
// 		// ind = index of cell
// 		// head = head particle of each cell
// 		// list[j] = returns the particle ahead that is within the same cell as j
// 		ind = (int)((xr/LX)*celli) + (int)((yr/LX)*celli)*M + (int)((zr/LX)*celli)*M*M;
// 		list[j] = head[ind];
// 		head[ind] = j;
// 	}
// 	return;
// }


///////////////////////////////////////////////////////////////////////////////
// Particle sorting
///////////////////////////////////////////////////////////////////////////////
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

void swap(int* a, int* b){
    int t = *a;
    *a = *b;
    *b = t;
}

void swap_Y(Real *Y, int i, int j){
    Real t0 = Y[3*i + 0];
    Real t1 = Y[3*i + 1];
    Real t2 = Y[3*i + 2];
    Y[3*i + 0] = Y[3*j + 0];
    Y[3*i + 1] = Y[3*j + 1];
    Y[3*i + 2] = Y[3*j + 2];
    Y[3*j + 0] = t0;
    Y[3*j + 1] = t1;
    Y[3*j + 2] = t2;
}

int partition (int arr[], Real *Y, int low, int high){
    int pivot = arr[high];  // selecting last element as pivot
    int i = (low - 1);  // index of smaller element
 
    for (int j = low; j <= high- 1; j++)
    {
        // If the current element is smaller than or equal to pivot
        if (arr[j] <= pivot)
        {
            i++;    // increment index of smaller element
            swap(&arr[i], &arr[j]);
            swap_Y(Y, i, j);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    swap_Y(Y, (i+1), high);
    return (i + 1);
}

/*  
    a[] is the key array, p is starting index, that is 0, 
    and r is the last index of array.  
*/
void quicksort(int a[], Real *Y, int p, int r){
    if(p < r)
    {
        int q;
        q = partition(a, Y, p, r);
        quicksort(a, Y, p, q-1);
        quicksort(a, Y, q+1, r);
    }
}


int partition_1D (int arr[], int *Y, int low, int high){
    int pivot = arr[high];  // selecting last element as pivot
    int i = (low - 1);  // index of smaller element
 
    for (int j = low; j <= high- 1; j++)
    {
        // If the current element is smaller than or equal to pivot
        if (arr[j] <= pivot)
        {
            i++;    // increment index of smaller element
            swap(&arr[i], &arr[j]);
            swap(&Y[i], &Y[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    swap(&Y[i + 1], &Y[high]);
    return (i + 1);
}

void quicksort_1D(int a[], int *Y, int p, int r){
    if(p < r)
    {
        int q;
        q = partition_1D(a, Y, p, r);
        quicksort_1D(a, Y, p, q-1);
        quicksort_1D(a, Y, q+1, r);
    }
}