#pragma once

#include <chrono>
#include <cmath>
#include <iostream>

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
                if(std::is_same<T, cufftDoubleReal>::value){
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
                if(std::is_same<T, cufftDoubleReal>::value){
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
				if(std::is_same<T, cufftDoubleComplex>::value){
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
				if(std::is_same<T, cufftDoubleComplex>::value){
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
	if(std::is_same<T, cufftDoubleComplex>::value){
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
__device__ __host__
int icell(int M, int x, int y, int z){
	double beta;
	int q;
	double celli = (double) M;
	beta = fmodf((x+M), M)+fmodf((y+M),M)*celli+fmodf((z+M), M)*celli*celli;
	q = (int)(beta);
	return q;
}

void bulkmap_loop(int* map, int M){
	int imap=0, tempmap=0;
	int iz = 0, iy = 0, ix = 0;
	for(iz = 0; iz < M; iz++){
		for(iy = 0; iy < M; iy++){
			for(ix = 0; ix < M; ix++){
				tempmap=icell(M, ix, iy, iz);
				imap=tempmap*13;
				map[imap]=icell(M, ix+1, iy, iz);
				map[imap+1]=icell(M, ix+1, iy+1, iz);
				map[imap+2]=icell(M, ix, iy+1, iz);
				map[imap+3]=icell(M, ix-1, iy+1, iz);
				map[imap+4]=icell(M, ix+1, iy, iz-1);
				map[imap+5]=icell(M, ix+1, iy+1, iz-1);
				map[imap+6]=icell(M, ix, iy+1, iz-1);
				map[imap+7]=icell(M, ix-1, iy+1, iz-1);
				map[imap+8]=icell(M, ix+1, iy, iz+1);
				map[imap+9]=icell(M, ix+1, iy+1, iz+1);
				map[imap+10]=icell(M, ix, iy+1, iz+1);
				map[imap+11]=icell(M, ix-1, iy+1, iz+1);
				map[imap+12]=icell(M, ix, iy, iz+1);
			}
		}
	}
	return;
}

void link_loop(int *list, int *head, double *Y, int M, int ncell, int N){
	int index = 0, i = 0, j = 0;
	double xr=0.0, yr=0.0, zr=0.0;
	double LX = PI2, celli;
	for(i = 0; i<ncell; i++){
		head[i]=-1;
	}
	celli = (double) (M);
	for(j = 0; j<N; j++){
		xr = Y[3*j + 0];
		yr = Y[3*j + 1];
		zr = Y[3*j + 2];
		// index = index of cell
		// head = head particle of each cell
		// list[j] = returns the particle ahead that is within the same cell as j
		index = (int)((xr/LX)*celli) + (int)((yr/LX)*celli)*M + (int)((zr/LX)*celli)*M*M;
		list[j] = head[index];
		head[index] = j;
	}
	return;
}

__global__
void bulkmap(int* map, int M){
	int imap=0, tempmap=0;
	int iz = 0, iy = 0, ix = 0;
	for(iz = 0; iz < M; iz++){
		for(iy = 0; iy < M; iy++){
			for(ix = 0; ix < M; ix++){
				tempmap=icell(M, ix, iy, iz);
				imap=tempmap*13;
				map[imap]=icell(M, ix+1, iy, iz);
				map[imap+1]=icell(M, ix+1, iy+1, iz);
				map[imap+2]=icell(M, ix, iy+1, iz);
				map[imap+3]=icell(M, ix-1, iy+1, iz);
				map[imap+4]=icell(M, ix+1, iy, iz-1);
				map[imap+5]=icell(M, ix+1, iy+1, iz-1);
				map[imap+6]=icell(M, ix, iy+1, iz-1);
				map[imap+7]=icell(M, ix-1, iy+1, iz-1);
				map[imap+8]=icell(M, ix+1, iy, iz+1);
				map[imap+9]=icell(M, ix+1, iy+1, iz+1);
				map[imap+10]=icell(M, ix, iy+1, iz+1);
				map[imap+11]=icell(M, ix-1, iy+1, iz+1);
				map[imap+12]=icell(M, ix, iy, iz+1);
			}
		}
	}
	return;
}

__global__
void link(int *list, int *head, double *Y, int M, int ncell, int N){
	const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

	int ind, i;
	double xr=0.0, yr=0.0, zr=0.0;
	double LX = PI2, celli;

	for(i = index; i < ncell; i += stride){
		head[i] = -1;
	}
	celli = (double) (M);

	for(int j = index; j < N; j += stride){
		xr=Y[3*j + 0];
		yr=Y[3*j + 1];
		zr=Y[3*j + 2];
		// ind = index of cell
		// head = head particle of each cell
		// list[j] = returns the particle ahead that is within the same cell as j
		ind = (int)((xr/LX)*celli) + (int)((yr/LX)*celli)*M + (int)((zr/LX)*celli)*M*M;
		list[j] = head[ind];
		head[ind] = j;
	}
	return;
}