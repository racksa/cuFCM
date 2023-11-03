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
    const int stride = blockDim.x*gridDim.x;

	for(int np = index; np < N; np+=stride){
		

		int xc = int(Y[3*np + 0]/Lx * Mx);
		int yc = int(Y[3*np + 1]/Ly * My);
		int zc = int(Y[3*np + 2]/Lz * Mz);

		hash[np] = xc + (yc + zc*My)*Mx;
	}
	return;
}

__global__
void verify_hash_gpu(int *hash, Real *Y, int N, int Mx, int My, int Mz,
					Real Lx, Real Ly, Real Lz){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

	for(int np = index; np < N; np+=stride){
        if(Y[3*np + 0]<0.0f || Y[3*np + 1]<0.0f || Y[3*np + 2]<0.0f || 
           Y[3*np + 0]>=Lx || Y[3*np + 1]>=Ly || Y[3*np + 2]>=Lz){
			printf("-------- create hash,\
                    particle %d (%.8f %.8f %.8f) not in box\n", 
			np, Y[3*np + 0], Y[3*np + 1], Y[3*np + 2]);
		}

		int xc = int(Y[3*np + 0]/Lx * Mx);
		int yc = int(Y[3*np + 1]/Ly * My);
		int zc = int(Y[3*np + 2]/Lz * Mz);

        if(hash[np] > Mx*My*Mz){
            printf("-------- verify hash,\
                     particle %d (%.8f %.8f %.8f) in cell %d (%d %d %d)\n", 
			np, Y[3*np + 0], Y[3*np + 1], Y[3*np + 2],
            hash[np], xc, yc, zc);
        }
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
        if(np > 0){
            c1 = particle_cellindex[np-1];
        }
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
void verify_cell_list(const int *particle_cellindex, const int *cell_start, const int *cell_end, const Real *Y, 
    int N , int Mx, int My, int Mz, Real Lx, Real Ly, Real Lz){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    unsigned int c1, c2;

    for(int np = index; np < N; np+=stride){
        c2 = particle_cellindex[np];
        if(np > 0){
            c1 = particle_cellindex[np-1];
            if(c1 != c2 && np>0){
                if(cell_start[c2] != cell_end[c1]){
                    Real x2 = Y[3*np];
                    Real y2 = Y[3*np+1];
                    Real z2 = Y[3*np+2];
                    Real x1 = Y[3*np-3];
                    Real y1 = Y[3*np-2];
                    Real z1 = Y[3*np-1];

                    Real xstart = Y[3*cell_start[c2] ];
                    Real ystart = Y[3*cell_start[c2] +1];
                    Real zstart = Y[3*cell_start[c2] +2];
                    Real xend = Y[3*cell_end[c1]-3];
                    Real yend = Y[3*cell_end[c1]-2];
                    Real zend = Y[3*cell_end[c1]-1];


                    int xc2 = int(x2/Lx * Mx);
                    int yc2 = int(y2/Ly * My);
                    int zc2 = int(z2/Lz * Mz);
                    int should_be2 = xc2 + (yc2 + zc2*My)*Mx;

                    int xc1 = int(x1/Lx * Mx);
                    int yc1 = int(y1/Ly * My);
                    int zc1 = int(z1/Lz * Mz);
                    int should_be1 = xc1 + (yc1 + zc1*My)*Mx;

                    int xcend = int(xend/Lx * Mx);
                    int ycend = int(yend/Ly * My);
                    int zcend = int(zend/Lz * Mz);
                    int should_beend = xcend + (ycend + zcend*My)*Mx;


                    int xcstart = int(xstart/Lx * Mx);
                    int ycstart = int(ystart/Ly * My);
                    int zcstart = int(zstart/Lz * Mz);
                    int should_bestart = xcstart + (ycstart + zcstart*My)*Mx;

                    

                    printf("-------- verify cell,\
                    icell[%d]=[%d %d],\
                    icell[%d]=[%d %d],\
                    Y[%d](%.2f %.2f %.2f) in cell [%d] but should be [%d](%d %d %d)\
                    Y[%d](%.2f %.2f %.2f) in cell [%d] but should be [%d](%d %d %d)\
                    Y[%d](%.2f %.2f %.2f) in cell [%d] but should be [%d](%d %d %d)\
                    Y[%d](%.2f %.2f %.2f) in cell [%d] but should be [%d](%d %d %d) \n", 
                    c1, cell_start[c1], cell_end[c1],
                    c2, cell_start[c2], cell_end[c2],
                    cell_end[c1], xend, yend, zend, particle_cellindex[cell_end[c1]], should_beend, xc1, yc1, zc1,
                    cell_start[c2], xstart, ystart, zstart, particle_cellindex[cell_start[c2]], should_bestart, xc2, yc2, zc2,
                    np-1, x1, y1, z1, particle_cellindex[np-1], should_be1, xcend, ycend, zcend, 
                    np, x2, y2, z2, particle_cellindex[np], should_be2, xcstart, ycstart, zcstart
                    );
                }
            }
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
                    fac *= Fref*fac*fac*fac;

                    const double dm1 = 1.0/rij;

                    Real fxij = fac*xij*dm1;
                    Real fyij = fac*yij*dm1;
                    Real fzij = fac*zij*dm1;

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
                    fac *= Fref*fac*fac*fac;

                    const double dm1 = 1.0/rij;

                    Real fxij = fac*xij*dm1;
                    Real fyij = fac*yij*dm1;
                    Real fzij = fac*zij*dm1;

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

    Real forgiving = 0.98; // Typical value is 1.0

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
                    if (rijsq < forgiving*rad*rad){
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
                        if (rijsq < forgiving*rad*rad){
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
