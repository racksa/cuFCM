#pragma once
#include "config.hpp"
#include <curand_kernel.h>
#include <curand.h>

void read_init_data(Real *Y, int N, const char *initpos_file_name);

void read_validate_data(Real *Y, Real *F, Real *V, Real *W, int N, const char *file_name);

void write_data(Real *Y, Real *F, Real *V, Real *W, int N, const char *file_name);

void write_init_data(Real *Y, Real *F, Real *T, int N);

void write_time(Real time_cuda_initialisation, 
                    Real time_readfile,
                    Real time_hashing, 
                    Real time_linklist,
                    Real time_precompute_gauss,
                    Real time_spreading,
                    Real time_FFT,
                    Real time_gathering,
                    Real time_correction,
                    Real time_compute,
                    const char *file_name);

void write_error(Real Verror,
                 Real Werror,
                 const char *file_name);

void read_config(Real *values, const char *file_name);

__global__
void init_wave_vector(Real *q, Real *qsq, Real *qpad, Real *qpadsq, int nptsh, int pad, Real nx, Real ny, Real nz);

void init_pos(Real *Y, Real rad, int N);

__global__
void init_pos_random_overlapping(Real *Y, int N, curandState *states);

__global__
void init_pos_lattice_random(Real *Y, Real rad, int N, curandState *states);

void init_force(Real *F, Real rad, int N);

void init_pos_gpu(Real *Y, Real rad, int N);

__global__
void check_overlap(Real x, Real y, Real z, Real *Y, Real rad, int np, int *check);

__global__
void init_force_kernel(Real *F, Real rad, int N, curandState *states);

__global__
void append(Real x, Real y, Real z, Real *Y, int np);
