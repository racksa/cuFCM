#pragma once
#include "config.hpp"
#include <curand_kernel.h>
#include <curand.h>
#include <string>
#include <vector>

void read_init_data(Real *Y, int N, const char *initpos_file_name);

void read_validate_data(Real *Y, Real *F, Real *T, Real *V, Real *W, int N, const char *file_name);

void write_pos(Real *Y, Real rh, int N, const char *file_name);

void write_data(Real *Y, Real *F, Real *T, Real *V, Real *W, int N, const char *file_name, const char *mode);

void write_init_data(Real *Y, Real *F, Real *T, int N);

void write_time(Real time_cuda_initialisation, 
                    Real time_readfile,
                    Real time_hashing,
                    Real time_spreading,
                    Real time_FFT,
                    Real time_gathering,
                    Real time_correction,
                    Real time_compute,
                    const char *file_name);

void write_error(Real Verror,
                 Real Werror,
                 const char *file_name);

void read_config(Real *values, std::vector<std::string>& datafile_names, const char *file_name);

__global__
void init_pos_random_overlapping(Real *Y, int N, Real boxsize, curandState *states);

__global__
void init_pos_lattice(Real *Y, int N, Real boxsize);

__global__
void interleaved2separate(Real *F_device_interleave,
                          Real *F_device, Real *T_device, int N);

__global__
void separate2interleaved(Real *F_device_interleave,
                          Real *F_device, Real *T_device, int N);

void init_random_force(Real *F, Real rad, int N);

__global__
void init_force_kernel(Real *F, Real rad, int N, curandState *states);

__global__
void box(Real *Y, int N, Real box_size);

__global__
void check_nan_in(Real* arr, int L, bool* result);

