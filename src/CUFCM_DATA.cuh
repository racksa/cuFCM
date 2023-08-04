#pragma once
#include "config.hpp"
#include <curand_kernel.h>
#include <curand.h>
#include <string>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

void read_init_data(Real *Y, int N, const char *initpos_file_name);

void read_init_data_thrust(thrust::host_vector<Real>& Y, const char *initpos_file_name);

void read_validate_data(Real *Y, Real *F, Real *T, Real *V, Real *W, int N, const char *file_name);

void read_validate_data_thrust(thrust::host_vector<Real>& Y,
                               thrust::host_vector<Real>& F,
                               thrust::host_vector<Real>& T, 
                               thrust::host_vector<Real>& V, 
                               thrust::host_vector<Real>& W,
                               const char *file_name);

Real percentage_error_magnitude_thrust(thrust::host_vector<Real> data, thrust::host_vector<Real> ref_data, int N);

void write_pos(Real *Y, Real rh, int N, const char *file_name);

void write_data(Real *Y, Real *F, Real *T, Real *V, Real *W, int N, const char *file_name, const char *mode);

void write_data_thrust(thrust::host_vector<Real> Y,
                        thrust::host_vector<Real> F,
                        thrust::host_vector<Real> T,
                        thrust::host_vector<Real> V,
                        thrust::host_vector<Real> W, 
                        int N, const char *file_name, const char *mode);

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

void write_celllist(int *cell_start_list, int *cell_end_list, int *map_list, int ncell, int Mx, int My, int Mz, const char *file_name);

void write_flow_field(Real *h, int Ngrid, const char *file_name);

void read_config(Real *values, std::vector<std::string>& datafile_names, const char *file_name);

void parser_config(Real *values, Pars& pars);

__global__
void init_pos_lattice(Real *Y, int N, Real Lx, Real Ly, Real Lz);

__global__
void interleaved2separate(Real *F_device_interleave,
                          Real *F_device, Real *T_device, int N);

__global__
void separate2interleaved(Real *F_device_interleave,
                          Real *F_device, Real *T_device, int N);

void init_random_force(Real *F, Real Fref, Real rad, int N);

__global__
void init_force_kernel(Real *F, Real Fref, int N, curandState *states);

__global__
void box(Real *Y, int N, Real Lx, Real Ly, Real Lz);

__global__
void check_nan_in(Real* arr, int L, bool* result);

__device__
void images(Real& x, Real box_size);