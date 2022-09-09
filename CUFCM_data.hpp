#pragma once
#include "config.hpp"
<<<<<<< HEAD
#include <curand_kernel.h>
#include <curand.h>

void read_init_data(Real *Y, int N, const char *initpos_file_name);

__global__
void init_wave_vector(Real *q, Real *qsq, Real *qpad, Real *qpadsq, int nptsh, int pad);

void init_pos(Real *Y, Real rad, int N);

void init_force(Real *F, Real rad, int N);

void init_pos_gpu(Real *Y, Real rad, int N);

__global__
void check_overlap(Real x, Real y, Real z, Real *Y, Real rad, int np, int *check);
=======

void read_init_data(Real *Y, int N, const char *initpos_file_name);
>>>>>>> d63b6cadcf22d8a392c4971f5119fad9eae22dd3
