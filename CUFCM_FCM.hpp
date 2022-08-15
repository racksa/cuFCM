#pragma once

void cufcm_force_distribution_loop(cufftReal* fx, cufftReal* fy, cufftReal* fz);

__global__
void cufcm_force_distribution(cufftReal* fx, cufftReal* fy, cufftReal* fz);

__global__
void cufcm_flow_solve(cufftComplex* fk_x, cufftComplex* fk_y, cufftComplex* fk_z,
                      cufftComplex* uk_x, cufftComplex* uk_y, cufftComplex* uk_z,
                      double* q, double* qpad, double* qsq, double* qpadsq);

__global__
void normalise_array(cufftReal* ux, cufftReal* uy, cufftReal* uz);


int index_3to1(int i, int j, int k);