#ifndef CONFIG_H
#define CONFIG_H

#define PI 3.14159265358979
#define PI2 6.28318530717959
#define PI2sqrt 2.5066282746310002
#define TWOoverPIsqrt 0.7978845608028654
#define PI2sqrt_inv 0.3989422804014327

#define NX 256.0
#define NY 256.0
#define NZ 256.0
#define NPTS 256.0
#define GRID_SIZE (NX*NY*NZ)
#define FFT_GRID_SIZE ((NX/2+1)*NY*NZ)

#define THREADS_PER_BLOCK 64

#define BATCH 10

#define RANK 1

#define HASH_ENCODE_FUNC linear_encode

#endif