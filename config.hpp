#ifndef CONFIG_H
#define CONFIG_H

#define NX 256.0
#define NY 256.0
#define NZ 256.0
#define NPTS 256.0
#define GRID_SIZE NX*NY*NZ
#define FFT_GRID_SIZE NX*NY*(NZ/2 + 1)

#define THREADS_PER_BLOCK 64

#define BATCH 10

#define RANK 1



#endif