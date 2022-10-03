#pragma once

#include <chrono>
#include <cmath>
#include <iostream>
#include <limits.h>
#include <stdint.h>

#include <cufft.h>
#include "../config.hpp"


///////////////////////////////////////////////////////////////////////////////
// Linklist
///////////////////////////////////////////////////////////////////////////////
/* Hash functions */

__host__ __device__
inline uint64_t morton_encode_for(unsigned int x, unsigned int y, unsigned int z, int M){
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



/* Linklist functions */

inline void link_loop(int *list, int *head, Real *Y, int M, int N, uint64_t (*f)(unsigned int, unsigned int, unsigned int, int)){
	uint64_t index;
	int ncell = M*M*M;
	unsigned int xi, yi, zi;
	Real xr=0.0, yr=0.0, zr=0.0;
	Real LX = PI2, celli;
	for(int i = 0; i<ncell; i++){
		head[i]=-1;
	}
	celli = (Real) (M);
	for(int j = 0; j < N; j++){

		xr = Y[3*j + 0];
		yr = Y[3*j + 1];
		zr = Y[3*j + 2];

		xi = (int)((xr/LX)*celli);
		yi = (int)((yr/LX)*celli);
		zi = (int)((zr/LX)*celli);
		
		index = f(xi, yi, zi, M);
		list[j] = head[index];
		head[index] = j;

	}
	return;
}
