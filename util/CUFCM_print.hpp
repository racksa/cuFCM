// #pragma once

// #include <chrono>
// #include <cmath>
// #include <iostream>
// #include <limits.h>
// #include <stdint.h>

// #include <cufft.h>
// #include "../config.hpp"


///////////////////////////////////////////////////////////////////////////////
// Print
///////////////////////////////////////////////////////////////////////////////
// template <typename T>
// void print_host_data_real(T* host_data){
//     printf("NOT IMPLEMENTED ERROR");
//     printf("[");
// 	for(int i = 0; i < NX; i++){
// 		printf("[");
// 		for(int j = 0; j < NY; j++){
// 			printf("[");
// 			for(int k = 0; k < NZ; k++){
// 				const int index = i*NY*NZ + j*NZ + k;
//                 if(std::is_same<T, myCufftReal>::value){
//                     printf("%.8f,\t", host_data[index]);
//                 }
// 			}
//             if(!(j == NY-1)){
//                 printf("],\n");
//             }
//             else
// 			    printf("]");
// 		}
// 		if(!(i == NX-1))
//             printf("],\n\n");
//         else
//             printf("]");
// 	}
// 	printf("]\n");
// }

// template <typename T>
// void print_host_data_real_3D_indexstyle(T* host_data1, T* host_data2, T* host_data3){
// 	for(int k = 0; k < NZ; k++){
// 		for(int j = 0; j < NY; j++){
// 			for(int i = 0; i < NX; i++){
// 				const int index = i + j*NX + k*NY*NX;
//                 if(std::is_same<T, myCufftReal>::value){
//                     printf("(%d %d %d) %d (%.8f %.8f %.8f) \n", i, j, k, index, host_data1[index], host_data2[index], host_data3[index]);
//                 }
// 			}
// 		}
// 	}
// }

// template <typename T>
// void print_host_data_complex(T* host_data){
//     printf("NOT IMPLEMENTED ERROR");
//     printf("[");
// 	for(int i = 0; i < NX; i++){
// 		printf("[");
// 		for(int j = 0; j < NY; j++){
// 			printf("[");
// 			for(int k = 0; k < (NZ/2+1); k++){
// 				const int index = i*NY*NZ + j*NZ + k;
// 				if(std::is_same<T, myCufftComplex>::value){
// 					printf("%.8f + i%.8f,\t", host_data[index].x, host_data[index].y);
// 				}
// 			}
//             if(!(j == NY-1)){
//                 printf("],\n");
//             }
//             else
// 			    printf("]");
// 		}
// 		if(!(i == NX-1))
//             printf("],\n\n");
//         else
//             printf("]");
// 	}
// 	printf("]\n");
// }

// template <typename T>
// void print_host_data_complex_3D_indexstyle(T* host_data1, T* host_data2, T* host_data3){
// 	for(int k = 0; k < NZ; k++){
// 		for(int j = 0; j < NY; j++){
// 			for(int i = 0; i < (NX/2+1); i++){
// 				const int index = i + j*(NX/2+1) + k*NY*(NX/2+1);
// 				if(std::is_same<T, myCufftComplex>::value){
// 					printf("(%d %d %d) %d (%.2f %.2f) (%.2f %.2f) (%.2f %.2f) \n", i, j, k, index, host_data1[index].x, host_data1[index].y, 
// 																						host_data2[index].x, host_data2[index].y, 
// 																						host_data3[index].x, host_data3[index].y);
// 				}
// 			}
// 		}
// 	}
// }

// template <typename T>
// void print_host_data_real_3D_flat(T* host_data, int N, int L){
// 	for(int np = 0; np < N; np++){
// 		printf("%d ( ", np);
// 		for(int l = 0; l < L; l++){
// 			printf("%.8f ", host_data[L*np + l]);
// 		}
// 		printf(")\n");
// 	}
// }

// template <typename T>
// void print_host_data_int_3D_flat(T* host_data, int N, int L){
// 	for(int np = 0; np < N; np++){
// 		printf("%d ( ", np);
// 		for(int l = 0; l < L; l++){
// 			printf("%d ", host_data[L*np + l]);
// 		}
// 		printf(")\n");
// 	}
// }

// template <typename T>
// void print_host_data_complex_3D_flat(T* host_data, int L1, int L2){
// 	if(std::is_same<T, myCufftComplex>::value){
// 		for(int l1 = 0; l1 < L1; l1++){
// 			printf("%d ( ", l1);
// 			for(int l2 = 0; l2 < L2; l2++){
// 				printf("( %.8f %.8f ) ", host_data[L2*l1 + l2].x, host_data[L2*l1 + l2].y );
// 			}
// 			printf(")\n");
// 		}
// 	}
// }