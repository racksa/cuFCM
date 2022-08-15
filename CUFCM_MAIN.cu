#include <iostream>
// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>

#include "cuda_util.hpp"
#include "config.hpp"
#include "CUFCM_FCM.hpp"
#include "CUFCM_util.hpp"

#define pi 3.1415926535

int main(int argc, char** argv) {
    cufftHandle plan, iplan;

	cufftReal* fx_device = malloc_device<cufftReal>(GRID_SIZE);
	cufftReal* fy_device = malloc_device<cufftReal>(GRID_SIZE);
	cufftReal* fz_device = malloc_device<cufftReal>(GRID_SIZE);
    cufftComplex* fk_x_device = malloc_device<cufftComplex>(FFT_GRID_SIZE);
	cufftComplex* fk_y_device = malloc_device<cufftComplex>(FFT_GRID_SIZE);
	cufftComplex* fk_z_device = malloc_device<cufftComplex>(FFT_GRID_SIZE);

	cufftReal* fx_host = malloc_host<cufftReal>(GRID_SIZE);
	cufftReal* fy_host = malloc_host<cufftReal>(GRID_SIZE);
	cufftReal* fz_host = malloc_host<cufftReal>(GRID_SIZE);
    cufftComplex* fk_x_host = malloc_host<cufftComplex>(FFT_GRID_SIZE);
    cufftComplex* fk_y_host = malloc_host<cufftComplex>(FFT_GRID_SIZE);
    cufftComplex* fk_z_host = malloc_host<cufftComplex>(FFT_GRID_SIZE);

	cufftReal* ux_device = malloc_device<cufftReal>(GRID_SIZE);
	cufftReal* uy_device = malloc_device<cufftReal>(GRID_SIZE);
	cufftReal* uz_device = malloc_device<cufftReal>(GRID_SIZE);
    cufftComplex* uk_x_device = malloc_device<cufftComplex>(FFT_GRID_SIZE);
	cufftComplex* uk_y_device = malloc_device<cufftComplex>(FFT_GRID_SIZE);
	cufftComplex* uk_z_device = malloc_device<cufftComplex>(FFT_GRID_SIZE);

	cufftReal* ux_host = malloc_host<cufftReal>(GRID_SIZE);
	cufftReal* uy_host = malloc_host<cufftReal>(GRID_SIZE);
	cufftReal* uz_host = malloc_host<cufftReal>(GRID_SIZE);
    cufftComplex* uk_x_host = malloc_host<cufftComplex>(FFT_GRID_SIZE);
    cufftComplex* uk_y_host = malloc_host<cufftComplex>(FFT_GRID_SIZE);
    cufftComplex* uk_z_host = malloc_host<cufftComplex>(FFT_GRID_SIZE);

	int pad = (NX/2 + 1);
	int nptsh = (NX/2);
	double* q_host = malloc_host<double>(NX);
	double* qpad_host = malloc_host<double>(pad);
	double* qsq_host = malloc_host<double>(NX);
	double* qpadsq_host = malloc_host<double>(pad);
	double* q_device = malloc_device<double>(NX);
	double* qpad_device = malloc_device<double>(pad);
	double* qsq_device = malloc_device<double>(NX);
	double* qpadsq_device = malloc_device<double>(pad);

	for(int i=0; i<NX; i++){
		if(i < nptsh || i == nptsh){
			q_host[i] = (double) i;
		}
		if(i > nptsh){
			q_host[i] = (double) (i - NX);
		}
		qsq_host[i] = q_host[i]*q_host[i];
	}
	
	for(int i=0; i<pad; i++){
		qpad_host[i] = (double) i;
		qpadsq_host[i] = qpad_host[i]*qpad_host[i];
	}
	copy_to_device<double>(q_host, q_device, NX);
	copy_to_device<double>(qpad_host, qpad_device, pad);
	copy_to_device<double>(qsq_host, qsq_device, NX);
	copy_to_device<double>(qpadsq_host, qpadsq_device, pad);

	
	///////////////////////////////////////////////////////////////////////////////
	// Create a 3D FFT plan
	///////////////////////////////////////////////////////////////////////////////
	cufftPlan3d(&plan, NX, NY, NZ, CUFFT_R2C);
	cufftPlan3d(&iplan, NX, NY, NZ, CUFFT_C2R);

	int num_thread_blocks = (GRID_SIZE + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;

	///////////////////////////////////////////////////////////////////////////////
	// Spreading
	///////////////////////////////////////////////////////////////////////////////
	// cufcm_force_distribution<<<num_thread_blocks, THREADS_PER_BLOCK>>>(fx_host, fy_host, fz_host);
	// print_host_data_real_3D_indexstyle<cufftReal>(fx_host, fy_host, fz_host);
	// /* Copy data to device */
	// copy_to_device<cufftReal>(fx_host, fx_device, GRID_SIZE);
	// copy_to_device<cufftReal>(fy_host, fy_device, GRID_SIZE);
	// copy_to_device<cufftReal>(fz_host, fz_device, GRID_SIZE);


	cufcm_force_distribution<<<num_thread_blocks, THREADS_PER_BLOCK>>>(fx_device, fy_device, fz_device);
	
	copy_to_host<cufftReal>(fx_device, fx_host, GRID_SIZE);
	copy_to_host<cufftReal>(fy_device, fy_host, GRID_SIZE);
	copy_to_host<cufftReal>(fz_device, fz_host, GRID_SIZE);
	print_host_data_real_3D_indexstyle(fx_host, fy_host, fz_host);

	///////////////////////////////////////////////////////////////////////////////
	// FFT
	///////////////////////////////////////////////////////////////////////////////
	if (cufftExecR2C(plan, fx_device, fk_x_device) != CUFFT_SUCCESS){
		printf("CUFFT error: ExecR2C Forward failed (fx)\n");
		return 0;	
	}
	if (cufftExecR2C(plan, fy_device, fk_y_device) != CUFFT_SUCCESS){
		printf("CUFFT error: ExecR2C Forward failed (fy)\n");
		return 0;	
	}
	if (cufftExecR2C(plan, fz_device, fk_z_device) != CUFFT_SUCCESS){
		printf("CUFFT error: ExecR2C Forward failed (fz)\n");
		return 0;	
	}

	/* Print FFT result */
	copy_to_host<cufftComplex>(fk_x_device, fk_x_host, FFT_GRID_SIZE);
	copy_to_host<cufftComplex>(fk_y_device, fk_y_host, FFT_GRID_SIZE);
	copy_to_host<cufftComplex>(fk_z_device, fk_z_host, FFT_GRID_SIZE);
	print_host_data_complex_3D_indexstyle(fk_x_host, fk_y_host, fk_z_host);



	///////////////////////////////////////////////////////////////////////////////
	// Solve for the flow
	///////////////////////////////////////////////////////////////////////////////
	cufcm_flow_solve<<<num_thread_blocks, THREADS_PER_BLOCK>>>(fk_x_device, fk_y_device, fk_z_device,
															   uk_x_device, uk_y_device, uk_z_device,
															   q_device, qpad_device, qsq_device, qpadsq_device);
															   
	/* Print Fourier flow result */
	copy_to_host<cufftComplex>(uk_x_device, uk_x_host, FFT_GRID_SIZE);
	copy_to_host<cufftComplex>(uk_y_device, uk_y_host, FFT_GRID_SIZE);
	copy_to_host<cufftComplex>(uk_z_device, uk_z_host, FFT_GRID_SIZE);
	print_host_data_complex_3D_indexstyle(uk_x_host, uk_y_host, uk_z_host);


	///////////////////////////////////////////////////////////////////////////////
	// IFFT
	///////////////////////////////////////////////////////////////////////////////
	if (cufftExecC2R(iplan, uk_x_device, ux_device) != CUFFT_SUCCESS){
		printf("CUFFT error: ExecR2C Backward failed (fx)\n");
		return 0;	
	}
	if (cufftExecC2R(iplan, uk_y_device, uy_device) != CUFFT_SUCCESS){
		printf("CUFFT error: ExecR2C Backward failed (fy)\n");
		return 0;	
	}
	if (cufftExecC2R(iplan, uk_z_device, uz_device) != CUFFT_SUCCESS){
		printf("CUFFT error: ExecC2R Backward failed (fz)\n");
		return 0;	
	}

	/* Normalise the result after IFFT */
	// normalise_array<<<num_thread_blocks, THREADS_PER_BLOCK>>>(ux_device, uy_device, uz_device);

	/* Print IFFT result */
	copy_to_host<cufftReal>(ux_device, ux_host, GRID_SIZE);
	copy_to_host<cufftReal>(uy_device, uy_host, GRID_SIZE);
	copy_to_host<cufftReal>(uz_device, uz_host, GRID_SIZE);
	print_host_data_real_3D_indexstyle(ux_host, uy_host, uz_host);



	

	///////////////////////////////////////////////////////////////////////////////
	// Finish
	///////////////////////////////////////////////////////////////////////////////
	cufftDestroy(plan);
	cudaFree(fx_device); cudaFree(fy_device); cudaFree(fz_device); 
	cudaFree(fk_x_device); cudaFree(fk_y_device); cudaFree(fk_z_device);
	cudaFree(ux_device); cudaFree(uy_device); cudaFree(uz_device); 
	cudaFree(uk_x_device); cudaFree(uk_y_device); cudaFree(uk_z_device);

	return 0;
}

