#include</data/hs2216/UAMMD/src/uammd.cuh>
#include<misc/IBM.cuh>


#include <cstdlib>
#include <iostream>
#include <cmath>
// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>
#include <curand_kernel.h>
#include <curand.h>

#include <cub/device/device_radix_sort.cuh>


#include "../config.hpp"
#include "../config_fcm.hpp"
#include "CUFCM_INCORPORATE.hpp"

#include "../util/cuda_util.hpp"
#include "../util/CUFCM_linklist.hpp"
#include "../util/CUFCM_print.hpp"
#include "../util/CUFCM_hashing.hpp"
#include "../util/maths_util.hpp"

using namespace uammd;
using std::make_shared;

//A simple Gaussian kernel compatible with the IBM module.
class Gaussian{
  const real prefactor;
  const real tau;
  const int support;
public:
  Gaussian(real width, int support):
    prefactor(pow(2.0*M_PI*width*width, -0.5)),
    tau(-0.5/(width*width)),
    support(support){}

 __host__ __device__ int3 getMaxSupport(){
   return {support, support, support};
 }

  __device__ int3 getSupport(real3 pos, int3 cell){
    return getMaxSupport();
  }

  __device__ real phi(real r, real3 pos) const{
    // printf("prefactor %.8f\n", prefactor);
    // printf("tau %.8f\n", tau);
    // printf("gauss %.8f\n", prefactor*exp(tau*r*r));
    return prefactor*exp(tau*r*r);
  }
};


int main(){
  auto time_start = get_time();

  auto sys = make_shared<System>();
  using Kernel = Gaussian;
  real L = PI2;
  real hydrodynamicRadius = 0.05;
  real upsampling = 1.5; //\sigma = \alpha * h
  int support = NGD_UAMMD;

  real sigma = hydrodynamicRadius/(sqrt(M_PI));
  real sigmasq = sigma*sigma;
  real anorm = pow(2.0*M_PI*sigmasq, -0.5);
  real anorm2 = 2.0*sigmasq;
  real prefactor = (pow(2.0*M_PI*sigmasq, -0.5));
  real tau = (-0.5/(sigmasq));
  // printf("prefactor ! %.8f\n", prefactor);
  // printf("tau ! %.8f\n", tau);
  // printf("gauss fcm ! %.8f\n", anorm*exp(-PI*PI/4.0/anorm2));
  // printf("gauss uammd ! %.8f\n", prefactor*exp(tau*PI*PI/4.0));


  real h = hydrodynamicRadius/(sqrt(M_PI)*upsampling);
  Grid grid(Box({L,L,L}), make_int3(L/h+1, L/h+1, L/h+1));
  int3 n = grid.cellDim;
  real dx = L/n.x;
  printf("dx %.8f\n", dx);
  printf("n.x %d\n", n.x);
  int numberParticles = 5e6;

  //The Grid will advice a certain ordering of the cell data, in particular, the data for cell(i,j,k) must be
  // at element i+n.x*(j+k*n.y). You may change this behavior (see the docs)
  //Initialize some arbitrary per particle data
  thrust::device_vector<real> markerData(numberParticles);
  thrust::fill(markerData.begin(), markerData.end(), 1.0);
  //The actual container for the positions is irrelevant as long as markerPositions[i] returns the 3d coordinates
  // of particle i. For instance, you may use a zip iterator if you have separated x,y,z arrays, or just pass a pointer to a real4 array in which the fourth component is ignored.
  //The same thing happens for the marker data, as long as it matches the grid data you may spread a real, a real3, or whatever custom iterator.
  thrust::device_vector<real3> markerPositions;
  { //Initialize some arbitrary positions
    std::vector<real3> h_pos(numberParticles);
    std::mt19937 gen(sys->rng().next());
    std::uniform_real_distribution<real> dist(-0.5, 0.5);
    auto rng = [&](){return dist(gen);};
    std::generate(h_pos.begin(), h_pos.end(), [&](){ return make_real3(rng(), rng(), rng())*L;});
    markerPositions = h_pos;
  }
  //Allocate the output grid data
  thrust::device_vector<real> gridData(n.x*n.y*n.z);
  thrust::fill(gridData.begin(), gridData.end(), 0);
  //Some arbitrary parameters for the Gaussian
  real width = hydrodynamicRadius/sqrt(M_PI);
  
  auto kernel = std::make_shared<Kernel>(width, support);
  //IBM is a very lightweight object, so you may recreate it instead of storing it.
  IBM<Kernel> ibm(kernel, grid);
  auto pos_ptr = thrust::raw_pointer_cast(markerPositions.data());
  auto markerData_ptr = thrust::raw_pointer_cast(markerData.data());
  auto gridData_ptr = thrust::raw_pointer_cast(gridData.data());

  ///////////////////////////////////////////////////////////////////////////////
  // UAMMD
  ///////////////////////////////////////////////////////////////////////////////
  cudaDeviceSynchronize();	time_start = get_time();
  ibm.spread(pos_ptr, markerData_ptr, gridData_ptr, numberParticles);
  //We can now go back to the particles, performing the inverse operation (interpolation).
  // thrust::fill(markerData.begin(), markerData.end(), 0);
  // ibm.gather(pos_ptr, markerData_ptr, gridData_ptr, numberParticles);
  cudaDeviceSynchronize();
  auto time_uammd = get_time() - time_start;
  std::cout << "uammd spread time\t" << time_uammd << std::endl;

  const int num_thread_blocks_N = (numberParticles + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;

  ///////////////////////////////////////////////////////////////////////////////
  // CUFCM
  ///////////////////////////////////////////////////////////////////////////////
  Real* Y_host = malloc_host<Real>(3*numberParticles);						Real* Y_device = malloc_device<Real>(3*numberParticles);
  Real* F_host = malloc_host<Real>(numberParticles);						  Real* F_device = malloc_device<Real>(numberParticles);
  Real* F3_host = malloc_host<Real>(3*numberParticles);						Real* F3_device = malloc_device<Real>(3*numberParticles);
  Real* hx_host = malloc_host<Real>(n.x*n.y*n.z);     Real* hx_device = malloc_device<Real>(n.x*n.y*n.z);
  Real* hy_host = malloc_host<Real>(n.x*n.y*n.z);     Real* hy_device = malloc_device<Real>(n.x*n.y*n.z);
  Real* hz_host = malloc_host<Real>(n.x*n.y*n.z);     Real* hz_device = malloc_device<Real>(n.x*n.y*n.z);
  
  for(int i = 0; i<numberParticles; i++){
    real temp = markerData[i];
    real3 temp3 = markerPositions[i];

    F_host[i] = (Real)temp;
    F3_host[3*i] =     (Real)1.0;
    F3_host[3*i + 1] = (Real)1.0;
    F3_host[3*i + 2] = (Real)1.0;

    Y_host[3*i] =     (Real)temp3.x + (Real)PI - (Real)dx*0.5;
    Y_host[3*i + 1] = (Real)temp3.y + (Real)PI - (Real)dx*0.5;
    Y_host[3*i + 2] = (Real)temp3.z + (Real)PI - (Real)dx*0.5;
    if(Y_host[3*i]>PI2){Y_host[3*i]-=PI2;}
    if(Y_host[3*i+1]>PI2){Y_host[3*i+1]-=PI2;}
    if(Y_host[3*i+2]>PI2){Y_host[3*i+2]-=PI2;}
    if(Y_host[3*i]<0){Y_host[3*i]+=PI2;}
    if(Y_host[3*i+1]<0){Y_host[3*i+1]+=PI2;}
    if(Y_host[3*i+2]<0){Y_host[3*i+2]+=PI2;}
    if(i<0){
      std::cout << markerData[i] << "\t" << F_host[i] << std::endl;
      real3 temp3 = markerPositions[i]; 
      std::cout << temp3.x << " " << temp3.y << " " << temp3.z << "\t" 
                << Y_host[3*i] << " " << Y_host[3*i + 1] << " " << Y_host[3*i + 2] << std::endl;
    }
  }

  copy_to_device<Real>(Y_host, Y_device, 3*numberParticles);
  copy_to_device<Real>(F_host, F_device, numberParticles);
  copy_to_device<Real>(F3_host, F3_device, 3*numberParticles);

  cudaDeviceSynchronize();	time_start = get_time();
  cufcm_mono_distribution_single_fx<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(hx_device, Y_device, 
                                              F_device, numberParticles, support,
                                              sigmasq,
                                              anorm, anorm2,
                                              dx, n.x);
  cudaDeviceSynchronize();
  auto time_fcm_shared = get_time() - time_start;
  std::cout << "cufcm spread time (shared)\t" << time_fcm_shared << std::endl;

  reset_device(hx_device, n.x*n.y*n.z);
  cudaDeviceSynchronize();	time_start = get_time();
  cufcm_mono_distribution_single_fx_recompute<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(hx_device, Y_device, 
                                              F_device, numberParticles, support,
                                              sigmasq,
                                              anorm, anorm2,
                                              dx, n.x);
  cudaDeviceSynchronize();
  auto time_fcm_recompute = get_time() - time_start;
  std::cout << "cufcm spread time (recompute)\t" << time_fcm_recompute << std::endl;

  reset_device(hx_device, n.x*n.y*n.z);
  cudaDeviceSynchronize();	time_start = get_time();
  cufcm_mono_distribution_regular_fxfyfz<<<num_thread_blocks_N, THREADS_PER_BLOCK>>>(hx_device, hy_device, hz_device,
                                              Y_device, 
                                              F3_device, numberParticles, support,
                                              sigmasq,
                                              anorm, anorm2,
                                              dx, n.x);
  cudaDeviceSynchronize();
  auto time_fcm_3d = get_time() - time_start;
  std::cout << "cufcm 3d spread time\t" << time_fcm_3d << std::endl;


  copy_to_host<Real>(hx_device, hx_host, n.x*n.y*n.z);
  copy_to_host<Real>(hy_device, hy_host, n.x*n.y*n.z);
  copy_to_host<Real>(hz_device, hz_host, n.x*n.y*n.z);

  Real error = 0;
  for(int t = 0; t<n.x; t++){
    real temp = gridData[t];
    error += abs(hx_host[t] - temp )/abs(temp );

    if(t<10){
      const int k = t/(n.x*n.y);
      const int j = (t - k*n.x*n.y)/n.x;
      const int i = t - k*n.x*n.y - j*n.x;
      printf("[%d %d %d]\t uammd_grid[%d] %.8f \t cufcm_grid[%d] %.8f\n", i, j, k, t, temp, t, hx_host[t]);

    }
  }
  std::cout<< "error\t" << error/(n.x) <<"\t";


  return 0;
}