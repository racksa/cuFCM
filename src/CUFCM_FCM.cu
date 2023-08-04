#include <cmath>
#include <iostream>
// Include CUDA runtime and CUFFT
#include <cufft.h>

#include "config.hpp"
#include "CUFCM_FCM.cuh"
#include <cub/cub.cuh>


///////////////////////////////////////////////////////////////////////////////
// Fast FCM
///////////////////////////////////////////////////////////////////////////////

__global__
void cufcm_mono_dipole_distribution_bpp_shared_dynamic(myCufftReal *fx, myCufftReal *fy, myCufftReal *fz,
              Real *Y, Real *T, Real *F,
              int N, int ngd, 
              Real sigma, Real sigmadip, Real Sigma,
              Real dx, int nx, int ny, int nz,
              int *particle_index, int start, int end,
              int rotation){
    
    int ngdh = ngd/2;

    extern __shared__ Integer s[];
    Integer *indx_shared = s;
    Integer *indy_shared = (Integer*)&indx_shared[ngd];
    Integer *indz_shared = (Integer*)&indy_shared[ngd];
    Real *gaussx_shared = (Real*)&indz_shared[ngd]; 
    Real *gaussy_shared = (Real*)&gaussx_shared[ngd];
    Real *gaussz_shared = (Real*)&gaussy_shared[ngd];
    #if USE_REGULARFCM
        Real *gaussx_dip_shared = (Real*)&gaussz_shared[ngd]; 
        Real *gaussy_dip_shared = (Real*)&gaussx_dip_shared[ngd];
        Real *gaussz_dip_shared = (Real*)&gaussy_dip_shared[ngd];
        Real *grad_gaussx_dip_shared = (Real*)&gaussz_dip_shared[ngd];
    #else
        Real *xdis_shared = (Real*)&gaussz_shared[ngd];
        Real *ydis_shared = (Real*)&xdis_shared[ngd];
        Real *zdis_shared = (Real*)&ydis_shared[ngd];
        Real *grad_gaussx_dip_shared = (Real*)&zdis_shared[ngd];
    #endif
    Real *grad_gaussy_dip_shared = (Real*)&grad_gaussx_dip_shared[ngd];
    Real *grad_gaussz_dip_shared = (Real*)&grad_gaussy_dip_shared[ngd];

    Real *Y_shared = (Real*)&grad_gaussz_dip_shared[ngd];
    Real *F_shared = (Real*)&Y_shared[3];
    Real *g_shared = (Real*)&F_shared[3];
    
    #if USE_REGULARFCM
        Real sigmasq = sigma*sigma;
        Real sigmadipsq = sigmadip*sigmadip;
        Real Sigmasq = sigmasq;
        Real Sigmadipsq = sigmadipsq;
        Real anorm = Real(1.0)/my_sqrt(Real(2.0)*Real(PI)*sigmasq);
        Real Anorm = anorm;
        Real anormdip = Real(1.0)/my_sqrt(Real(2.0)*Real(PI)*sigmadipsq);
    #else
        Real Sigmasq = Sigma*Sigma;
        Real Sigmadipsq = 0;
        if(rotation==1){
            Sigmadipsq = Sigmasq;
        }
        Real Anorm = Real(1.0)/my_sqrt(Real(PI2)*Sigmasq);
        Real pdmag = sigma*sigma - Sigmasq;
    #endif
    

    for(int np = blockIdx.x; (np < N) && (particle_index[np] >= start && particle_index[np] < end); np += gridDim.x){

        if(threadIdx.x == 0){
            Y_shared[0] = Y[3*np + 0];
            Y_shared[1] = Y[3*np + 1];
            Y_shared[2] = Y[3*np + 2];

            F_shared[0] = F[3*np + 0];
            F_shared[1] = F[3*np + 1];
            F_shared[2] = F[3*np + 2];

            // anti-symmetric G_lk = 0.5*epsilon_lkp*T_p
            if(rotation==1){
                g_shared[0] = + Real(0.0);
                g_shared[1] = + Real(0.0);
                g_shared[2] = + Real(0.0);
                g_shared[3] = + Real(0.5)*T[3*np + 2];
                g_shared[4] = - Real(0.5)*T[3*np + 2];
                g_shared[5] = + Real(-0.5)*T[3*np + 1];
                g_shared[6] = - Real(-0.5)*T[3*np + 1];
                g_shared[7] = + Real(0.5)*T[3*np + 0];
                g_shared[8] = - Real(0.5)*T[3*np + 0];
            }
        }
        __syncthreads();

        for(int i = threadIdx.x; i < 4*ngd; i += blockDim.x){
            Real xg = my_rint(Y_shared[0]/dx) - ngdh + my_fmod(Real(i), Real(ngd));
            Real yg = my_rint(Y_shared[1]/dx) - ngdh + my_fmod(Real(i), Real(ngd));
            Real zg = my_rint(Y_shared[2]/dx) - ngdh + my_fmod(Real(i), Real(ngd));

            Real xx = xg*dx - Y_shared[0];
            Real yy = yg*dx - Y_shared[1];
            Real zz = zg*dx - Y_shared[2];
            /* dis */
            if(i<ngd){
                gaussx_shared[i] = Anorm*my_exp(-xx*xx/(Real(2.0)*Sigmasq));
                gaussy_shared[i] = Anorm*my_exp(-yy*yy/(Real(2.0)*Sigmasq));
                gaussz_shared[i] = Anorm*my_exp(-zz*zz/(Real(2.0)*Sigmasq));
            }
            /* gauss */
            if(i>=ngd && i<2*ngd){
                #if USE_REGULARFCM
                    gaussx_dip_shared[i-ngd] = anormdip*my_exp(-xx*xx/(Real(2.0)*sigmadipsq));
                    gaussy_dip_shared[i-ngd] = anormdip*my_exp(-yy*yy/(Real(2.0)*sigmadipsq));
                    gaussz_dip_shared[i-ngd] = anormdip*my_exp(-zz*zz/(Real(2.0)*sigmadipsq));
                #else
                    xdis_shared[i-ngd] = xx;
                    ydis_shared[i-ngd] = yy;
                    zdis_shared[i-ngd] = zz;
                #endif
            }
            /* grad_gauss */
            if(i>=2*ngd && i<3*ngd){
                if(rotation==1){
                    grad_gaussx_dip_shared[i-2*ngd] = - xx / Sigmadipsq;
                    grad_gaussy_dip_shared[i-2*ngd] = - yy / Sigmadipsq;
                    grad_gaussz_dip_shared[i-2*ngd] = - zz / Sigmadipsq;
                }
            }
            /* ind */
            if(i>=3*ngd){
                indx_shared[i-3*ngd] = xg - nx * my_floor( xg / nx );
                indy_shared[i-3*ngd] = yg - ny * my_floor( yg / ny );
                indz_shared[i-3*ngd] = zg - nz * my_floor( zg / nz );
            }
        }
        __syncthreads();
        
        for(int t = threadIdx.x; t < ngd*ngd*ngd; t += blockDim.x){
            const int k = t/(ngd*ngd);
            const int j = (t - k*ngd*ngd)/ngd;
            const int i = t - k*ngd*ngd - j*ngd;

            Real gradx = 0, grady = 0, gradz = 0;
            if(rotation==1){
                gradx = grad_gaussx_dip_shared[i];
                grady = grad_gaussy_dip_shared[j];
                gradz = grad_gaussz_dip_shared[k];
            }

            int ind = indx_shared[i] + indy_shared[j]*nx + indz_shared[k]*nx*ny;

            #if USE_REGULARFCM
                Real temp = gaussx_shared[i]*gaussy_shared[j]*gaussz_shared[k];
                Real tempdip = gaussx_dip_shared[i]*gaussy_dip_shared[j]*gaussz_dip_shared[k];
            #else
                Real r2 = xdis_shared[i]*xdis_shared[i] + ydis_shared[j]*ydis_shared[j] + zdis_shared[k]*zdis_shared[k];
                Real temp1 = gaussx_shared[i]*gaussy_shared[j]*gaussz_shared[k];
                Real temp2 = Real(0.5) * pdmag / Sigmasq;
                Real temp3 = temp2 / Sigmasq;
                Real temp4 = Real(3.0)*temp2;
                Real temp = temp1*( Real(1.0) + temp3*r2 - temp4);
                Real tempdip = 0;
                if(rotation==1){
                    tempdip = temp1;
                }
            #endif

            Real dipole_x = 0;
            Real dipole_y = 0;
            Real dipole_z = 0;

            if(rotation==1){
               dipole_x = (g_shared[0]*gradx + g_shared[3]*grady + g_shared[5]*gradz)*tempdip;
               dipole_y = (g_shared[4]*gradx + g_shared[1]*grady + g_shared[7]*gradz)*tempdip;
               dipole_z = (g_shared[6]*gradx + g_shared[8]*grady + g_shared[2]*gradz)*tempdip;
            }

            atomicAdd(&fx[ind], F_shared[0]*temp + dipole_x);
            atomicAdd(&fy[ind], F_shared[1]*temp + dipole_y);
            atomicAdd(&fz[ind], F_shared[2]*temp + dipole_z);
        }
    }
}

__global__
void cufcm_flow_solve(myCufftComplex* fk_x, myCufftComplex* fk_y, myCufftComplex* fk_z,
                      myCufftComplex* uk_x, myCufftComplex* uk_y, myCufftComplex* uk_z,
                      int nx, int ny, int nz, Real Lx, Real Ly, Real Lz){
    const int i = threadIdx.x + blockIdx.x*blockDim.x;

    int fft_nx = nx/2 + 1;
    Real grid_size = nx*ny*nz;
    int fft_grid_size = fft_nx*ny*nz;

    if(i < fft_grid_size){
        const int indk = i/(ny*fft_nx);
        const int indj = (i - indk*(ny*fft_nx))/fft_nx;
        const int indi = i - (indj + indk*ny)*fft_nx;
        
        Real q1 = ( (indi <= nx/2)? Real(indi) : Real(indi - nx) ) * (Real(PI2)/Lx);
        Real q2 = ( (indj <= ny/2)? Real(indj) : Real(indj - ny) ) * (Real(PI2)/Ly);
        Real q3 = ( (indk <= nz/2)? Real(indk) : Real(indk - nz) ) * (Real(PI2)/Lz);
        Real qq = q1*q1 + q2*q2 + q3*q3;
        Real qq_inv = (Real)1.0/(qq);

        Real f1_re = fk_x[i].x;
        Real f1_im = fk_x[i].y;
        Real f2_re = fk_y[i].x;
        Real f2_im = fk_y[i].y;
        Real f3_re = fk_z[i].x;
        Real f3_im = fk_z[i].y;

        if(i==0){
            f1_re = (Real)0.0;
            f1_im = (Real)0.0;
            f2_re = (Real)0.0;
            f2_im = (Real)0.0;
            f3_re = (Real)0.0;
            f3_im = (Real)0.0;
        }

        Real kdotf_re = (q1*f1_re+q2*f2_re+q3*f3_re)*qq_inv;
        Real kdotf_im = (q1*f1_im+q2*f2_im+q3*f3_im)*qq_inv;
        Real norm = qq_inv / grid_size;
        
        // printf("ik_x[%d] = [%.4f %.4f]\n", i, uk_x[i].x, uk_y[i].y);

        uk_x[i].x = norm*(f1_re-q1*(kdotf_re));
        uk_x[i].y = norm*(f1_im-q1*(kdotf_im));
        uk_y[i].x = norm*(f2_re-q2*(kdotf_re));
        uk_y[i].y = norm*(f2_im-q2*(kdotf_im));
        uk_z[i].x = norm*(f3_re-q3*(kdotf_re));
        uk_z[i].y = norm*(f3_im-q3*(kdotf_im));

        if(i==0){
            uk_x[0].x = (Real)0.0;
            uk_x[0].y = (Real)0.0;
            uk_y[0].x = (Real)0.0;
            uk_y[0].y = (Real)0.0;
            uk_z[0].x = (Real)0.0;
            uk_z[0].y = (Real)0.0;
        }
    }
    return;
}

__global__
void cufcm_particle_velocities_bpp_shared_dynamic(myCufftReal *ux, myCufftReal *uy, myCufftReal *uz,
                                Real *Y, Real *VTEMP, Real *WTEMP,
                                int N, int ngd, 
                                Real sigma, Real sigmadip, Real Sigma,
                                Real dx, int nx, int ny, int nz,
                                int *particle_index, int start, int end,
                                int rotation){

    int ngdh = ngd/2;
    Real norm = dx*dx*dx;
    Real Vx = (Real) 0.0, Vy = (Real) 0.0, Vz = (Real) 0.0;
    Real Wx = (Real) 0.0, Wy = (Real) 0.0, Wz = (Real) 0.0;

    extern __shared__ Integer s[];
    Integer *indx_shared = s;
    Integer *indy_shared = (Integer*)&indx_shared[ngd];
    Integer *indz_shared = (Integer*)&indy_shared[ngd];
    Real *gaussx_shared = (Real*)&indz_shared[ngd]; 
    Real *gaussy_shared = (Real*)&gaussx_shared[ngd];
    Real *gaussz_shared = (Real*)&gaussy_shared[ngd];
    #if USE_REGULARFCM
        Real *gaussx_dip_shared = (Real*)&gaussz_shared[ngd]; 
        Real *gaussy_dip_shared = (Real*)&gaussx_dip_shared[ngd];
        Real *gaussz_dip_shared = (Real*)&gaussy_dip_shared[ngd];
        Real *grad_gaussx_dip_shared = (Real*)&gaussz_dip_shared[ngd];
    #else
        Real *xdis_shared = (Real*)&gaussz_shared[ngd];    
        Real *ydis_shared = (Real*)&xdis_shared[ngd];
        Real *zdis_shared = (Real*)&ydis_shared[ngd];
        Real *grad_gaussx_dip_shared = (Real*)&zdis_shared[ngd];
    #endif
    Real *grad_gaussy_dip_shared = (Real*)&grad_gaussx_dip_shared[ngd];
    Real *grad_gaussz_dip_shared = (Real*)&grad_gaussy_dip_shared[ngd];
    Real *Y_shared = (Real*)&grad_gaussz_dip_shared[ngd];

    #if USE_REGULARFCM
        Real sigmasq = sigma*sigma;
        Real sigmadipsq = sigmadip*sigmadip;
        Real Sigmasq = sigmasq;
        Real Sigmadipsq = sigmadipsq;
        Real anorm = Real(1.0)/my_sqrt(Real(2.0)*Real(PI)*sigmasq);
        Real Anorm = anorm;
        Real anormdip = Real(1.0)/my_sqrt(Real(2.0)*Real(PI)*sigmadipsq);
    #else
        Real Sigmasq = Sigma*Sigma;
        Real Sigmadipsq = 0;
        if(rotation==1){
            Sigmadipsq = Sigmasq;
        }
        Real Anorm = Real(1.0)/my_sqrt(Real(PI2)*Sigmasq);
        Real width2 = (Real(2.0)*Sigmasq);
        Real pdmag = sigma*sigma - Sigmasq;
    #endif
    

    // Specialize BlockReduce
    typedef cub::BlockReduce<Real, FCM_THREADS_PER_BLOCK> BlockReduce;
    // Allocate shared memory for BlockReduce
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    for(int np = blockIdx.x; (np < N) && (particle_index[np] >= start && particle_index[np] < end); np += gridDim.x){
        if(threadIdx.x == 0){
            Y_shared[0] = Y[3*np + 0];
            Y_shared[1] = Y[3*np + 1];
            Y_shared[2] = Y[3*np + 2];
        }
        __syncthreads();

        for(int i = threadIdx.x; i < 4*ngd; i += blockDim.x){
            Real xg = my_rint(Y_shared[0]/dx) - ngdh + my_fmod(Real(i), Real(ngd));
            Real yg = my_rint(Y_shared[1]/dx) - ngdh + my_fmod(Real(i), Real(ngd));
            Real zg = my_rint(Y_shared[2]/dx) - ngdh + my_fmod(Real(i), Real(ngd));

            Real xx = xg*dx - Y_shared[0];
            Real yy = yg*dx - Y_shared[1];
            Real zz = zg*dx - Y_shared[2];
            /* dis */
            if(i<ngd){
                gaussx_shared[i] = Anorm*my_exp(-xx*xx/(Real(2.0)*Sigmasq));
                gaussy_shared[i] = Anorm*my_exp(-yy*yy/(Real(2.0)*Sigmasq));
                gaussz_shared[i] = Anorm*my_exp(-zz*zz/(Real(2.0)*Sigmasq));
            }
            /* gauss */
            if(i>=ngd && i<2*ngd){
                #if USE_REGULARFCM
                    gaussx_dip_shared[i-ngd] = anormdip*my_exp(-xx*xx/(Real(2.0)*sigmadipsq));
                    gaussy_dip_shared[i-ngd] = anormdip*my_exp(-yy*yy/(Real(2.0)*sigmadipsq));
                    gaussz_dip_shared[i-ngd] = anormdip*my_exp(-zz*zz/(Real(2.0)*sigmadipsq));
                #else
                    xdis_shared[i-ngd] = xx;
                    ydis_shared[i-ngd] = yy;
                    zdis_shared[i-ngd] = zz;
                #endif
            }
            /* grad_gauss */
            if(i>=2*ngd && i<3*ngd){
                if(rotation==1){
                    grad_gaussx_dip_shared[i-2*ngd] = - xx / Sigmadipsq;
                    grad_gaussy_dip_shared[i-2*ngd] = - yy / Sigmadipsq;
                    grad_gaussz_dip_shared[i-2*ngd] = - zz / Sigmadipsq;
                }
            }
            /* ind */
            if(i>=3*ngd){
                indx_shared[i-3*ngd] = xg - nx * my_floor( xg / nx);
                indy_shared[i-3*ngd] = yg - ny * my_floor( yg / ny);
                indz_shared[i-3*ngd] = zg - nz * my_floor( zg / nz);
            }
        }
        __syncthreads();

        for(int t = threadIdx.x; t < ngd*ngd*ngd; t += blockDim.x){
            const int k = t/(ngd*ngd);
            const int j = (t - k*ngd*ngd)/ngd;
            const int i = t - k*ngd*ngd - j*ngd;

            Real gradx = 0, grady = 0, gradz = 0;
            if(rotation==1){
                gradx = grad_gaussx_dip_shared[i];
                grady = grad_gaussy_dip_shared[j];
                gradz = grad_gaussz_dip_shared[k];
            }

            int ind = indx_shared[i] + indy_shared[j]*int(nx) + indz_shared[k]*int(nx)*int(ny);
            
            #if USE_REGULARFCM
                Real temp = gaussx_shared[i]*gaussy_shared[j]*gaussz_shared[k]*norm;
                Real tempdip = gaussx_dip_shared[i]*gaussy_dip_shared[j]*gaussz_dip_shared[k]*norm;
            #else
                Real r2 = xdis_shared[i]*xdis_shared[i] + ydis_shared[j]*ydis_shared[j] + zdis_shared[k]*zdis_shared[k];
                Real temp1 = gaussx_shared[i]*gaussy_shared[j]*gaussz_shared[k]*norm;
                Real temp2 = Real(0.5) * pdmag / Sigmasq;
                Real temp3 = temp2 /Sigmasq;
                Real temp4 = Real(3.0)*temp2;
                Real temp5 = ( Real(1.0) + temp3*r2 - temp4);
                Real temp = temp1*temp5;
                Real tempdip = 0;
                if(rotation==1){
                    tempdip = temp1;
                }
            #endif

            Vx += ux[ind]*temp;
            Vy += uy[ind]*temp;
            Vz += uz[ind]*temp;
            if(rotation==1){
                Wx += -Real(0.5)*(uz[ind]*grady - uy[ind]*gradz)*tempdip;
                Wy += -Real(0.5)*(ux[ind]*gradz - uz[ind]*gradx)*tempdip;
                Wz += -Real(0.5)*(uy[ind]*gradx - ux[ind]*grady)*tempdip; 
            }
        }
        
        // Reduction
        Real total_Vx = BlockReduce(temp_storage).Sum(Vx);
        Real total_Vy = BlockReduce(temp_storage).Sum(Vy);
        Real total_Vz = BlockReduce(temp_storage).Sum(Vz);
        Real total_Wx = 0, total_Wy = 0, total_Wz = 0;
        if(rotation==1){
            total_Wx = BlockReduce(temp_storage).Sum(Wx);
            total_Wy = BlockReduce(temp_storage).Sum(Wy);
            total_Wz = BlockReduce(temp_storage).Sum(Wz);
        }
    
        if(threadIdx.x==0){
            VTEMP[3*np + 0] = total_Vx;  
            VTEMP[3*np + 1] = total_Vy;
            VTEMP[3*np + 2] = total_Vz;
            if(rotation==1){
                WTEMP[3*np + 0] = total_Wx;
                WTEMP[3*np + 1] = total_Wy;
                WTEMP[3*np + 2] = total_Wz;
            }
        }
    }
}
