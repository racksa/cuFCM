#include <iostream>
// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>

#include "config.hpp"
#include "CUFCM_CORRECTION.cuh"

// S_ij = S_I(r) delta_ij + S_xx(r) x_ix_j
/*
__device__ __host__
Real A_old(Real r, Real rsq, Real sigma, Real sigmasq, Real expS, Real erfS){
	return Real(1.0)/(Real(8.0)*Real(PI)*r) * ((Real(1.0) + sigmasq/rsq)*erfS - (Real(2.0)*sigma/r)/PI2sqrt * expS);
}

__device__ __host__
Real B_old(Real r, Real rsq, Real sigma, Real sigmasq, Real expS, Real erfS){
	return Real(1.0)/(Real(8.0)*Real(PI)*pow(r, 3)) * ((Real(1.0) - Real(3.0)*sigmasq/rsq)*erfS + (Real(6.0)*sigma/r)/PI2sqrt * expS);
}

__device__ __host__
Real f_old(Real r, Real rsq, Real sigma, Real sigmasq, Real expS, Real erfS){
	return Real(1.0)/(Real(8.0)*Real(PI)*pow(r, 3)) * ( erfS - r/sigma*Real(sqrt2oPI) * expS );
}

__device__ __host__
Real dfdr_old(Real r, Real rsq, Real sigma, Real sigmasq, Real expS, Real erfS){
	return Real(-3.0)/r*f_old(r, rsq, sigma, sigmasq, expS, erfS) + Real(1.0)/(Real(8.0)*Real(PI)*pow(r, 3)) * (rsq/(sigmasq*sigma)) * Real(sqrt2oPI) * expS;
}
*/

// __device__ __host__
// Real dAdr(Real r, Real rsq, Real sigma, Real sigmasq, Real expS, Real erfS){
// 	return (Real)-1.0/((Real)8.0*PI*pow(r, 2)) * (((Real)1.0+(Real)3.0*sigmasq/rsq)*erfS - ((Real)4.0*r/sigma + (Real)6.0*sigma/r)/PI2sqrt * expS );
// }

// __device__ __host__
// Real dBdr(Real r, Real rsq, Real sigma, Real sigmasq, Real expS, Real erfS){
// 	return (Real)-1.0/((Real)8.0*PI*pow(r, 4)) * (((Real)3.0-(Real)15.0*sigmasq/rsq)*erfS + ((Real)4.0*r/sigma + (Real)30.0*sigma/r)/PI2sqrt * expS);
// }

// S_ij = S_I(r) delta_ij + S_xx(r) x_ix_j
__device__ __host__
Real S_I(Real r, Real rsq, Real sigma, Real sigmasq, Real gaussgam, Real erfS){
	return Real(1.0)/(Real(8.0)*Real(PI)*r) * ((Real(1.0) + sigmasq/rsq)*erfS) - Real(0.5)*sigmasq*sigmasq/rsq * gaussgam;
}

__device__ __host__
Real S_xx(Real r, Real rsq, Real sigma, Real sigmasq, Real gaussgam, Real erfS){
	return Real(1.0)/(Real(8.0)*Real(PI)*pow(r, 3)) * ((Real(1.0) - Real(3.0)*sigmasq/rsq)*erfS) + Real(1.5)*sigmasq*sigmasq/rsq/rsq*gaussgam;
}

// R_ij = f(r) (-x cross)
__device__ __host__
Real f(Real r, Real rsq, Real sigma, Real sigmasq, Real gaussgam, Real erfS){
	return Real(1.0)/(Real(8.0)*Real(PI)*pow(r, 3)) * ( erfS - Real(4.0)*Real(PI)*r*sigmasq * gaussgam );
}

__device__ __host__
Real dfdr(Real r, Real rsq, Real sigma, Real sigmasq, Real gaussgam, Real erfS){
    return Real(-3.0)/r*f(r, rsq, sigma, sigmasq, gaussgam, erfS) + Real(0.5)/r*gaussgam;
}

// P_ij = Q_I(r) delta_ij + Q_xx(r) x_ix_j
__device__ __host__
Real Q_I(Real r, Real rsq, Real sigma, Real sigmasq, Real gaussgam, Real erfS){
	return Real(1.0)/(Real(4.0)*PI*pow(r, 3))*erfS - (Real(1.0) + sigmasq/rsq)*gaussgam;
}

__device__ __host__
Real Q_xx(Real r, Real rsq, Real sigma, Real sigmasq, Real gaussgam, Real erfS){
	return Real(-3.0)/(Real(4.0)*PI*pow(r, 5))*erfS + (Real(1.0) + Real(3.0)*sigmasq/rsq)/rsq*gaussgam;
}

// P_ij = P_I(r) delta_ij + P_xx(r) x_ix_j
__device__ __host__
Real P_I(Real r, Real rsq, Real sigma, Real sigmasq, Real gaussgam, Real erfS){
	return Real(-1.0)/(Real(8.0)*Real(PI)*pow(r, 3))*erfS + (Real(0.5)/rsq*(sigmasq+rsq))*gaussgam;
}

__device__ __host__
Real P_xx(Real r, Real rsq, Real sigma, Real sigmasq, Real gaussgam, Real erfS){
	return (Real(3.0)/(Real(8.0)*Real(PI)*pow(r, 4))*erfS - (Real(0.5)/rsq/r*(Real(3.0)*sigmasq+rsq))*gaussgam)/r;
}

// T_ij = T_I(r) delta_ij + T_xx(r) x_ix_j
__device__ __host__
Real T_I(Real r, Real rsq, Real sigma, Real sigmasq, Real gaussgam){
    return (Real(2.0) - rsq / sigmasq) / sigmasq * gaussgam;
}

__device__ __host__
Real T_xx(Real r, Real rsq, Real sigma, Real sigmasq, Real gaussgam){
    return Real(1.0) / sigmasq / sigmasq * gaussgam;
}

__device__ __host__
Real K(Real r, Real rsq, Real sigma, Real sigmasq, Real gaussgam){
    return Real(-0.5) / sigmasq * gaussgam;
}


__global__
void cufcm_pair_correction(Real* Y, Real* V, Real* W, Real* F, Real* T, int N, Real Lx, Real Ly, Real Lz,
                    int *particle_cellindex, int *cell_start, int *cell_end,
                    int *map,
                    int ncell, Real Rrefsq,
                    Real Sigma,
                    Real sigmaFCM,
                    Real sigmaFCMdip){

    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    int icell = 0, j = 0, jcello = 0, jcell = 0, nabor = 0;
    Real vxi = Real(0.0), vyi = Real(0.0), vzi = Real(0.0);
    #if ROTATION == 1
        Real wxi = (Real)0.0, wyi = (Real)0.0, wzi = (Real)0.0;
    #endif

    Real pdmag = sigmaFCM*sigmaFCM - Sigma*Sigma;
    Real pdmagsq_quarter = pdmag * pdmag * (Real)0.25;
    Real gamma = my_sqrt(Real(2.0))*Sigma;
    Real gammasq = gamma*gamma;
    Real gammaVF_FCM = my_sqrt(Real(2.0))*sigmaFCM;
    Real gammaVF_FCMsq = gammaVF_FCM*gammaVF_FCM;
    #if ROTATION == 1
        Real gammaVTWF_FCM = my_sqrt(sigmaFCM*sigmaFCM + sigmaFCMdip*sigmaFCMdip);
        Real gammaVTWF_FCMsq = gammaVTWF_FCM*gammaVTWF_FCM;
        Real gammaWT_FCM = my_sqrt(Real(2.0))*sigmaFCMdip;
        Real gammaWT_FCMsq = gammaWT_FCM*gammaWT_FCM;
    #endif

    for(int i = index; i < N; i += stride){
        icell = particle_cellindex[i];
    
        Real xi = Y[3*i + 0], yi = Y[3*i + 1], zi = Y[3*i + 2];
        Real xij = Real(0.0), yij = Real(0.0), zij = Real(0.0);
        /* intra-cell interactions */
        /* corrections only apply to particle i */
        for(j = cell_start[icell]; j < cell_end[icell]; j++){
            if(i != j){
                Real xij = xi - Y[3*j + 0];
                Real yij = yi - Y[3*j + 1];
                Real zij = zi - Y[3*j + 2];

                xij = xij - Lx * Real(int(xij/(Real(0.5)*Lx)));
                yij = yij - Ly * Real(int(yij/(Real(0.5)*Ly)));
                zij = zij - Lz * Real(int(zij/(Real(0.5)*Lz)));
                Real rijsq=xij*xij+yij*yij+zij*zij;
                if(rijsq < Rrefsq){
                    Real rij = my_sqrt(rijsq);

                    Real erfS = erf(Real(0.5)*rij/Sigma);
                    Real gaussgam = exp(-Real(0.5)*rijsq/gammasq)/pow(Real(PI2)*gammasq, Real(1.5));

                    Real erfS_VF_FCM = erf(rij/(my_sqrt(Real(2.0))*gammaVF_FCM));
                    Real gaussgam_VF_FCM = exp(-Real(0.5)*rijsq/(gammaVF_FCMsq))/pow(Real(PI2)*gammaVF_FCMsq, Real(1.5));

                    #if ROTATION == 1
                        Real erfS_VTWF_FCM = erf(rij/(my_sqrt(Real(2.0))*gammaVTWF_FCM));
                        Real gaussgam_VTWF_FCM = exp(-Real(0.5)*rijsq/(gammaVTWF_FCMsq))/pow(Real(PI2)*gammaVTWF_FCMsq, Real(1.5));

                        Real erfS_WT_FCM = erf(rij/(my_sqrt(Real(2.0))*gammaWT_FCM));
                        Real gaussgam_WT_FCM = exp(-Real(0.5)*rijsq/(gammaWT_FCMsq))/pow(Real(PI2)*gammaWT_FCMsq, Real(1.5));
                    #endif
                    // ------------VF------------
                    Real Fjdotx = xij*F[3*j + 0] + yij*F[3*j + 1] + zij*F[3*j + 2];
                    Real Fidotx = xij*F[3*i + 0] + yij*F[3*i + 1] + zij*F[3*i + 2];

                    Real AFCMtemp = S_I(rij, rijsq, gammaVF_FCM, gammaVF_FCMsq, gaussgam_VF_FCM, erfS_VF_FCM);
                    Real BFCMtemp = S_xx(rij, rijsq, gammaVF_FCM, gammaVF_FCMsq, gaussgam_VF_FCM, erfS_VF_FCM);
                    Real Atemp = S_I(rij, rijsq, gamma, gammasq, gaussgam, erfS);
                    Real Btemp = S_xx(rij, rijsq, gamma, gammasq, gaussgam, erfS);
                    Real Ctemp = Q_I(rij, rijsq, gamma, gammasq, gaussgam, erfS)*pdmag;
                    Real Dtemp = Q_xx(rij, rijsq, gamma, gammasq, gaussgam, erfS)*pdmag;
                    Real Ptemp = T_I(rij, rijsq, gamma, gammasq, gaussgam)*pdmagsq_quarter;
                    Real Qtemp = T_xx(rij, rijsq, gamma, gammasq, gaussgam)*pdmagsq_quarter;

                    Real temp1VF = (AFCMtemp - Atemp - Ctemp - Ptemp);
                    Real temp2VF = (BFCMtemp - Btemp - Dtemp - Qtemp);

                    // printf("%d (%.8f %.8f %.8f) %d (%.8f %.8f %.8f)  %.8f\n",
                    // i, Y[3*i + 0], Y[3*i + 0], Y[3*i + 0],
                    // j, Y[3*j + 0], Y[3*j + 0], Y[3*j + 0], rij);

                    Real tempVTWF = 0;
                    #if ROTATION == 1
                        // ------------WF+VT------------
                        Real fFCMtemp_VTWF = f(rij, rijsq, gammaVTWF_FCM, gammaVTWF_FCMsq, gaussgam_VTWF_FCM, erfS_VTWF_FCM);
                        Real ftemp = f(rij, rijsq, gamma, gammasq, gaussgam, erfS);
                        Real quatemp = Real(0.5)*pdmag*K(rij, rijsq, gamma, gammasq, gaussgam);

                        tempVTWF = (fFCMtemp_VTWF - ftemp - quatemp);

                        // ------------WT------------
                        // Real fFCMtemp_WT = f_g(rij, rijsq, gammaWT_FCM, gammaWT_FCMsq, gaussgam_WT_FCM, erfS_WT_FCM);
                        // Real dfdrFCMtemp = dfdr_g(rij, rijsq, gammaWT_FCM, gammaWT_FCMsq, gaussgam_WT_FCM, erfS_WT_FCM);
                        // Real dfdrtemp = dfdr_g(rij, rijsq, gamma, gammasq, gaussgam, erfS);

                        // Real temp1WT = (dfdrFCMtemp*rij + (Real)2.0*fFCMtemp_WT) - (dfdrtemp*rij + (Real)2.0*ftemp);
                        // Real temp2WT = dfdrFCMtemp/rij - dfdrtemp/rij;

                        Real Tidotx = (T[3*i + 0]*xij + T[3*i + 1]*yij + T[3*i + 2]*zij);
                        Real Tjdotx = (T[3*j + 0]*xij + T[3*j + 1]*yij + T[3*j + 2]*zij);

                        Real temp1WT = P_I(rij, rijsq, gammaWT_FCM, gammaWT_FCMsq, gaussgam_WT_FCM, erfS_WT_FCM)
                                        - P_I(rij, rijsq, gamma, gammasq, gaussgam, erfS);
                        Real temp2WT = P_xx(rij, rijsq, gammaWT_FCM, gammaWT_FCMsq, gaussgam_WT_FCM, erfS_WT_FCM) 
                                        - P_xx(rij, rijsq, gamma, gammasq, gaussgam, erfS);
                    #endif

                    // Summation                    
                    vxi = vxi + temp1VF*F[3*j + 0] + temp2VF*xij*Fjdotx + tempVTWF*( zij*T[3*j + 1] - yij*T[3*j + 2] );
                    vyi = vyi + temp1VF*F[3*j + 1] + temp2VF*yij*Fjdotx + tempVTWF*( xij*T[3*j + 2] - zij*T[3*j + 0] );
                    vzi = vzi + temp1VF*F[3*j + 2] + temp2VF*zij*Fjdotx + tempVTWF*( yij*T[3*j + 0] - xij*T[3*j + 1] );

                    #if ROTATION == 1
                        wxi = wxi + (Real)0.5*( T[3*j + 0]*temp1WT + xij*Tjdotx*temp2WT ) + tempVTWF*( zij*F[3*j + 1] - yij*F[3*j + 2] );
                        wyi = wyi + (Real)0.5*( T[3*j + 1]*temp1WT + yij*Tjdotx*temp2WT ) + tempVTWF*( xij*F[3*j + 2] - zij*F[3*j + 0] );
                        wzi = wzi + (Real)0.5*( T[3*j + 2]*temp1WT + zij*Tjdotx*temp2WT ) + tempVTWF*( yij*F[3*j + 0] - xij*F[3*j + 1] );
                    #endif
                }
            }
        }
        jcello = 13*icell;
        /* inter-cell interactions */
        /* corrections apply to both parties in different cells */
        for(nabor = 0; nabor < 13; nabor++){
            jcell = map[jcello + nabor];
            for(j = cell_start[jcell]; j < cell_end[jcell]; j++){
                xij = xi - Y[3*j + 0];
                yij = yi - Y[3*j + 1];
                zij = zi - Y[3*j + 2];

                xij = xij - Lx * Real(int(xij/(Real(0.5)*Lx)));
                yij = yij - Ly * Real(int(yij/(Real(0.5)*Ly)));
                zij = zij - Lz * Real(int(zij/(Real(0.5)*Lz)));
                Real rijsq=xij*xij+yij*yij+zij*zij;
                if(rijsq < Rrefsq){
                    Real rij = my_sqrt(rijsq);

                    Real erfS = erf(Real(0.5)*rij/Sigma);
                    Real gaussgam = exp(-Real(0.5)*rijsq/gammasq)/pow(Real(PI2)*gammasq, Real(1.5));

                    Real erfS_VF_FCM = erf(rij/(my_sqrt(Real(2.0))*gammaVF_FCM));
                    Real gaussgam_VF_FCM = exp(-Real(0.5)*rijsq/(gammaVF_FCMsq))/pow(Real(PI2)*gammaVF_FCMsq, Real(1.5));

                    #if ROTATION == 1
                        Real erfS_VTWF_FCM = erf(rij/(my_sqrt(Real(2.0))*gammaVTWF_FCM));
                        Real gaussgam_VTWF_FCM = exp(-Real(0.5)*rijsq/(gammaVTWF_FCMsq))/pow(Real(PI2)*gammaVTWF_FCMsq, Real(1.5));

                        Real erfS_WT_FCM = erf(rij/(my_sqrt(Real(2.0))*gammaWT_FCM));
                        Real gaussgam_WT_FCM = exp(-Real(0.5)*rijsq/(gammaWT_FCMsq))/pow(Real(PI2)*gammaWT_FCMsq, Real(1.5));
                    #endif
                    // ------------VF------------
                    Real Fjdotx = xij*F[3*j + 0] + yij*F[3*j + 1] + zij*F[3*j + 2];
                    Real Fidotx = xij*F[3*i + 0] + yij*F[3*i + 1] + zij*F[3*i + 2];

                    Real AFCMtemp = S_I(rij, rijsq, gammaVF_FCM, gammaVF_FCMsq, gaussgam_VF_FCM, erfS_VF_FCM);
                    Real BFCMtemp = S_xx(rij, rijsq, gammaVF_FCM, gammaVF_FCMsq, gaussgam_VF_FCM, erfS_VF_FCM);
                    Real Atemp = S_I(rij, rijsq, gamma, gammasq, gaussgam, erfS);
                    Real Btemp = S_xx(rij, rijsq, gamma, gammasq, gaussgam, erfS);
                    Real Ctemp = Q_I(rij, rijsq, gamma, gammasq, gaussgam, erfS)*pdmag;
                    Real Dtemp = Q_xx(rij, rijsq, gamma, gammasq, gaussgam, erfS)*pdmag;
                    Real Ptemp = T_I(rij, rijsq, gamma, gammasq, gaussgam)*pdmagsq_quarter;
                    Real Qtemp = T_xx(rij, rijsq, gamma, gammasq, gaussgam)*pdmagsq_quarter;

                    Real temp1VF = (AFCMtemp - Atemp - Ctemp - Ptemp);
                    Real temp2VF = (BFCMtemp - Btemp - Dtemp - Qtemp);

                    Real tempVTWF = 0;
                    #if ROTATION == 1
                        // ------------WF+VT------------
                        Real fFCMtemp_VTWF = f(rij, rijsq, gammaVTWF_FCM, gammaVTWF_FCMsq, gaussgam_VTWF_FCM, erfS_VTWF_FCM);
                        Real ftemp = f(rij, rijsq, gamma, gammasq, gaussgam, erfS);
                        Real quatemp = Real(0.5)*pdmag*K(rij, rijsq, gamma, gammasq, gaussgam);

                        tempVTWF = (fFCMtemp_VTWF - ftemp - quatemp);

                        // ------------WT------------
                        // Real fFCMtemp_WT = f_g(rij, rijsq, gammaWT_FCM, gammaWT_FCMsq, gaussgam_WT_FCM, erfS_WT_FCM);
                        // Real dfdrFCMtemp = dfdr_g(rij, rijsq, gammaWT_FCM, gammaWT_FCMsq, gaussgam_WT_FCM, erfS_WT_FCM);
                        // Real dfdrtemp = dfdr_g(rij, rijsq, gamma, gammasq, gaussgam, erfS);

                        // Real temp1WT = (dfdrFCMtemp*rij + (Real)2.0*fFCMtemp_WT) - (dfdrtemp*rij + (Real)2.0*ftemp);
                        // Real temp2WT = dfdrFCMtemp/rij - dfdrtemp/rij;

                        Real Tidotx = (T[3*i + 0]*xij + T[3*i + 1]*yij + T[3*i + 2]*zij);
                        Real Tjdotx = (T[3*j + 0]*xij + T[3*j + 1]*yij + T[3*j + 2]*zij);

                        Real temp1WT = P_I(rij, rijsq, gammaWT_FCM, gammaWT_FCMsq, gaussgam_WT_FCM, erfS_WT_FCM)
                                        - P_I(rij, rijsq, gamma, gammasq, gaussgam, erfS);
                        Real temp2WT = P_xx(rij, rijsq, gammaWT_FCM, gammaWT_FCMsq, gaussgam_WT_FCM, erfS_WT_FCM) 
                                        - P_xx(rij, rijsq, gamma, gammasq, gaussgam, erfS);
                    #endif

                    // Summation
                    vxi = vxi + temp1VF*F[3*j + 0] + temp2VF*xij*Fjdotx + tempVTWF*( zij*T[3*j + 1] - yij*T[3*j + 2] );
                    vyi = vyi + temp1VF*F[3*j + 1] + temp2VF*yij*Fjdotx + tempVTWF*( xij*T[3*j + 2] - zij*T[3*j + 0] );
                    vzi = vzi + temp1VF*F[3*j + 2] + temp2VF*zij*Fjdotx + tempVTWF*( yij*T[3*j + 0] - xij*T[3*j + 1] );

                    atomicAdd(&V[3*j + 0], temp1VF*F[3*i + 0] + temp2VF*xij*Fidotx - tempVTWF*( zij*T[3*i + 1] - yij*T[3*i + 2] ));
                    atomicAdd(&V[3*j + 1], temp1VF*F[3*i + 1] + temp2VF*yij*Fidotx - tempVTWF*( xij*T[3*i + 2] - zij*T[3*i + 0] ));
                    atomicAdd(&V[3*j + 2], temp1VF*F[3*i + 2] + temp2VF*zij*Fidotx - tempVTWF*( yij*T[3*i + 0] - xij*T[3*i + 1] ));

                    #if ROTATION == 1
                        wxi = wxi + (Real)0.5*( T[3*j + 0]*temp1WT + xij*Tjdotx*temp2WT ) + tempVTWF*( zij*F[3*j + 1] - yij*F[3*j + 2] );
                        wyi = wyi + (Real)0.5*( T[3*j + 1]*temp1WT + yij*Tjdotx*temp2WT ) + tempVTWF*( xij*F[3*j + 2] - zij*F[3*j + 0] );
                        wzi = wzi + (Real)0.5*( T[3*j + 2]*temp1WT + zij*Tjdotx*temp2WT ) + tempVTWF*( yij*F[3*j + 0] - xij*F[3*j + 1] );
                        atomicAdd(&W[3*j + 0], (Real)0.5*( T[3*i + 0]*temp1WT + xij*Tidotx*temp2WT ) - tempVTWF*( zij*F[3*i + 1] - yij*F[3*i + 2] ));
                        atomicAdd(&W[3*j + 1], (Real)0.5*( T[3*i + 1]*temp1WT + yij*Tidotx*temp2WT ) - tempVTWF*( xij*F[3*i + 2] - zij*F[3*i + 0] ));
                        atomicAdd(&W[3*j + 2], (Real)0.5*( T[3*i + 2]*temp1WT + zij*Tidotx*temp2WT ) - tempVTWF*( yij*F[3*i + 0] - xij*F[3*i + 1] ));
                    #endif
                }
                
            }
        }
        atomicAdd(&V[3*i + 0], vxi);
        atomicAdd(&V[3*i + 1], vyi);
        atomicAdd(&V[3*i + 2], vzi);
        #if ROTATION == 1
            atomicAdd(&W[3*i + 0], wxi);
            atomicAdd(&W[3*i + 1], wyi);
            atomicAdd(&W[3*i + 2], wzi);
        #endif
    }
    

    return;
}


__global__
void cufcm_self_correction(Real* V, Real* W, Real* F, Real* T, int N, 
                                Real StokesMob, Real ModStokesMob,
                                Real PDStokesMob, Real BiLapMob,
                                Real WT1Mob, Real WT2Mob){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;
    
    for(int np = index; np < N; np += stride){
        V[3*np + 0] = V[3*np + 0] + F[3*np + 0]*(StokesMob - ModStokesMob + PDStokesMob - BiLapMob) ;
        V[3*np + 1] = V[3*np + 1] + F[3*np + 1]*(StokesMob - ModStokesMob + PDStokesMob - BiLapMob) ;
        V[3*np + 2] = V[3*np + 2] + F[3*np + 2]*(StokesMob - ModStokesMob + PDStokesMob - BiLapMob) ;
        #if ROTATION == 1
            W[3*np + 0] = W[3*np + 0] + T[3*np + 0]*(WT1Mob - WT2Mob) ;
            W[3*np + 1] = W[3*np + 1] + T[3*np + 1]*(WT1Mob - WT2Mob) ;
            W[3*np + 2] = W[3*np + 2] + T[3*np + 2]*(WT1Mob - WT2Mob) ;
        #endif
    }

}


__global__
void cufcm_flowfield_correction(myCufftReal *fx, myCufftReal *fy, myCufftReal *fz,
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

    Real *xdis_shared = (Real*)&gaussz_shared[ngd];
    Real *ydis_shared = (Real*)&xdis_shared[ngd];
    Real *zdis_shared = (Real*)&ydis_shared[ngd];
    Real *grad_gaussx_dip_shared = (Real*)&zdis_shared[ngd];
    Real *grad_gaussy_dip_shared = (Real*)&grad_gaussx_dip_shared[ngd];
    Real *grad_gaussz_dip_shared = (Real*)&grad_gaussy_dip_shared[ngd];

    Real *Y_shared = (Real*)&grad_gaussz_dip_shared[ngd];
    Real *F_shared = (Real*)&Y_shared[3];
    Real *g_shared = (Real*)&F_shared[3];

    Real Sigmasq = Sigma*Sigma;
    Real sigmasq = sigma*sigma;
    Real Sigmadipsq = 0;
    if(rotation==1){
        Sigmadipsq = Sigmasq;
    }
    Real Anorm = Real(1.0)/my_sqrt(Real(PI2)*Sigmasq);
    Real pdmag = sigmasq - Sigmasq;
    

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
                xdis_shared[i-ngd] = xx;
                ydis_shared[i-ngd] = yy;
                zdis_shared[i-ngd] = zz;
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


            int ind = indx_shared[i] + indy_shared[j]*nx + indz_shared[k]*nx*ny;

            Real r2 = xdis_shared[i]*xdis_shared[i] + ydis_shared[j]*ydis_shared[j] + zdis_shared[k]*zdis_shared[k];
            Real r = my_sqrt(r2);
    
            Real Fidotx = xdis_shared[i]*F_shared[0] + ydis_shared[j]*F_shared[1] + zdis_shared[k]*F_shared[2];

            Real erfS_VF_FCM = erf(r/(my_sqrt(Real(2.0))*sigma));
            Real gaussgam_VF_FCM = exp(-Real(0.5)*r2/(sigmasq))/pow(Real(PI2)*sigmasq, Real(1.5));

            Real erfS = erf(r/(my_sqrt(Real(2.0))*Sigma));
            Real gaussgam = exp(-Real(0.5)*r2/Sigmasq)/pow(Real(PI2)*Sigmasq, Real(1.5));
            
            Real AFCMtemp = S_I(r, r2, sigma, sigmasq, gaussgam_VF_FCM, erfS_VF_FCM);
            Real BFCMtemp = S_xx(r, r2, sigma, sigmasq, gaussgam_VF_FCM, erfS_VF_FCM);
            Real Atemp = S_I(r, r2, Sigma, Sigmasq, gaussgam, erfS);
            Real Btemp = S_xx(r, r2, Sigmasq, Sigmasq, gaussgam, erfS);
            Real Ctemp = Q_I(r, r2, Sigma, Sigmasq, gaussgam, erfS)*pdmag*0.5;
            Real Dtemp = Q_xx(r, r2, Sigma, Sigmasq, gaussgam, erfS)*pdmag*0.5;
                    
            Real temp1VF = (AFCMtemp - Atemp - Ctemp);
            Real temp2VF = (BFCMtemp - Btemp - Dtemp);

            atomicAdd(&fx[ind], temp1VF*F_shared[0] + temp2VF*xdis_shared[i]*Fidotx);
            atomicAdd(&fy[ind], temp1VF*F_shared[1] + temp2VF*ydis_shared[j]*Fidotx);
            atomicAdd(&fz[ind], temp1VF*F_shared[2] + temp2VF*zdis_shared[k]*Fidotx);
        }
    }
}

// __global__
// void cufcm_compute_formula(Real* Y, Real* V, Real* W, Real* F, Real* T, int N,
//                     Real sigmaFCM,
//                     Real sigmaFCMdip,
//                     Real StokesMob,
//                     Real WT1Mob){
//     const int index = threadIdx.x + blockIdx.x*blockDim.x;
//     const int stride = blockDim.x*gridDim.x;
//     Real gammaVF_FCM = my_sqrt(Real(2.0))*sigmaFCM;
//     Real gammaVF_FCMsq = gammaVF_FCM*gammaVF_FCM;
//     Real gammaVTWF_FCM = my_sqrt(sigmaFCM*sigmaFCM + sigmaFCMdip*sigmaFCMdip);
//     Real gammaVTWF_FCMsq = gammaVTWF_FCM*gammaVTWF_FCM;
//     Real gammaWT_FCM = my_sqrt(Real(2.0))*sigmaFCMdip;
//     Real gammaWT_FCMsq = gammaWT_FCM*gammaWT_FCM;
//     Real vxi = (Real)0.0, vyi = (Real)0.0, vzi = (Real)0.0;
//     Real wxi = (Real)0.0, wyi = (Real)0.0, wzi = (Real)0.0;

//     for(int i = index; i < N; i += stride){
//         Real xi = Y[3*i + 0], yi = Y[3*i + 1], zi = Y[3*i + 2];
//         for(int j = 0; j < N; j++){
//             if(i != j){
//                 Real xij = xi - Y[3*j + 0];
//                 Real yij = yi - Y[3*j + 1];
//                 Real zij = zi - Y[3*j + 2];

//                 Real rijsq=xij*xij+yij*yij+zij*zij;
//                 Real rij = my_sqrt(rijsq);
//                 // Real erfS_VF_FCM = erf(rij/(my_sqrt(Real(2.0))*gammaVF_FCM));
//                 // Real expS_VF_FCM = exp(-rijsq/(Real(2.0)*gammaVF_FCMsq));
//                 // Real erfS_VTWF_FCM = erf(rij/(sqrtf(Real(2.0))*gammaVTWF_FCM));
//                 // Real expS_VTWF_FCM = exp(-rijsq/(Real(2.0)*gammaVTWF_FCMsq));
//                 // Real erfS_WT_FCM = erf(rij/(sqrtf(Real(2.0))*gammaWT_FCM));
//                 // Real expS_WT_FCM = exp(-rijsq/(Real(2.0)*gammaWT_FCMsq));

//                 Real erfS_VF_FCM = erf(rij/(my_sqrt(Real(2.0))*gammaVF_FCM));
//                 Real gaussgam_VF_FCM = exp(-Real(0.5)*rijsq/(gammaVF_FCMsq))/pow(Real(PI2)*gammaVF_FCMsq, Real(1.5));
//                 Real erfS_VTWF_FCM = erf(rij/(my_sqrt(Real(2.0))*gammaVTWF_FCM));
//                 Real gaussgam_VTWF_FCM = exp(-Real(0.5)*rijsq/(gammaVTWF_FCMsq))/pow(Real(PI2)*gammaVTWF_FCMsq, Real(1.5));
//                 Real erfS_WT_FCM = erf(rij/(my_sqrt(Real(2.0))*gammaWT_FCM));
//                 Real gaussgam_WT_FCM = exp(-Real(0.5)*rijsq/(gammaWT_FCMsq))/pow(Real(PI2)*gammaWT_FCMsq, Real(1.5));

//                 // ------------VF------------
//                 Real Fjdotx = xij*F[3*j + 0] + yij*F[3*j + 1] + zij*F[3*j + 2];
//                 Real AFCMtemp = S_I(rij, rijsq, gammaVF_FCM, gammaVF_FCMsq, gaussgam_VF_FCM, erfS_VF_FCM);
//                 Real BFCMtemp = S_xx(rij, rijsq, gammaVF_FCM, gammaVF_FCMsq, gaussgam_VF_FCM, erfS_VF_FCM);
//                 Real temp1VF = (AFCMtemp);
//                 Real temp2VF = (BFCMtemp);
//                 // ------------WF+VT------------
//                 Real fFCMtemp_VTWF = f(rij, rijsq, gammaVTWF_FCM, gammaVTWF_FCMsq, gaussgam_VTWF_FCM, erfS_VTWF_FCM);
//                 Real tempVTWF = (fFCMtemp_VTWF);
//                 // ------------WT------------
//                 Real Tjdotx = (T[3*j + 0]*xij + T[3*j + 1]*yij + T[3*j + 2]*zij);

//                 Real temp1WT = P_I(rij, rijsq, gammaWT_FCM, gammaWT_FCMsq, gaussgam_WT_FCM, erfS_WT_FCM);
//                 Real temp2WT = P_xx(rij, rijsq, gammaWT_FCM, gammaWT_FCMsq, gaussgam_WT_FCM, erfS_WT_FCM);
//                 // Summation
//                 wxi += (Real)0.5*( T[3*j + 0]*temp1WT - xij*Tjdotx*temp2WT ) + tempVTWF*( zij*F[3*j + 1] - yij*F[3*j + 2] );
//                 wyi += (Real)0.5*( T[3*j + 1]*temp1WT - yij*Tjdotx*temp2WT ) + tempVTWF*( xij*F[3*j + 2] - zij*F[3*j + 0] );
//                 wzi += (Real)0.5*( T[3*j + 2]*temp1WT - zij*Tjdotx*temp2WT ) + tempVTWF*( yij*F[3*j + 0] - xij*F[3*j + 1] );
//                 vxi += temp1VF*F[3*j + 0] + temp2VF*xij*Fjdotx + tempVTWF*( zij*T[3*j + 1] - yij*T[3*j + 2] );
//                 vyi += temp1VF*F[3*j + 1] + temp2VF*yij*Fjdotx + tempVTWF*( xij*T[3*j + 2] - zij*T[3*j + 0] );
//                 vzi += temp1VF*F[3*j + 2] + temp2VF*zij*Fjdotx + tempVTWF*( yij*T[3*j + 0] - xij*T[3*j + 1] );
//             }
//         }
//         vxi += F[3*i + 0]*(StokesMob);
//         vyi += F[3*i + 1]*(StokesMob);
//         vzi += F[3*i + 2]*(StokesMob);
//         wxi += T[3*i + 0]*(WT1Mob) ;
//         wyi += T[3*i + 1]*(WT1Mob) ;
//         wzi += T[3*i + 2]*(WT1Mob) ;
//         V[3*i + 0] = vxi;
//         V[3*i + 1] = vyi;
//         V[3*i + 2] = vzi;
//         W[3*i + 0] = wxi;
//         W[3*i + 1] = wyi;
//         W[3*i + 2] = wzi; 
//         return;
//     }
// }
