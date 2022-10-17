#include <iostream>
// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>

#include "config.hpp"
#include "CUFCM_CORRECTION.cuh"

__device__ __host__
Real f(Real r, Real rsq, Real sigma, Real sigmasq, Real expS, Real erfS){
	return (Real)1.0/((Real)8.0*PI*pow(r, 3)) * ( erfS - r/sigma*sqrt2oPI * expS );
}

__device__ __host__
Real dfdr(Real r, Real rsq, Real sigma, Real sigmasq, Real expS, Real erfS){
	return (Real)-3.0/r*f(r, rsq, sigma, sigmasq, expS, erfS) + (Real)1.0/((Real)8.0*PI*pow(r, 3)) * (rsq/(sigmasq*sigma)) * sqrt2oPI * expS;
}

// S_ij = A(r) delta_ij + B(r) x_ix_j
__device__ __host__
Real A(Real r, Real rsq, Real sigma, Real sigmasq, Real expS, Real erfS){
	return (Real)1.0/((Real)8.0*PI*r) * (((Real)1.0 + sigmasq/rsq)*erfS- ((Real)2.0*sigma/r)/PI2sqrt * expS);
}

__device__ __host__
Real B(Real r, Real rsq, Real sigma, Real sigmasq, Real expS, Real erfS){
	return (Real)1.0/((Real)8.0*PI*pow(r, 3)) * (((Real)1.0 - (Real)3.0*sigmasq/rsq)*erfS + ((Real)6.0*sigma/r)/PI2sqrt * expS);
}

__device__ __host__
Real dAdr(Real r, Real rsq, Real sigma, Real sigmasq, Real expS, Real erfS){
	return (Real)-1.0/((Real)8.0*PI*pow(r, 2)) * (((Real)1.0+(Real)3.0*sigmasq/rsq)*erfS - ((Real)4.0*r/sigma + (Real)6.0*sigma/r)/PI2sqrt * expS );
}

__device__ __host__
Real dBdr(Real r, Real rsq, Real sigma, Real sigmasq, Real expS, Real erfS){
	return (Real)-1.0/((Real)8.0*PI*pow(r, 4)) * (((Real)3.0-(Real)15.0*sigmasq/rsq)*erfS + ((Real)4.0*r/sigma + (Real)30.0*sigma/r)/PI2sqrt * expS);
}

// D_ij = C(r) delta_ij + F(r) x_ix_j
__device__ __host__
Real C(Real r, Real rsq, Real sigma, Real sigmasq, Real gaussgam, Real erfS){
	return (Real)1.0/((Real)4.0*PI*pow(r, 3))*erfS - ((Real)1.0 + sigmasq/rsq)*gaussgam;
}

__device__ __host__
Real D(Real r, Real rsq, Real sigma, Real sigmasq, Real gaussgam, Real erfS){
	return (Real)-3.0/((Real)4.0*PI*pow(r, 5))*erfS + ((Real)1.0/rsq + (Real)3.0*sigmasq/pow(r, 4))*gaussgam;
}

__device__ __host__
Real P(Real r, Real rsq, Real sigma, Real sigmasq, Real gaussgam){
	return ((Real)1.0 - rsq/(Real)4.0/sigmasq) * gaussgam / sigmasq;
}

__device__ __host__
Real Q(Real r, Real rsq, Real sigma, Real sigmasq, Real gaussgam){
	return (Real)1.0/(Real)4.0 * gaussgam / sigmasq / sigmasq;
}

__global__
void cufcm_pair_correction_spatial_hashing_tpp(Real* Y, Real* V, Real* W, Real* F, Real* T, int N, Real boxsize,
                    int *particle_cellindex, int *cell_start, int *cell_end,
                    int *map,
                    int ncell, Real Rrefsq,
                    Real pdmag,
                    Real sigma, Real sigmasq,
                    Real sigmaFCM, Real sigmaFCMsq,
                    Real sigmaFCMdip, Real sigmaFCMdipsq){

    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    int icell = 0, j = 0, jcello = 0, jcell = 0, nabor = 0;
    Real vxi = (Real)0.0, vyi = (Real)0.0, vzi = (Real)0.0;
    Real wxi = (Real)0.0, wyi = (Real)0.0, wzi = (Real)0.0;

    Real pdmagsq_quarter = pdmag * pdmag * (Real)0.25;
    Real gamma = sqrtf((Real)2.0)*sigma;
    Real gammasq = gamma*gamma;
    Real gammaVF_FCM = sqrtf((Real)2.0)*sigmaFCM;
    Real gammaVF_FCMsq = gammaVF_FCM*gammaVF_FCM;
    Real gammaVTWF_FCM = sqrtf(sigmaFCM*sigmaFCM + sigmaFCMdip*sigmaFCMdip);
    Real gammaVTWF_FCMsq = gammaVTWF_FCM*gammaVTWF_FCM;
    Real gammaWT_FCM = sqrtf((Real)2.0)*sigmaFCMdip;
    Real gammaWT_FCMsq = gammaWT_FCM*gammaWT_FCM;

    for(int i = index; i < N; i += stride){
        icell = particle_cellindex[i];
        
        Real xi = Y[3*i + 0], yi = Y[3*i + 1], zi = Y[3*i + 2];
        Real xij = (Real)0.0, yij = (Real)0.0, zij = (Real)0.0;
        /* intra-cell interactions */
        /* corrections only apply to particle i */
        for(j = cell_start[icell]; j < cell_end[icell]; j++){
            if(i != j){
                Real xij = xi - Y[3*j + 0];
                Real yij = yi - Y[3*j + 1];
                Real zij = zi - Y[3*j + 2];

                xij = xij - boxsize * (Real) ((int) (xij/(boxsize/Real(2.0))));
                yij = yij - boxsize * (Real) ((int) (yij/(boxsize/Real(2.0))));
                zij = zij - boxsize * (Real) ((int) (zij/(boxsize/Real(2.0))));

                Real rijsq=xij*xij+yij*yij+zij*zij;
                if(rijsq < Rrefsq){
                    Real rij = sqrtf(rijsq);
                    Real erfS = erf((Real)0.5*rij/sigma);
                    Real expS = exp(-rijsq/((Real)2.0*gammasq));
                    Real gaussgam = expS/pow((Real)2.0*PI*gammasq, (Real)1.5);

                    Real erfS_VF_FCM = erf(rij/(sqrtf((Real)2.0)*gammaVF_FCM));
                    Real expS_VF_FCM = exp(-rijsq/((Real)2.0*gammaVF_FCMsq));

                    Real erfS_VTWF_FCM = erf(rij/(sqrtf((Real)2.0)*gammaVTWF_FCM));
                    Real expS_VTWF_FCM = exp(-rijsq/((Real)2.0*gammaVTWF_FCMsq));

                    Real erfS_WT_FCM = erf(rij/(sqrtf((Real)2.0)*gammaWT_FCM));
                    Real expS_WT_FCM = exp(-rijsq/((Real)2.0*gammaWT_FCMsq));

                    // ------------VF------------
                    Real Fjdotx = xij*F[3*j + 0] + yij*F[3*j + 1] + zij*F[3*j + 2];
                    Real Fidotx = xij*F[3*i + 0] + yij*F[3*i + 1] + zij*F[3*i + 2];

                    Real AFCMtemp = A(rij, rijsq, gammaVF_FCM, gammaVF_FCMsq, expS_VF_FCM, erfS_VF_FCM);
                    Real BFCMtemp = B(rij, rijsq, gammaVF_FCM, gammaVF_FCMsq, expS_VF_FCM, erfS_VF_FCM);
                    Real Atemp = A(rij, rijsq, gamma, gammasq, expS, erfS);
                    Real Btemp = B(rij, rijsq, gamma, gammasq, expS, erfS);
                    Real Ctemp = C(rij, rijsq, gamma, gammasq, gaussgam, erfS)*pdmag;
                    Real Dtemp = D(rij, rijsq, gamma, gammasq, gaussgam, erfS)*pdmag;
                    Real Ptemp = P(rij, rijsq, sigma, sigmasq, gaussgam)*pdmagsq_quarter;
                    Real Qtemp = Q(rij, rijsq, sigma, sigmasq, gaussgam)*pdmagsq_quarter;

                    Real temp1VF = (AFCMtemp - Atemp - Ctemp - Ptemp);
                    Real temp2VF = (BFCMtemp - Btemp - Dtemp - Qtemp);

                    // ------------WF+VT------------
                    Real fFCMtemp_VTWF = f(rij, rijsq, gammaVTWF_FCM, gammaVTWF_FCMsq, expS_VTWF_FCM, erfS_VTWF_FCM);
                    Real ftemp = f(rij, rijsq, gamma, gammasq, expS, erfS);
                    Real quatemp = (Real)0.25*pdmag/(gammasq)*gaussgam;

                    Real tempVTWF = (fFCMtemp_VTWF - ftemp + quatemp);

                    // ------------WT------------
                    Real fFCMtemp_WT = f(rij, rijsq, gammaWT_FCM, gammaWT_FCMsq, expS_WT_FCM, erfS_WT_FCM);
                    Real dfdrFCMtemp = dfdr(rij, rijsq, gammaWT_FCM, gammaWT_FCMsq, expS_WT_FCM, erfS_WT_FCM);
                    Real dfdrtemp = dfdr(rij, rijsq, gamma, gammasq, expS, erfS);

                    Real Tidotx = (T[3*i + 0]*xij + T[3*i + 1]*yij + T[3*i + 2]*zij);
                    Real Tjdotx = (T[3*j + 0]*xij + T[3*j + 1]*yij + T[3*j + 2]*zij);

                    Real temp1WT = (dfdrFCMtemp*rij + (Real)2.0*fFCMtemp_WT) - (dfdrtemp*rij + (Real)2.0*ftemp);
                    Real temp2WT = dfdrFCMtemp/rij - dfdrtemp/rij;

                    // Summation
                    wxi = wxi + (Real)0.5*( T[3*j + 0]*temp1WT - xij*Tjdotx*temp2WT ) + tempVTWF*( zij*F[3*j + 1] - yij*F[3*j + 2] );
                    wyi = wyi + (Real)0.5*( T[3*j + 1]*temp1WT - yij*Tjdotx*temp2WT ) + tempVTWF*( xij*F[3*j + 2] - zij*F[3*j + 0] );
                    wzi = wzi + (Real)0.5*( T[3*j + 2]*temp1WT - zij*Tjdotx*temp2WT ) + tempVTWF*( yij*F[3*j + 0] - xij*F[3*j + 1] );

                    vxi = vxi + temp1VF*F[3*j + 0] + temp2VF*xij*Fjdotx + tempVTWF*( zij*T[3*j + 1] - yij*T[3*j + 2] );
                    vyi = vyi + temp1VF*F[3*j + 1] + temp2VF*yij*Fjdotx + tempVTWF*( xij*T[3*j + 2] - zij*T[3*j + 0] );
                    vzi = vzi + temp1VF*F[3*j + 2] + temp2VF*zij*Fjdotx + tempVTWF*( yij*T[3*j + 0] - xij*T[3*j + 1] );
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

                xij = xij - boxsize * ((Real) ((int) (xij/(boxsize/Real(2.0)))));
                yij = yij - boxsize * ((Real) ((int) (yij/(boxsize/Real(2.0)))));
                zij = zij - boxsize * ((Real) ((int) (zij/(boxsize/Real(2.0)))));
                Real rijsq=xij*xij+yij*yij+zij*zij;
                if(rijsq < Rrefsq){
                    Real rij = sqrtf(rijsq);
                    Real erfS = erf((Real)0.5*rij/sigma);
                    Real expS = exp(-rijsq/((Real)2.0*gammasq));
                    Real gaussgam = expS/pow((Real)2.0*PI*gammasq, (Real)1.5);

                    Real erfS_VF_FCM = erf(rij/(sqrtf((Real)2.0)*gammaVF_FCM));
                    Real expS_VF_FCM = exp(-rijsq/((Real)2.0*gammaVF_FCMsq));

                    Real erfS_VTWF_FCM = erf(rij/(sqrtf((Real)2.0)*gammaVTWF_FCM));
                    Real expS_VTWF_FCM = exp(-rijsq/((Real)2.0*gammaVTWF_FCMsq));

                    Real erfS_WT_FCM = erf(rij/(sqrtf((Real)2.0)*gammaWT_FCM));
                    Real expS_WT_FCM = exp(-rijsq/((Real)2.0*gammaWT_FCMsq));

                    // ------------VF------------
                    Real Fjdotx = xij*F[3*j + 0] + yij*F[3*j + 1] + zij*F[3*j + 2];
                    Real Fidotx = xij*F[3*i + 0] + yij*F[3*i + 1] + zij*F[3*i + 2];

                    Real AFCMtemp = A(rij, rijsq, gammaVF_FCM, gammaVF_FCMsq, expS_VF_FCM, erfS_VF_FCM);
                    Real BFCMtemp = B(rij, rijsq, gammaVF_FCM, gammaVF_FCMsq, expS_VF_FCM, erfS_VF_FCM);
                    Real Atemp = A(rij, rijsq, gamma, gammasq, expS, erfS);
                    Real Btemp = B(rij, rijsq, gamma, gammasq, expS, erfS);
                    Real Ctemp = C(rij, rijsq, gamma, gammasq, gaussgam, erfS)*pdmag;
                    Real Dtemp = D(rij, rijsq, gamma, gammasq, gaussgam, erfS)*pdmag;
                    Real Ptemp = P(rij, rijsq, sigma, sigmasq, gaussgam)*pdmagsq_quarter;
                    Real Qtemp = Q(rij, rijsq, sigma, sigmasq, gaussgam)*pdmagsq_quarter;

                    Real temp1VF = (AFCMtemp - Atemp - Ctemp - Ptemp);
                    Real temp2VF = (BFCMtemp - Btemp - Dtemp - Qtemp);

                    // ------------WF+VT------------
                    Real fFCMtemp_VTWF = f(rij, rijsq, gammaVTWF_FCM, gammaVTWF_FCMsq, expS_VTWF_FCM, erfS_VTWF_FCM);
                    Real ftemp = f(rij, rijsq, gamma, gammasq, expS, erfS);
                    Real quatemp = (Real)0.25*pdmag/(gammasq)*gaussgam;

                    Real tempVTWF = (fFCMtemp_VTWF - ftemp + quatemp);

                    // ------------WT------------
                    Real fFCMtemp_WT = f(rij, rijsq, gammaWT_FCM, gammaWT_FCMsq, expS_WT_FCM, erfS_WT_FCM);
                    Real dfdrFCMtemp = dfdr(rij, rijsq, gammaWT_FCM, gammaWT_FCMsq, expS_WT_FCM, erfS_WT_FCM);
                    Real dfdrtemp = dfdr(rij, rijsq, gamma, gammasq, expS, erfS);

                    Real Tidotx = (T[3*i + 0]*xij + T[3*i + 1]*yij + T[3*i + 2]*zij);
                    Real Tjdotx = (T[3*j + 0]*xij + T[3*j + 1]*yij + T[3*j + 2]*zij);

                    Real temp1WT = (dfdrFCMtemp*rij + (Real)2.0*fFCMtemp_WT) - (dfdrtemp*rij + (Real)2.0*ftemp);
                    Real temp2WT = dfdrFCMtemp/rij - dfdrtemp/rij;

                    // Summation
                    wxi = wxi + (Real)0.5*( T[3*j + 0]*temp1WT - xij*Tjdotx*temp2WT ) + tempVTWF*( zij*F[3*j + 1] - yij*F[3*j + 2] );
                    wyi = wyi + (Real)0.5*( T[3*j + 1]*temp1WT - yij*Tjdotx*temp2WT ) + tempVTWF*( xij*F[3*j + 2] - zij*F[3*j + 0] );
                    wzi = wzi + (Real)0.5*( T[3*j + 2]*temp1WT - zij*Tjdotx*temp2WT ) + tempVTWF*( yij*F[3*j + 0] - xij*F[3*j + 1] );

                    atomicAdd(&W[3*j + 0], (Real)0.5*( T[3*i + 0]*temp1WT - xij*Tidotx*temp2WT ) - tempVTWF*( zij*F[3*i + 1] - yij*F[3*i + 2] ));
                    atomicAdd(&W[3*j + 1], (Real)0.5*( T[3*i + 1]*temp1WT - yij*Tidotx*temp2WT ) - tempVTWF*( xij*F[3*i + 2] - zij*F[3*i + 0] ));
                    atomicAdd(&W[3*j + 2], (Real)0.5*( T[3*i + 2]*temp1WT - zij*Tidotx*temp2WT ) - tempVTWF*( yij*F[3*i + 0] - xij*F[3*i + 1] ));

                    vxi = vxi + temp1VF*F[3*j + 0] + temp2VF*xij*Fjdotx + tempVTWF*( zij*T[3*j + 1] - yij*T[3*j + 2] );
                    vyi = vyi + temp1VF*F[3*j + 1] + temp2VF*yij*Fjdotx + tempVTWF*( xij*T[3*j + 2] - zij*T[3*j + 0] );
                    vzi = vzi + temp1VF*F[3*j + 2] + temp2VF*zij*Fjdotx + tempVTWF*( yij*T[3*j + 0] - xij*T[3*j + 1] );
                    
                    atomicAdd(&V[3*j + 0], temp1VF*F[3*i + 0] + temp2VF*xij*Fidotx - tempVTWF*( zij*T[3*i + 1] - yij*T[3*i + 2] ));
                    atomicAdd(&V[3*j + 1], temp1VF*F[3*i + 1] + temp2VF*yij*Fidotx - tempVTWF*( xij*T[3*i + 2] - zij*T[3*i + 0] ));
                    atomicAdd(&V[3*j + 2], temp1VF*F[3*i + 2] + temp2VF*zij*Fidotx - tempVTWF*( yij*T[3*i + 0] - xij*T[3*i + 1] ));

                }
            }
        }
        atomicAdd(&V[3*i + 0], vxi);
        atomicAdd(&V[3*i + 1], vyi);
        atomicAdd(&V[3*i + 2], vzi);
        atomicAdd(&W[3*i + 0], wxi);
        atomicAdd(&W[3*i + 1], wyi);
        atomicAdd(&W[3*i + 2], wzi);

        return;
    }
}

__global__
void cufcm_self_correction(Real* V, Real* W, Real* F, Real* T, int N, Real boxsize,
                                Real StokesMob, Real ModStokesMob,
                                Real PDStokesMob, Real BiLapMob,
                                Real WT1Mob, Real WT2Mob){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;
    
    for(int np = index; np < N; np += stride){
        V[3*np + 0] = V[3*np + 0] + F[3*np + 0]*(StokesMob - ModStokesMob + PDStokesMob - BiLapMob) ;
        V[3*np + 1] = V[3*np + 1] + F[3*np + 1]*(StokesMob - ModStokesMob + PDStokesMob - BiLapMob) ;
        V[3*np + 2] = V[3*np + 2] + F[3*np + 2]*(StokesMob - ModStokesMob + PDStokesMob - BiLapMob) ;

        W[3*np + 0] = W[3*np + 0] + T[3*np + 0]*(WT1Mob - WT2Mob) ;
        W[3*np + 1] = W[3*np + 1] + T[3*np + 1]*(WT1Mob - WT2Mob) ;
        W[3*np + 2] = W[3*np + 2] + T[3*np + 2]*(WT1Mob - WT2Mob) ;
    }

}

__global__
void cufcm_compute_formula(Real* Y, Real* V, Real* W, Real* F, Real* T, int N, int N_truncate, Real boxsize,
                    Real sigmaFCM,
                    Real sigmaFCMdip,
                    Real StokesMob,
                    Real WT1Mob,
                    Real hasimoto_ratio){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    Real gammaVF_FCM = sqrtf((Real)2.0)*sigmaFCM;
    Real gammaVF_FCMsq = gammaVF_FCM*gammaVF_FCM;
    Real gammaVTWF_FCM = sqrtf(sigmaFCM*sigmaFCM + sigmaFCMdip*sigmaFCMdip);
    Real gammaVTWF_FCMsq = gammaVTWF_FCM*gammaVTWF_FCM;
    Real gammaWT_FCM = sqrtf((Real)2.0)*sigmaFCMdip;
    Real gammaWT_FCMsq = gammaWT_FCM*gammaWT_FCM;
    Real vxi = (Real)0.0, vyi = (Real)0.0, vzi = (Real)0.0;
    Real wxi = (Real)0.0, wyi = (Real)0.0, wzi = (Real)0.0;

    for(int i = index; i < N_truncate; i += stride){
        Real xi = Y[3*i + 0], yi = Y[3*i + 1], zi = Y[3*i + 2];

        for(int j = 0; j < N; j++){
            if(i != j){
                Real xij = xi - Y[3*j + 0];
                Real yij = yi - Y[3*j + 1];
                Real zij = zi - Y[3*j + 2];

                xij = xij - boxsize * (Real) ((int) (xij/(boxsize/Real(2.0))));
                yij = yij - boxsize * (Real) ((int) (yij/(boxsize/Real(2.0))));
                zij = zij - boxsize * (Real) ((int) (zij/(boxsize/Real(2.0))));

                Real rijsq=xij*xij+yij*yij+zij*zij;
                Real rij = sqrtf(rijsq);

                Real erfS_VF_FCM = erf(rij/(sqrtf(Real(2.0))*gammaVF_FCM));
                Real expS_VF_FCM = exp(-rijsq/(Real(2.0)*gammaVF_FCMsq));

                Real erfS_VTWF_FCM = erf(rij/(sqrtf(Real(2.0))*gammaVTWF_FCM));
                Real expS_VTWF_FCM = exp(-rijsq/(Real(2.0)*gammaVTWF_FCMsq));

                Real erfS_WT_FCM = erf(rij/(sqrtf(Real(2.0))*gammaWT_FCM));
                Real expS_WT_FCM = exp(-rijsq/(Real(2.0)*gammaWT_FCMsq));

                // ------------VF------------
                Real Fjdotx = xij*F[3*j + 0] + yij*F[3*j + 1] + zij*F[3*j + 2];

                Real AFCMtemp = A(rij, rijsq, gammaVF_FCM, gammaVF_FCMsq, expS_VF_FCM, erfS_VF_FCM);
                Real BFCMtemp = B(rij, rijsq, gammaVF_FCM, gammaVF_FCMsq, expS_VF_FCM, erfS_VF_FCM);

                Real temp1VF = (AFCMtemp);
                Real temp2VF = (BFCMtemp);

                // ------------WF+VT------------
                Real fFCMtemp_VTWF = f(rij, rijsq, gammaVTWF_FCM, gammaVTWF_FCMsq, expS_VTWF_FCM, erfS_VTWF_FCM);

                Real tempVTWF = (fFCMtemp_VTWF);

                // ------------WT------------
                Real fFCMtemp_WT = f(rij, rijsq, gammaWT_FCM, gammaWT_FCMsq, expS_WT_FCM, erfS_WT_FCM);
                Real dfdrFCMtemp = dfdr(rij, rijsq, gammaWT_FCM, gammaWT_FCMsq, expS_WT_FCM, erfS_WT_FCM);

                Real Tjdotx = (T[3*j + 0]*xij + T[3*j + 1]*yij + T[3*j + 2]*zij);

                Real temp1WT = (dfdrFCMtemp*rij + Real(2.0)*fFCMtemp_WT);
                Real temp2WT = dfdrFCMtemp/rij;

                // Summation
                wxi += (Real)0.5*( T[3*j + 0]*temp1WT - xij*Tjdotx*temp2WT ) + tempVTWF*( zij*F[3*j + 1] - yij*F[3*j + 2] );
                wyi += (Real)0.5*( T[3*j + 1]*temp1WT - yij*Tjdotx*temp2WT ) + tempVTWF*( xij*F[3*j + 2] - zij*F[3*j + 0] );
                wzi += (Real)0.5*( T[3*j + 2]*temp1WT - zij*Tjdotx*temp2WT ) + tempVTWF*( yij*F[3*j + 0] - xij*F[3*j + 1] );

                vxi += temp1VF*F[3*j + 0]*hasimoto_ratio + temp2VF*xij*Fjdotx*hasimoto_ratio + tempVTWF*( zij*T[3*j + 1] - yij*T[3*j + 2] );
                vyi += temp1VF*F[3*j + 1]*hasimoto_ratio + temp2VF*yij*Fjdotx*hasimoto_ratio + tempVTWF*( xij*T[3*j + 2] - zij*T[3*j + 0] );
                vzi += temp1VF*F[3*j + 2]*hasimoto_ratio + temp2VF*zij*Fjdotx*hasimoto_ratio + tempVTWF*( yij*T[3*j + 0] - xij*T[3*j + 1] );
            }
        }

        vxi += F[3*i + 0]*(StokesMob)*hasimoto_ratio;
        vyi += F[3*i + 1]*(StokesMob)*hasimoto_ratio;
        vzi += F[3*i + 2]*(StokesMob)*hasimoto_ratio;

        wxi += T[3*i + 0]*(WT1Mob) ;
        wyi += T[3*i + 1]*(WT1Mob) ;
        wzi += T[3*i + 2]*(WT1Mob) ;

        V[3*i + 0] += vxi;
        V[3*i + 1] += vyi;
        V[3*i + 2] += vzi;
        W[3*i + 0] += wxi;
        W[3*i + 1] += wyi;
        W[3*i + 2] += wzi;        

        // printf("%d (%.8f %.8f %.8f) (%.8f %.8f %.8f) \n", i, vxi, vyi, vzi, wxi, wyi, wzi);

        return;
    }
}
