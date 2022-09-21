#include <iostream>
// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>

#include "config.hpp"
#include "fcmmacro.hpp"
#include "CUFCM_CORRECTION.hpp"

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
void cufcm_pair_correction_linklist(Real* Y, Real* V, Real* W, Real* F, Real* T, int N,
                    int *map, int *head, int *list,
                    int ncell, Real Rrefsq,
                    Real pdmag,
                    Real sigma, Real sigmasq,
                    Real sigmaFCM, Real sigmaFCMsq,
                    Real sigmaFCMdip, Real sigmaFCMdipsq){

    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    int icell = 0, i = 0, j = 0, jcello = 0, jcell = 0, nabor = 0;
    Real xi = 0.0, yi = 0.0, zi = 0.0, vxi = 0.0, vyi = 0.0, vzi = 0.0, wxi = 0.0, wyi = 0.0, wzi = 0.0, xij = 0.0, yij = 0.0, zij = 0.0, rijsq = 0.0, rij = 0.0;
    Real pdmagsq_quarter = pdmag * pdmag * 0.25;
    Real quatemp = 0.0;
    Real temp1VF, temp2VF;
    Real tempVTWF;
    Real temp1WT, temp2WT;

    Real gamma, gammasq, gammaVF_FCM, gammaVF_FCMsq, gammaVTWF_FCM, gammaVTWF_FCMsq, gammaWT_FCM, gammaWT_FCMsq;
    Real erfS, expS, gaussgam, erfS_VF_FCM, expS_VF_FCM, erfS_VTWF_FCM, expS_VTWF_FCM, erfS_WT_FCM, expS_WT_FCM;

    Real Atemp, Btemp, Ctemp, Dtemp, AFCMtemp, BFCMtemp, Ptemp, Qtemp;
    Real ftemp, fFCMtemp_VTWF, fFCMtemp_WT, dfdrtemp, dfdrFCMtemp;
    Real Fjdotx, Fidotx;
    Real Tjdotx, Tidotx;

    gamma = sqrtf((Real)2.0)*sigma;
    gammasq = gamma*gamma;

    gammaVF_FCM = sqrtf((Real)2.0)*sigmaFCM;
    gammaVF_FCMsq = gammaVF_FCM*gammaVF_FCM;

    gammaVTWF_FCM = sqrtf(sigmaFCM*sigmaFCM + sigmaFCMdip*sigmaFCMdip);
    gammaVTWF_FCMsq = gammaVTWF_FCM*gammaVTWF_FCM;

    gammaWT_FCM = sqrtf((Real)2.0)*sigmaFCMdip;
    gammaWT_FCMsq = gammaWT_FCM*gammaWT_FCM;

    int m = 0;

    for(int np = index; np < N; np += stride){
        // Create mapping from np to i
        i = head[icell];
        while(m <= np){
            if(i > -1){
                i = list[i];
            }
            else{
                if(icell < ncell){
                    icell++;
                    i = head[icell];
                }
            }
            if(i > -1){
                m++;
            }

        }

        xi = Y[3*i + 0];
        yi = Y[3*i + 1];
        zi = Y[3*i + 2];

        j = list[i];
        /* intra-cell interactions */
        /* corrections apply to both parties */
        while(j > -1){
            xij = xi - Y[3*j + 0];
            yij = yi - Y[3*j + 1];
            zij = zi - Y[3*j + 2];

            xij = xij - PI2 * ((Real) ((int) (xij/PI)));
            yij = yij - PI2 * ((Real) ((int) (yij/PI)));
            zij = zij - PI2 * ((Real) ((int) (zij/PI)));

            rijsq=xij*xij+yij*yij+zij*zij;
            if(rijsq < Rrefsq){
                rij = sqrtf(rijsq);
                erfS = erf(0.5*rij/sigma);
                expS = exp(-rijsq/(2.0*gammasq));
                gaussgam = expS/pow(2.0*PI*gammasq, 1.5);

                erfS_VF_FCM = erf(rij/(sqrtf(2)*gammaVF_FCM));
                expS_VF_FCM = exp(-rijsq/(2.0*gammaVF_FCMsq));

                erfS_VTWF_FCM = erf(rij/(sqrtf(2)*gammaVTWF_FCM));
                expS_VTWF_FCM = exp(-rijsq/(2.0*gammaVTWF_FCMsq));

                erfS_WT_FCM = erf(rij/(sqrtf(2)*gammaWT_FCM));
                expS_WT_FCM = exp(-rijsq/(2.0*gammaWT_FCMsq));

                // ------------VF------------
                Fjdotx = xij*F[3*j + 0] + yij*F[3*j + 1] + zij*F[3*j + 2];
                Fidotx = xij*F[3*i + 0] + yij*F[3*i + 1] + zij*F[3*i + 2];

                AFCMtemp = A(rij, rijsq, gammaVF_FCM, gammaVF_FCMsq, expS_VF_FCM, erfS_VF_FCM);
                BFCMtemp = B(rij, rijsq, gammaVF_FCM, gammaVF_FCMsq, expS_VF_FCM, erfS_VF_FCM);
                Atemp = A(rij, rijsq, gamma, gammasq, expS, erfS);
                Btemp = B(rij, rijsq, gamma, gammasq, expS, erfS);
                Ctemp = C(rij, rijsq, gamma, gammasq, gaussgam, erfS)*pdmag;
                Dtemp = D(rij, rijsq, gamma, gammasq, gaussgam, erfS)*pdmag;
                Ptemp = P(rij, rijsq, sigma, sigmasq, gaussgam)*pdmagsq_quarter;
                Qtemp = Q(rij, rijsq, sigma, sigmasq, gaussgam)*pdmagsq_quarter;

                temp1VF = (AFCMtemp - Atemp - Ctemp - Ptemp);
                temp2VF = (BFCMtemp - Btemp - Dtemp - Qtemp);

                // ------------WF+VT------------
                fFCMtemp_VTWF = f(rij, rijsq, gammaVTWF_FCM, gammaVTWF_FCMsq, expS_VTWF_FCM, erfS_VTWF_FCM);
                ftemp = f(rij, rijsq, gamma, gammasq, expS, erfS);
                quatemp = 0.25*pdmag/(gammasq)*gaussgam;

                tempVTWF = (fFCMtemp_VTWF - ftemp + quatemp);

                // ------------WT------------
                fFCMtemp_WT = f(rij, rijsq, gammaWT_FCM, gammaWT_FCMsq, expS_WT_FCM, erfS_WT_FCM);
                dfdrFCMtemp = dfdr(rij, rijsq, gammaWT_FCM, gammaWT_FCMsq, expS_WT_FCM, erfS_WT_FCM);
                dfdrtemp = dfdr(rij, rijsq, gamma, gammasq, expS, erfS);

                Tidotx = (T[3*i + 0]*xij + T[3*i + 1]*yij + T[3*i + 2]*zij);
                Tjdotx = (T[3*j + 0]*xij + T[3*j + 1]*yij + T[3*j + 2]*zij);

                temp1WT = (dfdrFCMtemp*rij + 2.0*fFCMtemp_WT) - (dfdrtemp*rij + 2.0*ftemp);
                temp2WT = dfdrFCMtemp/rij - dfdrtemp/rij;

                // Summation
                wxi = wxi + 0.5*( T[3*j + 0]*temp1WT - xij*Tjdotx*temp2WT ) + tempVTWF*( zij*F[3*j + 1] - yij*F[3*j + 2] );
                wyi = wyi + 0.5*( T[3*j + 1]*temp1WT - yij*Tjdotx*temp2WT ) + tempVTWF*( xij*F[3*j + 2] - zij*F[3*j + 0] );
                wzi = wzi + 0.5*( T[3*j + 2]*temp1WT - zij*Tjdotx*temp2WT ) + tempVTWF*( yij*F[3*j + 0] - xij*F[3*j + 1] );

                atomicAdd(&W[3*j + 0], 0.5*( T[3*i + 0]*temp1WT - xij*Tidotx*temp2WT ) - tempVTWF*( zij*F[3*i + 1] - yij*F[3*i + 2] ));
                atomicAdd(&W[3*j + 1], 0.5*( T[3*i + 1]*temp1WT - yij*Tidotx*temp2WT ) - tempVTWF*( xij*F[3*i + 2] - zij*F[3*i + 0] ));
                atomicAdd(&W[3*j + 2], 0.5*( T[3*i + 2]*temp1WT - zij*Tidotx*temp2WT ) - tempVTWF*( yij*F[3*i + 0] - xij*F[3*i + 1] ));

                vxi = vxi + temp1VF*F[3*j + 0] + temp2VF*xij*Fjdotx + tempVTWF*( zij*T[3*j + 1] - yij*T[3*j + 2] );
                vyi = vyi + temp1VF*F[3*j + 1] + temp2VF*yij*Fjdotx + tempVTWF*( xij*T[3*j + 2] - zij*T[3*j + 0] );
                vzi = vzi + temp1VF*F[3*j + 2] + temp2VF*zij*Fjdotx + tempVTWF*( yij*T[3*j + 0] - xij*T[3*j + 1] );

                atomicAdd(&V[3*j + 0], temp1VF*F[3*i + 0] + temp2VF*xij*Fidotx - tempVTWF*( zij*T[3*i + 1] - yij*T[3*i + 2] ));
                atomicAdd(&V[3*j + 1], temp1VF*F[3*i + 1] + temp2VF*yij*Fidotx - tempVTWF*( xij*T[3*i + 2] - zij*T[3*i + 0] ));
                atomicAdd(&V[3*j + 2], temp1VF*F[3*i + 2] + temp2VF*zij*Fidotx - tempVTWF*( yij*T[3*i + 0] - xij*T[3*i + 1] ));

            }
            j = list[j];
        }
        jcello = 13*icell;
        // inter-cell interactions
        for(nabor = 0; nabor < 13; nabor++){
            jcell = map[jcello + nabor];
            j = head[jcell];
            while(j > -1){
                xij = xi - Y[3*j + 0];
                yij = yi - Y[3*j + 1];
                zij = zi - Y[3*j + 2];

                xij = xij - PI2 * ((Real) ((int) (xij/PI)));
                yij = yij - PI2 * ((Real) ((int) (yij/PI)));
                zij = zij - PI2 * ((Real) ((int) (zij/PI)));
                rijsq=xij*xij+yij*yij+zij*zij;
                if(rijsq < Rrefsq){
                    rij = sqrtf(rijsq);
                    erfS = erf(0.5*rij/sigma);
                    expS = exp(-rijsq/(2.0*gammasq));
                    gaussgam = expS/pow(2.0*PI*gammasq, 1.5);

                    erfS_VF_FCM = erf(rij/(sqrtf(2)*gammaVF_FCM));
                    expS_VF_FCM = exp(-rijsq/(2.0*gammaVF_FCMsq));

                    erfS_VTWF_FCM = erf(rij/(sqrtf(2)*gammaVTWF_FCM));
                    expS_VTWF_FCM = exp(-rijsq/(2.0*gammaVTWF_FCMsq));

                    erfS_WT_FCM = erf(rij/(sqrtf(2)*gammaWT_FCM));
                    expS_WT_FCM = exp(-rijsq/(2.0*gammaWT_FCMsq));

                    // ------------VF------------
                    Fjdotx = xij*F[3*j + 0] + yij*F[3*j + 1] + zij*F[3*j + 2];
                    Fidotx = xij*F[3*i + 0] + yij*F[3*i + 1] + zij*F[3*i + 2];

                    AFCMtemp = A(rij, rijsq, gammaVF_FCM, gammaVF_FCMsq, expS_VF_FCM, erfS_VF_FCM);
                    BFCMtemp = B(rij, rijsq, gammaVF_FCM, gammaVF_FCMsq, expS_VF_FCM, erfS_VF_FCM);
                    Atemp = A(rij, rijsq, gamma, gammasq, expS, erfS);
                    Btemp = B(rij, rijsq, gamma, gammasq, expS, erfS);
                    Ctemp = C(rij, rijsq, gamma, gammasq, gaussgam, erfS)*pdmag;
                    Dtemp = D(rij, rijsq, gamma, gammasq, gaussgam, erfS)*pdmag;
                    Ptemp = P(rij, rijsq, sigma, sigmasq, gaussgam)*pdmagsq_quarter;
                    Qtemp = Q(rij, rijsq, sigma, sigmasq, gaussgam)*pdmagsq_quarter;

                    temp1VF = (AFCMtemp - Atemp - Ctemp - Ptemp);
                    temp2VF = (BFCMtemp - Btemp - Dtemp - Qtemp);

                    // ------------WF+VT------------
                    fFCMtemp_VTWF = f(rij, rijsq, gammaVTWF_FCM, gammaVTWF_FCMsq, expS_VTWF_FCM, erfS_VTWF_FCM);
                    ftemp = f(rij, rijsq, gamma, gammasq, expS, erfS);
                    quatemp = 0.25*pdmag/(gammasq)*gaussgam;

                    tempVTWF = (fFCMtemp_VTWF - ftemp + quatemp);

                    // ------------WT------------
                    fFCMtemp_WT = f(rij, rijsq, gammaWT_FCM, gammaWT_FCMsq, expS_WT_FCM, erfS_WT_FCM);
                    dfdrFCMtemp = dfdr(rij, rijsq, gammaWT_FCM, gammaWT_FCMsq, expS_WT_FCM, erfS_WT_FCM);
                    dfdrtemp = dfdr(rij, rijsq, gamma, gammasq, expS, erfS);

                    Tidotx = (T[3*i + 0]*xij + T[3*i + 1]*yij + T[3*i + 2]*zij);
                    Tjdotx = (T[3*j + 0]*xij + T[3*j + 1]*yij + T[3*j + 2]*zij);

                    temp1WT = (dfdrFCMtemp*rij + 2.0*fFCMtemp_WT) - (dfdrtemp*rij + 2.0*ftemp);
                    temp2WT = dfdrFCMtemp/rij - dfdrtemp/rij;

                    // Summation
                    wxi = wxi + 0.5*( T[3*j + 0]*temp1WT - xij*Tjdotx*temp2WT ) + tempVTWF*( zij*F[3*j + 1] - yij*F[3*j + 2] );
                    wyi = wyi + 0.5*( T[3*j + 1]*temp1WT - yij*Tjdotx*temp2WT ) + tempVTWF*( xij*F[3*j + 2] - zij*F[3*j + 0] );
                    wzi = wzi + 0.5*( T[3*j + 2]*temp1WT - zij*Tjdotx*temp2WT ) + tempVTWF*( yij*F[3*j + 0] - xij*F[3*j + 1] );

                    atomicAdd(&W[3*j + 0], 0.5*( T[3*i + 0]*temp1WT - xij*Tidotx*temp2WT ) - tempVTWF*( zij*F[3*i + 1] - yij*F[3*i + 2] ));
                    atomicAdd(&W[3*j + 1], 0.5*( T[3*i + 1]*temp1WT - yij*Tidotx*temp2WT ) - tempVTWF*( xij*F[3*i + 2] - zij*F[3*i + 0] ));
                    atomicAdd(&W[3*j + 2], 0.5*( T[3*i + 2]*temp1WT - zij*Tidotx*temp2WT ) - tempVTWF*( yij*F[3*i + 0] - xij*F[3*i + 1] ));

                    vxi = vxi + temp1VF*F[3*j + 0] + temp2VF*xij*Fjdotx + tempVTWF*( zij*T[3*j + 1] - yij*T[3*j + 2] );
                    vyi = vyi + temp1VF*F[3*j + 1] + temp2VF*yij*Fjdotx + tempVTWF*( xij*T[3*j + 2] - zij*T[3*j + 0] );
                    vzi = vzi + temp1VF*F[3*j + 2] + temp2VF*zij*Fjdotx + tempVTWF*( yij*T[3*j + 0] - xij*T[3*j + 1] );
                    
                    atomicAdd(&V[3*j + 0], temp1VF*F[3*i + 0] + temp2VF*xij*Fidotx - tempVTWF*( zij*T[3*i + 1] - yij*T[3*i + 2] ));
                    atomicAdd(&V[3*j + 1], temp1VF*F[3*i + 1] + temp2VF*yij*Fidotx - tempVTWF*( xij*T[3*i + 2] - zij*T[3*i + 0] ));
                    atomicAdd(&V[3*j + 2], temp1VF*F[3*i + 2] + temp2VF*zij*Fidotx - tempVTWF*( yij*T[3*i + 0] - xij*T[3*i + 1] ));

                }
                j = list[j];
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
void cufcm_pair_correction_spatial_hashing_tpp(Real* Y, Real* V, Real* W, Real* F, Real* T, int N,
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

                xij = xij - PI2 * (Real) ((int) (xij/PI));
                yij = yij - PI2 * (Real) ((int) (yij/PI));
                zij = zij - PI2 * (Real) ((int) (zij/PI));

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

                xij = xij - PI2 * ((Real) ((int) (xij/PI)));
                yij = yij - PI2 * ((Real) ((int) (yij/PI)));
                zij = zij - PI2 * ((Real) ((int) (zij/PI)));
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

        W[3*np + 0] = W[3*np + 0] + T[3*np + 0]*(WT1Mob - WT2Mob) ;
        W[3*np + 1] = W[3*np + 1] + T[3*np + 1]*(WT1Mob - WT2Mob) ;
        W[3*np + 2] = W[3*np + 2] + T[3*np + 2]*(WT1Mob - WT2Mob) ;
    }

}

void cufcm_pair_correction_loop(Real* Y, Real* V, Real* W, Real* F, Real* T, int N,
                    int *map, int *head, int *list,
                    int ncell, Real Rrefsq,
                    Real pdmag,
                    Real sigma, Real sigmasq,
                    Real sigmaFCM, Real sigmaFCMsq,
                    Real sigmaFCMdip, Real sigmaFCMdipsq){
    int icell = 0, i = 0, j = 0, jcello = 0, jcell = 0, nabor = 0;
    Real xi = 0.0, yi = 0.0, zi = 0.0, vxi = 0.0, vyi = 0.0, vzi = 0.0, wxi = 0.0, wyi = 0.0, wzi = 0.0, xij = 0.0, yij = 0.0, zij = 0.0, rijsq = 0.0, rij = 0.0;
    Real pdmagsq_quarter = pdmag * pdmag * 0.25;
    Real quatemp = 0.0;
    Real temp1VF, temp2VF;
    Real tempVTWF;
    Real temp1WT, temp2WT;

    Real gamma, gammasq, gammaVF_FCM, gammaVF_FCMsq, gammaVTWF_FCM, gammaVTWF_FCMsq, gammaWT_FCM, gammaWT_FCMsq;
    Real erfS, expS, gaussgam, erfS_VF_FCM, expS_VF_FCM, erfS_VTWF_FCM, expS_VTWF_FCM, erfS_WT_FCM, expS_WT_FCM;

    Real Atemp, Btemp, Ctemp, Dtemp, AFCMtemp, BFCMtemp, Ptemp, Qtemp;
    Real ftemp, fFCMtemp_VTWF, fFCMtemp_WT, dfdrtemp, dfdrFCMtemp;
    Real Fjdotx, Fidotx;
    Real Tjdotx, Tidotx;

    gamma = sqrt(2.0)*sigma;
    gammasq = gamma*gamma;

    gammaVF_FCM = sqrt(2.0)*sigmaFCM;
    gammaVF_FCMsq = gammaVF_FCM*gammaVF_FCM;

    gammaVTWF_FCM = sqrt(sigmaFCM*sigmaFCM + sigmaFCMdip*sigmaFCMdip);
    gammaVTWF_FCMsq = gammaVTWF_FCM*gammaVTWF_FCM;

    gammaWT_FCM = sqrt(2.0)*sigmaFCMdip;
    gammaWT_FCMsq = gammaWT_FCM*gammaWT_FCM;

    for(icell = 0; icell < ncell; icell++){
        i = head[icell];
        while(i > -1){
            xi = Y[3*i + 0];
            yi = Y[3*i + 1];
            zi = Y[3*i + 2];
            vxi = V[3*i + 0];
            vyi = V[3*i + 1];
            vzi = V[3*i + 2];
            wxi = W[3*i + 0];
            wyi = W[3*i + 1];
            wzi = W[3*i + 2];
            j = list[i];
            // intra-cell interactions
            // corrections apply to both parties
            while(j > -1){
                // printf("interaction between %d and %d\n", i, j);
                xij = xi - Y[3*j + 0];
                yij = yi - Y[3*j + 1];
                zij = zi - Y[3*j + 2];

                xij = xij - PI2 * ((Real) ((int) (xij/PI)));
                yij = yij - PI2 * ((Real) ((int) (yij/PI)));
                zij = zij - PI2 * ((Real) ((int) (zij/PI)));

                rijsq=xij*xij+yij*yij+zij*zij;
                if(rijsq < Rrefsq){
                    rij = sqrt(rijsq);
                    erfS = erf(0.5*rij/sigma);
                    expS = exp(-rijsq/(2.0*gammasq));
                    gaussgam = expS/pow(2.0*PI*gammasq, 1.5);

                    erfS_VF_FCM = erf(rij/(sqrt(2)*gammaVF_FCM));
                    expS_VF_FCM = exp(-rijsq/(2.0*gammaVF_FCMsq));

                    erfS_VTWF_FCM = erf(rij/(sqrt(2)*gammaVTWF_FCM));
                    expS_VTWF_FCM = exp(-rijsq/(2.0*gammaVTWF_FCMsq));

                    erfS_WT_FCM = erf(rij/(sqrt(2)*gammaWT_FCM));
                    expS_WT_FCM = exp(-rijsq/(2.0*gammaWT_FCMsq));

                    // ------------VF------------
                    Fjdotx = xij*F[3*j + 0] + yij*F[3*j + 1] + zij*F[3*j + 2];
                    Fidotx = xij*F[3*i + 0] + yij*F[3*i + 1] + zij*F[3*i + 2];

                    AFCMtemp = A(rij, rijsq, gammaVF_FCM, gammaVF_FCMsq, expS_VF_FCM, erfS_VF_FCM);
                    BFCMtemp = B(rij, rijsq, gammaVF_FCM, gammaVF_FCMsq, expS_VF_FCM, erfS_VF_FCM);
                    Atemp = A(rij, rijsq, gamma, gammasq, expS, erfS);
                    Btemp = B(rij, rijsq, gamma, gammasq, expS, erfS);
                    Ctemp = C(rij, rijsq, gamma, gammasq, gaussgam, erfS)*pdmag;
                    Dtemp = D(rij, rijsq, gamma, gammasq, gaussgam, erfS)*pdmag;
                    Ptemp = P(rij, rijsq, sigma, sigmasq, gaussgam)*pdmagsq_quarter;
                    Qtemp = Q(rij, rijsq, sigma, sigmasq, gaussgam)*pdmagsq_quarter;

                    temp1VF = (AFCMtemp - Atemp - Ctemp - Ptemp);
                    temp2VF = (BFCMtemp - Btemp - Dtemp - Qtemp);

                    // ------------WF+VT------------
                    fFCMtemp_VTWF = f(rij, rijsq, gammaVTWF_FCM, gammaVTWF_FCMsq, expS_VTWF_FCM, erfS_VTWF_FCM);
                    ftemp = f(rij, rijsq, gamma, gammasq, expS, erfS);
                    quatemp = 0.25*pdmag/(gammasq)*gaussgam;

                    tempVTWF = (fFCMtemp_VTWF - ftemp + quatemp);

                    // ------------WT------------
                    fFCMtemp_WT = f(rij, rijsq, gammaWT_FCM, gammaWT_FCMsq, expS_WT_FCM, erfS_WT_FCM);
                    dfdrFCMtemp = dfdr(rij, rijsq, gammaWT_FCM, gammaWT_FCMsq, expS_WT_FCM, erfS_WT_FCM);
                    dfdrtemp = dfdr(rij, rijsq, gamma, gammasq, expS, erfS);

                    Tidotx = (T[3*i + 0]*xij + T[3*i + 1]*yij + T[3*i + 2]*zij);
                    Tjdotx = (T[3*j + 0]*xij + T[3*j + 1]*yij + T[3*j + 2]*zij);

                    temp1WT = (dfdrFCMtemp*rij + 2.0*fFCMtemp_WT) - (dfdrtemp*rij + 2.0*ftemp);
                    temp2WT = dfdrFCMtemp/rij - dfdrtemp/rij;

                    // Summation
                    wxi = wxi + 0.5*( T[3*j + 0]*temp1WT - xij*Tjdotx*temp2WT ) + tempVTWF*( zij*F[3*j + 1] - yij*F[3*j + 2] );
                    wyi = wyi + 0.5*( T[3*j + 1]*temp1WT - yij*Tjdotx*temp2WT ) + tempVTWF*( xij*F[3*j + 2] - zij*F[3*j + 0] );
                    wzi = wzi + 0.5*( T[3*j + 2]*temp1WT - zij*Tjdotx*temp2WT ) + tempVTWF*( yij*F[3*j + 0] - xij*F[3*j + 1] );

                    W[3*j + 0] = W[3*j + 0] + 0.5*( T[3*i + 0]*temp1WT - xij*Tidotx*temp2WT ) - tempVTWF*( zij*F[3*i + 1] - yij*F[3*i + 2] );
                    W[3*j + 1] = W[3*j + 1] + 0.5*( T[3*i + 1]*temp1WT - yij*Tidotx*temp2WT ) - tempVTWF*( xij*F[3*i + 2] - zij*F[3*i + 0] );
                    W[3*j + 2] = W[3*j + 2] + 0.5*( T[3*i + 2]*temp1WT - zij*Tidotx*temp2WT ) - tempVTWF*( yij*F[3*i + 0] - xij*F[3*i + 1] );

                    vxi = vxi + temp1VF*F[3*j + 0] + temp2VF*xij*Fjdotx + tempVTWF*( zij*T[3*j + 1] - yij*T[3*j + 2] );
                    vyi = vyi + temp1VF*F[3*j + 1] + temp2VF*yij*Fjdotx + tempVTWF*( xij*T[3*j + 2] - zij*T[3*j + 0] );
                    vzi = vzi + temp1VF*F[3*j + 2] + temp2VF*zij*Fjdotx + tempVTWF*( yij*T[3*j + 0] - xij*T[3*j + 1] );

                    V[3*j + 0] = V[3*j + 0] + temp1VF*F[3*i + 0] + temp2VF*xij*Fidotx - tempVTWF*( zij*T[3*i + 1] - yij*T[3*i + 2] );
                    V[3*j + 1] = V[3*j + 1] + temp1VF*F[3*i + 1] + temp2VF*yij*Fidotx - tempVTWF*( xij*T[3*i + 2] - zij*T[3*i + 0] );
                    V[3*j + 2] = V[3*j + 2] + temp1VF*F[3*i + 2] + temp2VF*zij*Fidotx - tempVTWF*( yij*T[3*i + 0] - xij*T[3*i + 1] );
                }
                j = list[j];
            }
            jcello = 13*icell;
            // inter-cell interactions
            for(nabor = 0; nabor < 13; nabor++){
                jcell = map[jcello + nabor];
                j = head[jcell];
                while(j > -1){
                    xij = xi - Y[3*j + 0];
                    yij = yi - Y[3*j + 1];
                    zij = zi - Y[3*j + 2];

                    xij = xij - PI2 * ((Real) ((int) (xij/PI)));
                    yij = yij - PI2 * ((Real) ((int) (yij/PI)));
                    zij = zij - PI2 * ((Real) ((int) (zij/PI)));
                    rijsq=xij*xij+yij*yij+zij*zij;
                    if(rijsq < Rrefsq){
                        rij = sqrt(rijsq);
                        erfS = erf(0.5*rij/sigma);
                        expS = exp(-rijsq/(2.0*gammasq));
                        gaussgam = expS/pow(2.0*PI*gammasq, 1.5);

                        erfS_VF_FCM = erf(rij/(sqrt(2)*gammaVF_FCM));
                        expS_VF_FCM = exp(-rijsq/(2.0*gammaVF_FCMsq));

                        erfS_VTWF_FCM = erf(rij/(sqrt(2)*gammaVTWF_FCM));
                        expS_VTWF_FCM = exp(-rijsq/(2.0*gammaVTWF_FCMsq));

                        erfS_WT_FCM = erf(rij/(sqrt(2)*gammaWT_FCM));
                        expS_WT_FCM = exp(-rijsq/(2.0*gammaWT_FCMsq));

                        // ------------VF------------
                        Fjdotx = xij*F[3*j + 0] + yij*F[3*j + 1] + zij*F[3*j + 2];
                        Fidotx = xij*F[3*i + 0] + yij*F[3*i + 1] + zij*F[3*i + 2];

                        AFCMtemp = A(rij, rijsq, gammaVF_FCM, gammaVF_FCMsq, expS_VF_FCM, erfS_VF_FCM);
                        BFCMtemp = B(rij, rijsq, gammaVF_FCM, gammaVF_FCMsq, expS_VF_FCM, erfS_VF_FCM);
                        Atemp = A(rij, rijsq, gamma, gammasq, expS, erfS);
                        Btemp = B(rij, rijsq, gamma, gammasq, expS, erfS);
                        Ctemp = C(rij, rijsq, gamma, gammasq, gaussgam, erfS)*pdmag;
                        Dtemp = D(rij, rijsq, gamma, gammasq, gaussgam, erfS)*pdmag;
                        Ptemp = P(rij, rijsq, sigma, sigmasq, gaussgam)*pdmagsq_quarter;
                        Qtemp = Q(rij, rijsq, sigma, sigmasq, gaussgam)*pdmagsq_quarter;

                        temp1VF = (AFCMtemp - Atemp - Ctemp - Ptemp);
                        temp2VF = (BFCMtemp - Btemp - Dtemp - Qtemp);

                        // ------------WF+VT------------
                        fFCMtemp_VTWF = f(rij, rijsq, gammaVTWF_FCM, gammaVTWF_FCMsq, expS_VTWF_FCM, erfS_VTWF_FCM);
                        ftemp = f(rij, rijsq, gamma, gammasq, expS, erfS);
                        quatemp = 0.25*pdmag/(gammasq)*gaussgam;

                        tempVTWF = (fFCMtemp_VTWF - ftemp + quatemp);

                        // ------------WT------------
                        fFCMtemp_WT = f(rij, rijsq, gammaWT_FCM, gammaWT_FCMsq, expS_WT_FCM, erfS_WT_FCM);
                        dfdrFCMtemp = dfdr(rij, rijsq, gammaWT_FCM, gammaWT_FCMsq, expS_WT_FCM, erfS_WT_FCM);
                        dfdrtemp = dfdr(rij, rijsq, gamma, gammasq, expS, erfS);

                        Tidotx = (T[3*i + 0]*xij + T[3*i + 1]*yij + T[3*i + 2]*zij);
                        Tjdotx = (T[3*j + 0]*xij + T[3*j + 1]*yij + T[3*j + 2]*zij);

                        temp1WT = (dfdrFCMtemp*rij + 2.0*fFCMtemp_WT) - (dfdrtemp*rij + 2.0*ftemp);
                        temp2WT = dfdrFCMtemp/rij - dfdrtemp/rij;

                        // Summation
                        wxi = wxi + 0.5*( T[3*j + 0]*temp1WT - xij*Tjdotx*temp2WT ) + tempVTWF*( zij*F[3*j + 1] - yij*F[3*j + 2] );
                        wyi = wyi + 0.5*( T[3*j + 1]*temp1WT - yij*Tjdotx*temp2WT ) + tempVTWF*( xij*F[3*j + 2] - zij*F[3*j + 0] );
                        wzi = wzi + 0.5*( T[3*j + 2]*temp1WT - zij*Tjdotx*temp2WT ) + tempVTWF*( yij*F[3*j + 0] - xij*F[3*j + 1] );

                        W[3*j + 0] = W[3*j + 0] + 0.5*( T[3*i + 0]*temp1WT - xij*Tidotx*temp2WT ) - tempVTWF*( zij*F[3*i + 1] - yij*F[3*i + 2] );
                        W[3*j + 1] = W[3*j + 1] + 0.5*( T[3*i + 1]*temp1WT - yij*Tidotx*temp2WT ) - tempVTWF*( xij*F[3*i + 2] - zij*F[3*i + 0] );
                        W[3*j + 2] = W[3*j + 2] + 0.5*( T[3*i + 2]*temp1WT - zij*Tidotx*temp2WT ) - tempVTWF*( yij*F[3*i + 0] - xij*F[3*i + 1] );

                        vxi = vxi + temp1VF*F[3*j + 0] + temp2VF*xij*Fjdotx + tempVTWF*( zij*T[3*j + 1] - yij*T[3*j + 2] );
                        vyi = vyi + temp1VF*F[3*j + 1] + temp2VF*yij*Fjdotx + tempVTWF*( xij*T[3*j + 2] - zij*T[3*j + 0] );
                        vzi = vzi + temp1VF*F[3*j + 2] + temp2VF*zij*Fjdotx + tempVTWF*( yij*T[3*j + 0] - xij*T[3*j + 1] );

                        V[3*j + 0] = V[3*j + 0] + temp1VF*F[3*i + 0] + temp2VF*xij*Fidotx - tempVTWF*( zij*T[3*i + 1] - yij*T[3*i + 2] );
                        V[3*j + 1] = V[3*j + 1] + temp1VF*F[3*i + 1] + temp2VF*yij*Fidotx - tempVTWF*( xij*T[3*i + 2] - zij*T[3*i + 0] );
                        V[3*j + 2] = V[3*j + 2] + temp1VF*F[3*i + 2] + temp2VF*zij*Fidotx - tempVTWF*( yij*T[3*i + 0] - xij*T[3*i + 1] );
                    }
                    j = list[j];
                }
            }
            V[3*i + 0] = vxi;
            V[3*i + 1] = vyi;
            V[3*i + 2] = vzi;
            W[3*i + 0] = wxi;
            W[3*i + 1] = wyi;
            W[3*i + 2] = wzi;
            i = list[i];
        }
    }
    return;
}

void cufcm_self_correction_loop(Real* V, Real* W, Real* F, Real* T, int N,
                                Real StokesMob, Real ModStokesMob,
                                Real PDStokesMob, Real BiLapMob,
                                Real WT1Mob, Real WT2Mob){
    for(int i = 0; i < N; i++){
        V[3*i + 0] = V[3*i + 0] + F[3*i + 0]*(StokesMob - ModStokesMob + PDStokesMob - BiLapMob) ;
        V[3*i + 1] = V[3*i + 1] + F[3*i + 1]*(StokesMob - ModStokesMob + PDStokesMob - BiLapMob) ;
        V[3*i + 2] = V[3*i + 2] + F[3*i + 2]*(StokesMob - ModStokesMob + PDStokesMob - BiLapMob) ;

        W[3*i + 0] = W[3*i + 0] + T[3*i + 0]*(WT1Mob - WT2Mob) ;
        W[3*i + 1] = W[3*i + 1] + T[3*i + 1]*(WT1Mob - WT2Mob) ;
        W[3*i + 2] = W[3*i + 2] + T[3*i + 2]*(WT1Mob - WT2Mob) ;
    }
}