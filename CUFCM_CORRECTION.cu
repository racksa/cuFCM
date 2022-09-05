#include <iostream>
// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>

#include "config.hpp"
#include "CUFCM_CORRECTION.hpp"

__device__ __host__
double f(double r, double rsq, double sigma, double sigmasq, double expS, double erfS){
	return 1.0/(8.0*PI*pow(r, 3)) * ( erfS - r/sigma*sqrt(2.0/PI) * expS );
}

__device__ __host__
double dfdr(double r, double rsq, double sigma, double sigmasq, double expS, double erfS){
	return -3.0/r*f(r, rsq, sigma, sigmasq, expS, erfS) + 1.0/(8.0*PI*pow(r, 3)) * (rsq/(sigmasq*sigma)) * sqrt(2.0/PI) * expS;
}

// S_ij = A(r) delta_ij + B(r) x_ix_j
__device__ __host__
double A(double r, double rsq, double sigma, double sigmasq, double expS, double erfS){
	return 1.0/(8.0*PI*r) * ((1.0 + sigmasq/rsq)*erfS- (2.0*sigma/r)/sqrt(PI2) * expS);
}

__device__ __host__
double B(double r, double rsq, double sigma, double sigmasq, double expS, double erfS){
	return 1.0/(8.0*PI*pow(r, 3)) * ((1.0 - 3.0*sigmasq/rsq)*erfS + (6.0*sigma/r)/sqrt(PI2) * expS);
}

__device__ __host__
double dAdr(double r, double rsq, double sigma, double sigmasq, double expS, double erfS){
	return -1.0/(8.0*PI*pow(r, 2)) * ((1.0+3.0*sigmasq/rsq)*erfS - (4.0*r/sigma + 6*sigma/r)/sqrt(PI2) * expS );
}

__device__ __host__
double dBdr(double r, double rsq, double sigma, double sigmasq, double expS, double erfS){
	return -1.0/(8.0*PI*pow(r, 4)) * ((3.0-15.0*sigmasq/rsq)*erfS + (4.0*r/sigma + 30.0*sigma/r)/sqrt(PI2) * expS);
}

// D_ij = C(r) delta_ij + F(r) x_ix_j
__device__ __host__
double C(double r, double rsq, double sigma, double sigmasq, double gaussgam, double erfS){
	return 1.0/(4.0*PI*pow(r, 3))*erfS - (1.0 + sigmasq/rsq)*gaussgam;
}

__device__ __host__
double D(double r, double rsq, double sigma, double sigmasq, double gaussgam, double erfS){
	return -3.0/(4.0*PI*pow(r, 5))*erfS + (1.0/rsq + 3.0*sigmasq/pow(r, 4))*gaussgam;
}

__device__ __host__
double P(double r, double rsq, double sigma, double sigmasq, double gaussgam){
	return (1.0 - rsq/4.0/sigmasq) * gaussgam / sigmasq;
}

__device__ __host__
double Q(double r, double rsq, double sigma, double sigmasq, double gaussgam){
	return 1.0/4.0 * gaussgam / sigmasq / sigmasq;
}

__global__
void cufcm_pair_correction(double* Y, double* V, double* W, double* F, double* T, int N,
                    int *map, int *head, int *list,
                    int ncell, double Rrefsq,
                    double pdmag,
                    double sigma, double sigmasq,
                    double sigmaFCM, double sigmaFCMsq,
                    double sigmaFCMdip, double sigmaFCMdipsq){

    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    int icell = 0, i = 0, j = 0, jcello = 0, jcell = 0, nabor = 0;
    double xi = 0.0, yi = 0.0, zi = 0.0, vxi = 0.0, vyi = 0.0, vzi = 0.0, wxi = 0.0, wyi = 0.0, wzi = 0.0, xij = 0.0, yij = 0.0, zij = 0.0, rijsq = 0.0, rij = 0.0;
    double pdmagsq_quarter = pdmag * pdmag * 0.25;
    double quatemp = 0.0;
    double temp1VF, temp2VF;
    double tempVTWF;
    double temp1WT, temp2WT;

    double gamma, gammasq, gammaVF_FCM, gammaVF_FCMsq, gammaVTWF_FCM, gammaVTWF_FCMsq, gammaWT_FCM, gammaWT_FCMsq;
    double erfS, expS, gaussgam, erfS_VF_FCM, expS_VF_FCM, erfS_VTWF_FCM, expS_VTWF_FCM, erfS_WT_FCM, expS_WT_FCM;

    double Atemp, Btemp, Ctemp, Dtemp, AFCMtemp, BFCMtemp, Ptemp, Qtemp;
    double ftemp, fFCMtemp_VTWF, fFCMtemp_WT, dfdrtemp, dfdrFCMtemp;
    double Fjdotx, Fidotx;
    double Tjdotx, Tidotx;

    gamma = sqrtf(2.0)*sigma;
    gammasq = gamma*gamma;

    gammaVF_FCM = sqrtf(2.0)*sigmaFCM;
    gammaVF_FCMsq = gammaVF_FCM*gammaVF_FCM;

    gammaVTWF_FCM = sqrtf(sigmaFCM*sigmaFCM + sigmaFCMdip*sigmaFCMdip);
    gammaVTWF_FCMsq = gammaVTWF_FCM*gammaVTWF_FCM;

    gammaWT_FCM = sqrtf(2.0)*sigmaFCMdip;
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
            // if(np == 300){
                // printf("--- thread tracker:%d\tm:%d\ti:%d\ticell/ncell:\t %d / %d \n", np, m, i, icell, ncell);
            // }
        }

        // 9993 ( 0.54540205 0.08480939 0.09774844 )
        // 9994 ( 0.88209362 0.12272217 0.49171341 )
        // 9995 ( -0.17118156 0.76744805 -0.64195434 )
        // 9996 ( -0.16255797 0.14025717 -0.18016408 )
        // 9997 ( 1.40436664 0.00166609 0.70345968 )
        // 9997 ( 0.33007566 1.19916520 -1.61376982 )
        // 9999 ( 2.12668145 -0.10840189 0.03746328 )

        // printf("--- thread:%d\ti:%d\ticell/ncell:\t %d / %d \n", np, i, icell, ncell);
    
        
        xi = Y[3*i + 0];
        yi = Y[3*i + 1];
        zi = Y[3*i + 2];
        // vxi = V[3*i + 0];
        // vyi = V[3*i + 1];
        // vzi = V[3*i + 2];
        // wxi = W[3*i + 0];
        // wyi = W[3*i + 1];
        // wzi = W[3*i + 2];
        j = list[i];
        // intra-cell interactions
        // corrections apply to both parties
        while(j > -1){
            xij = xi - Y[3*j + 0];
            yij = yi - Y[3*j + 1];
            zij = zi - Y[3*j + 2];

            xij = xij - PI2 * ((double) ((int) (xij/PI)));
            yij = yij - PI2 * ((double) ((int) (yij/PI)));
            zij = zij - PI2 * ((double) ((int) (zij/PI)));

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
                // W[3*j + 0] = W[3*j + 0] + 0.5*( T[3*i + 0]*temp1WT - xij*Tidotx*temp2WT ) - tempVTWF*( zij*F[3*i + 1] - yij*F[3*i + 2] );
                // W[3*j + 1] = W[3*j + 1] + 0.5*( T[3*i + 1]*temp1WT - yij*Tidotx*temp2WT ) - tempVTWF*( xij*F[3*i + 2] - zij*F[3*i + 0] );
                // W[3*j + 2] = W[3*j + 2] + 0.5*( T[3*i + 2]*temp1WT - zij*Tidotx*temp2WT ) - tempVTWF*( yij*F[3*i + 0] - xij*F[3*i + 1] );

                vxi = vxi + temp1VF*F[3*j + 0] + temp2VF*xij*Fjdotx + tempVTWF*( zij*T[3*j + 1] - yij*T[3*j + 2] );
                vyi = vyi + temp1VF*F[3*j + 1] + temp2VF*yij*Fjdotx + tempVTWF*( xij*T[3*j + 2] - zij*T[3*j + 0] );
                vzi = vzi + temp1VF*F[3*j + 2] + temp2VF*zij*Fjdotx + tempVTWF*( yij*T[3*j + 0] - xij*T[3*j + 1] );

                atomicAdd(&V[3*j + 0], temp1VF*F[3*i + 0] + temp2VF*xij*Fidotx - tempVTWF*( zij*T[3*i + 1] - yij*T[3*i + 2] ));
                atomicAdd(&V[3*j + 1], temp1VF*F[3*i + 1] + temp2VF*yij*Fidotx - tempVTWF*( xij*T[3*i + 2] - zij*T[3*i + 0] ));
                atomicAdd(&V[3*j + 2], temp1VF*F[3*i + 2] + temp2VF*zij*Fidotx - tempVTWF*( yij*T[3*i + 0] - xij*T[3*i + 1] ));
                // V[3*j + 0] = V[3*j + 0] + temp1VF*F[3*i + 0] + temp2VF*xij*Fidotx - tempVTWF*( zij*T[3*i + 1] - yij*T[3*i + 2] );
                // V[3*j + 1] = V[3*j + 1] + temp1VF*F[3*i + 1] + temp2VF*yij*Fidotx - tempVTWF*( xij*T[3*i + 2] - zij*T[3*i + 0] );
                // V[3*j + 2] = V[3*j + 2] + temp1VF*F[3*i + 2] + temp2VF*zij*Fidotx - tempVTWF*( yij*T[3*i + 0] - xij*T[3*i + 1] );

                // if(i==9997){
                //     printf("\tinteraction (%d %d)\t cell(%d %d) vi(%.8f %.8f %.8f) \n", i, j, icell, icell, 
                //     temp1VF*F[3*j + 0] + temp2VF*xij*Fjdotx + tempVTWF*( zij*T[3*j + 1] - yij*T[3*j + 2] ), 
                //     temp1VF*F[3*j + 1] + temp2VF*yij*Fjdotx + tempVTWF*( xij*T[3*j + 2] - zij*T[3*j + 0] ), 
                //     temp1VF*F[3*j + 2] + temp2VF*zij*Fjdotx + tempVTWF*( yij*T[3*j + 0] - xij*T[3*j + 1] ));
                // }
                // if(j==9997){
                //     printf("\tinteraction (%d %d)\t cell(%d %d) vj(%.8f %.8f %.8f) \n", i, j, icell, icell, 
                //     temp1VF*F[3*i + 0] + temp2VF*xij*Fidotx - tempVTWF*( zij*T[3*i + 1] - yij*T[3*i + 2] ), 
                //     temp1VF*F[3*i + 1] + temp2VF*yij*Fidotx - tempVTWF*( xij*T[3*i + 2] - zij*T[3*i + 0] ), 
                //     temp1VF*F[3*i + 2] + temp2VF*zij*Fidotx - tempVTWF*( yij*T[3*i + 0] - xij*T[3*i + 1] ));
                // }

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

                xij = xij - PI2 * ((double) ((int) (xij/PI)));
                yij = yij - PI2 * ((double) ((int) (yij/PI)));
                zij = zij - PI2 * ((double) ((int) (zij/PI)));
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
                    // W[3*j + 0] = W[3*j + 0] + 0.5*( T[3*i + 0]*temp1WT - xij*Tidotx*temp2WT ) - tempVTWF*( zij*F[3*i + 1] - yij*F[3*i + 2] );
                    // W[3*j + 1] = W[3*j + 1] + 0.5*( T[3*i + 1]*temp1WT - yij*Tidotx*temp2WT ) - tempVTWF*( xij*F[3*i + 2] - zij*F[3*i + 0] );
                    // W[3*j + 2] = W[3*j + 2] + 0.5*( T[3*i + 2]*temp1WT - zij*Tidotx*temp2WT ) - tempVTWF*( yij*F[3*i + 0] - xij*F[3*i + 1] );

                    vxi = vxi + temp1VF*F[3*j + 0] + temp2VF*xij*Fjdotx + tempVTWF*( zij*T[3*j + 1] - yij*T[3*j + 2] );
                    vyi = vyi + temp1VF*F[3*j + 1] + temp2VF*yij*Fjdotx + tempVTWF*( xij*T[3*j + 2] - zij*T[3*j + 0] );
                    vzi = vzi + temp1VF*F[3*j + 2] + temp2VF*zij*Fjdotx + tempVTWF*( yij*T[3*j + 0] - xij*T[3*j + 1] );
                    
                    atomicAdd(&V[3*j + 0], temp1VF*F[3*i + 0] + temp2VF*xij*Fidotx - tempVTWF*( zij*T[3*i + 1] - yij*T[3*i + 2] ));
                    atomicAdd(&V[3*j + 1], temp1VF*F[3*i + 1] + temp2VF*yij*Fidotx - tempVTWF*( xij*T[3*i + 2] - zij*T[3*i + 0] ));
                    atomicAdd(&V[3*j + 2], temp1VF*F[3*i + 2] + temp2VF*zij*Fidotx - tempVTWF*( yij*T[3*i + 0] - xij*T[3*i + 1] ));
                    // V[3*j + 0] = V[3*j + 0] + temp1VF*F[3*i + 0] + temp2VF*xij*Fidotx - tempVTWF*( zij*T[3*i + 1] - yij*T[3*i + 2] );
                    // V[3*j + 1] = V[3*j + 1] + temp1VF*F[3*i + 1] + temp2VF*yij*Fidotx - tempVTWF*( xij*T[3*i + 2] - zij*T[3*i + 0] );
                    // V[3*j + 2] = V[3*j + 2] + temp1VF*F[3*i + 2] + temp2VF*zij*Fidotx - tempVTWF*( yij*T[3*i + 0] - xij*T[3*i + 1] );
                    
                    // if(i==9997){
                    //     printf("\tinteraction (%d %d)\t cell(%d %d) vi(%.8f %.8f %.8f) \n", i, j, icell, jcell, 
                    //     temp1VF*F[3*j + 0] + temp2VF*xij*Fjdotx + tempVTWF*( zij*T[3*j + 1] - yij*T[3*j + 2] ), 
                    //     temp1VF*F[3*j + 1] + temp2VF*yij*Fjdotx + tempVTWF*( xij*T[3*j + 2] - zij*T[3*j + 0] ), 
                    //     temp1VF*F[3*j + 2] + temp2VF*zij*Fjdotx + tempVTWF*( yij*T[3*j + 0] - xij*T[3*j + 1] ));
                    // }
                    // if(j==9997){
                    //     printf("\tinteraction (%d %d)\t cell(%d %d) vj(%.8f %.8f %.8f) \n", i, j, icell, jcell, 
                    //     temp1VF*F[3*i + 0] + temp2VF*xij*Fidotx - tempVTWF*( zij*T[3*i + 1] - yij*T[3*i + 2] ), 
                    //     temp1VF*F[3*i + 1] + temp2VF*yij*Fidotx - tempVTWF*( xij*T[3*i + 2] - zij*T[3*i + 0] ), 
                    //     temp1VF*F[3*i + 2] + temp2VF*zij*Fidotx - tempVTWF*( yij*T[3*i + 0] - xij*T[3*i + 1] ));
                    // }

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
void cufcm_pair_correction_spatial_hashing(double* Y, double* V, double* W, double* F, double* T, int N,
                    int *map, int *head, int *list,
                    int ncell, double Rrefsq,
                    double pdmag,
                    double sigma, double sigmasq,
                    double sigmaFCM, double sigmaFCMsq,
                    double sigmaFCMdip, double sigmaFCMdipsq){

    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int stride = blockDim.x*gridDim.x;

    int icell = 0, i = 0, j = 0, jcello = 0, jcell = 0, nabor = 0;
    double xi = 0.0, yi = 0.0, zi = 0.0, vxi = 0.0, vyi = 0.0, vzi = 0.0, wxi = 0.0, wyi = 0.0, wzi = 0.0, xij = 0.0, yij = 0.0, zij = 0.0, rijsq = 0.0, rij = 0.0;
    double pdmagsq_quarter = pdmag * pdmag * 0.25;
    double quatemp = 0.0;
    double temp1VF, temp2VF;
    double tempVTWF;
    double temp1WT, temp2WT;

    double gamma, gammasq, gammaVF_FCM, gammaVF_FCMsq, gammaVTWF_FCM, gammaVTWF_FCMsq, gammaWT_FCM, gammaWT_FCMsq;
    double erfS, expS, gaussgam, erfS_VF_FCM, expS_VF_FCM, erfS_VTWF_FCM, expS_VTWF_FCM, erfS_WT_FCM, expS_WT_FCM;

    double Atemp, Btemp, Ctemp, Dtemp, AFCMtemp, BFCMtemp, Ptemp, Qtemp;
    double ftemp, fFCMtemp_VTWF, fFCMtemp_WT, dfdrtemp, dfdrFCMtemp;
    double Fjdotx, Fidotx;
    double Tjdotx, Tidotx;

    gamma = sqrtf(2.0)*sigma;
    gammasq = gamma*gamma;

    gammaVF_FCM = sqrtf(2.0)*sigmaFCM;
    gammaVF_FCMsq = gammaVF_FCM*gammaVF_FCM;

    gammaVTWF_FCM = sqrtf(sigmaFCM*sigmaFCM + sigmaFCMdip*sigmaFCMdip);
    gammaVTWF_FCMsq = gammaVTWF_FCM*gammaVTWF_FCM;

    gammaWT_FCM = sqrtf(2.0)*sigmaFCMdip;
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
        // intra-cell interactions
        // corrections apply to both parties
        while(j > -1){
            xij = xi - Y[3*j + 0];
            yij = yi - Y[3*j + 1];
            zij = zi - Y[3*j + 2];

            xij = xij - PI2 * ((double) ((int) (xij/PI)));
            yij = yij - PI2 * ((double) ((int) (yij/PI)));
            zij = zij - PI2 * ((double) ((int) (zij/PI)));

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

                xij = xij - PI2 * ((double) ((int) (xij/PI)));
                yij = yij - PI2 * ((double) ((int) (yij/PI)));
                zij = zij - PI2 * ((double) ((int) (zij/PI)));
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
void cufcm_self_correction(double* V, double* W, double* F, double* T, int N,
                                double StokesMob, double ModStokesMob,
                                double PDStokesMob, double BiLapMob,
                                double WT1Mob, double WT2Mob){
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

void cufcm_pair_correction_loop(double* Y, double* V, double* W, double* F, double* T, int N,
                    int *map, int *head, int *list,
                    int ncell, double Rrefsq,
                    double pdmag,
                    double sigma, double sigmasq,
                    double sigmaFCM, double sigmaFCMsq,
                    double sigmaFCMdip, double sigmaFCMdipsq){
    int icell = 0, i = 0, j = 0, jcello = 0, jcell = 0, nabor = 0;
    double xi = 0.0, yi = 0.0, zi = 0.0, vxi = 0.0, vyi = 0.0, vzi = 0.0, wxi = 0.0, wyi = 0.0, wzi = 0.0, xij = 0.0, yij = 0.0, zij = 0.0, rijsq = 0.0, rij = 0.0;
    double pdmagsq_quarter = pdmag * pdmag * 0.25;
    double quatemp = 0.0;
    double temp1VF, temp2VF;
    double tempVTWF;
    double temp1WT, temp2WT;

    double gamma, gammasq, gammaVF_FCM, gammaVF_FCMsq, gammaVTWF_FCM, gammaVTWF_FCMsq, gammaWT_FCM, gammaWT_FCMsq;
    double erfS, expS, gaussgam, erfS_VF_FCM, expS_VF_FCM, erfS_VTWF_FCM, expS_VTWF_FCM, erfS_WT_FCM, expS_WT_FCM;

    double Atemp, Btemp, Ctemp, Dtemp, AFCMtemp, BFCMtemp, Ptemp, Qtemp;
    double ftemp, fFCMtemp_VTWF, fFCMtemp_WT, dfdrtemp, dfdrFCMtemp;
    double Fjdotx, Fidotx;
    double Tjdotx, Tidotx;

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

                xij = xij - PI2 * ((double) ((int) (xij/PI)));
                yij = yij - PI2 * ((double) ((int) (yij/PI)));
                zij = zij - PI2 * ((double) ((int) (zij/PI)));

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

                    xij = xij - PI2 * ((double) ((int) (xij/PI)));
                    yij = yij - PI2 * ((double) ((int) (yij/PI)));
                    zij = zij - PI2 * ((double) ((int) (zij/PI)));
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

void cufcm_self_correction_loop(double* V, double* W, double* F, double* T, int N,
                                double StokesMob, double ModStokesMob,
                                double PDStokesMob, double BiLapMob,
                                double WT1Mob, double WT2Mob){
    for(int i = 0; i < N; i++){
        V[3*i + 0] = V[3*i + 0] + F[3*i + 0]*(StokesMob - ModStokesMob + PDStokesMob - BiLapMob) ;
        V[3*i + 1] = V[3*i + 1] + F[3*i + 1]*(StokesMob - ModStokesMob + PDStokesMob - BiLapMob) ;
        V[3*i + 2] = V[3*i + 2] + F[3*i + 2]*(StokesMob - ModStokesMob + PDStokesMob - BiLapMob) ;

        W[3*i + 0] = W[3*i + 0] + T[3*i + 0]*(WT1Mob - WT2Mob) ;
        W[3*i + 1] = W[3*i + 1] + T[3*i + 1]*(WT1Mob - WT2Mob) ;
        W[3*i + 2] = W[3*i + 2] + T[3*i + 2]*(WT1Mob - WT2Mob) ;
    }
}