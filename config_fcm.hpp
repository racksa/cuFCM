// #include <cufft.h>
// #include "config.hpp"

// #ifndef FCMMACRO_H
// #define FCMMACRO_H


// // #define RH 0.008089855908506678
// #define RH 0.02609300415934458
// #define NP 500000

// #if SOLVER_MODE == 1

//     #define NPTS (256.0)
//     #define DX (PI2/NPTS)

//     #if TOL == 0
//         #define ALPHA 0.9352    /* SIGMA/dx */
//         #define BETA 9.706      /* (NGD.DX)/SIGMA */
//         #define ETA 5.573     /* RC/SIGMA */
//     #elif TOL == 1
//         #define ALPHA 1.0509    /* SIGMA/dx */
//         #define BETA 10.873     /* (NGD.DX)/SIGMA */
//         #define ETA 6.373     /* RC/SIGMA */
//     #elif TOL == 2
//         #define ALPHA 1.1545    /* SIGMA/dx */
//         #define BETA 11.773     /* (NGD.DX)/SIGMA */
//         #define ETA 7.090     /* RC/SIGMA */
//     #elif TOL == 3
//         #define ALPHA 1.2526    /* SIGMA/dx */
//         #define BETA 12.639     /* (NGD.DX)/SIGMA */
//         #define ETA 7.717     /* RC/SIGMA */
//     #endif
//     // #define ALPHA 1.3450    /* SIGMA/dx */
//     // #define BETA 13.410     /* (NGD.DX)/SIGMA */
//     // #define ETA 8.326     /* RC/SIGMA */

//     #define SIGMA_FCM (RH/PIsqrt)
//     #define SIGMA_FCMDIP (RH/2.6613400789829376)
//     #define SIGMA (DX*ALPHA)
//     #define SIGMA_FAC (SIGMA/SIGMA_FCM)
//     #define NGD (int)(BETA*ALPHA)
//     #define RREF_FAC (ETA*ALPHA)

// #elif SOLVER_MODE == 0

//     #define SIGMA (RH/PIsqrt)
//     #define SIGMA_FAC 1

//     #if TOL == 0
//         #define ALPHA 1.0409    /* SIGMA/dx */
//         #define BETA 10.873     /* (NGD.DX)/SIGMA */
//     #elif TOL == 1
//         #define ALPHA 1.1545    /* SIGMA/dx */
//         #define BETA 11.773     /* (NGD.DX)/SIGMA */
//     #elif TOL == 2
//         #define ALPHA 1.2526    /* SIGMA/dx */
//         #define BETA 12.639     /* (NGD.DX)/SIGMA */
//     #elif TOL == 3
//         #define ALPHA 1.3850    /* SIGMA/dx */
//         #define BETA 13.410     /* (NGD.DX)/SIGMA */
//     #endif
    
//     #define DX_C (SIGMA/ALPHA)
//     #define NPTS ((int)(PI2/DX_C)%2 == 0? (int)(PI2/DX_C) : (int)(PI2/DX_C) + 1)
//     #define DX (PI2/NPTS)
//     #define NGD (int)(BETA*ALPHA)

//     #define ETA 5.573

// #endif

// #define NX NPTS
// #define NY NPTS
// #define NZ NPTS


// #define GRID_SIZE (int)(NX*NY*NZ)
// #define FFT_GRID_SIZE ((NX/2+1)*NY*NZ)


// #define NGD_UAMMD 9
// #endif