#pragma once

#include "../config.hpp"
#include <stdlib.h>
#include <math.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

inline Real percentage_error(Real *data, Real *ref_data, int N){
    Real ret;
    Real x_ref, y_ref, z_ref;
    Real x, y, z;

    for(int np = 0; np < N; np++){
        x = data[3*np];
        y = data[3*np + 1];
        z = data[3*np + 2];
        x_ref = ref_data[3*np];
        y_ref = ref_data[3*np + 1];
        z_ref = ref_data[3*np + 2];

        // printf("(%.8f %.8f %.8f) (%.8f %.8f %.8f)\n", x-x_ref, y-y_ref, z-z_ref, 
        // abs((x - x_ref)/x_ref), abs((y - y_ref)/y_ref), abs((z - z_ref)/z_ref) );

        // printf("(%.8f %.8f %.8f) (%.8f %.8f %.8f)\n", x, y, z, x_ref, y_ref, z_ref );

        ret += 1.0/3.0 * ( abs((x - x_ref)/x_ref) 
                         + abs((y - y_ref)/y_ref) 
                         + abs((z - z_ref)/z_ref) );
    }
    ret = ret / (Real) N;

    return ret;
}


inline Real percentage_error_magnitude(Real *data, Real *ref_data, int N){
    Real ret = Real(0.0);

    for(int np = 0; np < N; np++){
        Real x = data[3*np];
        Real y = data[3*np + 1];
        Real z = data[3*np + 2];
        Real x_ref = ref_data[3*np];
        Real y_ref = ref_data[3*np + 1];
        Real z_ref = ref_data[3*np + 2];

        ret += sqrt( (x - x_ref)*(x - x_ref) + 
                     (y - y_ref)*(y - y_ref) + 
                     (z - z_ref)*(z - z_ref) ) / 
               sqrt( x_ref*x_ref + 
                     y_ref*y_ref + 
                     z_ref*z_ref );
    }
    ret = ret / (Real) N;

    return ret;
}


inline Real mean(Real *data, int length){
    Real ret = 0;
    for(int i = 0; i<length; i++){
        ret += data[i];
    }
    return ret/Real(length);
}


inline Real stdv(Real *data, int length){
    Real avg = mean(data, length);
    Real ret = 0;
    for(int i = 0; i<length; i++){
        ret += (data[i] - avg)*(data[i] - avg);
    }
    return sqrtf(ret/Real(length));
}