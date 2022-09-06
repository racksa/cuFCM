#include "CUFCM_data.hpp"
#include <cstdio>
#include "config.hpp"

void read_init_data(Real *Y, int N, const char *file_name){
    FILE *ifile;
    ifile = fopen(file_name, "r");
    for(int np = 0; np < N; np++){
        #if USE_DOUBLE_PRECISION == true
        if(fscanf(ifile, "%lf %lf %lf", &Y[3*np + 0], &Y[3*np + 1], &Y[3*np + 2]) == 0){
            printf("fscanf error: Unable to read data");
        }
        #else
        if(fscanf(ifile, "%f %f %f", &Y[3*np + 0], &Y[3*np + 1], &Y[3*np + 2]) == 0){
            printf("fscanf error: Unable to read data");
        }
        #endif
    }
    fclose(ifile);

    return;
}