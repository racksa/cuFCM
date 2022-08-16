#include "CUFCM_data.hpp"
#include <cstdio>

void read_init_data(double *Y, int N, char *file_name){
    FILE *ifile;
    ifile = fopen(file_name, "r");
    for(int np = 0; np < N; np++){
        if(fscanf(ifile, "%lf %lf %lf", &Y[3*np + 0], &Y[3*np + 1], &Y[3*np + 2]) == 0){
            printf("fscanf error: Unable to read data");
        }
    }
    fclose(ifile);

    return;
}