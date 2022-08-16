#include "CUFCM_data.hpp"
#include <cstdio>

void read_init_data(double *Y, int Np, char *file_name){
    FILE *ifile;
    ifile = fopen(file_name, "r");
    for(int n = 0; n < Np; n++){
        if(fscanf(ifile, "%lf %lf %lf", &Y[3*n + 0], &Y[3*n + 1], &Y[3*n + 2]) == 0){
            printf("fscanf error: Unable to read data");
        }
    }
    fclose(ifile);

    return;
}