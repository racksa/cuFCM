NVCC_FLAGS=-arch=sm_60 -std=c++14 -O3 -I../include

LINK=-lcublas -lcufft -llapacke -lcblas -lcurand -lcuda -lineinfo

CUFCM_FILES = CUFCM_MAIN.cu CUFCM_FCM.cu CUFCM_data.cu CUFCM_CORRECTION.cu



UAMMDINCLUDEFLAGS=-I$(CUDA_ROOT)/include -I$(UAMMD_ROOT)/src -I$(UAMMD_ROOT)/src/third_party


CUFCM : CUFCM_MAIN.cu
	nvcc $(NVCC_FLAGS) $(CUFCM_FILES) $(LINK) -o bin/CUFCM

clean :
	rm -f bin/CUFCM


# UAMMD makefile
# Default log level is 5, which prints up to MESSAGE, 0 will only print critical errors and 14 will print everything up to the most low level debug information
LOG_LEVEL=5

NVCC=nvcc
CXX=g++

CUDA_ROOT=$(shell dirname `which nvcc`)/..
UAMMD_ROOT= ../UAMMD/
#Uncomment to compile in double precision mode
#DOUBLE_PRECISION=-DDOUBLE_PRECISION
INCLUDEFLAGS=-I$(CUDA_ROOT)/include -I$(UAMMD_ROOT)/src -I$(UAMMD_ROOT)/src/third_party
NVCCFLAGS=-ccbin=$(CXX) -std=c++14 -O3 $(INCLUDEFLAGS) -DMAXLOGLEVEL=$(LOG_LEVEL) $(DOUBLE_PRECISION) --extended-lambda

CUFCM_FILES_NOMAIN = incorporate/CUFCM_INCORPORATE.cu
# all: $(patsubst %.cu, %, $(wildcard *.cu))

# %: %.cu Makefile
# 	$(NVCC) $(NVCCFLAGS) $< -o $(@:.out=)

# clean: $(patsubst %.cu, %.clean, $(wildcard *.cu))

# %.clean:
# 	rm -f $(@:.clean=)

spread_with_UAMMD : incorporate/spread_interpolate.cu
	nvcc $(NVCC_FLAGS) incorporate/spread_interpolate.cu $(CUFCM_FILES_NOMAIN) $(NVCCFLAGS) -o bin/spread $(LINK)