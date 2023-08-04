VPATH = src/ test/
NVCC_FLAGS= -arch=sm_75 -std=c++14 -O0
USER_FLAGS= 

LINK= -lcufft -lcurand

RANDOM_GENERATOR_FILES = CUFCM_RANDOM_MAIN.cu CUFCM_RANDOMPACKER.cu CUFCM_DATA.cu CUFCM_CELLLIST.cu
CUFCM_FILES =  CUFCM_MAIN.cu CUFCM_CELLLIST.cu CUFCM_FCM.cu CUFCM_DATA.cu CUFCM_CORRECTION.cu CUFCM_SOLVER.cu
TEST_FILES = CUFCM_TEST.cu CUFCM_CELLLIST.cu CUFCM_FCM.cu CUFCM_DATA.cu CUFCM_CORRECTION.cu CUFCM_SOLVER.cu


CUFCM : $(CUFCM_FILES)
	nvcc $^ $(NVCC_FLAGS) $(LINK) -o bin/$@

CUFCM_DOUBLE : $(CUFCM_FILES)
	nvcc $^ -DUSE_DOUBLE_PRECISION $(NVCC_FLAGS) $(LINK) -o bin/CUFCM

FCM : $(CUFCM_FILES)
	nvcc $^ -DUSE_REGULARFCM $(NVCC_FLAGS) $(LINK) -o bin/$@

FLOWFIELD : $(CUFCM_FILES)
	nvcc $^ $(NVCC_FLAGS) $(LINK) -o bin/$@

RANDOM_GENERATOR : $(RANDOM_GENERATOR_FILES)
	nvcc $^ $(NVCC_FLAGS) $(LINK) -o bin/RANDOM

TEST : $(TEST_FILES)
	nvcc $^ $(NVCC_FLAGS) $(LINK) -o bin/$@

clean :
	rm -f bin/TEST
	rm -f bin/FCM
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
NVCCFLAGS=-ccbin=$(CXX) -std=c++14 -O3 $(INCLUDEFLAGS) -DMAXLOGLEVEL=$(LOG_LEVEL) $(DOUBLE_PRECISION) --extended-lambda -lineinfo

CUFCM_FILES_NOMAIN = incorporate/CUFCM_INCORPORATE.cu

spread_with_UAMMD : incorporate/spread_interpolate.cu
	nvcc $(NVCC_FLAGS) incorporate/spread_interpolate.cu $(CUFCM_FILES_NOMAIN) $(NVCCFLAGS) -o bin/spread $(LINK)