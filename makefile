NVCC_FLAGS=-arch=sm_60 -std=c++14 -O3 -g -I../include

LINK=-lcublas -lcufft -llapacke -lcblas  -lcurand -lcuda

CUFCM_FILES = CUFCM_MAIN.cu CUFCM_FCM.cu

# CUFFT_TEST_Main : CUFFT_TEST_main.cu
# 	nvcc $(NVCC_FLAGS) CUFFT_TEST_main.cu -o CUFFT_TEST_main $(LINK)

CUFCM : CUFCM_MAIN.cu
	nvcc $(NVCC_FLAGS) $(CUFCM_FILES) -o CUFCM $(LINK)

clean :
	rm -f CUFCM
