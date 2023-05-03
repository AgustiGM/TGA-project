CUDA_HOME   = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3

NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = --allow-unsupported-compiler -O3 -Wno-deprecated-gpu-targets  -gencode arch=compute_86,code=sm_86 --ptxas-options=-v 
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib
PROG_FLAGS  = -DPINNED=1 -DDUMMY=100

EXE00	        = main.exe




default: $(EXE00)


$(EXE00): main.cu
	$(NVCC) -o $@ main.cu $(NVCC_FLAGS) $(PROG_FLAGS)




all:	$(EXE00)

clean:
	rm -rf *.o kernel*.exe

