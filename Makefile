CUDA_HOME   = /usr

NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O3 -Wno-deprecated-gpu-targets -I$(CUDA_HOME)/include -gencode arch=compute_86,code=sm_86 --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib

PROG_FLAGS  = -DPINNED=1 -DSIZE=32

EXE00	        = main.exe

OBJ00	        = main.o

default: $(EXE00)

main.o: main.o
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS) $(PROG_FLAGS)



$(EXE00): $(OBJ00)
	$(NVCC) $(OBJ00) -o $(EXE00) $(LD_FLAGS)




all:	$(EXE00)

clean:
	rm -rf *.o main.exe

