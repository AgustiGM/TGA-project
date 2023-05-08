CUDA_HOME   = /usr

NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O3 -Wno-deprecated-gpu-targets -I$(CUDA_HOME)/include --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc -I./
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib -I./

PROG_FLAGS  = -DPINNED=1 -DSIZE=32

EXE00	        = main.exe

EXE01			= test.exe

OBJ00	        = main.o nnfunctions.o utils.o
OBJ01	        = test.o nnfunctions.o utils.o

default: $(EXE00)


main.o: main.cu nnfunctions.h utils.h
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS) $(PROG_FLAGS)

test.o: test.cu nnfunctions.h utils.h
	$(NVCC) -c -o $@ test.cu $(NVCC_FLAGS) $(PROG_FLAGS)

nnfunctions.o: nnfunctions.cu nnfunctions.h
	$(NVCC) -c -o $@ nnfunctions.cu $(NVCC_FLAGS)

utils.o: utils.cu utils.h
	$(NVCC) -c -o $@ utils.cu $(NVCC_FLAGS)


$(EXE00): $(OBJ00)
	$(NVCC) $(OBJ00) -o $(EXE00) $(LD_FLAGS)

$(EXE01): $(OBJ01)
	$(NVCC) $(OBJ01) -o $(EXE01) $(LD_FLAGS)




all:	$(EXE00) $(EXE01)

test: $(EXE01)

clean:
	rm -rf *.o main.exe test.exe

