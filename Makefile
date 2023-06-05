CUDA_HOME   = /usr/local/cuda-12.1

SELF_DIR 	= $(dir $(lastword $(MAKEFILE_LIST)))

NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O3 -Wno-deprecated-gpu-targets -I$(CUDA_HOME)/include --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc -I$(SELF_DIR)  
LD_FLAGS    = -lcudart -Xlinker "-rpath,$(CUDA_HOME)/lib64" -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib -I$(CUDA_HOME)/include -I$(SELF_DIR) 

# NVCC        = $(CUDA_HOME)/bin/nvcc
# NVCC_FLAGS  = -O3 -Wno-deprecated-gpu-targets -I$(CUDA_HOME)/include -gencode arch=compute_86,code=sm_86 --ptxas-options=-v -I$(SELF_DIR) -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
# LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib -L/usr/local/cuda/lib64 -I$(SELF_DIR)

PROG_FLAGS  = -DPINNED=1 -DSIZE=32

EXE00	        = main.exe
EXE01			= test.exe
EXE02			= forwardtest.exe
EXE03			= forward_primitives.exe
EXE04			= reader.exe
EXE05			= nn_main.exe

OBJ00	        = main.o nnfunctions.o utils.o
OBJ01	        = test.o nnfunctions.o utils.o
OBJ02	        = forwardtest.o nnfunctions.o utils.o
OBJ03	        = forward_primitives.o nnfunctions.o utils.o primitives.o 
OBJ04	        = reader.o 
OBJ05	        = nn_main.o primitives.o nnfunctions.o utils.o

default: $(EXE00)


main.o: main.cu nnfunctions.h utils.h
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS) $(PROG_FLAGS)

test.o: test.cu nnfunctions.h utils.h
	$(NVCC) -c -o $@ test.cu $(NVCC_FLAGS) $(PROG_FLAGS)

forwardtest.o: forwardtest.cu nnfunctions.h utils.h
	$(NVCC) -c -o $@ forwardtest.cu $(NVCC_FLAGS) $(PROG_FLAGS)

forward_primitives.o: forward_primitives.cu nnfunctions.h utils.h primitives.h
	$(NVCC) -c -o $@ forward_primitives.cu $(NVCC_FLAGS) $(PROG_FLAGS)

reader.o: reader.cu 
	$(NVCC) -c -o $@ reader.cu $(NVCC_FLAGS) $(PROG_FLAGS)


nn_main.o: nn_main.cu nnfunctions.h utils.h primitives.h
	$(NVCC) -c -o $@ nn_main.cu $(NVCC_FLAGS) $(PROG_FLAGS)



input_utils.o: input_utils.cu input_utils.h
	$(NVCC) -c -o $@ input_utils.cu $(NVCC_FLAGS)

nnfunctions.o: nnfunctions.cu nnfunctions.h
	$(NVCC) -c -o $@ nnfunctions.cu $(NVCC_FLAGS)

primitives.o: primitives.cu primitives.h
	$(NVCC) -c -o $@ primitives.cu $(NVCC_FLAGS)

utils.o: utils.cu utils.h
	$(NVCC) -c -o $@ utils.cu $(NVCC_FLAGS)



$(EXE00): $(OBJ00)
	$(NVCC) $(OBJ00) -o $(EXE00) $(LD_FLAGS)

$(EXE01): $(OBJ01)
	$(NVCC) $(OBJ01) -o $(EXE01) $(LD_FLAGS)

$(EXE02): $(OBJ02)
	$(NVCC) $(OBJ02) -o $(EXE02) $(LD_FLAGS)

$(EXE03): $(OBJ03)
	$(NVCC) $(OBJ03) -o $(EXE03) $(LD_FLAGS)

$(EXE04): $(OBJ04)
	$(NVCC) $(OBJ04) -o $(EXE04) $(LD_FLAGS)

$(EXE05): $(OBJ05)
	$(NVCC) $(OBJ05) -o $(EXE05) $(LD_FLAGS)




all:	$(EXE00) $(EXE01) $(EXE02) $(EXE03) 

test:	$(EXE01)

reader:	$(EXE04)

nn_main:	$(EXE05)

forwardtest:	$(EXE02)

primitives:	$(EXE03)

clean:
	rm -rf *.o main.exe test.exe forwardtest.exe forward_primitives.exe reader.exe nn_main.exe

