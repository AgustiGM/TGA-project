#!/bin/bash

### Directivas para el gestor de colas
#SBATCH --job-name=MultiGPU
#SBATCH -D .
#SBATCH --output=submit-STREAMS.o%j
#SBATCH --error=submit-STREAMS.e%j
#SBATCH -A cuda
#SBATCH -p cuda
### Se piden 4 GPUs 
#SBATCH --gres=gpu:4

export PATH=/Soft/cuda/11.2.1/bin:$PATH


./kernel00.exe 10000 Y
./kernel00.exe 300000 N
./kernel00.exe 300000 N
./kernel00.exe 300000 N

#./kernel01.exe 10000 Y
#./kernel01.exe 300000 N
#./kernel01.exe 300000 N
#./kernel01.exe 300000 N

#./kernel02.exe 10000 Y
#./kernel02.exe 300000 N
#./kernel02.exe 300000 N
#./kernel02.exe 300000 N


#nsys nvprof --print-gpu-trace ./kernel00.exe 300000 N
#nsys vprof --print-gpu-trace ./kernel01.exe 300000 N
#nsys nvprof --print-gpu-trace ./kernel02.exe 300000 N


