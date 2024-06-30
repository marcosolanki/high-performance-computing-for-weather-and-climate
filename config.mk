REALSIZE=8

CXX=clang++
CXXFLAGS=-std=c++11 -Ofast -march=native -Wall -Wextra -Wpedantic -Wshadow

ACCCXX=nvc++
ACCFLAGS=-std=c++11 -O4 -tp=native -acc=gpu -gpu=cc60 # -Minfo=accel

CUDACXX=nvcc
CUDAFLAGS=-std=c++11 -O3 --use_fast_math -dlto -arch=sm_60 -Xcompiler "-Ofast -march=native -Wall -Wextra -Wshadow"

# FC=ftn
# FFLAGS=-O3 -hfp3 -eZ -ffree -N255 -ec -eC -eI -eF -rm -h omp

RM=rm -f
