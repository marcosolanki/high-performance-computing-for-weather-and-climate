REALSIZE=8

CXX=clang++
CXXFLAGS=-std=c++11 -Ofast -march=native -Wall -Wextra -Wpedantic -Wshadow

ACCCXX=nvc++
ACCFLAGS=-std=c++11 -O4 -tp=native -acc=gpu -gpu=cc60 # -Minfo=accel

CUDACXX=nvcc
CUDAFLAGS=-std=c++11 -O3 -use_fast_math -extra-device-vectorization -arch=sm_60 -restrict -dlto -Xcompiler "-Ofast -march=native -Wall -Wextra -Wshadow"

RM=rm -f
