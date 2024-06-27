REALSIZE=8

CXX=clang++
CXXFLAGS=-std=c++11 -Ofast -march=native -Wall -Wextra -Wpedantic -Wshadow

CUDACXX=nvcc
CUDAFLAGS=-std=c++11 -O3 --use_fast_math -dlto -arch=sm_61 -Xcompiler "-Ofast -march=native -Wall -Wextra -Wshadow"

RM=rm -f
