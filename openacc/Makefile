-include ../config.mk

ALLFLAGS=-std=c++11 -Ofast -tp=native -acc=gpu -gpu=cc60
VERSIONS=kernels parallel optimize jacobi main stencil acc

CXX=nvc++
FLAGS=-tp=native -acc -gpu=managed -gpu=managed -Minfo=accel

FT=ftn
FFLAGS=-O3 -hfp3 -eZ -ffree -N255 -ec -eC -eI -eF -rm -h omp

.PHONY: all

all: $(VERSIONS)

main: ../cpu/main.cpp utils.hpp
	$(CXX) -o $@ $< $(FLAGS)
    
jacobi: jacobi.cpp
	$(CXX) -o $@ $< $(FLAGS)
    
stencil: utils.F90 stencil2d.F90
	$(FT) -o $@ $? $(FFLAGS)
    
acc: utils.F90 stencil2d-acc.F90
	$(FT) -o $@ $? $(FFLAGS)

kernels: main-kernels.cpp utils.hpp
	$(CXX) -o $@ $< $(FLAGS)
    
parallel: main-parallel.cpp utils.hpp
	$(CXX) -o $@ $< $(FLAGS)
    
optimize: main-optimize.cpp utils.hpp
	$(CXX) -o $@ $< $(FLAGS)

.PHONY: again
again: clean all
    
.PHONY: clean
clean:
	$(RM) $(VERSIONS) in_field.csv out_field.csv plot.png
	$(RM) *.i *.lst *.mod *.o
