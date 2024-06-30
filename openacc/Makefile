-include ../config.mk

VERSIONS=main jacobi # stencil

.PHONY: all
all: $(VERSIONS)

main: main.cpp kernels.hpp parallel.hpp optimised.hpp utils.hpp
	$(ACCCXX) -o $@ $< $(ACCFLAGS)

jacobi: jacobi.cpp timer.h
	$(ACCCXX) -o $@ $< $(ACCFLAGS)

# stencil: utils.F90 stencil2d.F90
# 	$(FC) -o $@ $? $(FFLAGS)

.PHONY: again
again: clean all

.PHONY: clean
clean:
	$(RM) $(VERSIONS) in_field.csv out_field.csv plot.png *.i *.lst *.mod *.o
