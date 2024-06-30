-include ../config.mk

VERSIONS=main jacobi kernels parallel optimised # stencil

.PHONY: all
all: $(VERSIONS)

main: ../cpu/main.cpp ../cpu/utils.hpp
	$(ACCCXX) -o $@ $< $(ACCFLAGS)

jacobi: jacobi.cpp
	$(ACCCXX) -o $@ $< $(ACCFLAGS)

kernels: main-kernels.cpp utils.hpp
	$(ACCCXX) -o $@ $< $(ACCFLAGS)

parallel: main-parallel.cpp utils.hpp
	$(ACCCXX) -o $@ $< $(ACCFLAGS)

optimised: main-optimised.cpp utils.hpp
	$(ACCCXX) -o $@ $< $(ACCFLAGS)

# stencil: utils.F90 stencil2d.F90
# 	$(FC) -o $@ $? $(FFLAGS)

.PHONY: again
again: clean all

.PHONY: clean
clean:
	$(RM) $(VERSIONS) in_field.csv out_field.csv plot.png *.i *.lst *.mod *.o
