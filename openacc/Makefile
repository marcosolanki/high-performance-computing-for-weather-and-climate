-include ../config.mk


VERSIONS=kernels parallel optimised

.PHONY: all
all: $(VERSIONS)

kernels: main-kernels.cpp utils.hpp
	$(ACCCXX) -o $@ $< $(ACCFLAGS)

parallel: main-parallel.cpp utils.hpp
	$(ACCCXX) -o $@ $< $(ACCFLAGS)

optimised: main-optimised.cpp utils.hpp
	$(ACCCXX) -o $@ $< $(ACCFLAGS)

.PHONY: clean
clean:
	$(RM) $(VERSIONS) in_field.csv out_field.csv plot.png
