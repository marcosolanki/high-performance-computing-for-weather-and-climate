-include ../config.mk

CXX=CC
VERSIONS=kernels parallel optimize

.PHONY: all

all: $(VERSIONS)

kernels: main-kernels.cpp utils.hpp
	$(CXX) -o $@ $< -fopenacc $(CXXFLAGS)
    
parallel: main-parallel.cpp utils.hpp
	$(CXX) -o $@ $< -fopenacc $(CXXFLAGS)
    
optimize: main-optimize.cpp utils.hpp
	$(CXX) -o $@ $< -fopenacc $(CXXFLAGS)

.PHONY: clean
clean:
	$(RM) $(VERSIONS) in_field.csv out_field.csv plot.png
