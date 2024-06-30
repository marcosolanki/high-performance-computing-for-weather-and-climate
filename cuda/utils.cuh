#pragma once

#include <iostream>


namespace {

static inline __host__ __device__ std::size_t index(std::size_t i, std::size_t j, std::size_t k,
                                                    std::size_t xsize, std::size_t ysize) {
    return i + j * xsize + k * xsize * ysize;
}


static inline void check(cudaError_t error) {
    if(error != cudaSuccess) {
        std::cerr << "ERROR: A CUDA runtime API call returned a cudaError_t != cudaSuccess.\n"
                  << "Error name:   \"" << cudaGetErrorName(error) << "\"\n"
                  << "Error string: \"" << cudaGetErrorString(error) << "\"\n";
        std::cout << "================================================================================\n";
        exit(EXIT_FAILURE);
    }
}

} // namespace


enum class Mode {
    laplap_global,
    laplap_shared,
    biharm_global,
    invalid
};


namespace utils {

void print_args_errmsg() {
    std::cerr << "================================================================================\n";
    std::cerr << "                             Welcome to stencil2d!\n";
    std::cerr << " nx  :: Amount of (interior) points in x-direction. Must be >0.\n";
    std::cerr << " ny  :: Amount of (interior) points in y-direction. Must be >0.\n";
    std::cerr << " nz  :: Amount of (interior) points in z-direction. Must be >0.\n";
    std::cerr << "itrs :: Number of diffusive timesteps to perform. Must be >0.\n";
    std::cerr << "mode :: Computation mode. Must be \"laplap-{global/shared}\", or \"biharm-global\".\n";
    std::cerr << "================================================================================\n";
    std::cerr << "Input syntax: ./main <nx> <ny> <nz> <itrs> <mode>\n";
    std::cerr << "================================================================================\n";
}


Mode mode_from_string(const char *s) {
    std::string mode(s);
    if(mode == "laplap-global") return Mode::laplap_global;
    if(mode == "laplap-shared") return Mode::laplap_shared;
    if(mode == "biharm-global") return Mode::biharm_global;
    return Mode::invalid;
}


std::string get_mode_desc(Mode mode) {
    switch(mode) {
        case Mode::laplap_global: return "Double 5-point laplacian stencil. Uses global memory only.";
        case Mode::laplap_shared: return "Double 5-point laplacian stencil. Uses shared memory.";
        case Mode::biharm_global: return "Single 13-point biharmonic stencil. Uses global memory only.";
        default: __builtin_unreachable();
    }
}

} // namespace utils
