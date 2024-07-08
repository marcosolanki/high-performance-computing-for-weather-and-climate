#pragma once

#include <iostream>


namespace {

// Translates a 3D index to a 1D/linear index.
static inline std::size_t index(std::size_t i, std::size_t j, std::size_t k,
                                std::size_t xsize, std::size_t ysize) {
    return i + j * xsize + k * xsize * ysize;
}

} // namespace


enum class Mode {
    kernels,
    parallel,
    optimised,
    invalid
};


namespace utils {

// Prints an explanation of the input syntax. Called when invalid input is detected.
void print_args_errmsg() {
    std::cerr << "================================================================================\n";
    std::cerr << "                             Welcome to stencil2d!\n";
    std::cerr << " nx  :: Amount of (interior) points in x-direction. Must be >0.\n";
    std::cerr << " ny  :: Amount of (interior) points in y-direction. Must be >0.\n";
    std::cerr << " nz  :: Amount of (interior) points in z-direction. Must be >0.\n";
    std::cerr << "bdry :: Boundary/halo width/size. Must be >1.\n";
    std::cerr << "itrs :: Number of diffusive timesteps to perform. Must be >0.\n";
    std::cerr << "mode :: Computation mode. Must be \"kernels\", \"parallel\" or \"optimised\".\n";
    std::cerr << "================================================================================\n";
    std::cerr << "Input syntax: ./main <nx> <ny> <nz> <bdry> <itrs> <mode>\n";
    std::cerr << "================================================================================\n";
}


// Translates an input string to an acceleration mode.
Mode mode_from_string(const char *s) {
    std::string mode(s);
    if(mode == "kernels") return Mode::kernels;
    if(mode == "parallel") return Mode::parallel;
    if(mode == "optimised") return Mode::optimised;
    return Mode::invalid;
}


// Returns a brief description of a given acceleration mode.
std::string get_mode_desc(Mode mode) {
    switch(mode) {
        case Mode::kernels: return "OpenACC acceleration using only \"kernels\" pragmas.";
        case Mode::parallel: return "OpenACC acceleration using only \"parallel loop collapse\" pragmas.";
        case Mode::optimised: return "OpenACC acceleration using all possible pragmas.";
        default: __builtin_unreachable();
    }
}

} // namespace utils
