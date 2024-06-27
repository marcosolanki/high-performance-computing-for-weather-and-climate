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
