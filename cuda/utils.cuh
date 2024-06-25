#pragma once

#include <fstream>


static inline __host__ __device__ std::size_t index(std::size_t i, std::size_t j, std::size_t k,
                                                    std::size_t xsize, std::size_t ysize) {
    return i + j * xsize + k * xsize * ysize;
}


template<typename T>
void write_file(std::ostream &os, T *u,
                std::size_t xsize, std::size_t ysize, std::size_t zsize) {

    T *u_host = new T[xsize * ysize * zsize];
    cudaMemcpy(u_host, u, xsize * ysize * zsize * sizeof(T), cudaMemcpyDeviceToHost);

    os << xsize << ',' << ysize << ',' << zsize << '\n';

    for(std::size_t k = 0; k < zsize; ++k)
        for(std::size_t j = 0; j < ysize; ++j)
            for(std::size_t i = 0; i < xsize; ++i)
                os << u_host[index(i, j, k, xsize, ysize)]
                   << ((k < zsize - 1 || j < ysize - 1 || i < xsize - 1) ? ',' : '\n');

    delete[] u_host;
}
