#pragma once

#include <ostream>


namespace {

static inline std::size_t index(std::size_t i, std::size_t j, std::size_t k,
                                std::size_t xsize, std::size_t ysize) {
    return i + j * xsize + k * xsize * ysize;
}

} // namespace


namespace utils {

template<typename T>
void initialise(T *u, std::size_t xsize, std::size_t ysize, std::size_t zsize) {

    const std::size_t imin = static_cast<T>(0.25 * xsize + 0.5);
    const std::size_t imax = static_cast<T>(0.75 * xsize + 0.5);
    const std::size_t jmin = static_cast<T>(0.25 * ysize + 0.5);
    const std::size_t jmax = static_cast<T>(0.75 * ysize + 0.5);
    const std::size_t kmin = static_cast<T>(0.25 * zsize + 0.5);
    const std::size_t kmax = static_cast<T>(0.75 * zsize + 0.5);

    for(std::size_t k = 0; k < zsize; ++k)
        for(std::size_t j = 0; j < ysize; ++j)
            for(std::size_t i = 0; i < xsize; ++i)
                u[index(i, j, k, xsize, ysize)] =
                    static_cast<T>(imin <= i && i <= imax && jmin <= j && j <= jmax && kmin <= k && k <= kmax);
}


template<typename T>
void write_file(std::ostream &os, T *u,
                std::size_t xsize, std::size_t ysize, std::size_t zsize) {

    os << xsize << ',' << ysize << ',' << zsize << '\n';

    for(std::size_t k = 0; k < zsize; ++k)
        for(std::size_t j = 0; j < ysize; ++j)
            for(std::size_t i = 0; i < xsize; ++i)
                os << u[index(i, j, k, xsize, ysize)]
                   << ((k < zsize - 1 || j < ysize - 1 || i < xsize - 1) ? ',' : '\n');
}

} // namespace utils
