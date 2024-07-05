#pragma once

#include "utils.cuh"

#include <ostream>


namespace host {

// initialise<T>():
// Initialises the domain.
//
// Input:   u_host              :: Uninitialised field (located on the host)
//          xsize, ysize, zsize :: Dimensions of the domain (including boundary points)
//          T                   :: Numeric real type
// Output:  u_host              :: Initialised field (located on the host)
template<typename T>
void initialise(T *u_host, std::size_t xsize, std::size_t ysize, std::size_t zsize) {

    const std::size_t imin = static_cast<T>(0.25 * xsize + 0.5);
    const std::size_t imax = static_cast<T>(0.75 * xsize + 0.5);
    const std::size_t jmin = static_cast<T>(0.25 * ysize + 0.5);
    const std::size_t jmax = static_cast<T>(0.75 * ysize + 0.5);
    const std::size_t kmin = static_cast<T>(0.25 * zsize + 0.5);
    const std::size_t kmax = static_cast<T>(0.75 * zsize + 0.5);

    for(std::size_t k = kmin; k <= kmax; ++k)
        for(std::size_t j = jmin; j <= jmax; ++j)
            for(std::size_t i = imin; i <= imax; ++i)
                u_host[index(i, j, k, xsize, ysize)] =
                    static_cast<T>(imin <= i && i <= imax && jmin <= j && j <= jmax && kmin <= k && k <= kmax);
}


// write_file<T>():
// Writes field to a *.csv file.
//
// Input:   os                  :: Output stream
//          u_host              :: Field (located on the host)
//          xsize, ysize, zsize :: Dimensions of the domain (including boundary points)
//          T                   :: Numeric real type
// Output:  os                  :: Output stream (updated)
template<typename T>
void write_file(std::ostream &os, const T *u_host,
                std::size_t xsize, std::size_t ysize, std::size_t zsize) {

    os << xsize << ',' << ysize << ',' << zsize << '\n';

    for(std::size_t k = 0; k < zsize; ++k)
        for(std::size_t j = 0; j < ysize; ++j)
            for(std::size_t i = 0; i < xsize; ++i)
                os << std::setprecision(std::numeric_limits<double>::digits10)
                   << u_host[index(i, j, k, xsize, ysize)]
                   << ((k < zsize - 1 || j < ysize - 1 || i < xsize - 1) ? ',' : '\n');
}

} // namespace host
