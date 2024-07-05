#pragma once

#include "utils.hpp"


namespace kernels {

// update_boundaries<T>():
// Enforces periodic boundary conditions in x and y.
//
// Input:   u               :: Input field (located on the device)
//          xmin, xmax      :: i must be in [xmin, xmax[ to access an interior point (i, j, k)
//          ymin, ymax      :: j must be in [ymin, ymax[ to access an interior point (i, j, k)
//          zmax            :: Alias for zsize
//          xsize, ysize    :: Dimensions of the domain (including boundary points)
//          T               :: Numeric real type
// Output:  u               :: Output field (located on the device)
template<typename T>
void update_boundaries(T *u, std::size_t xmin, std::size_t xmax,
                       std::size_t ymin, std::size_t ymax, std::size_t zmax,
                       std::size_t xsize, std::size_t ysize) {

    const std::size_t xint = xmax - xmin;
    const std::size_t yint = ymax - ymin;

    #pragma acc kernels present(u)
    {
        // South edge (without corners):
        for(std::size_t k = 0; k < zmax; ++k)
            for(std::size_t j = 0; j < ymin; ++j)
                for(std::size_t i = xmin; i < xmax; ++i)
                    u[index(i, j, k, xsize, ysize)] = u[index(i, j + yint, k, xsize, ysize)];

        // North edge (without corners):
        for(std::size_t k = 0; k < zmax; ++k)
            for(std::size_t j = ymax; j < ysize; ++j)
                for(std::size_t i = xmin; i < xmax; ++i)
                    u[index(i, j, k, xsize, ysize)] = u[index(i, j - yint, k, xsize, ysize)];

        // West edge (including corners):
        for(std::size_t k = 0; k < zmax; ++k)
            for(std::size_t j = ymin; j < ymax; ++j)
                for(std::size_t i = 0; i < xmin; ++i)
                    u[index(i, j, k, xsize, ysize)] = u[index(i + xint, j, k, xsize, ysize)];

        // East edge (including corners):
        for(std::size_t k = 0; k < zmax; ++k)
            for(std::size_t j = ymin; j < ymax; ++j)
                for(std::size_t i = xmax; i < xsize; ++i)
                    u[index(i, j, k, xsize, ysize)] = u[index(i - xint, j, k, xsize, ysize)];
    }
}


// update_interior<T>():
// Performs the fourth-order diffusion update in the interior of the domain using two consecutive 5-point Laplacian stencils.
//
// Input:   u               :: Input field (located on the device)
//          v               :: Temporary field to store intermediate results in (located on the device)
//          xmin, xmax      :: i must be in [xmin, xmax[ to access an interior point (i, j, k)
//          ymin, ymax      :: j must be in [ymin, ymax[ to access an interior point (i, j, k)
//          zmax            :: Alias for zsize
//          xsize, ysize    :: Dimensions of the domain (including boundary points)
//          T               :: Numeric real type
// Output:  u               :: Output field (located on the device)
template<typename T>
void update_interior(T *u, T *v, T alpha, std::size_t xmin, std::size_t xmax, std::size_t ymin,
                     std::size_t ymax, std::size_t zmax, std::size_t xsize, std::size_t ysize) {

    #pragma acc kernels present(u) deviceptr(v)
    {
        // Apply the initial laplacian:
        for(std::size_t k = 0; k < zmax; ++k)
            for(std::size_t j = ymin - 1; j < ymax + 1; ++j)
                for(std::size_t i = xmin - 1; i < xmax + 1; ++i)
                    v[index(i, j, k, xsize, ysize)] = -static_cast<T>(4) * u[index(i, j, k, xsize, ysize)]
                                                                         + u[index(i - 1, j, k, xsize, ysize)]
                                                                         + u[index(i + 1, j, k, xsize, ysize)]
                                                                         + u[index(i, j - 1, k, xsize, ysize)]
                                                                         + u[index(i, j + 1, k, xsize, ysize)];

        // Apply the second laplacian and update the field:
        for(std::size_t k = 0; k < zmax; ++k)
            for(std::size_t j = ymin; j < ymax; ++j)
                for(std::size_t i = xmin; i < xmax; ++i)
                    u[index(i, j, k, xsize, ysize)] += alpha * (static_cast<T>(4) * v[index(i, j, k, xsize, ysize)]
                                                                                  - v[index(i - 1, j, k, xsize, ysize)]
                                                                                  - v[index(i + 1, j, k, xsize, ysize)]
                                                                                  - v[index(i, j - 1, k, xsize, ysize)]
                                                                                  - v[index(i, j + 1, k, xsize, ysize)]);
    }
}

} // namespace kernels
