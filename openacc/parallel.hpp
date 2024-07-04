#pragma once

#include "utils.hpp"


namespace parallel {

template<typename T>
void update_boundaries(T *u, std::size_t xmin, std::size_t xmax,
                       std::size_t ymin, std::size_t ymax, std::size_t zmax,
                       std::size_t xsize, std::size_t ysize) {

    const std::size_t xint = xmax - xmin;
    const std::size_t yint = ymax - ymin;

    // South edge (without corners):
    #pragma acc parallel loop collapse(3) present(u)
    for(std::size_t k = 0; k < zmax; ++k)
        for(std::size_t j = 0; j < ymin; ++j)
            for(std::size_t i = xmin; i < xmax; ++i)
                u[index(i, j, k, xsize, ysize)] = u[index(i, j + yint, k, xsize, ysize)];

    // North edge (without corners):
    #pragma acc parallel loop collapse(3) present(u)
    for(std::size_t k = 0; k < zmax; ++k)
        for(std::size_t j = ymax; j < ysize; ++j)
            for(std::size_t i = xmin; i < xmax; ++i)
                u[index(i, j, k, xsize, ysize)] = u[index(i, j - yint, k, xsize, ysize)];

    // West edge (including corners):
    #pragma acc parallel loop collapse(3) present(u)
    for(std::size_t k = 0; k < zmax; ++k)
        for(std::size_t j = ymin; j < ymax; ++j)
            for(std::size_t i = 0; i < xmin; ++i)
                u[index(i, j, k, xsize, ysize)] = u[index(i + xint, j, k, xsize, ysize)];

    // East edge (including corners):
    #pragma acc parallel loop collapse(3) present(u)
    for(std::size_t k = 0; k < zmax; ++k)
        for(std::size_t j = ymin; j < ymax; ++j)
            for(std::size_t i = xmax; i < xsize; ++i)
                u[index(i, j, k, xsize, ysize)] = u[index(i - xint, j, k, xsize, ysize)];
}


template<typename T>
void update_interior(T *u, T *v, T alpha, std::size_t xmin, std::size_t xmax, std::size_t ymin,
                     std::size_t ymax, std::size_t zmax, std::size_t xsize, std::size_t ysize) {

    // Apply the initial laplacian:
    #pragma acc parallel loop collapse(3) present(u) deviceptr(v)
    for(std::size_t k = 0; k < zmax; ++k)
        for(std::size_t j = ymin - 1; j < ymax + 1; ++j)
            for(std::size_t i = xmin - 1; i < xmax + 1; ++i)
                v[index(i, j, k, xsize, ysize)] = -static_cast<T>(4) * u[index(i, j, k, xsize, ysize)]
                                                                     + u[index(i - 1, j, k, xsize, ysize)]
                                                                     + u[index(i + 1, j, k, xsize, ysize)]
                                                                     + u[index(i, j - 1, k, xsize, ysize)]
                                                                     + u[index(i, j + 1, k, xsize, ysize)];

    // Apply the second laplacian and update the field:
    #pragma acc parallel loop collapse(3) present(u) deviceptr(v)
    for(std::size_t k = 0; k < zmax; ++k)
        for(std::size_t j = ymin; j < ymax; ++j)
            for(std::size_t i = xmin; i < xmax; ++i)
                u[index(i, j, k, xsize, ysize)] += alpha * (static_cast<T>(4) * v[index(i, j, k, xsize, ysize)]
                                                                              - v[index(i - 1, j, k, xsize, ysize)]
                                                                              - v[index(i + 1, j, k, xsize, ysize)]
                                                                              - v[index(i, j - 1, k, xsize, ysize)]
                                                                              - v[index(i, j + 1, k, xsize, ysize)]);
}

} // namespace parallel
