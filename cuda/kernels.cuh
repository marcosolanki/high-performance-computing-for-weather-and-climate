#pragma once

#include "utils.cuh"


namespace kernels {

template<typename T>
__global__ void update_south(T *u, std::size_t xmin, std::size_t xmax, std::size_t ymin,
                             std::size_t zmin, std::size_t yint, std::size_t xsize, std::size_t ysize) {

    // Ranges:
    // i in [xmin, xmax[
    // j in [0, ymin[
    // k in [0, zmin[

    const std::size_t i = blockDim.x * blockIdx.x + threadIdx.x + xmin;
    const std::size_t j = blockDim.y * blockIdx.y + threadIdx.y;
    const std::size_t k = blockDim.z * blockIdx.z + threadIdx.z;

    if(i < xmax && j < ymin && k < zmin)
        u[index(i, j, k, xsize, ysize)] = u[index(i, j + yint, k, xsize, ysize)];
}


template<typename T>
__global__ void update_north(T *u, std::size_t xmin, std::size_t xmax, std::size_t ymax,
                             std::size_t zmin, std::size_t yint, std::size_t xsize, std::size_t ysize) {

    // Ranges:
    // i in [xmin, xmax[
    // j in [ymax, ysize[
    // k in [0, zmin[

    const std::size_t i = blockDim.x * blockIdx.x + threadIdx.x + xmin;
    const std::size_t j = blockDim.y * blockIdx.y + threadIdx.y + ymax;
    const std::size_t k = blockDim.z * blockIdx.z + threadIdx.z;

    if(i < xmax && j < ysize && k < zmin)
        u[index(i, j, k, xsize, ysize)] = u[index(i, j - yint, k, xsize, ysize)];
}


template<typename T>
__global__ void update_west(T *u, std::size_t xmin, std::size_t ymin, std::size_t ymax,
                            std::size_t zmin, std::size_t xint, std::size_t xsize, std::size_t ysize) {

    // Ranges:
    // i in [0, xmin[
    // j in [ymin, ymax[
    // k in [0, zmin[

    const std::size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    const std::size_t j = blockDim.y * blockIdx.y + threadIdx.y + ymin;
    const std::size_t k = blockDim.z * blockIdx.z + threadIdx.z;

    if(i < xmin && j < ymax && k < zmin)
        u[index(i, j, k, xsize, ysize)] = u[index(i + xint, j, k, xsize, ysize)];
}


template<typename T>
__global__ void update_east(T *u, std::size_t xmax, std::size_t ymin, std::size_t ymax,
                            std::size_t zmin, std::size_t xint, std::size_t xsize, std::size_t ysize) {

    // Ranges:
    // i in [xmax, xsize[
    // j in [ymin, ymax[
    // k in [0, zmin[

    const std::size_t i = blockDim.x * blockIdx.x + threadIdx.x + xmax;
    const std::size_t j = blockDim.y * blockIdx.y + threadIdx.y + ymin;
    const std::size_t k = blockDim.z * blockIdx.z + threadIdx.z;

    if(i < xsize && j < ymax && k < zmin)
        u[index(i, j, k, xsize, ysize)] = u[index(i - xint, j, k, xsize, ysize)];
}


template<typename T>
__global__ void biharmonic_operator(const T *u, T *v, std::size_t xmin, std::size_t xmax, std::size_t ymin,
                                    std::size_t ymax, std::size_t zmax, std::size_t xsize, std::size_t ysize) {

    const std::size_t i = blockDim.x * blockIdx.x + threadIdx.x + xmin;
    const std::size_t j = blockDim.y * blockIdx.y + threadIdx.y + ymin;
    const std::size_t k = blockDim.z * blockIdx.z + threadIdx.z;

    // Stencil:
    //         1
    //     2  -8   2
    // 1  -8  20  -8  1
    //     2  -8   2
    //         1

    if(i < xmax && j < ymax && k < zmax)
        v[index(i, j, k, xsize, ysize)]
            = 1 * (u[index(i + 2, j, k, xsize, ysize)] + u[index(i, j + 2, k, xsize, ysize)]
                +  u[index(i - 2, j, k, xsize, ysize)] + u[index(i, j - 2, k, xsize, ysize)])
            + 2 * (u[index(i + 1, j + 1, k, xsize, ysize)] + u[index(i + 1, j - 1, k, xsize, ysize)]
                +  u[index(i - 1, j + 1, k, xsize, ysize)] + u[index(i - 1, j - 1, k, xsize, ysize)])
            - 8 * (u[index(i + 1, j, k, xsize, ysize)] + u[index(i, j + 1, k, xsize, ysize)]
                +  u[index(i - 1, j, k, xsize, ysize)] + u[index(i, j - 1, k, xsize, ysize)])
           + 20 * u[index(i, j, k, xsize, ysize)];
}


template<typename T>
__global__ void update_interior(T *u, const T *v, T alpha, std::size_t xmin, std::size_t xmax, std::size_t ymin,
                                std::size_t ymax, std::size_t zmax, std::size_t xsize, std::size_t ysize) {

    const std::size_t i = blockDim.x * blockIdx.x + threadIdx.x + xmin;
    const std::size_t j = blockDim.y * blockIdx.y + threadIdx.y + ymin;
    const std::size_t k = blockDim.z * blockIdx.z + threadIdx.z;

    if(i < xmax && j < ymax && k < zmax)
        u[index(i, j, k, xsize, ysize)] -= alpha * v[index(i, j, k, xsize, ysize)];
}

} // namespace kernels
