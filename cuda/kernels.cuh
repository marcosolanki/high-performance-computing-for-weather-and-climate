#pragma once

#include "utils.cuh"


template<typename T>
__global__ void initialize_kernel(T *u,
                                  std::size_t imin, std::size_t imax, std::size_t jmin,
                                  std::size_t jmax, std::size_t kmin, std::size_t kmax,
                                  std::size_t xsize, std::size_t ysize, std::size_t zsize) {

    const std::size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    const std::size_t j = blockDim.y * blockIdx.y + threadIdx.y;
    const std::size_t k = blockDim.z * blockIdx.z + threadIdx.z;

    if(imin <= i && i <= imax && jmin <= j && j <= jmax && kmin <= k && k <= kmax)
        u[index(i, j, k, xsize, ysize)] = 1;
    else if(i < xsize && j < ysize && k < zsize)
        u[index(i, j, k, xsize, ysize)] = 0;
}


template<typename T>
__global__ void update_halo_bottom_kernel(T *u, std::size_t xmin, std::size_t xmax, std::size_t ymin,
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
__global__ void update_halo_top_kernel(T *u, std::size_t xmin, std::size_t xmax, std::size_t ymax,
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
__global__ void update_halo_left_kernel(T *u, std::size_t xmin, std::size_t ymin, std::size_t ymax,
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
__global__ void update_halo_right_kernel(T *u, std::size_t xmax, std::size_t ymin, std::size_t ymax,
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
