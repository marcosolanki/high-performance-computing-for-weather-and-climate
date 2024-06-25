#pragma once

#include "kernels.cuh"


template<typename T>
void initialize(cudaStream_t &stream, T *u,
                std::size_t xsize, std::size_t ysize, std::size_t zsize) {

    const std::size_t imin = static_cast<T>(0.25 * xsize + 0.5);
    const std::size_t imax = static_cast<T>(0.75 * xsize + 0.5);
    const std::size_t jmin = static_cast<T>(0.25 * ysize + 0.5);
    const std::size_t jmax = static_cast<T>(0.75 * ysize + 0.5);
    const std::size_t kmin = static_cast<T>(0.25 * zsize + 0.5);
    const std::size_t kmax = static_cast<T>(0.75 * zsize + 0.5);

    constexpr dim3 block_dim(4, 4, 4);
    const dim3 grid_dim((xsize + (block_dim.x - 1)) / block_dim.x,
                        (ysize + (block_dim.y - 1)) / block_dim.y,
                        (zsize + (block_dim.z - 1)) / block_dim.z);

    initialize_kernel<<<grid_dim, block_dim, 0, stream>>>(u,
                                                          imin, imax, jmin,
                                                          jmax, kmin, kmax,
                                                          xsize, ysize, zsize);
}


template<typename T>
void update_halo(cudaStream_t &stream, T *u,
                 std::size_t xsize, std::size_t ysize,
                 std::size_t xmin, std::size_t xmax,
                 std::size_t ymin, std::size_t ymax, std::size_t zmin) {

    const std::size_t xint = xmax - xmin;
    const std::size_t yint = ymax - ymin;

    constexpr dim3 block_dim(4, 4, 4);

    // Ranges:
    // i in [xmin, xmax[
    // j in [0, ymin[
    // k in [0, zmin[
    const dim3 grid_dim_bottom((xmax - xmin + (block_dim.x - 1)) / block_dim.x,
                               (ymin + (block_dim.y - 1)) / block_dim.y,
                               (zmin + (block_dim.z - 1)) / block_dim.z);

    // Ranges:
    // i in [xmin, xmax[
    // j in [ymax, ysize[
    // k in [0, zmin[
    const dim3 grid_dim_top((xmax - xmin + (block_dim.x - 1)) / block_dim.x,
                            (ysize - ymax + (block_dim.y - 1)) / block_dim.y,
                            (zmin + (block_dim.z - 1)) / block_dim.z);

    // Ranges:
    // i in [0, xmin[
    // j in [ymin, ymax[
    // k in [0, zmin[
    const dim3 grid_dim_left((xmin + (block_dim.x - 1)) / block_dim.x,
                             (ymax - ymin + (block_dim.y - 1)) / block_dim.y,
                             (zmin + (block_dim.z - 1)) / block_dim.z);

    // Ranges:
    // i in [xmax, xsize[
    // j in [ymin, ymax[
    // k in [0, zmin[
    const dim3 grid_dim_right((xsize - xmax + (block_dim.x - 1)) / block_dim.x,
                              (ymax - ymin + (block_dim.y - 1)) / block_dim.y,
                              (zmin + (block_dim.z - 1)) / block_dim.z);

    update_halo_bottom_kernel<<<grid_dim_bottom, block_dim, 0, stream>>>(u, xmin, xmax, ymin, zmin, yint, xsize, ysize);
    update_halo_top_kernel<<<grid_dim_top, block_dim, 0, stream>>>(u, xmin, xmax, ymax, zmin, yint, xsize, ysize);
    update_halo_left_kernel<<<grid_dim_left, block_dim, 0, stream>>>(u, xmin, ymin, ymax, zmin, xint, xsize, ysize);
    update_halo_right_kernel<<<grid_dim_right, block_dim, 0, stream>>>(u, xmax, ymin, ymax, zmin, xint, xsize, ysize);
}
