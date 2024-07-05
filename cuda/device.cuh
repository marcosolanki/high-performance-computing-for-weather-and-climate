#pragma once

#include "kernels.cuh"


namespace device {

// update_boundaries<T>():
// Enforces periodic boundary conditions in x and y.
//
// Input:   stream          :: CUDA stream used
//          u               :: Input field (located on the device)
//          xmin, xmax      :: i must be in [xmin, xmax[ to access an interior point (i, j, k)
//          ymin, ymax      :: j must be in [ymin, ymax[ to access an interior point (i, j, k)
//          zmax            :: Alias for zsize
//          xsize, ysize    :: Dimensions of the domain (including boundary points)
//          T               :: Numeric real type
// Output:  u               :: Output field (located on the device)
template<typename T>
void update_boundaries(cudaStream_t &stream, T *u,
                       std::size_t xmin, std::size_t xmax,
                       std::size_t ymin, std::size_t ymax, std::size_t zmax,
                       std::size_t xsize, std::size_t ysize) {

    const std::size_t xint = xmax - xmin;
    const std::size_t yint = ymax - ymin;

    // Ranges:
    // i in [xmin, xmax[
    // j in [0, ymin[
    // k in [0, zmax[
    constexpr dim3 block_dim_south(64, 1, 1);
    const dim3 grid_dim_south((xmax - xmin + (block_dim_south.x - 1)) / block_dim_south.x,
                              (ymin + (block_dim_south.y - 1)) / block_dim_south.y,
                              (zmax + (block_dim_south.z - 1)) / block_dim_south.z);

    // Ranges:
    // i in [xmin, xmax[
    // j in [ymax, ysize[
    // k in [0, zmax[
    constexpr dim3 block_dim_north(64, 1, 1);
    const dim3 grid_dim_north((xmax - xmin + (block_dim_north.x - 1)) / block_dim_north.x,
                              (ysize - ymax + (block_dim_north.y - 1)) / block_dim_north.y,
                              (zmax + (block_dim_north.z - 1)) / block_dim_north.z);

    // Ranges:
    // i in [0, xmin[
    // j in [ymin, ymax[
    // k in [0, zmax[
    constexpr dim3 block_dim_west(3, 32, 1);
    const dim3 grid_dim_west((xmin + (block_dim_west.x - 1)) / block_dim_west.x,
                             (ymax - ymin + (block_dim_west.y - 1)) / block_dim_west.y,
                             (zmax + (block_dim_west.z - 1)) / block_dim_west.z);

    // Ranges:
    // i in [xmax, xsize[
    // j in [ymin, ymax[
    // k in [0, zmax[
    constexpr dim3 block_dim_east(3, 32, 1);
    const dim3 grid_dim_east((xsize - xmax + (block_dim_east.x - 1)) / block_dim_east.x,
                             (ymax - ymin + (block_dim_east.y - 1)) / block_dim_east.y,
                             (zmax + (block_dim_east.z - 1)) / block_dim_east.z);

    kernels::update_south<<<grid_dim_south, block_dim_south, 0, stream>>>(u, xmin, xmax, ymin, zmax, yint, xsize, ysize);
    kernels::update_north<<<grid_dim_north, block_dim_north, 0, stream>>>(u, xmin, xmax, ymax, zmax, yint, xsize, ysize);
    kernels::update_west<<<grid_dim_west, block_dim_west, 0, stream>>>(u, xmin, ymin, ymax, zmax, xint, xsize, ysize);
    kernels::update_east<<<grid_dim_east, block_dim_east, 0, stream>>>(u, xmax, ymin, ymax, zmax, xint, xsize, ysize);
}


// update_interior_double_laplacian<T>():
// Performs the fourth-order diffusion update in the interior of the domain using two consecutive 5-point Laplacian stencils and no shared memory.
//
// Input:   stream          :: CUDA stream used
//          u               :: Input field (located on the device)
//          v               :: Temporary field to store intermediate results in (located on the device)
//          xmin, xmax      :: i must be in [xmin, xmax[ to access an interior point (i, j, k)
//          ymin, ymax      :: j must be in [ymin, ymax[ to access an interior point (i, j, k)
//          zmax            :: Alias for zsize
//          xsize, ysize    :: Dimensions of the domain (including boundary points)
//          T               :: Numeric real type
// Output:  u               :: Output field (located on the device)
template<typename T>
void update_interior_double_laplacian(cudaStream_t &stream, T *u, T *v, T alpha, std::size_t xmin,
                                      std::size_t xmax, std::size_t ymin, std::size_t ymax,
                                      std::size_t zmax, std::size_t xsize, std::size_t ysize) {

    constexpr dim3 block_dim(8, 8, 1);

    const dim3 grid_dim_lap((xmax - xmin + 2 + (block_dim.x - 1)) / block_dim.x,
                            (ymax - ymin + 2 + (block_dim.y - 1)) / block_dim.y,
                            (zmax + (block_dim.z - 1)) / block_dim.z);
    const dim3 grid_dim_int((xmax - xmin + (block_dim.x - 1)) / block_dim.x,
                            (ymax - ymin + (block_dim.y - 1)) / block_dim.y,
                            (zmax + (block_dim.z - 1)) / block_dim.z);

    kernels::laplacian<<<grid_dim_lap, block_dim, 0, stream>>>(u, v, xmin - 1, xmax + 1, ymin - 1, ymax + 1, zmax, xsize, ysize);
    kernels::laplacian_update<<<grid_dim_int, block_dim, 0, stream>>>(u, v, alpha, xmin, xmax, ymin, ymax, zmax, xsize, ysize);
}


// update_interior_double_laplacian_shared<T>():
// Performs the fourth-order diffusion update in the interior of the domain using two consecutive 5-point Laplacian stencils and shared memory.
//
// Input:   stream          :: CUDA stream used
//          u               :: Input field (located on the device)
//          v               :: Temporary field to store intermediate results in (located on the device)
//          xmin, xmax      :: i must be in [xmin, xmax[ to access an interior point (i, j, k)
//          ymin, ymax      :: j must be in [ymin, ymax[ to access an interior point (i, j, k)
//          zmax            :: Alias for zsize
//          xsize, ysize    :: Dimensions of the domain (including boundary points)
//          T               :: Numeric real type
// Output:  u               :: Output field (located on the device)
template<typename T>
void update_interior_double_laplacian_shared(cudaStream_t &stream, T *u, T *v, T alpha, std::size_t xmin,
                                             std::size_t xmax, std::size_t ymin, std::size_t ymax,
                                             std::size_t zmax, std::size_t xsize, std::size_t ysize) {

    constexpr dim3 block_dim(8, 8, 1);

    const dim3 grid_dim_lap((xmax - xmin + 2 + (block_dim.x - 1)) / block_dim.x,
                            (ymax - ymin + 2 + (block_dim.y - 1)) / block_dim.y,
                            (zmax + (block_dim.z - 1)) / block_dim.z);
    const dim3 grid_dim_int((xmax - xmin + (block_dim.x - 1)) / block_dim.x,
                            (ymax - ymin + (block_dim.y - 1)) / block_dim.y,
                            (zmax + (block_dim.z - 1)) / block_dim.z);

    constexpr std::size_t shared_size = (block_dim.x + 2) * (block_dim.y + 2) * sizeof(T);

    kernels::laplacian_shared<<<grid_dim_lap, block_dim, shared_size, stream>>>(u, v, xmin - 1, xmax + 1, ymin - 1, ymax + 1, zmax, xsize, ysize);
    kernels::laplacian_shared_update<<<grid_dim_int, block_dim, shared_size, stream>>>(u, v, alpha, xmin, xmax, ymin, ymax, zmax, xsize, ysize);
}


// update_interior_biharmonic<T>():
// Performs the fourth-order diffusion update in the interior of the domain using a single 13-point biharmonic stencil and no shared memory.
//
// Input:   stream          :: CUDA stream used
//          u               :: Input field (located on the device)
//          v               :: Temporary field to store intermediate results in (located on the device)
//          xmin, xmax      :: i must be in [xmin, xmax[ to access an interior point (i, j, k)
//          ymin, ymax      :: j must be in [ymin, ymax[ to access an interior point (i, j, k)
//          zmax            :: Alias for zsize
//          xsize, ysize    :: Dimensions of the domain (including boundary points)
//          T               :: Numeric real type
// Output:  u               :: Output field (located on the device)
template<typename T>
void update_interior_biharmonic(cudaStream_t &stream, T *u, T *v, T alpha, std::size_t xmin,
                                std::size_t xmax, std::size_t ymin, std::size_t ymax,
                                std::size_t zmax, std::size_t xsize, std::size_t ysize) {

    constexpr dim3 block_dim(8, 8, 1);
    const dim3 grid_dim((xmax - xmin + (block_dim.x - 1)) / block_dim.x,
                        (ymax - ymin + (block_dim.y - 1)) / block_dim.y,
                        (zmax + (block_dim.z - 1)) / block_dim.z);

    kernels::biharmonic_operator<<<grid_dim, block_dim, 0, stream>>>(u, v, xmin, xmax, ymin, ymax, zmax, xsize, ysize);
    kernels::update_interior<<<grid_dim, block_dim, 0, stream>>>(u, v, alpha, xmin, xmax, ymin, ymax, zmax, xsize, ysize);
}

} // namespace device
