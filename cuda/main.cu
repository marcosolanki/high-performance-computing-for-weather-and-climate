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


template<typename T>
__global__ void initialize_kernel(T *u,
                                  std::size_t imin, std::size_t imax, std::size_t jmin,
                                  std::size_t jmax, std::size_t kmin, std::size_t kmax,
                                  std::size_t xsize, std::size_t ysize) {

    const std::size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    const std::size_t j = blockDim.y * blockIdx.y + threadIdx.y;
    const std::size_t k = blockDim.z * blockIdx.z + threadIdx.z;

    if(imin <= i && i <= imax && jmin <= j && j <= jmax && kmin <= k && k <= kmax)
        u[index(i, j, k, xsize, ysize)] = 1;
    else
        u[index(i, j, k, xsize, ysize)] = 0;
}


template<typename T>
void initialize(cudaStream_t &stream, T *u,
                std::size_t xsize, std::size_t ysize, std::size_t zsize) {

    const std::size_t imin = static_cast<T>(0.25 * xsize + 0.5);
    const std::size_t imax = static_cast<T>(0.75 * xsize + 0.5);
    const std::size_t jmin = static_cast<T>(0.25 * ysize + 0.5);
    const std::size_t jmax = static_cast<T>(0.75 * ysize + 0.5);
    const std::size_t kmin = static_cast<T>(0.25 * zsize + 0.5);
    const std::size_t kmax = static_cast<T>(0.75 * zsize + 0.5);

    const dim3 block_dim(4, 4, 4);
    const dim3 grid_dim((xsize + (block_dim.x - 1)) / block_dim.x,
                        (ysize + (block_dim.y - 1)) / block_dim.y,
                        (zsize + (block_dim.z - 1)) / block_dim.z);

    initialize_kernel<<<grid_dim, block_dim, 0, stream>>>(u,
                                                          imin, imax, jmin,
                                                          jmax, kmin, kmax,
                                                          xsize, ysize);
}


int main(void) {

    constexpr std::size_t xsize = 128, ysize = 128, zsize = 64;
    cudaStream_t stream;
    double *u;
    std::ofstream os;

    os.open("out_field.csv");
    cudaStreamCreate(&stream);
    cudaMalloc(&u, xsize * ysize * zsize * sizeof(double));
    initialize(stream, u, xsize, ysize, zsize);
    write_file(os, u, xsize, ysize, zsize);
    cudaFree(u);
    cudaStreamDestroy(stream);
    os.close();

    return EXIT_SUCCESS;
}
