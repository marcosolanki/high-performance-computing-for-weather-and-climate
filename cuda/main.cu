#include "wrappers.cuh"


int main(void) {

    constexpr std::size_t xsize = 128, ysize = 128, zsize = 64;
    constexpr std::size_t halo = 3, iters = 1024;
    constexpr double alpha = 1.0 / 32;

    constexpr std::size_t xmin = halo, xmax = xsize - halo;
    constexpr std::size_t ymin = halo, ymax = ysize - halo;
    constexpr std::size_t zmin = 0, zmax = zsize;

    cudaStream_t stream;
    double *u, *v;
    std::ofstream in_os, out_os;

    cudaStreamCreate(&stream);
    cudaMalloc(&u, xsize * ysize * zsize * sizeof(double));
    cudaMalloc(&v, xsize * ysize * zsize * sizeof(double));
    initialise(stream, u, xsize, ysize, zsize);

    in_os.open("in_field.csv");
    write_file(in_os, u, xsize, ysize, zsize);
    in_os.close();

    for(std::size_t i = 0; i < iters; ++i) {
        update_boundaries(stream, u, xmin, xmax, ymin, ymax, zmin, xsize, ysize);
        update_interior(stream, u, v, alpha, xmin, xmax, ymin, ymax, zmax, xsize, ysize);
    }

    out_os.open("out_field.csv");
    write_file(out_os, u, xsize, ysize, zsize);
    out_os.close();

    cudaFree(u);
    cudaFree(v);
    cudaStreamDestroy(stream);

    return EXIT_SUCCESS;
}
