#include "wrappers.cuh"


int main(void) {

    constexpr std::size_t xsize = 128, ysize = 128, zsize = 64, halo = 3;
    constexpr std::size_t xmin = halo, xmax = xsize - halo;
    constexpr std::size_t ymin = halo, ymax = ysize - halo;
    constexpr std::size_t zmin = 0;

    cudaStream_t stream;
    double *u;
    std::ofstream os;

    os.open("out_field.csv");
    cudaStreamCreate(&stream);
    cudaMalloc(&u, xsize * ysize * zsize * sizeof(double));

    initialize(stream, u, xsize, ysize, zsize);
    update_halo(stream, u, xsize, ysize, xmin, xmax, ymin, ymax, zmin);
    write_file(os, u, xsize, ysize, zsize);

    cudaFree(u);
    cudaStreamDestroy(stream);
    os.close();

    return EXIT_SUCCESS;
}
