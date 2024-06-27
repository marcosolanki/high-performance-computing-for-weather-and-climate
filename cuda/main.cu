#include "device.cuh"
#include "host.cuh"

#include <chrono>
#include <fstream>
#include <sstream>


using time_point = std::chrono::time_point<std::chrono::steady_clock>;


template<typename T>
double run_simulation(std::size_t xsize, std::size_t ysize, std::size_t zsize, std::size_t iters, std::size_t halo) {

    constexpr T alpha = static_cast<T>(1) / 32;
    const std::size_t xmin = halo, xmax = xsize - halo;
    const std::size_t ymin = halo, ymax = ysize - halo;
    const std::size_t zmax = zsize;

    cudaStream_t stream;
    T *u, *v, *w, *u_host;
    std::ofstream in_os, out_os;

    check(cudaMallocHost(&u_host, xsize * ysize * zsize * sizeof(T)));
    host::initialise(u_host, xsize, ysize, zsize);

    in_os.open("in_field.csv");
    host::write_file(in_os, u_host, xsize, ysize, zsize);
    in_os.close();

    const time_point begin = std::chrono::steady_clock::now();

    check(cudaStreamCreate(&stream));
    check(cudaMallocAsync(&u, xsize * ysize * zsize * sizeof(T), stream));
    check(cudaMallocAsync(&v, xsize * ysize * zsize * sizeof(T), stream));
    check(cudaMallocAsync(&w, xsize * ysize * zsize * sizeof(T), stream));
    check(cudaMemcpyAsync(u, u_host, xsize * ysize * zsize * sizeof(T), cudaMemcpyHostToDevice, stream));

    for(std::size_t i = 0; i < iters; ++i) {
        device::update_boundaries(stream, u, xmin, xmax, ymin, ymax, zmax, xsize, ysize);
        device::update_interior_double_laplacian(stream, u, v, w, alpha, xmin, xmax, ymin, ymax, zmax, xsize, ysize);
    }

    check(cudaMemcpyAsync(u_host, u, xsize * ysize * zsize * sizeof(T), cudaMemcpyDeviceToHost, stream));
    check(cudaFreeAsync(u, stream));
    check(cudaFreeAsync(v, stream));
    check(cudaFreeAsync(w, stream));
    check(cudaStreamDestroy(stream));

    check(cudaDeviceSynchronize());
    const time_point end = std::chrono::steady_clock::now();

    out_os.open("out_field.csv");
    host::write_file(out_os, u_host, xsize, ysize, zsize);
    out_os.close();

    check(cudaFreeHost(u_host));

    return std::chrono::duration<double, std::milli>(end - begin).count() / 1000;
}


template<typename T>
int templated_main(int argc, char const **argv) {
    constexpr std::size_t halo = 3;

    if(argc == 5) {
        std::size_t x, y, z, iters;

        {
            std::istringstream x_ss(argv[1]);     x_ss >> x;
            std::istringstream y_ss(argv[2]);     y_ss >> y;
            std::istringstream z_ss(argv[3]);     z_ss >> z;
            std::istringstream iters_ss(argv[4]); iters_ss >> iters;

            if(x_ss.fail() || y_ss.fail() || z_ss.fail() || iters_ss.fail()) {
                std::cerr << "Input syntax: ./main <nx> <ny> <nz> <iters>\n";
                return EXIT_FAILURE;
            }
        }

        std::cout << "================================================================================\n";
        std::cout << "                             Welcome to stencil2d!\n";
        std::cout << "Version    :: C++ with CUDA\n";
        std::cout << "Interior   :: (" << x << ", " << y << ", " << z << ")\n";
        std::cout << "Boundaries :: (" << halo << ", " << halo << ", " << 0 << ")\n";
        std::cout << "Iterations :: " << iters << '\n';
        std::cout << "Real size  :: " << sizeof(T) << '\n';
        std::cout << "================================================================================\n";

        const double time = run_simulation<T>(x + 2 * halo, y + 2 * halo, z, iters, halo);

        std::cout << "Runtime    :: " << time << "s\n";
        std::cout << "================================================================================\n";
    }
    else {
        std::cerr << "Input syntax: ./main <nx> <ny> <nz> <iters>\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


int main(int argc, char const **argv) {
    #if !defined(REALSIZE) || REALSIZE == 8
    return templated_main<double>(argc, argv);
    #elif REALSIZE == 4
    return templated_main<float>(argc, argv);
    #else
    #error "Selected REALSIZE not supported!"
    #endif
}
