#include "kernels.hpp"

#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

#include <openacc.h>


using time_point = std::chrono::time_point<std::chrono::steady_clock>;


template<typename T>
double run_simulation(std::size_t xsize, std::size_t ysize, std::size_t zsize, std::size_t itrs, std::size_t halo) {
    assert(0 < xsize && 0 < ysize && 0 < zsize && 0 < itrs);

    constexpr T alpha = static_cast<T>(1) / 32;
    const std::size_t xmin = halo, xmax = xsize - halo;
    const std::size_t ymin = halo, ymax = ysize - halo;
    const std::size_t zmax = zsize;

    T *u, *v;
    std::ofstream os;

    u = new T[xsize * ysize * zsize];
    v = new T[xsize * ysize];

    utils::initialise(u, xsize, ysize, zsize);

    os.open("in_field.csv");
    utils::write_file(os, u, xsize, ysize, zsize);
    os.close();

    const time_point start = std::chrono::steady_clock::now();

    #pragma acc data copy(u[0:xsize*ysize*zsize]) create(v[0:xsize*ysize])
    {
        for(std::size_t i = 0; i < itrs; ++i) {
            kernels::update_boundaries(u, xmin, xmax, ymin, ymax, zmax, xsize, ysize);
            kernels::update_interior(u, v, alpha, xmin, xmax, ymin, ymax, zmax, xsize, ysize);
        }
        kernels::update_boundaries(u, xmin, xmax, ymin, ymax, zmax, xsize, ysize);
    }

    const time_point end = std::chrono::steady_clock::now();

    os.open("out_field.csv");
    utils::write_file(os, u, xsize, ysize, zsize);
    os.close();

    return std::chrono::duration<double, std::milli>(end - start).count() / 1000;
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
        std::cout << "   / \\"                                               << '\n';
        std::cout << "  / | \\                                 _ _"          << '\n';
        std::cout << "  | | | +-+-+  +-+  +   +  +-+  +  +   /   \\  +--"    << '\n';
        std::cout << "  /\\ \\/   |    |    |\\  |  |    |  |       /  |  |" << '\n';
        std::cout << "  | | |   |    +-+  | \\ |  |    |  |      /   |  |"   << '\n';
        std::cout << "  \\ | /   |    |    |  \\|  |    |  |     /    |  |"  << '\n';
        std::cout << "   \\ /    +    +-+  +   +  +-+  +  +-+  +--+  +--"    << '\n';
        std::cout << "================================================================================\n";
        std::cout << "                             Welcome to stencil2d!\n";
        std::cout << "Version    :: C++ with OpenACC\n";
        std::cout << "Interior   :: (" << x << ", " << y << ", " << z << ")\n";
        std::cout << "Boundaries :: (" << halo << ", " << halo << ", " << 0 << ")\n";
        std::cout << "Iterations :: " << iters << '\n';
        std::cout << "Real size  :: " << sizeof(T) << '\n';
        std::cout << "OpenACC    :: "
        #ifdef _OPENACC
                  << "Enabled\n";
        #else
                  << "Disabled\n";
        #endif
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
