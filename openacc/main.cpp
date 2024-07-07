#include "host.hpp"
#include "kernels.hpp"
#include "parallel.hpp"
#include "optimised.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

#ifdef _OPENACC
#include <openacc.h>
#endif


using time_point = std::chrono::time_point<std::chrono::steady_clock>;


// run_simulation<T>():
// Runs the 4th-order diffusion simulation on a CUDA-enabled GPU and writes its results to some output files.
//
// Input:   xsize, ysize, zsize :: Dimensions of the domain (including boundary points)
//          itrs                :: Number of timestep iterations
//          bdry                :: Number of boundary points (halo size)
//          mode                :: Acceleration mode (kernels-based, parallel-based or unrestricted)
//          T                   :: Numeric real type
// Output:  return (...)        :: Measured time (memory transfer + device allocation + computation) in seconds
template<typename T>
double run_simulation(std::size_t xsize, std::size_t ysize, std::size_t zsize, std::size_t itrs, std::size_t bdry, Mode mode) {

    constexpr T alpha = static_cast<T>(1) / 32;
    const std::size_t xmin = bdry, xmax = xsize - bdry;
    const std::size_t ymin = bdry, ymax = ysize - bdry;

    T *u, *v;
    std::ofstream os;

    u = new T[xsize * ysize * zsize];

    host::initialise(u, xsize, ysize, zsize);

    os.open("in_field.csv");
    host::write_file(os, u, xsize, ysize, zsize);
    os.close();

    const time_point start = std::chrono::steady_clock::now();

    #ifdef _OPENACC
    v = static_cast<T*>(acc_malloc(xsize * ysize * zsize * sizeof(T)));
    #else
    v = new T[xsize * ysize * zsize];
    #endif

    if(v == NULL) {
        std::cerr << "ERROR: acc_malloc() returned NULL.\n";
        exit(EXIT_FAILURE);
    }

    #pragma acc data copy(u[0:xsize*ysize*zsize])
    {
        switch(mode) {
            case Mode::kernels: {
                for(std::size_t i = 0; i < itrs; ++i) {
                    kernels::update_boundaries(u, xmin, xmax, ymin, ymax, xsize, ysize, zsize);
                    kernels::update_interior(u, v, alpha, xmin, xmax, ymin, ymax, xsize, ysize, zsize);
                }
                kernels::update_boundaries(u, xmin, xmax, ymin, ymax, xsize, ysize, zsize);
                break;
            }
            case Mode::parallel: {
                for(std::size_t i = 0; i < itrs; ++i) {
                    parallel::update_boundaries(u, xmin, xmax, ymin, ymax, xsize, ysize, zsize);
                    parallel::update_interior(u, v, alpha, xmin, xmax, ymin, ymax, xsize, ysize, zsize);
                }
                parallel::update_boundaries(u, xmin, xmax, ymin, ymax, xsize, ysize, zsize);
                break;
            }
            case Mode::optimised: {
                for(std::size_t i = 0; i < itrs; ++i) {
                    optimised::update_boundaries(u, xmin, xmax, ymin, ymax, xsize, ysize, zsize);
                    optimised::update_interior(u, v, alpha, xmin, xmax, ymin, ymax, xsize, ysize, zsize);
                }
                optimised::update_boundaries(u, xmin, xmax, ymin, ymax, xsize, ysize, zsize);
                break;
            }
            default: __builtin_unreachable();
        }
    }

    #ifdef _OPENACC
    acc_free(v);
    #else
    delete[] v;
    #endif

    const time_point end = std::chrono::steady_clock::now();

    os.open("out_field.csv");
    host::write_file(os, u, xsize, ysize, zsize);
    os.close();

    return std::chrono::duration<double, std::milli>(end - start).count() / 1000;
}


// templated_main<T>():
// Main function with flexible numeric real type.
//
// Input:   argv            :: Input arguments
//          argc            :: Number of input arguments
//          T               :: Numeric real type
// Output:  return (...)    :: Exit code (EXIT_SUCCESS or EXIT_FAILURE)
template<typename T>
int templated_main(int argc, char const **argv) {
    constexpr std::size_t bdry = 3;

    if(argc == 6) {
        std::size_t x, y, z, itrs;
        Mode mode;

        {
            std::istringstream x_ss(argv[1]), y_ss(argv[2]), z_ss(argv[3]), itrs_ss(argv[4]);
            x_ss >> x; y_ss >> y; z_ss >> z; itrs_ss >> itrs;
            mode = utils::mode_from_string(argv[5]);

            if(x_ss.fail() || y_ss.fail() || z_ss.fail() || itrs_ss.fail() ||
               x == 0 || y == 0 || z == 0 || itrs == 0 || mode == Mode::invalid) {

                utils::print_args_errmsg();
                return EXIT_FAILURE;
            }
        }

        std::cout << "================================================================================\n";
        std::cout << "                             Welcome to stencil2d!\n";
        std::cout << "Version    :: C++ with OpenACC\n";
        std::cout << "Interior   :: (" << x << ", " << y << ", " << z << ")\n";
        std::cout << "Boundaries :: (" << bdry << ", " << bdry << ", " << 0 << ")\n";
        std::cout << "Iterations :: " << itrs << '\n';
        std::cout << "Real size  :: " << sizeof(T) << '\n';
        std::cout << "OpenACC    :: "
        #ifdef _OPENACC
                  << "Enabled\n";
        #else
                  << "Disabled\n";
        #endif
        std::cout << "Exec. mode :: " << utils::get_mode_desc(mode) << '\n';
        std::cout << "================================================================================\n";

        const double time = run_simulation<T>(x + 2 * bdry, y + 2 * bdry, z, itrs, bdry, mode);

        std::cout << "Runtime    :: " << time << "s\n";
        std::cout << "================================================================================\n";
    }
    else {
        utils::print_args_errmsg();
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
