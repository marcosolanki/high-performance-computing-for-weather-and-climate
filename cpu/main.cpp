#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

#include "utils.hpp"


using time_point = std::chrono::time_point<std::chrono::steady_clock>;


template<typename T>
void update_halo(Storage3D<T> &in_field) {

    const std::size_t x_interior = in_field.x_max() - in_field.x_min();
    const std::size_t y_interior = in_field.y_max() - in_field.y_min();

    // Bottom edge (without corners):
    for(std::size_t k = 0; k < in_field.z_max(); ++k)
        for(std::size_t j = 0; j < in_field.y_min(); ++j)
            for(std::size_t i = in_field.x_min(); i < in_field.x_max(); ++i)
                in_field(i, j, k) = in_field(i, j + y_interior, k);

    // Top edge (without corners):
    for(std::size_t k = 0; k < in_field.z_max(); ++k)
        for(std::size_t j = in_field.y_max(); j < in_field.y_size(); ++j)
            for(std::size_t i = in_field.x_min(); i < in_field.x_max(); ++i)
                in_field(i, j, k) = in_field(i, j - y_interior, k);

    // Left edge (including corners):
    for(std::size_t k = 0; k < in_field.z_max(); ++k)
        for(std::size_t j = in_field.y_min(); j < in_field.y_max(); ++j)
            for(std::size_t i = 0; i < in_field.x_min(); ++i)
                in_field(i, j, k) = in_field(i + x_interior, j, k);

    // Right edge (including corners):
    for(std::size_t k = 0; k < in_field.z_max(); ++k)
        for(std::size_t j = in_field.y_min(); j < in_field.y_max(); ++j)
            for(std::size_t i = in_field.x_max(); i < in_field.x_size(); ++i)
                in_field(i, j, k) = in_field(i - x_interior, j, k);
}


template<typename T>
void apply_diffusion(Storage3D<T> &in_field, Storage3D<T> &out_field, T alpha,
                     std::size_t num_iter, std::size_t x, std::size_t y, std::size_t halo) {

    Storage3D<T> tmp_field(x, y, 1, halo);

    for(std::size_t iter = 0; iter < num_iter; ++iter) {
        update_halo(in_field);

        for(std::size_t k = 0; k < in_field.z_max(); ++k) {

            // Apply the initial laplacian:
            for(std::size_t j = in_field.y_min() - 1; j < in_field.y_max() + 1; ++j)
                for(std::size_t i = in_field.x_min() - 1; i < in_field.x_max() + 1; ++i)
                    tmp_field(i, j, 0) = -static_cast<T>(4) * in_field(i, j, k) + in_field(i - 1, j, k) + in_field(i + 1, j, k)
                                                                                + in_field(i, j - 1, k) + in_field(i, j + 1, k);

            // Apply the second laplacian:
            for(std::size_t j = in_field.y_min(); j < in_field.y_max(); ++j) {
                for(std::size_t i = in_field.x_min(); i < in_field.x_max(); ++i) {
                    T laplap = -static_cast<T>(4) * tmp_field(i, j, 0) + tmp_field(i - 1, j, 0) + tmp_field(i + 1, j, 0)
                                                                       + tmp_field(i, j - 1, 0) + tmp_field(i, j + 1, 0);

                    // ...and update the field:
                    if(iter == num_iter - 1) out_field(i, j, k) = in_field(i, j, k) - alpha * laplap;
                    else in_field(i, j, k) = in_field(i, j, k) - alpha * laplap;
                }
            }
        }
    }
}


template<typename T>
void report_time(const Storage3D<T> &storage, std::size_t num_iter, double time) {
    std::cout << " nx  = " << storage.x_max() - storage.x_min() << '\n'
              << " ny  = " << storage.y_max() - storage.y_min() << '\n'
              << " nz  = " << storage.z_max() << '\n'
              << "iter = " << num_iter << '\n'
              << "time = " << time << "s\n";
}


template<typename T>
void run_simulation(std::size_t x, std::size_t y, std::size_t z, std::size_t iter, std::size_t halo) {
    assert(0 < x && 0 < y && 0 < z && 0 < iter);

    Storage3D<T> input(x, y, z, halo);
    input.initialize();
    Storage3D<T> output(x, y, z, halo);
    output.initialize();

    constexpr T alpha = 1 / static_cast<T>(32);

    std::ofstream fout;
    fout.open("in_field.csv");
    input.write_file(fout);
    fout.close();
    const time_point start = std::chrono::steady_clock::now();

    apply_diffusion<T>(input, output, alpha, iter, x, y, halo);

    const time_point end = std::chrono::steady_clock::now();
    update_halo<T>(output);
    fout.open("out_field.csv");
    output.write_file(fout);
    fout.close();

    const double time = std::chrono::duration<double, std::milli>(end - start).count() / 1000;
    report_time(output, iter, time);
}


int main(int argc, char const **argv) {

    constexpr std::size_t halo = 3;

    if(argc == 5) {
        std::size_t x, y, z, iter;

        {
            std::istringstream x_ss(argv[1]);    x_ss >> x;
            std::istringstream y_ss(argv[2]);    y_ss >> y;
            std::istringstream z_ss(argv[3]);    z_ss >> z;
            std::istringstream iter_ss(argv[4]); iter_ss >> iter;

            if(x_ss.fail() || y_ss.fail() || z_ss.fail() || iter_ss.fail()) {
                std::cerr << "Input syntax: ./main <nx> <ny> <nz> <iter>\n";
                return EXIT_FAILURE;
            }
        }

        run_simulation<double>(x, y, z, iter, halo);
    }
    else {
        std::cerr << "Input syntax: ./main <nx> <ny> <nz> <iter>\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
