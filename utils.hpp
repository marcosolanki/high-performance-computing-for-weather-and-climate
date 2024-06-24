#pragma once

#include <cmath>
#include <ostream>
#include <vector>


template<typename T>
class Storage3D {
    public:
    Storage3D(std::size_t x, std::size_t y, std::size_t z, std::size_t halo, T value=0)
        : xsize_(x + 2 * halo), ysize_(y + 2 * halo), zsize_(z), halosize_(halo),
          data_((x + 2 * halo) * (y + 2 * halo) * (z + 2 * halo), value) {}

    T &operator()(std::size_t i, std::size_t j, std::size_t k) { return data_[i + j * xsize_ + k * xsize_ * ysize_]; }

    void write_file(std::ostream &os) {

        os << xsize_ << ',' << ysize_ << ',' << zsize_ << '\n';

        for(std::size_t k = 0; k < zsize_; ++k)
            for(std::size_t j = 0; j < ysize_; ++j)
                for(std::size_t i = 0; i < xsize_; ++i)
                    os << operator()(i, j, k)
                       << ((k < zsize_ - 1 || j < ysize_ - 1 || i < xsize_ - 1) ? ',' : '\n');
    }

    void initialize() {
        const std::size_t kmin = std::round(0.25 * zsize_);
        const std::size_t kmax = std::round(0.75 * zsize_);
        const std::size_t jmin = std::round(0.25 * ysize_);
        const std::size_t jmax = std::round(0.75 * ysize_);
        const std::size_t imin = std::round(0.25 * xsize_);
        const std::size_t imax = std::round(0.75 * xsize_);

        for(std::size_t k = kmin; k <= kmax; ++k)
            for(std::size_t j = jmin; j <= jmax; ++j)
                for(std::size_t i = imin; i <= imax; ++i)
                    operator()(i, j, k) = 1;
    }

    std::size_t x_min()  const { return halosize_; }
    std::size_t x_max()  const { return xsize_ - halosize_; }
    std::size_t x_size() const { return xsize_; }
    std::size_t y_min()  const { return halosize_; }
    std::size_t y_max()  const { return ysize_ - halosize_; }
    std::size_t y_size() const { return ysize_; }
    std::size_t z_min()  const { return 0; }
    std::size_t z_max()  const { return zsize_; }

    private:
    std::size_t xsize_, ysize_, zsize_, halosize_;
    std::vector<T> data_;
};
