#pragma once


namespace {

static inline __host__ __device__ std::size_t index(std::size_t i, std::size_t j, std::size_t k,
                                                    std::size_t xsize, std::size_t ysize) {
    return i + j * xsize + k * xsize * ysize;
}

} // namespace
