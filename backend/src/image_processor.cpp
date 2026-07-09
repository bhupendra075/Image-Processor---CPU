#include "image_processor.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include <stdexcept>

namespace image_processor {

// Helper: clamp value to [0, 255]
inline uint8_t clamp(int val) {
    return static_cast<uint8_t>(std::max(0, std::min(255, val)));
}

// --- Box Blur (Separable 2D Convolution) ---
// Algorithm: 1D horizontal pass -> 1D vertical pass
// Uses OpenMP parallel for with collapse(2) for maximum CPU utilization
// Zero-copy: directly accesses NumPy array memory via pybind11 buffer protocol
void box_blur(pybind11::array_t<uint8_t>& image, int kernel_size) {
    // Validate kernel size (must be odd and >= 3)
    if (kernel_size < 3 || kernel_size % 2 == 0) {
        kernel_size = 3; // fallback to default
    }

    // Get buffer info for zero-copy access
    pybind11::buffer_info buf = image.request();

    // Ensure 2D grayscale image
    if (buf.ndim != 2) {
        throw std::runtime_error("Expected 2D grayscale image");
    }

    const int height = static_cast<int>(buf.shape[0]);
    const int width = static_cast<int>(buf.shape[1]);
    const int stride = static_cast<int>(buf.strides[0] / sizeof(uint8_t));

    uint8_t* data = static_cast<uint8_t*>(buf.ptr);
    const int radius = kernel_size / 2;

    // Temporary buffer for horizontal pass (same size as image)
    std::vector<uint8_t> temp(height * width);

    // --- Horizontal Pass ---
    // Each thread processes a full row (or multiple rows)
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int sum = 0;
            int count = 0;

            // Convolve horizontally within kernel radius
            for (int k = -radius; k <= radius; ++k) {
                int xk = x + k;
                if (xk >= 0 && xk < width) {
                    sum += data[y * stride + xk];
                    ++count;
                }
            }
            // Average using actual count at borders to avoid darkening
            float avg = count > 0 ? static_cast<float>(sum) / count : 0.0f;
            temp[y * width + x] = clamp(static_cast<int>(avg + 0.5f));
        }
    }

    // --- Vertical Pass ---
    // Write results back to original array (in-place)
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int sum = 0;
            int count = 0;

            // Convolve vertically within kernel radius
            for (int k = -radius; k <= radius; ++k) {
                int yk = y + k;
                if (yk >= 0 && yk < height) {
                    sum += temp[yk * width + x];
                    ++count;
                }
            }
            float avg = count > 0 ? static_cast<float>(sum) / count : 0.0f;
            data[y * stride + x] = clamp(static_cast<int>(avg + 0.5f));
        }
    }
}

// --- Sobel Edge Detection ---
// Computes gradient magnitude: sqrt(Gx^2 + Gy^2)
// Gx = [-1 0 1; -2 0 2; -1 0 1] * image
// Gy = [-1 -2 -1; 0 0 0; 1 2 1] * image
// Zero-copy: directly modifies NumPy array memory
void sobel_edge(pybind11::array_t<uint8_t>& image) {
    pybind11::buffer_info buf = image.request();

    if (buf.ndim != 2) {
        throw std::runtime_error("Expected 2D grayscale image");
    }

    const int height = static_cast<int>(buf.shape[0]);
    const int width = static_cast<int>(buf.shape[1]);
    const int stride = static_cast<int>(buf.strides[0] / sizeof(uint8_t));

    uint8_t* data = static_cast<uint8_t*>(buf.ptr);

    // Temporary buffer for gradient magnitude
    std::vector<float> magnitude(height * width);

    // Sobel kernels (3x3)
    // Gx: horizontal gradient
    // Gy: vertical gradient
    const int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    // Compute gradients with OpenMP parallelization
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int gx = 0, gy = 0;

            // Apply 3x3 Sobel kernels
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    uint8_t pixel = data[(y + ky) * stride + (x + kx)];
                    gx += pixel * Gx[ky + 1][kx + 1];
                    gy += pixel * Gy[ky + 1][kx + 1];
                }
            }

            // Gradient magnitude
            magnitude[y * width + x] = std::sqrt(static_cast<float>(gx * gx + gy * gy));
        }
    }

    // Normalize and write back to original array (in-place)
    // Find max for normalization
    float max_mag = 0.0f;
    for (float m : magnitude) {
        if (m > max_mag) max_mag = m;
    }

    if (max_mag > 0.0f) {
        const float scale = 255.0f / max_mag;
        #pragma omp parallel for collapse(2) schedule(static)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                data[y * stride + x] = clamp(static_cast<int>(magnitude[y * width + x] * scale + 0.5f));
            }
        }
    }
}

} // namespace image_processor