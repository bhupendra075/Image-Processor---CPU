#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "image_processor.hpp"

namespace py = pybind11;

// Module definition with zero-copy NumPy array support
// Using py::array_t<uint8_t> ensures:
// 1. Input is a C-contiguous uint8 NumPy array (or will be converted)
// 2. Direct memory access via buffer protocol - NO data copying
// 3. Modifications happen in-place on the original NumPy array
PYBIND11_MODULE(image_processor, m) {
    m.doc() = "High-performance image processing with OpenMP and zero-copy NumPy integration";

    // Box Blur: in-place modification of input array
    // Parameters: image (2D uint8 array), kernel_size (odd integer >= 3)
    m.def("box_blur", &image_processor::box_blur,
          py::arg("image"), py::arg("kernel_size") = 3,
          R"pbdoc(
            Apply Box Blur filter to a grayscale image (in-place).

            Uses separable 2D convolution with OpenMP parallelization.
            Modifies the input NumPy array directly via zero-copy buffer protocol.

            Args:
                image: 2D uint8 NumPy array (height x width), modified in-place
                kernel_size: Size of blur kernel (must be odd, >= 3), default=3

            Returns:
                None (modifies input array in-place)
          )pbdoc");

    // Sobel Edge Detection: in-place modification of input array
    // Parameters: image (2D uint8 array)
    m.def("sobel_edge", &image_processor::sobel_edge,
          py::arg("image"),
          R"pbdoc(
            Apply Sobel Edge Detection filter to a grayscale image (in-place).

            Computes gradient magnitude using 3x3 Sobel kernels with OpenMP parallelization.
            Modifies the input NumPy array directly via zero-copy buffer protocol.

            Args:
                image: 2D uint8 NumPy array (height x width), modified in-place

            Returns:
                None (modifies input array in-place)
          )pbdoc");
}