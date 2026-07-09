cmake_minimum_required(VERSION 3.15)
project(image_processor LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type (default to Release if not specified)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Platform-specific settings
if(WIN32)
    # Windows: output .pyd extension for Python module
    set(CMAKE_SHARED_MODULE_PREFIX "")
    set(CMAKE_SHARED_MODULE_SUFFIX ".pyd")
else()
    # Linux/macOS: standard .so extension
    set(CMAKE_SHARED_MODULE_PREFIX "")
    set(CMAKE_SHARED_MODULE_SUFFIX ".so")
endif()

# Find pybind11 (prefer system package, fallback to fetchcontent)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found via find_package, using FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)

# Python development headers (optional - pybind11 handles this)
find_package(Python3 COMPONENTS Development REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${Python3_INCLUDE_DIRS})

# Source files
set(SOURCES
    src/image_processor.cpp
    src/pybind11_bindings.cpp
)

# Create the Python extension module
pybind11_add_module(image_processor ${SOURCES})

# Link OpenMP
target_link_libraries(image_processor PRIVATE OpenMP::OpenMP_CXX)

# Link Python libraries
target_link_libraries(image_processor PRIVATE ${Python3_LIBRARIES})

# Set output directory for the built module
set_target_properties(image_processor PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Installation (optional)
install(TARGETS image_processor
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)