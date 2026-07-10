cmake_minimum_required(VERSION 3.18)

project(clpipe_native LANGUAGES CXX)

# ---------------------------------------------------------------------------
# CUDA is optional and auto-detected.
#
# On a Jetson (or any box with the toolkit installed) this finds nvcc and builds
# the GPU backend. On a laptop it finds nothing, prints one line, and builds the
# CPU-only extension. Nothing downstream has to know which happened: the module
# reports its backend at runtime and the Python layer adapts.
#
# Override explicitly with -DCLPIPE_WITH_CUDA=ON/OFF.
# ---------------------------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)

if(DEFINED CLPIPE_WITH_CUDA)
    set(_clpipe_cuda_default ${CLPIPE_WITH_CUDA})
elseif(CMAKE_CUDA_COMPILER)
    set(_clpipe_cuda_default ON)
else()
    set(_clpipe_cuda_default OFF)
endif()

option(CLPIPE_WITH_CUDA "Build the CUDA background-subtraction backend" ${_clpipe_cuda_default})

if(CLPIPE_WITH_CUDA AND NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CLPIPE_WITH_CUDA=ON but no CUDA compiler was found.")
endif()

if(CLPIPE_WITH_CUDA)
    enable_language(CUDA)
    message(STATUS "clpipe: CUDA backend enabled (${CMAKE_CUDA_COMPILER})")
else()
    message(STATUS "clpipe: CUDA not found, building the CPU-only extension")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    # A debug build would make the CPU baseline look far worse than it is and
    # turn every benchmark number into fiction.
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(CLPIPE_SOURCES
    cpp/src/background.cpp
    cpp/src/background_cpu.cpp
    cpp/src/bindings.cpp
)

if(CLPIPE_WITH_CUDA)
    list(APPEND CLPIPE_SOURCES cpp/src/background_cuda.cu)
endif()

pybind11_add_module(_native ${CLPIPE_SOURCES})

target_include_directories(_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)

if(CLPIPE_WITH_CUDA)
    target_compile_definitions(_native PRIVATE CLPIPE_WITH_CUDA)
    set_target_properties(_native PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        # 53 = TX1/Nano, 62 = TX2, 72 = Xavier, 87 = Orin. Trim to the one board
        # you actually target to cut compile time.
        CUDA_ARCHITECTURES "53;62;72;87"
    )
endif()

if(MSVC)
    target_compile_options(_native PRIVATE /O2)
else()
    target_compile_options(_native PRIVATE -O3 -Wall -Wextra)
endif()

# Drop the built module next to the Python package so `import clpipe` picks it
# up with no install step and no PYTHONPATH fiddling.
set_target_properties(_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/clpipe
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/clpipe
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/clpipe
)
