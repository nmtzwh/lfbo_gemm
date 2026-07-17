cmake_minimum_required(VERSION 3.18)
project(MatOptOpenBLASBenchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT MATOPT_PACKAGE OR NOT MATOPT_KERNEL_ID)
  message(FATAL_ERROR "MATOPT_PACKAGE and MATOPT_KERNEL_ID are required")
endif()

find_package(benchmark 1.9.5 CONFIG QUIET)
if(NOT benchmark_FOUND)
  if(NOT MATOPT_FETCH_GOOGLE_BENCHMARK)
    message(FATAL_ERROR "Google Benchmark not found; explicitly enable MATOPT_FETCH_GOOGLE_BENCHMARK")
  endif()
  include(FetchContent)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.5
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(googlebenchmark)
endif()

find_path(OPENBLAS_INCLUDE_DIR NAMES cblas.h PATH_SUFFIXES
  openblas openblas-pthread openblas-openmp
  openblas64 openblas64-pthread openblas64-openmp)
find_library(OPENBLAS_LIBRARY NAMES openblas openblas64)
if(NOT OPENBLAS_INCLUDE_DIR OR NOT OPENBLAS_LIBRARY)
  message(FATAL_ERROR "A system OpenBLAS CBLAS implementation is required")
endif()

find_package(MatOptKernel CONFIG REQUIRED
  PATHS "${MATOPT_PACKAGE}/lib/cmake/MatOptKernel" NO_DEFAULT_PATH)
configure_file(src/kernel_bridge.hpp.in generated/kernel_bridge.hpp @ONLY)

add_executable(matopt-openblas-benchmark src/main.cpp)
target_include_directories(matopt-openblas-benchmark PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/generated" "${OPENBLAS_INCLUDE_DIR}")
target_link_libraries(matopt-openblas-benchmark PRIVATE
  benchmark::benchmark MatOptKernel::${MATOPT_KERNEL_ID} ${OPENBLAS_LIBRARY})
set_target_properties(matopt-openblas-benchmark PROPERTIES
  BUILD_RPATH "${MATOPT_PACKAGE}/lib")
