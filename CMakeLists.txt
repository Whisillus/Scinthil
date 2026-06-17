cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90 100)
endif()

project(scinthil LANGUAGES CXX CUDA)

option(SCINTHIL_BUILD_EXAMPLES "Build Scinthil examples" ON)
option(SCINTHIL_BUILD_EXAMPLE_PTX "Build PTX dump artifacts for Scinthil examples" ON)
option(SCINTHIL_BUILD_EXAMPLE_SASS "Build SASS dump artifacts for Scinthil examples" ON)
option(SCINTHIL_BUILD_TESTS "Build Scinthil tests" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ScinthilCuda)

add_subdirectory(csrc)

if(SCINTHIL_BUILD_EXAMPLES)
  add_subdirectory(example)
endif()

if(SCINTHIL_BUILD_TESTS)
  add_subdirectory(test)
endif()
