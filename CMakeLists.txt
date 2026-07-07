cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90 100)
endif()

project(scinthil LANGUAGES CXX CUDA)

option(SCINTHIL_BUILD_PROFILES "Build Scinthil profiles" ON)
option(SCINTHIL_BUILD_PROFILE_PTX "Build PTX dump artifacts for Scinthil profiles" ON)
option(SCINTHIL_BUILD_PROFILE_SASS "Build SASS dump artifacts for Scinthil profiles" ON)
option(SCINTHIL_BUILD_TESTS "Build Scinthil tests" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ScinthilCuda)

add_subdirectory(csrc)

if(SCINTHIL_BUILD_PROFILES)
  add_subdirectory(profile)
endif()

if(SCINTHIL_BUILD_TESTS)
  add_subdirectory(test)
endif()
