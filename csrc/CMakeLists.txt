add_library(scinthil INTERFACE)

find_package(CUDAToolkit REQUIRED)

get_filename_component(SCINTHIL_CSRC_REPOSITORY_ROOT
                       "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

target_include_directories(
  scinthil
  INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
            "$<BUILD_INTERFACE:${SCINTHIL_CSRC_REPOSITORY_ROOT}/3rdparty/cutlass/include>")

target_link_libraries(scinthil INTERFACE CUDA::cudart CUDA::cuda_driver)

add_subdirectory(utils)
add_subdirectory(cuda)
add_subdirectory(gemm)
add_subdirectory(microbenchmark)
