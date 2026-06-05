add_library(scinthil INTERFACE)

get_filename_component(SCINTHIL_CSRC_REPOSITORY_ROOT
                       "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

target_include_directories(
  scinthil
  INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
            "$<BUILD_INTERFACE:${SCINTHIL_CSRC_REPOSITORY_ROOT}/3rdparty/cutlass/include>")

add_subdirectory(util)
add_subdirectory(cuda/binary)
