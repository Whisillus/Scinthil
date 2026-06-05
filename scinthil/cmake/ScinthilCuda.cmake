include_guard(GLOBAL)

get_filename_component(SCINTHIL_REPOSITORY_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
get_filename_component(SCINTHIL_CSRC_DIR "${SCINTHIL_REPOSITORY_ROOT}/csrc" ABSOLUTE)

function(scinthil_add_cuda_executable target_name)
  add_executable(${target_name})
  target_sources(${target_name} PRIVATE ${ARGN})
  target_include_directories(
    ${target_name}
    PRIVATE ${SCINTHIL_CSRC_DIR} ${SCINTHIL_CSRC_DIR}/cuda/binary
            ${SCINTHIL_REPOSITORY_ROOT}/3rdparty/cutlass/include
            ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_options(
    ${target_name}
    PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-std=c++20>"
            "$<$<COMPILE_LANGUAGE:CUDA>:--std=c++20>")
endfunction()
