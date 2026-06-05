include_guard(GLOBAL)

get_filename_component(SCINTHIL_REPOSITORY_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
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

function(scinthil_add_cuda_example target_name)
  scinthil_add_cuda_executable(${target_name} ${ARGN})

  if(NOT SCINTHIL_BUILD_EXAMPLE_PTX AND NOT SCINTHIL_BUILD_EXAMPLE_SASS)
    return()
  endif()

  find_program(
    SCINTHIL_CUOBJDUMP_EXECUTABLE
    NAMES cuobjdump
    HINTS "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/bin" "$ENV{CUDA_PATH}/bin")

  if(NOT SCINTHIL_CUOBJDUMP_EXECUTABLE)
    message(
      FATAL_ERROR
        "CUDA dump artifacts are enabled for example '${target_name}', "
        "but cuobjdump was not found. Set CMAKE_CUDA_COMPILER_TOOLKIT_ROOT, "
        "CUDA_PATH, or disable SCINTHIL_BUILD_EXAMPLE_PTX and "
        "SCINTHIL_BUILD_EXAMPLE_SASS.")
  endif()

  set(artifact_dir "${CMAKE_CURRENT_BINARY_DIR}/${target_name}.cuda")
  set(ptx_output "${artifact_dir}/${target_name}.ptx")
  set(sass_output "${artifact_dir}/${target_name}.sass")
  set(capture_script "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/ScinthilCaptureCudaDump.cmake")

  set(cuda_artifact_outputs)

  if(SCINTHIL_BUILD_EXAMPLE_PTX)
    add_custom_command(
      OUTPUT "${ptx_output}"
      COMMAND
        "${CMAKE_COMMAND}" "-DSCINTHIL_CUDA_DUMP_TOOL=${SCINTHIL_CUOBJDUMP_EXECUTABLE}"
        -DSCINTHIL_CUDA_DUMP_MODE=--dump-ptx
        "-DSCINTHIL_CUDA_DUMP_INPUT=$<TARGET_FILE:${target_name}>"
        "-DSCINTHIL_CUDA_DUMP_OUTPUT=${ptx_output}" -P "${capture_script}"
      DEPENDS ${target_name}
      VERBATIM
      COMMENT "Writing CUDA PTX dump for ${target_name}")
    list(APPEND cuda_artifact_outputs "${ptx_output}")
  endif()

  if(SCINTHIL_BUILD_EXAMPLE_SASS)
    add_custom_command(
      OUTPUT "${sass_output}"
      COMMAND
        "${CMAKE_COMMAND}" "-DSCINTHIL_CUDA_DUMP_TOOL=${SCINTHIL_CUOBJDUMP_EXECUTABLE}"
        -DSCINTHIL_CUDA_DUMP_MODE=--dump-sass
        "-DSCINTHIL_CUDA_DUMP_INPUT=$<TARGET_FILE:${target_name}>"
        "-DSCINTHIL_CUDA_DUMP_OUTPUT=${sass_output}" -P "${capture_script}"
      DEPENDS ${target_name}
      VERBATIM
      COMMENT "Writing CUDA SASS dump for ${target_name}")
    list(APPEND cuda_artifact_outputs "${sass_output}")
  endif()

  add_custom_target(
    ${target_name}_cuda_artifacts ALL
    DEPENDS ${cuda_artifact_outputs})
endfunction()
