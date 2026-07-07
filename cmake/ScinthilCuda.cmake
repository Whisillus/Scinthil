include_guard(GLOBAL)

function(scinthil_add_cuda_executable target_name)
  if(NOT TARGET scinthil)
    message(
      FATAL_ERROR
        "scinthil_add_cuda_executable requires the scinthil target. "
        "Call add_subdirectory(csrc) before adding CUDA executables.")
  endif()

  add_executable(${target_name})
  target_sources(${target_name} PRIVATE ${ARGN})
  target_link_libraries(${target_name} PRIVATE scinthil)
  target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  set_target_properties(
    ${target_name}
    PROPERTIES CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
               CUDA_EXTENSIONS OFF)
endfunction()

function(scinthil_add_cuda_dump_artifacts target_name ptx_enabled sass_enabled artifact_label)
  if(NOT ${ptx_enabled} AND NOT ${sass_enabled})
    return()
  endif()

  find_program(
    SCINTHIL_CUOBJDUMP_EXECUTABLE
    NAMES cuobjdump
    HINTS "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/bin" "$ENV{CUDA_PATH}/bin")

  if(NOT SCINTHIL_CUOBJDUMP_EXECUTABLE)
    message(
      FATAL_ERROR
        "CUDA dump artifacts are enabled for ${artifact_label} '${target_name}', "
        "but cuobjdump was not found. Set CMAKE_CUDA_COMPILER_TOOLKIT_ROOT, "
        "CUDA_PATH, or disable ${ptx_enabled} and ${sass_enabled}.")
  endif()

  get_target_property(cuda_architectures ${target_name} CUDA_ARCHITECTURES)
  if(NOT cuda_architectures)
    set(cuda_architectures ${CMAKE_CUDA_ARCHITECTURES})
  endif()

  set(artifact_dir "${CMAKE_CURRENT_BINARY_DIR}/${target_name}.cuda")
  set(capture_script "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/ScinthilCaptureCudaDump.cmake")

  set(cuda_artifact_outputs)

  foreach(cuda_architecture IN LISTS cuda_architectures)
    if(NOT cuda_architecture MATCHES "^([0-9]+[a-z]?)(-(real|virtual))?$")
      message(
        FATAL_ERROR
          "CUDA dump artifacts for ${artifact_label} '${target_name}' require numeric CUDA_ARCHITECTURES, "
          "but found '${cuda_architecture}'.")
    endif()

    set(cuda_architecture_name "sm_${CMAKE_MATCH_1}")
    set(cuda_architecture_label "sm${CMAKE_MATCH_1}")
    set(cuda_architecture_kind "${CMAKE_MATCH_3}")
    set(ptx_output "${artifact_dir}/${target_name}.${cuda_architecture_label}.ptx")
    set(sass_output "${artifact_dir}/${target_name}.${cuda_architecture_label}.sass")

    if(${ptx_enabled} AND NOT cuda_architecture_kind STREQUAL "real")
      add_custom_command(
        OUTPUT "${ptx_output}"
        COMMAND
          "${CMAKE_COMMAND}" "-DSCINTHIL_CUDA_DUMP_TOOL=${SCINTHIL_CUOBJDUMP_EXECUTABLE}"
          -DSCINTHIL_CUDA_DUMP_MODE=--dump-ptx
          "-DSCINTHIL_CUDA_DUMP_ARCHITECTURE=${cuda_architecture_name}"
          "-DSCINTHIL_CUDA_DUMP_INPUT=$<TARGET_FILE:${target_name}>"
          "-DSCINTHIL_CUDA_DUMP_OUTPUT=${ptx_output}" -P "${capture_script}"
        DEPENDS ${target_name}
        VERBATIM
        COMMENT "Writing CUDA PTX dump for ${target_name} ${cuda_architecture_name}")
      list(APPEND cuda_artifact_outputs "${ptx_output}")
    endif()

    if(${sass_enabled} AND NOT cuda_architecture_kind STREQUAL "virtual")
      add_custom_command(
        OUTPUT "${sass_output}"
        COMMAND
          "${CMAKE_COMMAND}" "-DSCINTHIL_CUDA_DUMP_TOOL=${SCINTHIL_CUOBJDUMP_EXECUTABLE}"
          -DSCINTHIL_CUDA_DUMP_MODE=--dump-sass
          "-DSCINTHIL_CUDA_DUMP_ARCHITECTURE=${cuda_architecture_name}"
          "-DSCINTHIL_CUDA_DUMP_INPUT=$<TARGET_FILE:${target_name}>"
          "-DSCINTHIL_CUDA_DUMP_OUTPUT=${sass_output}" -P "${capture_script}"
        DEPENDS ${target_name}
        VERBATIM
        COMMENT "Writing CUDA SASS dump for ${target_name} ${cuda_architecture_name}")
      list(APPEND cuda_artifact_outputs "${sass_output}")
    endif()
  endforeach()

  add_custom_target(
    ${target_name}_cuda_artifacts ALL
    DEPENDS ${cuda_artifact_outputs})
endfunction()

function(scinthil_add_cuda_profile target_name)
  scinthil_add_cuda_executable(${target_name} ${ARGN})
  scinthil_add_cuda_dump_artifacts(${target_name} SCINTHIL_BUILD_PROFILE_PTX SCINTHIL_BUILD_PROFILE_SASS profile)
endfunction()
