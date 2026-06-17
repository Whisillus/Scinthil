if(NOT DEFINED SCINTHIL_CUDA_DUMP_TOOL)
  message(FATAL_ERROR "SCINTHIL_CUDA_DUMP_TOOL is required")
endif()

if(NOT DEFINED SCINTHIL_CUDA_DUMP_MODE)
  message(FATAL_ERROR "SCINTHIL_CUDA_DUMP_MODE is required")
endif()

if(NOT DEFINED SCINTHIL_CUDA_DUMP_INPUT)
  message(FATAL_ERROR "SCINTHIL_CUDA_DUMP_INPUT is required")
endif()

if(NOT DEFINED SCINTHIL_CUDA_DUMP_OUTPUT)
  message(FATAL_ERROR "SCINTHIL_CUDA_DUMP_OUTPUT is required")
endif()

get_filename_component(output_dir "${SCINTHIL_CUDA_DUMP_OUTPUT}" DIRECTORY)
file(MAKE_DIRECTORY "${output_dir}")

set(dump_command "${SCINTHIL_CUDA_DUMP_TOOL}" "${SCINTHIL_CUDA_DUMP_MODE}")
if(DEFINED SCINTHIL_CUDA_DUMP_ARCHITECTURE AND NOT SCINTHIL_CUDA_DUMP_ARCHITECTURE STREQUAL "")
  list(APPEND dump_command --gpu-architecture "${SCINTHIL_CUDA_DUMP_ARCHITECTURE}")
endif()
list(APPEND dump_command "${SCINTHIL_CUDA_DUMP_INPUT}")

execute_process(
  COMMAND ${dump_command}
  OUTPUT_FILE "${SCINTHIL_CUDA_DUMP_OUTPUT}"
  ERROR_VARIABLE dump_error
  RESULT_VARIABLE dump_result)

if(NOT dump_result EQUAL 0)
  message(
    FATAL_ERROR
      "Failed to write CUDA dump '${SCINTHIL_CUDA_DUMP_OUTPUT}' from "
      "'${SCINTHIL_CUDA_DUMP_INPUT}': ${dump_error}")
endif()
