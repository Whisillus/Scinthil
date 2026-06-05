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

execute_process(
  COMMAND "${SCINTHIL_CUDA_DUMP_TOOL}" "${SCINTHIL_CUDA_DUMP_MODE}" "${SCINTHIL_CUDA_DUMP_INPUT}"
  OUTPUT_FILE "${SCINTHIL_CUDA_DUMP_OUTPUT}"
  ERROR_VARIABLE dump_error
  RESULT_VARIABLE dump_result)

if(NOT dump_result EQUAL 0)
  message(
    FATAL_ERROR
      "Failed to write CUDA dump '${SCINTHIL_CUDA_DUMP_OUTPUT}' from "
      "'${SCINTHIL_CUDA_DUMP_INPUT}': ${dump_error}")
endif()
