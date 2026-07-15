find_package(GTest REQUIRED)

scinthil_add_cuda_executable(scinthil_test_binary test_binary.cu)

if(TARGET GTest::gtest_main)
  target_link_libraries(scinthil_test_binary PRIVATE GTest::gtest_main)
elseif(TARGET GTest::GTest AND TARGET GTest::Main)
  target_link_libraries(scinthil_test_binary PRIVATE GTest::GTest GTest::Main)
else()
  message(FATAL_ERROR "GTest main imported target not found")
endif()
