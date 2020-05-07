# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/tmp)
add_executable(smoke_test smoke_test.cpp)
target_link_libraries(smoke_test PUBLIC armadillo)
enable_testing()

add_test(NAME smoke_test COMMAND smoke_test)

# add_test(NAME smoke_test COMMAND smoke_test WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/tmp)

# add_custom_target(run_smoke_test smoke_test COMMAND $<TARGET_FILE:smoke_test> DEPENDS smoke_test)
# add_custom_target(run_smoke_test smoke_test COMMAND $<TARGET_FILE:smoke_test>)
