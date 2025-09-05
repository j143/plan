# Download and build GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Add test executable
add_executable(dist_sim_tests
  training_simulator_test.cpp
  communication_model_test.cpp
  compute_model_test.cpp
  memory_model_test.cpp
)

target_link_libraries(dist_sim_tests
  PRIVATE
  dist_sim_core
  gtest_main
)

# Register tests
include(GoogleTest)
gtest_discover_tests(dist_sim_tests)