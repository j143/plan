# Download and build GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Add test executable
add_executable(dist_sim_tests
  training_simulator_test.cpp
)

# Link against core C++ library and gtest
target_link_libraries(dist_sim_tests
  PRIVATE
  dist_sim_lib   # <-- Not dist_sim_core
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(dist_sim_tests)