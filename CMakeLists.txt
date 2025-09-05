cmake_minimum_required(VERSION 3.14)
project(dist_training_sim VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  pybind11
  URL https://github.com/pybind/pybind11/archive/v2.10.4.tar.gz
)
FetchContent_MakeAvailable(pybind11)

add_library(dist_sim_lib
  src/training_simulator.cpp
  src/communication_model.cpp
  src/compute_model.cpp
  src/memory_model.cpp
)
target_include_directories(dist_sim_lib PUBLIC src)

add_subdirectory(src/bindings)

option(ENABLE_TESTING "Enable testing" ON)
if(ENABLE_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()