cmake_minimum_required(VERSION 3.14)
project(dist_training_sim VERSION 0.1.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Find packages
find_package(pybind11 REQUIRED)

# Main library
add_library(dist_sim_core
  src/training_simulator.cpp
)

target_include_directories(dist_sim_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
)

# Python bindings
add_subdirectory(src/bindings)

# Tests
option(ENABLE_TESTING "Enable testing" ON)
if(ENABLE_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

# Installation
install(TARGETS dist_sim_core
  EXPORT dist_sim_coreTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(DIRECTORY src/
  DESTINATION include/dist_sim
  FILES_MATCHING PATTERN "*.hpp"
)