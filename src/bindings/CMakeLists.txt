pybind11_add_module(dist_sim_core bindings.cpp)

target_link_libraries(dist_sim_core PRIVATE dist_sim_core)

# Set output directory
set_target_properties(dist_sim_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/dist_training_sim"
)