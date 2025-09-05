#include <pybind11/pybind11.h>

PYBIND11_MODULE(dist_sim_core, m) {
    m.doc() = "Minimal test module";
}