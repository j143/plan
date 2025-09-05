#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "training_simulator.hpp"

namespace py = pybind11;

PYBIND11_MODULE(dist_sim_core, m) {
    m.doc() = "Distributed training simulator core";
    
    // Bind GpuNode struct
    py::class_<dist_sim::GpuNode>(m, "GpuNode")
        .def(py::init<>())
        .def_readwrite("type", &dist_sim::GpuNode::type)
        .def_readwrite("compute_tflops", &dist_sim::GpuNode::compute_tflops)
        .def_readwrite("memory_gb", &dist_sim::GpuNode::memory_gb)
        .def_readwrite("device_id", &dist_sim::GpuNode::device_id)
        .def_readwrite("node_id", &dist_sim::GpuNode::node_id);
    
    // Bind NetworkLink struct
    py::class_<dist_sim::NetworkLink>(m, "NetworkLink")
        .def(py::init<>())
        .def_readwrite("source_id", &dist_sim::NetworkLink::source_id)
        .def_readwrite("target_id", &dist_sim::NetworkLink::target_id)
        .def_readwrite("bandwidth_gbps", &dist_sim::NetworkLink::bandwidth_gbps)
        .def_readwrite("latency_us", &dist_sim::NetworkLink::latency_us);
    
    // Bind ModelConfig struct
    py::class_<dist_sim::ModelConfig>(m, "ModelConfig")
        .def(py::init<>())
        .def_readwrite("model_size_gb", &dist_sim::ModelConfig::model_size_gb)
        .def_readwrite("batch_size", &dist_sim::ModelConfig::batch_size)
        .def_readwrite("seq_length", &dist_sim::ModelConfig::seq_length);
    
    // Bind ParallelismConfig struct
    py::class_<dist_sim::ParallelismConfig>(m, "ParallelismConfig")
        .def(py::init<>())
        .def_readwrite("tensor_parallel", &dist_sim::ParallelismConfig::tensor_parallel)
        .def_readwrite("pipeline_parallel", &dist_sim::ParallelismConfig::pipeline_parallel)
        .def_readwrite("data_parallel", &dist_sim::ParallelismConfig::data_parallel);
    
    // Bind SimulationResult struct
    py::class_<dist_sim::TrainingSimulator::SimulationResult>(m, "SimulationResult")
        .def(py::init<>())
        .def_readwrite("time_per_iteration_ms", &dist_sim::TrainingSimulator::SimulationResult::time_per_iteration_ms)
        .def_readwrite("compute_time_percent", &dist_sim::TrainingSimulator::SimulationResult::compute_time_percent)
        .def_readwrite("communication_time_percent", &dist_sim::TrainingSimulator::SimulationResult::communication_time_percent)
        .def_readwrite("memory_usage_gb", &dist_sim::TrainingSimulator::SimulationResult::memory_usage_gb)
        .def_readwrite("memory_bottlenecked", &dist_sim::TrainingSimulator::SimulationResult::memory_bottlenecked)
        .def_readwrite("bottleneck_description", &dist_sim::TrainingSimulator::SimulationResult::bottleneck_description);
    
    // Bind TrainingSimulator class
    py::class_<dist_sim::TrainingSimulator>(m, "TrainingSimulator")
        .def(py::init<>())
        .def("add_node", &dist_sim::TrainingSimulator::add_node)
        .def("add_network_link", &dist_sim::TrainingSimulator::add_network_link)
        .def("set_model_config", &dist_sim::TrainingSimulator::set_model_config)
        .def("set_parallelism_config", &dist_sim::TrainingSimulator::set_parallelism_config)
        .def("run_simulation", &dist_sim::TrainingSimulator::run_simulation);
}