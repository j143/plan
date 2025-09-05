#include "training_simulator.hpp"
#include "communication_model.hpp"
#include "compute_model.hpp"
#include "memory_model.hpp"
#include <stdexcept>

namespace dist_sim {

TrainingSimulator::TrainingSimulator() 
    : comm_model_(nullptr), compute_model_(nullptr), memory_model_(nullptr) {
}

TrainingSimulator::~TrainingSimulator() = default;

void TrainingSimulator::add_node(const GpuNode& node) {
    nodes_.push_back(node);
}

void TrainingSimulator::add_network_link(const NetworkLink& link) {
    links_.push_back(link);
}

void TrainingSimulator::set_model_config(const ModelConfig& config) {
    model_config_ = config;
}

void TrainingSimulator::set_parallelism_config(const ParallelismConfig& config) {
    parallelism_config_ = config;
}

TrainingSimulator::SimulationResult TrainingSimulator::run_simulation(int num_iterations) {
    // Lazy initialization of models
    if (!comm_model_) {
        comm_model_ = std::make_unique<CommunicationModel>(nodes_, links_);
    }
    
    if (!compute_model_) {
        compute_model_ = std::make_unique<ComputeModel>(nodes_);
    }
    
    if (!memory_model_) {
        memory_model_ = std::make_unique<MemoryModel>();
    }
    
    // Validate configuration
    if (nodes_.empty()) {
        throw std::runtime_error("No GPU nodes configured");
    }
    
    if (model_config_.model_size_gb == 0) {
        throw std::runtime_error("Model size not configured");
    }
    
    // Run simulation
    SimulationResult result;
    
    // Calculate compute time
    double compute_time = compute_model_->predict_computation_time(
        model_config_, parallelism_config_, num_iterations);
    
    // Calculate communication time
    double comm_time = comm_model_->predict_communication_time(
        model_config_, parallelism_config_, num_iterations);
    
    // Calculate memory usage
    double memory_usage = memory_model_->predict_memory_usage(
        model_config_, parallelism_config_);
    
    // Check for memory bottleneck
    bool memory_bottlenecked = false;
    for (const auto& node : nodes_) {
        if (memory_usage > node.memory_gb) {
            memory_bottlenecked = true;
            break;
        }
    }
    
    // Populate result
    result.time_per_iteration_ms = (compute_time + comm_time) / num_iterations;
    result.compute_time_percent = compute_time / (compute_time + comm_time) * 100.0;
    result.communication_time_percent = comm_time / (compute_time + comm_time) * 100.0;
    result.memory_usage_gb = memory_usage;
    result.memory_bottlenecked = memory_bottlenecked;
    
    // Determine bottleneck
    if (memory_bottlenecked) {
        result.bottleneck_description = "Memory usage exceeds available GPU memory";
    } else if (result.compute_time_percent > 70.0) {
        result.bottleneck_description = "Computation bound";
    } else if (result.communication_time_percent > 70.0) {
        result.bottleneck_description = "Communication bound";
    } else {
        result.bottleneck_description = "Balanced workload";
    }
    
    return result;
}

} // namespace dist_sim