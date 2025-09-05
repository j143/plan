#pragma once
#include <memory>
#include <string>
#include <vector>

namespace dist_sim {

// Forward declarations
class CommunicationModel;
class ComputeModel;
class MemoryModel;

// Key data structures
struct GpuNode {
    std::string type;         // e.g., "H100", "A100"
    double compute_tflops;    // Peak TFLOPS
    double memory_gb;         // GPU memory in GB
    int device_id;            // Device ID within node
    int node_id;              // Physical node ID
};

struct NetworkLink {
    int source_id;
    int target_id;
    double bandwidth_gbps;
    double latency_us;
};

struct ModelConfig {
    size_t model_size_gb;     // Model size in GB
    int batch_size;           // Global batch size
    int seq_length;           // Sequence length
};

struct ParallelismConfig {
    int tensor_parallel;      // Tensor parallelism degree
    int pipeline_parallel;    // Pipeline parallelism degree
    int data_parallel;        // Data parallelism degree
};

// Main simulation engine
class TrainingSimulator {
public:
    TrainingSimulator();
    ~TrainingSimulator();
    
    // Configuration methods
    void add_node(const GpuNode& node);
    void add_network_link(const NetworkLink& link);
    void set_model_config(const ModelConfig& config);
    void set_parallelism_config(const ParallelismConfig& config);
    
    // Simulation results
    struct SimulationResult {
        double time_per_iteration_ms;
        double compute_time_percent;
        double communication_time_percent;
        double memory_usage_gb;
        bool memory_bottlenecked;
        std::string bottleneck_description;
    };
    
    // Run simulation
    SimulationResult run_simulation(int num_iterations);
    
private:
    std::vector<GpuNode> nodes_;
    std::vector<NetworkLink> links_;
    ModelConfig model_config_;
    ParallelismConfig parallelism_config_;
    
    // Component models
    std::unique_ptr<CommunicationModel> comm_model_;
    std::unique_ptr<ComputeModel> compute_model_;
    std::unique_ptr<MemoryModel> memory_model_;
};

} // namespace dist_sim
