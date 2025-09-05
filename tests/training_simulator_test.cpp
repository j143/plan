#include <gtest/gtest.h>
#include "training_simulator.hpp"

namespace dist_sim {
namespace testing {

class TrainingSimulatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up a standard test configuration
        simulator_ = std::make_unique<TrainingSimulator>();
        
        // Add 4 GPUs
        for (int i = 0; i < 4; ++i) {
            GpuNode node;
            node.type = "H100";
            node.compute_tflops = 312.0;
            node.memory_gb = 80.0;
            node.device_id = i;
            node.node_id = i / 2;  // 2 GPUs per node
            simulator_->add_node(node);
        }
        
        // Add network links
        for (int i = 0; i < 4; ++i) {
            for (int j = i + 1; j < 4; ++j) {
                NetworkLink link;
                link.source_id = i;
                link.target_id = j;
                link.bandwidth_gbps = (i/2 == j/2) ? 300.0 : 50.0;  // Higher bandwidth within node
                link.latency_us = (i/2 == j/2) ? 1.0 : 10.0;       // Lower latency within node
                simulator_->add_network_link(link);
                
                // Bidirectional
                link.source_id = j;
                link.target_id = i;
                simulator_->add_network_link(link);
            }
        }
        
        // Set model config
        ModelConfig model_config;
        model_config.model_size_gb = 20;
        model_config.batch_size = 32;
        model_config.seq_length = 2048;
        simulator_->set_model_config(model_config);
        
        // Set parallelism config
        ParallelismConfig parallelism_config;
        parallelism_config.tensor_parallel = 2;
        parallelism_config.pipeline_parallel = 2;
        parallelism_config.data_parallel = 1;
        simulator_->set_parallelism_config(parallelism_config);
    }
    
    std::unique_ptr<TrainingSimulator> simulator_;
};

TEST_F(TrainingSimulatorTest, BasicSimulationRuns) {
    // Test that simulation runs without crashing
    TrainingSimulator::SimulationResult result = simulator_->run_simulation(100);
    
    // Basic validation
    EXPECT_GT(result.time_per_iteration_ms, 0.0);
    EXPECT_GE(result.compute_time_percent, 0.0);
    EXPECT_LE(result.compute_time_percent, 100.0);
    EXPECT_GE(result.communication_time_percent, 0.0);
    EXPECT_LE(result.communication_time_percent, 100.0);
    EXPECT_GT(result.memory_usage_gb, 0.0);
    EXPECT_FALSE(result.memory_bottlenecked);  // Should not be memory bottlenecked
    EXPECT_FALSE(result.bottleneck_description.empty());
}

TEST_F(TrainingSimulatorTest, ScalesWithIterations) {
    // Simulation time should scale linearly with iterations
    TrainingSimulator::SimulationResult result1 = simulator_->run_simulation(100);
    TrainingSimulator::SimulationResult result2 = simulator_->run_simulation(200);
    
    // Iteration time should be the same (within floating point error)
    EXPECT_NEAR(result1.time_per_iteration_ms, result2.time_per_iteration_ms, 0.01);
}

TEST_F(TrainingSimulatorTest, DetectsMemoryBottleneck) {
    // Set very large model that exceeds GPU memory
    ModelConfig model_config;
    model_config.model_size_gb = 100;  // Larger than 80GB GPU memory
    model_config.batch_size = 32;
    model_config.seq_length = 2048;
    simulator_->set_model_config(model_config);
    
    // Run simulation
    TrainingSimulator::SimulationResult result = simulator_->run_simulation(100);
    
    // Should detect memory bottleneck
    EXPECT_TRUE(result.memory_bottlenecked);
    EXPECT_TRUE(result.bottleneck_description.find("Memory") != std::string::npos);
}

} // namespace testing
} // namespace dist_sim