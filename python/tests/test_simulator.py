import pytest
import numpy as np
from dist_training_sim import DistributedTrainingSimulator

class TestDistributedTrainingSimulator:
    """Test suite for the Python wrapper of the simulator."""
    
    @pytest.fixture
    def simulator(self):
        """Create a simulator with a standard test configuration."""
        sim = DistributedTrainingSimulator()
        # Add a 2-node, 4-GPU cluster
        sim.add_cluster(num_nodes=2, gpus_per_node=2, gpu_type="H100")
        return sim
    
    def test_add_cluster(self, simulator):
        """Test that cluster creation works."""
        # Reset and create a new cluster
        simulator = DistributedTrainingSimulator()
        simulator.add_cluster(num_nodes=3, gpus_per_node=4, gpu_type="A100", 
                             compute_tflops=156.0, memory_gb=40.0)
        
        # Run a simple simulation to verify it works
        result = simulator.simulate_training(
            model_size_gb=20,
            batch_size=32,
            seq_length=1024,
            tensor_parallel=4,
            pipeline_parallel=3,
            data_parallel=1
        )
        
        # Basic validation
        assert result.time_per_iteration_ms > 0
        assert 0 <= result.compute_time_percent <= 100
        assert 0 <= result.communication_time_percent <= 100
        assert result.memory_usage_gb > 0
    
    def test_memory_bottleneck(self, simulator):
        """Test that memory bottlenecks are correctly identified."""
        # Set model size larger than available memory
        result = simulator.simulate_training(
            model_size_gb=100,  # Larger than 80GB per GPU
            batch_size=32,
            seq_length=2048,
            tensor_parallel=2,
            pipeline_parallel=2,
            data_parallel=1
        )
        
        # Should detect memory bottleneck
        assert result.memory_bottlenecked
        assert "Memory" in result.bottleneck_description
    
    def test_compare_configurations(self, simulator):
        """Test configuration comparison functionality."""
        configs = [
            {"model_size_gb": 20, "batch_size": 32, "seq_length": 2048, "tp": 2, "pp": 2, "dp": 1},
            {"model_size_gb": 20, "batch_size": 32, "seq_length": 2048, "tp": 4, "pp": 1, "dp": 1},
            {"model_size_gb": 20, "batch_size": 32, "seq_length": 2048, "tp": 1, "pp": 4, "dp": 1}
        ]
        
        results = simulator.compare_configurations(configs)
        
        # Should return results in order of performance (fastest first)
        assert len(results) == 3
        
        # Check that results are sorted by iteration time
        for i in range(1, len(results)):
            assert results[i-1]["result"].time_per_iteration_ms <= results[i]["result"].time_per_iteration_ms