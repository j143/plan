import numpy as np
from .dist_sim_core import (
    TrainingSimulator,
    GpuNode,
    NetworkLink,
    ModelConfig,
    ParallelismConfig,
    SimulationResult,
)

class DistributedTrainingSimulator:
    """Python wrapper for the C++ simulation engine with additional utilities."""
    
    def __init__(self):
        """Initialize the simulator."""
        self.simulator = TrainingSimulator()
        self.results = []
        
    def add_gpu_node(self, gpu_type, compute_tflops, memory_gb, device_id, node_id=0):
        """Add a GPU node to the simulation."""
        node = GpuNode()
        node.type = gpu_type
        node.compute_tflops = compute_tflops
        node.memory_gb = memory_gb
        node.device_id = device_id
        node.node_id = node_id
        self.simulator.add_node(node)
        
    def add_network_link(self, source_id, target_id, bandwidth_gbps, latency_us):
        """Add a network link between two GPUs."""
        link = NetworkLink()
        link.source_id = source_id
        link.target_id = target_id
        link.bandwidth_gbps = bandwidth_gbps
        link.latency_us = latency_us
        self.simulator.add_network_link(link)
    
    def add_cluster(self, num_nodes, gpus_per_node, gpu_type="H100", 
                   compute_tflops=312.0, memory_gb=80.0):
        """Add a homogeneous cluster configuration."""
        for i in range(num_nodes):
            for j in range(gpus_per_node):
                device_id = i * gpus_per_node + j
                self.add_gpu_node(gpu_type, compute_tflops, memory_gb, device_id, i)
                
        # Add fully connected network within nodes (high bandwidth)
        for i in range(num_nodes):
            for j in range(gpus_per_node):
                for k in range(j+1, gpus_per_node):
                    src_id = i * gpus_per_node + j
                    dst_id = i * gpus_per_node + k
                    # High bandwidth, low latency for intra-node
                    self.add_network_link(src_id, dst_id, 300.0, 1.0)
                    self.add_network_link(dst_id, src_id, 300.0, 1.0)
        
        # Add node-to-node connections (lower bandwidth)
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                src_base = i * gpus_per_node
                dst_base = j * gpus_per_node
                # Connect first GPU of each node with lower bandwidth
                self.add_network_link(src_base, dst_base, 50.0, 10.0)
                self.add_network_link(dst_base, src_base, 50.0, 10.0)
    
    def simulate_training(self, model_size_gb, batch_size, seq_length,
                        tensor_parallel, pipeline_parallel, data_parallel,
                        num_iterations=100):
        """Run a training simulation with the given parameters."""
        model_config = ModelConfig()
        model_config.model_size_gb = model_size_gb
        model_config.batch_size = batch_size
        model_config.seq_length = seq_length
        
        parallelism_config = ParallelismConfig()
        parallelism_config.tensor_parallel = tensor_parallel
        parallelism_config.pipeline_parallel = pipeline_parallel
        parallelism_config.data_parallel = data_parallel
        
        self.simulator.set_model_config(model_config)
        self.simulator.set_parallelism_config(parallelism_config)
        
        result = self.simulator.run_simulation(num_iterations)
        
        # Store result for comparison
        self.results.append({
            "model_size_gb": model_size_gb,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "parallelism": (tensor_parallel, pipeline_parallel, data_parallel),
            "time_per_iteration_ms": result.time_per_iteration_ms,
            "memory_usage_gb": result.memory_usage_gb,
            "bottleneck": result.bottleneck_description
        })
        
        return result
    
    def compare_configurations(self, configs):
        """Compare multiple parallelism configurations."""
        results = []
        for config in configs:
            result = self.simulate_training(
                model_size_gb=config["model_size_gb"],
                batch_size=config["batch_size"],
                seq_length=config["seq_length"],
                tensor_parallel=config["tp"],
                pipeline_parallel=config["pp"],
                data_parallel=config["dp"],
                num_iterations=config.get("iterations", 100)
            )
            results.append({
                "config": config,
                "result": result
            })
        
        # Sort by iteration time
        results.sort(key=lambda x: x["result"].time_per_iteration_ms)
        
        return results