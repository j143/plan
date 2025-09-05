# Distributed Training Simulator

A high-performance simulator for distributed deep learning training systems that models computation, communication, and memory patterns to predict performance and scaling behavior of training workloads.

## Features

- **Modular Architecture**: Clean separation between C++ core and Python interface
- **Performance Modeling**:
  - Communication patterns (point-to-point, all-reduce, broadcast)
  - Compute operations (forward/backward pass, weight updates)
  - Memory systems (allocation, transfer, optimization strategies)
- **Visualization**: Comprehensive visualization tools for simulation results
- **Cross-platform**: Works on Linux, macOS, and Windows

## Installation

### Requirements

- C++17 compatible compiler (GCC 7+, Clang 5+, or MSVC 2019+)
- CMake 3.14+
- Python 3.7+
- pybind11 (installed automatically during setup)

### Building from Source

1. Clone the repository:
   ```bash
   git clone https://github.com/j143/plan.git
   cd plan
   ```

2. Create a build directory and run CMake:
   ```bash
   mkdir build
   cd build
   cmake ..
   ```

3. Build the project:
   ```bash
   cmake --build .
   ```

4. Install the Python package:
   ```bash
   cd ..
   pip install -e python/
   ```

### Using the Docker Image

For convenience, you can use our Docker image:

```bash
docker pull j143/dist-training-sim:latest
docker run -it j143/dist-training-sim:latest
```

## Getting Started

### Basic Usage

```python
import dist_training_sim as dts

# Create a communication model
comm_model = dts.CommunicationModel(num_nodes=8, topology=dts.Topology.RING)

# Create a compute model
compute_model = dts.ComputeModel(num_devices=8, device_type=dts.DeviceType.GPU)

# Create a memory model
memory_model = dts.MemoryModel(num_devices=8)

# Create a simulator
simulator = dts.TrainingSimulator(comm_model, compute_model, memory_model)

# Configure the simulation
simulator.set_batch_size(256)
simulator.set_model_size(parameters=250e6, activations=100e6)

# Run the simulation
results = simulator.simulate_training_iteration()

# Get results
print(f"Iteration time: {results.total_time_ms:.2f} ms")
print(f"Compute time: {results.compute_time_ms:.2f} ms")
print(f"Communication time: {results.communication_time_ms:.2f} ms")
```

### Project Structure

```
plan/
├── CMakeLists.txt               # Main CMake configuration
├── src/                         # C++ source files
│   ├── communication_model.hpp
│   ├── communication_model.cpp
│   ├── compute_model.hpp
│   ├── compute_model.cpp
│   ├── memory_model.hpp
│   ├── memory_model.cpp
│   ├── training_simulator.hpp
│   ├── training_simulator.cpp
│   └── bindings/                # Python bindings
│       ├── CMakeLists.txt
│       └── bindings.cpp
├── python/                      # Python package
│   ├── setup.py
│   ├── dist_training_sim/
│   │   ├── __init__.py
│   │   ├── simulator.py
│   │   └── visualization.py
│   └── tests/                   # Python tests
│       ├── __init__.py
│       ├── test_simulator.py
│       └── test_visualization.py
├── tests/                       # C++ tests
│   ├── CMakeLists.txt
│   ├── communication_model_test.cpp
│   ├── compute_model_test.cpp
│   ├── memory_model_test.cpp
│   └── training_simulator_test.cpp
├── .github/
│   └── workflows/
│       └── ci.yml               # CI pipeline
└── Dockerfile                   # Docker configuration
```

### Visualization

The visualization module provides tools to analyze and visualize simulation results:

```python
from dist_training_sim.visualization import Visualizer

# Create a visualizer
vis = Visualizer()

# Plot training timeline
fig = vis.plot_training_timeline(results.events)

# Plot communication graph
fig = vis.plot_communication_graph(results.comm_adjacency)

# Plot scaling efficiency
fig = vis.plot_scaling_efficiency(
    num_devices=[1, 2, 4, 8, 16, 32],
    throughput=[100, 195, 380, 740, 1400, 2600]
)

# Save figure
vis.save_figure(fig, "scaling_efficiency.png")
```

## Documentation

For complete API documentation, see the [API Reference](https://github.com/j143/plan/docs/api/) and [User Guide](https://github.com/j143/plan/docs/guide/).

## Examples

The `examples/` directory contains Jupyter notebooks demonstrating different use cases:

- Basic simulation workflow
- Scaling analysis
- Communication optimization
- Memory optimization strategies
- Custom model configuration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

