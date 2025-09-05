# plan


### folder structure

```
ai-plan/
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

