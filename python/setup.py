from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "dist_training_sim.dist_sim_core",
        ["../src/bindings/bindings.cpp",
         "../src/training_simulator.cpp",
         "../src/communication_model.cpp",
         "../src/compute_model.cpp",
         "../src/memory_model.cpp"],
        include_dirs=["../src"],
        extra_compile_args=["-std=c++17"],
    ),
]

setup(
    name="dist_training_sim",
    version="0.1.0",
    author="Janardhan",
    author_email="j143@example.com",
    description="Distributed Training Simulator",
    long_description="A high-performance simulation tool for distributed AI training infrastructure optimization.",
    ext_modules=ext_modules,
    packages=find_packages(),
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "plotly>=5.10.0",
        "dash>=2.6.0",
    ],
)