#include "compute_model.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace dist_sim {

ComputeModel::ComputeModel(int num_devices, DeviceType device_type)
    : num_devices_(num_devices), device_type_(device_type), batch_size_(64),
      model_parameters_(0), model_activations_(0) {
    
    if (num_devices <= 0) {
        throw std::invalid_argument("Number of devices must be positive");
    }
    
    // Initialize device properties based on type
    device_properties_.resize(num_devices);
    setDeviceType(device_type);
    
    // Set default operation profiles
    OperationProfile forward_profile = {
        1e10,   // 10 GFLOPS
        1e8,    // 100 MB memory read
        5e7     // 50 MB memory write
    };
    
    OperationProfile backward_profile = {
        2e10,   // 20 GFLOPS (typically 2x forward pass)
        1.5e8,  // 150 MB memory read
        1e8     // 100 MB memory write
    };
    
    OperationProfile weight_update_profile = {
        5e9,    // 5 GFLOPS
        1e8,    // 100 MB memory read
        1e8     // 100 MB memory write
    };
    
    operation_profiles_[Operation::FORWARD_PASS] = forward_profile;
    operation_profiles_[Operation::BACKWARD_PASS] = backward_profile;
    operation_profiles_[Operation::WEIGHT_UPDATE] = weight_update_profile;
}

ComputeModel::~ComputeModel() = default;

void ComputeModel::setDeviceType(DeviceType device_type) {
    device_type_ = device_type;
    
    // Set default properties based on device type
    DeviceProperties properties;
    
    switch (device_type) {
        case DeviceType::CPU:
            properties = {
                1e12,   // 1 TFLOPS
                100.0,  // 100 GB/s memory bandwidth
                32768,  // 32 GB memory
                32      // 32 cores
            };
            break;
            
        case DeviceType::GPU:
            properties = {
                1e13,   // 10 TFLOPS
                900.0,  // 900 GB/s memory bandwidth
                16384,  // 16 GB memory
                80      // 80 SMs (streaming multiprocessors)
            };
            break;
            
        case DeviceType::TPU:
            properties = {
                1.4e14, // 140 TFLOPS
                600.0,  // 600 GB/s memory bandwidth
                32768,  // 32 GB memory
                2       // 2 cores
            };
            break;
            
        case DeviceType::CUSTOM:
            properties = {
                1e12,   // 1 TFLOPS
                500.0,  // 500 GB/s memory bandwidth
                8192,   // 8 GB memory
                16      // 16 cores
            };
            break;
    }
    
    setUniformDeviceProperties(properties);
}

void ComputeModel::setUniformDeviceProperties(const DeviceProperties& properties) {
    for (int i = 0; i < num_devices_; ++i) {
        device_properties_[i] = properties;
    }
}

void ComputeModel::setDeviceProperties(int device_id, const DeviceProperties& properties) {
    if (device_id < 0 || device_id >= num_devices_) {
        throw std::out_of_range("Device ID must be within range [0, num_devices-1]");
    }
    
    device_properties_[device_id] = properties;
}

void ComputeModel::setOperationProfile(Operation op, const OperationProfile& profile) {
    operation_profiles_[op] = profile;
}

void ComputeModel::setCustomOperationProfile(const std::string& op_name, const OperationProfile& profile) {
    custom_operation_profiles_[op_name] = profile;
}

void ComputeModel::setBatchSize(int batch_size) {
    if (batch_size <= 0) {
        throw std::invalid_argument("Batch size must be positive");
    }
    
    batch_size_ = batch_size;
}

void ComputeModel::setModelSize(size_t parameters, size_t activations) {
    model_parameters_ = parameters;
    model_activations_ = activations;
}

double ComputeModel::simulateOperation(Operation op, int device_id, int local_batch_size) {
    if (device_id < 0 || device_id >= num_devices_) {
        throw std::out_of_range("Device ID must be within range [0, num_devices-1]");
    }
    
    auto it = operation_profiles_.find(op);
    if (it == operation_profiles_.end()) {
        throw std::invalid_argument("Operation profile not found");
    }
    
    // If local_batch_size is not specified, use global batch_size / num_devices
    if (local_batch_size <= 0) {
        local_batch_size = batch_size_ / num_devices_;
        if (local_batch_size <= 0) local_batch_size = 1;
    }
    
    OperationProfile scaled_profile = scaleProfileForBatchSize(it->second, local_batch_size);
    return calculateExecutionTime(device_id, scaled_profile);
}

double ComputeModel::simulateCustomOperation(const std::string& op_name, int device_id, int local_batch_size) {
    if (device_id < 0 || device_id >= num_devices_) {
        throw std::out_of_range("Device ID must be within range [0, num_devices-1]");
    }
    
    auto it = custom_operation_profiles_.find(op_name);
    if (it == custom_operation_profiles_.end()) {
        throw std::invalid_argument("Custom operation profile not found: " + op_name);
    }
    
    // If local_batch_size is not specified, use global batch_size / num_devices
    if (local_batch_size <= 0) {
        local_batch_size = batch_size_ / num_devices_;
        if (local_batch_size <= 0) local_batch_size = 1;
    }
    
    OperationProfile scaled_profile = scaleProfileForBatchSize(it->second, local_batch_size);
    return calculateExecutionTime(device_id, scaled_profile);
}

double ComputeModel::simulateFullTrainingIteration(int device_id, int local_batch_size) {
    if (device_id < 0 || device_id >= num_devices_) {
        throw std::out_of_range("Device ID must be within range [0, num_devices-1]");
    }
    
    // If local_batch_size is not specified, use global batch_size / num_devices
    if (local_batch_size <= 0) {
        local_batch_size = batch_size_ / num_devices_;
        if (local_batch_size <= 0) local_batch_size = 1;
    }
    
    // A full training iteration includes forward pass, backward pass, and weight update
    double forward_time = simulateOperation(Operation::FORWARD_PASS, device_id, local_batch_size);
    double backward_time = simulateOperation(Operation::BACKWARD_PASS, device_id, local_batch_size);
    double weight_update_time = simulateOperation(Operation::WEIGHT_UPDATE, device_id, local_batch_size);
    
    return forward_time + backward_time + weight_update_time;
}

int ComputeModel::getNumDevices() const {
    return num_devices_;
}

ComputeModel::DeviceType ComputeModel::getDeviceType() const {
    return device_type_;
}

ComputeModel::DeviceProperties ComputeModel::getDeviceProperties(int device_id) const {
    if (device_id < 0 || device_id >= num_devices_) {
        throw std::out_of_range("Device ID must be within range [0, num_devices-1]");
    }
    
    return device_properties_[device_id];
}

int ComputeModel::getBatchSize() const {
    return batch_size_;
}

std::pair<size_t, size_t> ComputeModel::getModelSize() const {
    return {model_parameters_, model_activations_};
}

ComputeModel::OperationProfile ComputeModel::scaleProfileForBatchSize(
    const OperationProfile& profile, int local_batch_size) const {
    
    // Scale the profile linearly with batch size
    double scale_factor = static_cast<double>(local_batch_size) / 
                         (batch_size_ / num_devices_);
    
    OperationProfile scaled_profile = {
        profile.flops * scale_factor,
        profile.memory_read * scale_factor,
        profile.memory_write * scale_factor
    };
    
    return scaled_profile;
}

double ComputeModel::calculateExecutionTime(int device_id, const OperationProfile& profile) const {
    const DeviceProperties& device = device_properties_[device_id];
    
    // Calculate compute time: FLOPS / device_flops
    double compute_time = profile.flops / device.flops;
    
    // Calculate memory time: (memory_read + memory_write) / memory_bandwidth
    double total_memory = profile.memory_read + profile.memory_write;
    double memory_time = total_memory / (device.memory_bandwidth * 1e9); // Convert GB/s to B/s
    
    // Simple roofline model: execution time is max of compute-bound and memory-bound times
    double execution_time = std::max(compute_time, memory_time);
    
    // Convert to milliseconds
    return execution_time * 1000.0;
}

} // namespace dist_sim
