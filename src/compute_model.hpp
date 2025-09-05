#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>

namespace dist_sim {

/**
 * @brief Models computational operations in a distributed training system
 * 
 * Simulates various computational operations including forward/backward passes,
 * gradient calculations, weight updates, and other ML training operations.
 */
class ComputeModel {
public:
    enum class DeviceType {
        CPU,
        GPU,
        TPU,
        CUSTOM
    };
    
    struct DeviceProperties {
        double flops;            // FLOPs (floating point operations per second)
        double memory_bandwidth; // Memory bandwidth in GB/s
        int memory_size;         // Memory size in MB
        int cores;               // Number of cores/streaming multiprocessors
    };
    
    struct OperationProfile {
        double flops;            // Required floating point operations
        double memory_read;      // Required memory reads in bytes
        double memory_write;     // Required memory writes in bytes
    };
    
    enum class Operation {
        FORWARD_PASS,
        BACKWARD_PASS,
        WEIGHT_UPDATE,
        CUSTOM
    };

public:
    // Constructor
    ComputeModel(int num_devices, DeviceType device_type = DeviceType::GPU);
    
    // Destructor
    virtual ~ComputeModel();
    
    // Configure devices
    void setDeviceType(DeviceType device_type);
    void setUniformDeviceProperties(const DeviceProperties& properties);
    void setDeviceProperties(int device_id, const DeviceProperties& properties);
    
    // Set operation profiles
    void setOperationProfile(Operation op, const OperationProfile& profile);
    void setCustomOperationProfile(const std::string& op_name, const OperationProfile& profile);
    
    // Batch size configuration
    void setBatchSize(int batch_size);
    void setModelSize(size_t parameters, size_t activations);
    
    // Simulation methods
    double simulateOperation(Operation op, int device_id, int local_batch_size = -1);
    double simulateCustomOperation(const std::string& op_name, int device_id, int local_batch_size = -1);
    double simulateFullTrainingIteration(int device_id, int local_batch_size = -1);
    
    // Utility functions
    int getNumDevices() const;
    DeviceType getDeviceType() const;
    DeviceProperties getDeviceProperties(int device_id) const;
    int getBatchSize() const;
    std::pair<size_t, size_t> getModelSize() const;

private:
    int num_devices_;
    DeviceType device_type_;
    std::vector<DeviceProperties> device_properties_;
    std::unordered_map<Operation, OperationProfile> operation_profiles_;
    std::unordered_map<std::string, OperationProfile> custom_operation_profiles_;
    int batch_size_;
    size_t model_parameters_;
    size_t model_activations_;
    
    // Helper methods
    OperationProfile scaleProfileForBatchSize(const OperationProfile& profile, int local_batch_size) const;
    double calculateExecutionTime(int device_id, const OperationProfile& profile) const;
};

} // namespace dist_sim
