#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstdint>

namespace dist_sim {

/**
 * @brief Models memory systems and hierarchies in a distributed training system
 * 
 * Simulates various memory operations including allocation, deallocation, transfer
 * between different memory types (VRAM, HBM, RAM, disk), and memory management strategies.
 */
class MemoryModel {
public:
    enum class MemoryType {
        GPU_VRAM,   // GPU video memory
        HBM,        // High Bandwidth Memory
        CPU_RAM,    // CPU main memory
        DISK,       // Storage
        REMOTE      // Remote memory (other node)
    };
    
    struct MemoryTier {
        MemoryType type;
        uint64_t capacity_bytes;
        double bandwidth_gbps;
        double latency_ns;
    };
    
    struct MemoryAllocation {
        uint64_t size_bytes;
        uint64_t address;
        MemoryType type;
        bool is_pinned;
    };
    
    struct TransferProperties {
        double bandwidth_gbps;
        double latency_ns;
    };

public:
    // Constructor
    MemoryModel(int num_devices);
    
    // Destructor
    virtual ~MemoryModel();
    
    // Configure memory system
    void addMemoryTier(int device_id, const MemoryTier& tier);
    void setTransferProperties(MemoryType source, MemoryType destination, const TransferProperties& props);
    
    // Memory operations simulation
    double simulateAllocation(int device_id, MemoryType type, uint64_t size_bytes);
    double simulateDeallocation(int device_id, MemoryType type, uint64_t size_bytes);
    double simulateTransfer(int source_device, MemoryType source_type, 
                            int dest_device, MemoryType dest_type, uint64_t size_bytes);
    
    // Memory management
    bool canAllocate(int device_id, MemoryType type, uint64_t size_bytes) const;
    uint64_t getAvailableMemory(int device_id, MemoryType type) const;
    uint64_t getTotalMemory(int device_id, MemoryType type) const;
    
    // Memory optimization simulation
    double simulateMemoryOptimization(const std::string& strategy, uint64_t model_size_bytes);

private:
    struct DeviceMemory {
        std::unordered_map<MemoryType, MemoryTier> tiers;
        std::unordered_map<MemoryType, uint64_t> allocated;
    };
    
    int num_devices_;
    std::vector<DeviceMemory> device_memories_;
    std::unordered_map<MemoryType, std::unordered_map<MemoryType, TransferProperties>> transfer_properties_;
    
    // Helper methods
    bool hasTier(int device_id, MemoryType type) const;
    double calculateTransferTime(MemoryType source, MemoryType destination, uint64_t size_bytes) const;
};

} // namespace dist_sim
