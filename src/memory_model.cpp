#include "memory_model.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace dist_sim {

MemoryModel::MemoryModel(int num_devices)
    : num_devices_(num_devices) {
    
    if (num_devices <= 0) {
        throw std::invalid_argument("Number of devices must be positive");
    }
    
    device_memories_.resize(num_devices);
    
    // Set up default memory tiers for each device
    for (int device_id = 0; device_id < num_devices_; ++device_id) {
        // Default GPU VRAM - 16GB with high bandwidth
        MemoryTier vram = {
            MemoryType::GPU_VRAM,
            16ULL * 1024 * 1024 * 1024,  // 16 GB
            900.0,  // 900 GB/s bandwidth
            100.0   // 100 ns latency
        };
        
        // Default CPU RAM - 128GB with medium bandwidth
        MemoryTier ram = {
            MemoryType::CPU_RAM,
            128ULL * 1024 * 1024 * 1024,  // 128 GB
            100.0,  // 100 GB/s bandwidth
            100000.0 // 100,000 ns (100 us) latency
        };
        
        // Default disk - 1TB with low bandwidth
        MemoryTier disk = {
            MemoryType::DISK,
            1024ULL * 1024 * 1024 * 1024,  // 1 TB
            5.0,    // 5 GB/s bandwidth (NVMe SSD)
            100000000.0  // 100,000,000 ns (100 ms) latency
        };
        
        addMemoryTier(device_id, vram);
        addMemoryTier(device_id, ram);
        addMemoryTier(device_id, disk);
        
        // Initialize allocated memory to 0
        device_memories_[device_id].allocated[MemoryType::GPU_VRAM] = 0;
        device_memories_[device_id].allocated[MemoryType::CPU_RAM] = 0;
        device_memories_[device_id].allocated[MemoryType::DISK] = 0;
    }
    
    // Set up default transfer properties
    
    // GPU VRAM -> CPU RAM (PCIe)
    TransferProperties vram_to_ram = {
        16.0,   // 16 GB/s (PCIe 4.0 x16)
        1000.0  // 1,000 ns (1 us) latency
    };
    
    // CPU RAM -> GPU VRAM (PCIe)
    TransferProperties ram_to_vram = {
        16.0,   // 16 GB/s (PCIe 4.0 x16)
        1000.0  // 1,000 ns (1 us) latency
    };
    
    // CPU RAM -> Disk
    TransferProperties ram_to_disk = {
        5.0,    // 5 GB/s (NVMe SSD)
        100000.0  // 100,000 ns (100 us) latency
    };
    
    // Disk -> CPU RAM
    TransferProperties disk_to_ram = {
        5.0,    // 5 GB/s (NVMe SSD)
        100000.0  // 100,000 ns (100 us) latency
    };
    
    setTransferProperties(MemoryType::GPU_VRAM, MemoryType::CPU_RAM, vram_to_ram);
    setTransferProperties(MemoryType::CPU_RAM, MemoryType::GPU_VRAM, ram_to_vram);
    setTransferProperties(MemoryType::CPU_RAM, MemoryType::DISK, ram_to_disk);
    setTransferProperties(MemoryType::DISK, MemoryType::CPU_RAM, disk_to_ram);
}

MemoryModel::~MemoryModel() = default;

void MemoryModel::addMemoryTier(int device_id, const MemoryTier& tier) {
    if (device_id < 0 || device_id >= num_devices_) {
        throw std::out_of_range("Device ID must be within range [0, num_devices-1]");
    }
    
    device_memories_[device_id].tiers[tier.type] = tier;
    device_memories_[device_id].allocated[tier.type] = 0;
}

void MemoryModel::setTransferProperties(MemoryType source, MemoryType destination, 
                                        const TransferProperties& props) {
    transfer_properties_[source][destination] = props;
}

double MemoryModel::simulateAllocation(int device_id, MemoryType type, uint64_t size_bytes) {
    if (device_id < 0 || device_id >= num_devices_) {
        throw std::out_of_range("Device ID must be within range [0, num_devices-1]");
    }
    
    if (!hasTier(device_id, type)) {
        throw std::invalid_argument("Memory type not available on specified device");
    }
    
    if (!canAllocate(device_id, type, size_bytes)) {
        throw std::runtime_error("Not enough memory available for allocation");
    }
    
    // Update allocated memory
    device_memories_[device_id].allocated[type] += size_bytes;
    
    // Calculate allocation time based on memory type and size
    const MemoryTier& tier = device_memories_[device_id].tiers[type];
    
    // Simple model: base latency + size/bandwidth
    double base_latency_s = tier.latency_ns * 1e-9;  // Convert ns to s
    double bandwidth_bytes_per_s = tier.bandwidth_gbps * 1e9;  // Convert GB/s to B/s
    double size_dependent_time = static_cast<double>(size_bytes) / bandwidth_bytes_per_s;
    
    // Small overhead for allocation operations
    double allocation_overhead_factor = 0.01;  // 1% overhead
    
    double time_s = base_latency_s + size_dependent_time * allocation_overhead_factor;
    
    // Convert to milliseconds
    return time_s * 1000.0;
}

double MemoryModel::simulateDeallocation(int device_id, MemoryType type, uint64_t size_bytes) {
    if (device_id < 0 || device_id >= num_devices_) {
        throw std::out_of_range("Device ID must be within range [0, num_devices-1]");
    }
    
    if (!hasTier(device_id, type)) {
        throw std::invalid_argument("Memory type not available on specified device");
    }
    
    uint64_t& allocated = device_memories_[device_id].allocated[type];
    if (allocated < size_bytes) {
        throw std::invalid_argument("Cannot deallocate more memory than is allocated");
    }
    
    // Update allocated memory
    allocated -= size_bytes;
    
    // Deallocation is typically much faster than allocation
    // We'll model it as a fixed small cost
    double time_ms = 0.01;  // 0.01 ms
    
    return time_ms;
}

double MemoryModel::simulateTransfer(int source_device, MemoryType source_type,
                                    int dest_device, MemoryType dest_type, uint64_t size_bytes) {
    if (source_device < 0 || source_device >= num_devices_ ||
        dest_device < 0 || dest_device >= num_devices_) {
        throw std::out_of_range("Device ID must be within range [0, num_devices-1]");
    }
    
    if (!hasTier(source_device, source_type)) {
        throw std::invalid_argument("Source memory type not available on source device");
    }
    
    if (!hasTier(dest_device, dest_type)) {
        throw std::invalid_argument("Destination memory type not available on destination device");
    }
    
    if (!canAllocate(dest_device, dest_type, size_bytes)) {
        throw std::runtime_error("Not enough memory available for transfer on destination");
    }
    
    // Update allocated memory on destination
    device_memories_[dest_device].allocated[dest_type] += size_bytes;
    
    // For same device, different memory types (e.g., VRAM to RAM on same device)
    if (source_device == dest_device) {
        return calculateTransferTime(source_type, dest_type, size_bytes);
    }
    
    // For transfers between devices, model as:
    // 1. Transfer from source_type to CPU_RAM on source device
    // 2. Transfer from source CPU_RAM to dest CPU_RAM (network or interconnect)
    // 3. Transfer from CPU_RAM to dest_type on dest device
    
    double time = 0.0;
    
    // Step 1: Only if source_type is not CPU_RAM
    if (source_type != MemoryType::CPU_RAM) {
        time += calculateTransferTime(source_type, MemoryType::CPU_RAM, size_bytes);
    }
    
    // Step 2: Transfer between devices (simplified model for now)
    // Assume a fixed interconnect bandwidth between devices
    double interconnect_bandwidth_gbps = 50.0;  // 50 GB/s (e.g., NVLink)
    double interconnect_latency_ns = 1000.0;    // 1000 ns (1 μs)
    
    double interconnect_bandwidth_bytes_per_s = interconnect_bandwidth_gbps * 1e9;
    double interconnect_latency_s = interconnect_latency_ns * 1e-9;
    
    time += interconnect_latency_s + static_cast<double>(size_bytes) / interconnect_bandwidth_bytes_per_s;
    
    // Step 3: Only if dest_type is not CPU_RAM
    if (dest_type != MemoryType::CPU_RAM) {
        time += calculateTransferTime(MemoryType::CPU_RAM, dest_type, size_bytes);
    }
    
    // Convert to milliseconds
    return time * 1000.0;
}

bool MemoryModel::canAllocate(int device_id, MemoryType type, uint64_t size_bytes) const {
    if (device_id < 0 || device_id >= num_devices_) {
        return false;
    }
    
    if (!hasTier(device_id, type)) {
        return false;
    }
    
    const auto& device_memory = device_memories_[device_id];
    uint64_t available = device_memory.tiers.at(type).capacity_bytes - device_memory.allocated.at(type);
    
    return size_bytes <= available;
}

uint64_t MemoryModel::getAvailableMemory(int device_id, MemoryType type) const {
    if (device_id < 0 || device_id >= num_devices_) {
        throw std::out_of_range("Device ID must be within range [0, num_devices-1]");
    }
    
    if (!hasTier(device_id, type)) {
        throw std::invalid_argument("Memory type not available on specified device");
    }
    
    const auto& device_memory = device_memories_[device_id];
    return device_memory.tiers.at(type).capacity_bytes - device_memory.allocated.at(type);
}

uint64_t MemoryModel::getTotalMemory(int device_id, MemoryType type) const {
    if (device_id < 0 || device_id >= num_devices_) {
        throw std::out_of_range("Device ID must be within range [0, num_devices-1]");
    }
    
    if (!hasTier(device_id, type)) {
        throw std::invalid_argument("Memory type not available on specified device");
    }
    
    return device_memories_[device_id].tiers.at(type).capacity_bytes;
}

double MemoryModel::simulateMemoryOptimization(const std::string& strategy, uint64_t model_size_bytes) {
    double savings_factor = 0.0;
    double overhead_factor = 0.0;
    
    // Different memory optimization strategies
    if (strategy == "activation_checkpointing") {
        // Typical savings: 30-60% of activation memory, with recomputation overhead
        savings_factor = 0.4;  // 40% memory savings
        overhead_factor = 0.3;  // 30% compute overhead
    }
    else if (strategy == "mixed_precision") {
        // Typical savings: 50% for weights and activations, minimal overhead
        savings_factor = 0.5;  // 50% memory savings
        overhead_factor = 0.05;  // 5% compute overhead (tensor core benefits might actually make it faster)
    }
    else if (strategy == "gradient_accumulation") {
        // Memory savings proportional to accumulation steps, with minimal overhead
        int accumulation_steps = 8;  // Default
        savings_factor = 1.0 - (1.0 / accumulation_steps);
        overhead_factor = 0.01;  // 1% overhead
    }
    else if (strategy == "zero_redundancy_optimizer") {
        // ZeRO stages 1, 2, and 3 have different impacts
        int zero_stage = 2;  // Default to stage 2
        
        if (zero_stage == 1) {
            // Stage 1: Optimizer state partitioning
            savings_factor = 0.33;  // ~33% memory savings
            overhead_factor = 0.05;  // 5% communication overhead
        }
        else if (zero_stage == 2) {
            // Stage 2: Optimizer + gradient partitioning
            savings_factor = 0.66;  // ~66% memory savings
            overhead_factor = 0.1;  // 10% communication overhead
        }
        else if (zero_stage == 3) {
            // Stage 3: Optimizer + gradient + parameter partitioning
            savings_factor = 0.9;  // Up to 90% memory savings
            overhead_factor = 0.2;  // 20% communication overhead
        }
    }
    else if (strategy == "offload_to_cpu") {
        // Offload to CPU: significant memory savings, high overhead
        savings_factor = 0.7;  // 70% GPU memory savings
        overhead_factor = 0.5;  // 50% performance overhead
    }
    else if (strategy == "offload_to_nvme") {
        // Offload to NVMe: even more memory savings, even higher overhead
        savings_factor = 0.9;  // 90% GPU memory savings
        overhead_factor = 2.0;  // 200% performance overhead
    }
    else {
        throw std::invalid_argument("Unknown memory optimization strategy: " + strategy);
    }
    
    // Calculate memory savings
    uint64_t memory_saved = static_cast<uint64_t>(model_size_bytes * savings_factor);
    
    // Return the overhead as a factor (higher is worse)
    return overhead_factor;
}

bool MemoryModel::hasTier(int device_id, MemoryType type) const {
    if (device_id < 0 || device_id >= num_devices_) {
        return false;
    }
    
    const auto& device_memory = device_memories_[device_id];
    return device_memory.tiers.find(type) != device_memory.tiers.end();
}

double MemoryModel::calculateTransferTime(MemoryType source, MemoryType destination, uint64_t size_bytes) const {
    // Check if we have transfer properties for this pair
    auto source_it = transfer_properties_.find(source);
    if (source_it == transfer_properties_.end()) {
        throw std::invalid_argument("No transfer properties defined for source memory type");
    }
    
    auto dest_it = source_it->second.find(destination);
    if (dest_it == source_it->second.end()) {
        throw std::invalid_argument("No transfer properties defined for destination memory type");
    }
    
    const TransferProperties& props = dest_it->second;
    
    // Calculate transfer time: latency + (size / bandwidth)
    double bandwidth_bytes_per_s = props.bandwidth_gbps * 1e9;  // Convert GB/s to B/s
    double latency_s = props.latency_ns * 1e-9;  // Convert ns to s
    
    double transfer_time_s = latency_s + static_cast<double>(size_bytes) / bandwidth_bytes_per_s;
    
    return transfer_time_s;
}

} // namespace dist_sim
