#include "communication_model.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace dist_sim {

CommunicationModel::CommunicationModel(int num_nodes, Topology topology)
    : num_nodes_(num_nodes), topology_(topology) {
    
    if (num_nodes <= 0) {
        throw std::invalid_argument("Number of nodes must be positive");
    }
    
    // Initialize with default topology
    setTopology(topology);
}

CommunicationModel::~CommunicationModel() = default;

void CommunicationModel::setTopology(Topology topology) {
    topology_ = topology;
    link_properties_.clear();
    
    // Default link properties
    LinkProperties default_properties = {
        10.0,  // 10 Gbps bandwidth
        10.0,  // 10 microseconds latency
        0.0    // 0% error rate
    };
    
    // Create links based on topology
    switch (topology) {
        case Topology::RING:
            for (int i = 0; i < num_nodes_; ++i) {
                int next = (i + 1) % num_nodes_;
                link_properties_[{i, next}] = default_properties;
                link_properties_[{next, i}] = default_properties;
            }
            break;
            
        case Topology::TREE:
            for (int i = 0; i < num_nodes_ / 2; ++i) {
                int left = 2 * i + 1;
                int right = 2 * i + 2;
                
                if (left < num_nodes_) {
                    link_properties_[{i, left}] = default_properties;
                    link_properties_[{left, i}] = default_properties;
                }
                
                if (right < num_nodes_) {
                    link_properties_[{i, right}] = default_properties;
                    link_properties_[{right, i}] = default_properties;
                }
            }
            break;
            
        case Topology::FULLY_CONNECTED:
            for (int i = 0; i < num_nodes_; ++i) {
                for (int j = 0; j < num_nodes_; ++j) {
                    if (i != j) {
                        link_properties_[{i, j}] = default_properties;
                    }
                }
            }
            break;
            
        case Topology::MESH_2D: {
            int side = static_cast<int>(std::sqrt(num_nodes_));
            if (side * side != num_nodes_) {
                throw std::invalid_argument("Number of nodes must be a perfect square for 2D mesh topology");
            }
            
            for (int i = 0; i < num_nodes_; ++i) {
                int row = i / side;
                int col = i % side;
                
                // Connect to neighbors (up, down, left, right)
                if (row > 0) { // Up
                    int up = (row - 1) * side + col;
                    link_properties_[{i, up}] = default_properties;
                    link_properties_[{up, i}] = default_properties;
                }
                
                if (row < side - 1) { // Down
                    int down = (row + 1) * side + col;
                    link_properties_[{i, down}] = default_properties;
                    link_properties_[{down, i}] = default_properties;
                }
                
                if (col > 0) { // Left
                    int left = row * side + (col - 1);
                    link_properties_[{i, left}] = default_properties;
                    link_properties_[{left, i}] = default_properties;
                }
                
                if (col < side - 1) { // Right
                    int right = row * side + (col + 1);
                    link_properties_[{i, right}] = default_properties;
                    link_properties_[{right, i}] = default_properties;
                }
            }
            break;
        }
            
        case Topology::TORUS_2D: {
            int side = static_cast<int>(std::sqrt(num_nodes_));
            if (side * side != num_nodes_) {
                throw std::invalid_argument("Number of nodes must be a perfect square for 2D torus topology");
            }
            
            for (int i = 0; i < num_nodes_; ++i) {
                int row = i / side;
                int col = i % side;
                
                // Connect to neighbors with wrap-around
                int up = ((row - 1 + side) % side) * side + col;
                int down = ((row + 1) % side) * side + col;
                int left = row * side + ((col - 1 + side) % side);
                int right = row * side + ((col + 1) % side);
                
                link_properties_[{i, up}] = default_properties;
                link_properties_[{up, i}] = default_properties;
                link_properties_[{i, down}] = default_properties;
                link_properties_[{down, i}] = default_properties;
                link_properties_[{i, left}] = default_properties;
                link_properties_[{left, i}] = default_properties;
                link_properties_[{i, right}] = default_properties;
                link_properties_[{right, i}] = default_properties;
            }
            break;
        }
            
        case Topology::CUSTOM:
            // Custom topology is set through setCustomTopology
            break;
    }
}

void CommunicationModel::setCustomTopology(const std::vector<std::pair<int, int>>& links) {
    topology_ = Topology::CUSTOM;
    link_properties_.clear();
    
    LinkProperties default_properties = {
        10.0,  // 10 Gbps bandwidth
        10.0,  // 10 microseconds latency
        0.0    // 0% error rate
    };
    
    for (const auto& link : links) {
        if (link.first < 0 || link.first >= num_nodes_ || 
            link.second < 0 || link.second >= num_nodes_) {
            throw std::out_of_range("Node indices must be within range [0, num_nodes-1]");
        }
        
        link_properties_[{link.first, link.second}] = default_properties;
    }
}

void CommunicationModel::setUniformLinkProperties(const LinkProperties& properties) {
    for (auto& link : link_properties_) {
        link.second = properties;
    }
}

void CommunicationModel::setLinkProperties(int source, int destination, const LinkProperties& properties) {
    if (source < 0 || source >= num_nodes_ || destination < 0 || destination >= num_nodes_) {
        throw std::out_of_range("Node indices must be within range [0, num_nodes-1]");
    }
    
    NodePair pair = {source, destination};
    if (link_properties_.find(pair) == link_properties_.end()) {
        throw std::invalid_argument("No direct link exists between specified nodes");
    }
    
    link_properties_[pair] = properties;
}

double CommunicationModel::simulateSendRecv(int source, int destination, size_t bytes) {
    if (source == destination) {
        return 0.0;  // No communication needed
    }
    
    if (!nodesAreConnected(source, destination)) {
        throw std::invalid_argument("No direct link exists between specified nodes");
    }
    
    return calculateTransferTime(source, destination, bytes);
}

double CommunicationModel::simulateAllReduce(size_t bytes_per_node, CollectiveAlgorithm algorithm) {
    double time = 0.0;
    
    switch (algorithm) {
        case CollectiveAlgorithm::RING_ALLREDUCE: {
            // Ring AllReduce has two phases: scatter-reduce and allgather
            // Each phase takes (n-1) steps where n is the number of nodes
            
            // In each step, each node sends/receives bytes_per_node / num_nodes_ bytes
            size_t chunk_size = bytes_per_node / num_nodes_;
            
            // Time for scatter-reduce phase
            for (int i = 0; i < num_nodes_ - 1; ++i) {
                double max_step_time = 0.0;
                for (int node = 0; node < num_nodes_; ++node) {
                    int target = (node + 1) % num_nodes_;
                    max_step_time = std::max(max_step_time, 
                                            calculateTransferTime(node, target, chunk_size));
                }
                time += max_step_time;
            }
            
            // Time for allgather phase
            for (int i = 0; i < num_nodes_ - 1; ++i) {
                double max_step_time = 0.0;
                for (int node = 0; node < num_nodes_; ++node) {
                    int target = (node + 1) % num_nodes_;
                    max_step_time = std::max(max_step_time, 
                                            calculateTransferTime(node, target, chunk_size));
                }
                time += max_step_time;
            }
            break;
        }
            
        case CollectiveAlgorithm::RECURSIVE_DOUBLING: {
            // Only works for power-of-2 number of nodes
            if ((num_nodes_ & (num_nodes_ - 1)) != 0) {
                throw std::invalid_argument("Recursive doubling requires power-of-2 number of nodes");
            }
            
            // log(n) steps
            int steps = static_cast<int>(std::log2(num_nodes_));
            for (int step = 0; step < steps; ++step) {
                double max_step_time = 0.0;
                int distance = 1 << step;
                
                for (int node = 0; node < num_nodes_; ++node) {
                    int partner = node ^ distance;
                    if (partner > node && partner < num_nodes_) {
                        max_step_time = std::max(max_step_time, 
                                               calculateTransferTime(node, partner, bytes_per_node));
                    }
                }
                time += max_step_time;
            }
            break;
        }
            
        case CollectiveAlgorithm::BINARY_TREE: {
            // Reduce + Broadcast
            // For reduce: log(n) steps with half the nodes sending in each step
            // For broadcast: log(n) steps
            
            double reduce_time = 0.0;
            double broadcast_time = 0.0;
            
            int levels = static_cast<int>(std::ceil(std::log2(num_nodes_)));
            
            // Reduce phase (bottom-up)
            for (int level = levels - 1; level >= 0; --level) {
                double max_step_time = 0.0;
                int nodes_at_level = 1 << level;
                
                for (int i = 0; i < nodes_at_level && i + nodes_at_level < num_nodes_; ++i) {
                    int child = i + nodes_at_level;
                    int parent = i;
                    
                    if (child < num_nodes_ && nodesAreConnected(child, parent)) {
                        max_step_time = std::max(max_step_time, 
                                               calculateTransferTime(child, parent, bytes_per_node));
                    }
                }
                reduce_time += max_step_time;
            }
            
            // Broadcast phase (top-down)
            for (int level = 0; level < levels; ++level) {
                double max_step_time = 0.0;
                int nodes_at_level = 1 << level;
                
                for (int i = 0; i < nodes_at_level && i < num_nodes_; ++i) {
                    int parent = i;
                    int left_child = 2 * i + 1;
                    int right_child = 2 * i + 2;
                    
                    if (left_child < num_nodes_ && nodesAreConnected(parent, left_child)) {
                        max_step_time = std::max(max_step_time, 
                                               calculateTransferTime(parent, left_child, bytes_per_node));
                    }
                    
                    if (right_child < num_nodes_ && nodesAreConnected(parent, right_child)) {
                        max_step_time = std::max(max_step_time, 
                                               calculateTransferTime(parent, right_child, bytes_per_node));
                    }
                }
                broadcast_time += max_step_time;
            }
            
            time = reduce_time + broadcast_time;
            break;
        }
            
        case CollectiveAlgorithm::RABENSEIFNER: {
            // Rabenseifner's algorithm combines recursive doubling with a ring algorithm
            // Only works for power-of-2 number of nodes
            if ((num_nodes_ & (num_nodes_ - 1)) != 0) {
                throw std::invalid_argument("Rabenseifner's algorithm requires power-of-2 number of nodes");
            }
            
            // For simplicity, we'll approximate it as 1.5 * recursive doubling time
            return 1.5 * simulateAllReduce(bytes_per_node, CollectiveAlgorithm::RECURSIVE_DOUBLING);
        }
    }
    
    return time;
}

double CommunicationModel::simulateBroadcast(int root, size_t bytes) {
    if (root < 0 || root >= num_nodes_) {
        throw std::out_of_range("Root node index must be within range [0, num_nodes-1]");
    }
    
    double time = 0.0;
    
    // Simple binary tree broadcast
    std::vector<bool> received(num_nodes_, false);
    received[root] = true;
    
    std::vector<int> current_senders = {root};
    
    while (!current_senders.empty()) {
        std::vector<int> next_senders;
        double max_step_time = 0.0;
        
        for (int sender : current_senders) {
            for (int receiver = 0; receiver < num_nodes_; ++receiver) {
                if (!received[receiver] && nodesAreConnected(sender, receiver)) {
                    received[receiver] = true;
                    next_senders.push_back(receiver);
                    
                    double transfer_time = calculateTransferTime(sender, receiver, bytes);
                    max_step_time = std::max(max_step_time, transfer_time);
                }
            }
        }
        
        time += max_step_time;
        current_senders = next_senders;
    }
    
    return time;
}

double CommunicationModel::simulateGather(int root, size_t bytes_per_node) {
    if (root < 0 || root >= num_nodes_) {
        throw std::out_of_range("Root node index must be within range [0, num_nodes-1]");
    }
    
    double time = 0.0;
    
    // Simple direct gather (everyone sends to root)
    double max_transfer_time = 0.0;
    
    for (int node = 0; node < num_nodes_; ++node) {
        if (node != root) {
            if (!nodesAreConnected(node, root)) {
                throw std::invalid_argument("No direct link exists between node and root");
            }
            
            double transfer_time = calculateTransferTime(node, root, bytes_per_node);
            max_transfer_time = std::max(max_transfer_time, transfer_time);
        }
    }
    
    time = max_transfer_time;
    return time;
}

double CommunicationModel::simulateAllGather(size_t bytes_per_node) {
    // Ring algorithm for allgather
    double time = 0.0;
    
    for (int step = 0; step < num_nodes_ - 1; ++step) {
        double max_step_time = 0.0;
        
        for (int node = 0; node < num_nodes_; ++node) {
            int sender = (node - step - 1 + num_nodes_) % num_nodes_;
            int receiver = (node - step + num_nodes_) % num_nodes_;
            
            if (!nodesAreConnected(sender, receiver)) {
                throw std::invalid_argument("Ring topology required for this implementation of allgather");
            }
            
            double transfer_time = calculateTransferTime(sender, receiver, bytes_per_node);
            max_step_time = std::max(max_step_time, transfer_time);
        }
        
        time += max_step_time;
    }
    
    return time;
}

int CommunicationModel::getNumNodes() const {
    return num_nodes_;
}

CommunicationModel::Topology CommunicationModel::getTopology() const {
    return topology_;
}

CommunicationModel::LinkProperties CommunicationModel::getLinkProperties(int source, int destination) const {
    if (source < 0 || source >= num_nodes_ || destination < 0 || destination >= num_nodes_) {
        throw std::out_of_range("Node indices must be within range [0, num_nodes-1]");
    }
    
    NodePair pair = {source, destination};
    auto it = link_properties_.find(pair);
    
    if (it == link_properties_.end()) {
        throw std::invalid_argument("No direct link exists between specified nodes");
    }
    
    return it->second;
}

bool CommunicationModel::nodesAreConnected(int source, int destination) const {
    if (source < 0 || source >= num_nodes_ || destination < 0 || destination >= num_nodes_) {
        return false;
    }
    
    if (source == destination) {
        return true;  // Node is connected to itself
    }
    
    NodePair pair = {source, destination};
    return link_properties_.find(pair) != link_properties_.end();
}

double CommunicationModel::calculateTransferTime(int source, int destination, size_t bytes) const {
    NodePair pair = {source, destination};
    auto it = link_properties_.find(pair);
    
    if (it == link_properties_.end()) {
        throw std::invalid_argument("No direct link exists between specified nodes");
    }
    
    const LinkProperties& props = it->second;
    
    // Calculate transfer time: latency + (bytes / bandwidth)
    // Convert bytes to bits (× 8) and Gbps to bps (× 10^9)
    double bandwidth_bps = props.bandwidth_gbps * 1e9;
    double transfer_time_s = props.latency_us * 1e-6 + (bytes * 8.0) / bandwidth_bps;
    
    // Convert to milliseconds
    return transfer_time_s * 1000.0;
}

} // namespace dist_sim
