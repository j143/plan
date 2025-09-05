#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>

namespace dist_sim {

/**
 * @brief Models communication between compute nodes in a distributed system
 * 
 * Simulates various network topologies and communication patterns
 * including point-to-point, all-reduce, broadcast, and gather operations.
 */
class CommunicationModel {
public:
    enum class Topology {
        RING,
        TREE,
        FULLY_CONNECTED,
        MESH_2D,
        TORUS_2D,
        CUSTOM
    };
    
    enum class CollectiveAlgorithm {
        RING_ALLREDUCE,
        RECURSIVE_DOUBLING,
        BINARY_TREE,
        RABENSEIFNER
    };

    struct LinkProperties {
        double bandwidth_gbps;
        double latency_us;
        double error_rate;
    };

    struct NodePair {
        int source;
        int destination;
        
        bool operator==(const NodePair& other) const {
            return source == other.source && destination == other.destination;
        }
    };

    // Custom hash function for NodePair
    struct NodePairHash {
        std::size_t operator()(const NodePair& pair) const {
            return std::hash<int>()(pair.source) ^ std::hash<int>()(pair.destination);
        }
    };

public:
    // Constructor
    CommunicationModel(int num_nodes, Topology topology = Topology::FULLY_CONNECTED);
    
    // Destructor
    virtual ~CommunicationModel();
    
    // Configure network topology
    void setTopology(Topology topology);
    void setCustomTopology(const std::vector<std::pair<int, int>>& links);
    
    // Configure link properties
    void setUniformLinkProperties(const LinkProperties& properties);
    void setLinkProperties(int source, int destination, const LinkProperties& properties);
    
    // Communication operations
    double simulateSendRecv(int source, int destination, size_t bytes);
    double simulateAllReduce(size_t bytes_per_node, CollectiveAlgorithm algorithm = CollectiveAlgorithm::RING_ALLREDUCE);
    double simulateBroadcast(int root, size_t bytes);
    double simulateGather(int root, size_t bytes_per_node);
    double simulateAllGather(size_t bytes_per_node);
    
    // Utility functions
    int getNumNodes() const;
    Topology getTopology() const;
    LinkProperties getLinkProperties(int source, int destination) const;

private:
    int num_nodes_;
    Topology topology_;
    std::unordered_map<NodePair, LinkProperties, NodePairHash> link_properties_;
    
    // Helper methods
    bool nodesAreConnected(int source, int destination) const;
    double calculateTransferTime(int source, int destination, size_t bytes) const;
};

} // namespace dist_sim
