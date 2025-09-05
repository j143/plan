import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx

class Visualizer:
    """
    Visualization tools for distributed training simulation results.
    
    This class provides methods to visualize various aspects of distributed training
    simulations, including communication patterns, compute performance, memory usage,
    and overall training progress.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6), style: str = "darkgrid"):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size for plots
            style: Seaborn style for plots (darkgrid, whitegrid, dark, white, ticks)
        """
        self.figsize = figsize
        sns.set_style(style)
        plt.rcParams["figure.figsize"] = figsize
        
    def plot_training_timeline(self, 
                              events: List[Dict],
                              show_communication: bool = True,
                              show_computation: bool = True,
                              show_memory: bool = True) -> plt.Figure:
        """
        Plot a timeline of training events.
        
        Args:
            events: List of event dictionaries with keys:
                   - 'start_time': Start time of event in ms
                   - 'end_time': End time of event in ms
                   - 'type': Type of event ('compute', 'communication', 'memory')
                   - 'name': Name of the event
                   - 'device_id': ID of the device where event occurred
            show_communication: Whether to show communication events
            show_computation: Whether to show computation events
            show_memory: Whether to show memory events
            
        Returns:
            Matplotlib Figure object
        """
        # Filter events based on parameters
        filtered_events = []
        for event in events:
            if (event['type'] == 'communication' and show_communication or
                event['type'] == 'compute' and show_computation or
                event['type'] == 'memory' and show_memory):
                filtered_events.append(event)
        
        if not filtered_events:
            raise ValueError("No events to display after filtering")
        
        # Determine number of devices
        device_ids = set(event['device_id'] for event in filtered_events)
        num_devices = max(device_ids) + 1
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Define colors for different event types
        colors = {
            'compute': 'tab:blue',
            'communication': 'tab:red',
            'memory': 'tab:green'
        }
        
        # Plot each event as a horizontal bar
        for event in filtered_events:
            y_position = event['device_id']
            start = event['start_time']
            duration = event['end_time'] - event['start_time']
            
            ax.barh(y_position, duration, left=start, height=0.5,
                   color=colors[event['type']], alpha=0.7,
                   label=event['type'])
            
            # Add text label if duration is long enough
            if duration > (ax.get_xlim()[1] - ax.get_xlim()[0]) / 50:
                ax.text(start + duration/2, y_position, event['name'],
                       ha='center', va='center', fontsize=8)
        
        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        # Set labels and title
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Device ID')
        ax.set_title('Distributed Training Timeline')
        ax.set_yticks(range(num_devices))
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        return fig
        
    def plot_communication_graph(self, 
                                adjacency_matrix: np.ndarray,
                                node_labels: Optional[List[str]] = None,
                                edge_weights: Optional[np.ndarray] = None,
                                layout: str = 'spring') -> plt.Figure:
        """
        Plot the communication graph between nodes.
        
        Args:
            adjacency_matrix: Square matrix representing connections between nodes
            node_labels: Optional list of labels for nodes
            edge_weights: Optional matrix of edge weights (e.g., bandwidth)
            layout: Graph layout algorithm ('spring', 'circular', 'kamada_kawai', etc.)
            
        Returns:
            Matplotlib Figure object
        """
        # Create graph from adjacency matrix
        G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph())
        
        # Set node labels
        if node_labels:
            mapping = {i: label for i, label in enumerate(node_labels)}
            G = nx.relabel_nodes(G, mapping)
        
        # Set edge weights
        if edge_weights is not None:
            for i, j in G.edges():
                G[i][j]['weight'] = edge_weights[i][j]
                
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(G)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)  # Default to spring layout
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', ax=ax)
        
        # Draw edges with width based on weight
        if edge_weights is not None:
            # Normalize edge weights for width
            edge_list = list(G.edges())
            weights = [G[u][v]['weight'] for u, v in edge_list]
            min_weight, max_weight = min(weights), max(weights)
            norm_weights = [(w - min_weight) / (max_weight - min_weight) * 3 + 1 for w in weights]
            
            nx.draw_networkx_edges(G, pos, width=norm_weights, edge_color='gray', 
                                 arrows=True, arrowsize=15, ax=ax)
        else:
            nx.draw_networkx_edges(G, pos, width=1.5, edge_color='gray', 
                                 arrows=True, arrowsize=15, ax=ax)
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', ax=ax)
        
        # Set title and remove axis
        ax.set_title('Communication Graph')
        ax.axis('off')
        
        return fig
    
    def plot_scaling_efficiency(self, 
                               num_devices: List[int], 
                               throughput: List[float], 
                               efficiency: Optional[List[float]] = None,
                               reference_line: bool = True) -> plt.Figure:
        """
        Plot scaling efficiency as number of devices increases.
        
        Args:
            num_devices: List of device counts
            throughput: List of throughput measurements (e.g., samples/sec)
            efficiency: Optional list of scaling efficiency percentages
            reference_line: Whether to show ideal linear scaling reference
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax1 = plt.subplots(figsize=self.figsize)
        
        # Plot throughput
        color = 'tab:blue'
        ax1.set_xlabel('Number of Devices')
        ax1.set_ylabel('Throughput (samples/sec)', color=color)
        ax1.plot(num_devices, throughput, 'o-', color=color, label='Throughput')
        
        # Plot reference line for ideal scaling
        if reference_line:
            # Use first measurement as baseline
            baseline = throughput[0] / num_devices[0]
            ideal = [n * baseline for n in num_devices]
            ax1.plot(num_devices, ideal, '--', color='gray', label='Ideal Scaling')
        
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Plot efficiency on secondary y-axis if provided
        if efficiency:
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Efficiency (%)', color=color)
            ax2.plot(num_devices, efficiency, 's-', color=color, label='Efficiency')
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_ylim([0, 105])  # Efficiency is 0-100%
            
            # Add legend for both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax1.legend(loc='upper left')
        
        ax1.set_title('Scaling Performance')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        return fig
    
    def plot_memory_usage(self, 
                         device_ids: List[int],
                         memory_data: Dict[str, List[float]],
                         memory_capacity: Optional[List[float]] = None) -> plt.Figure:
        """
        Plot memory usage across devices.
        
        Args:
            device_ids: List of device IDs
            memory_data: Dictionary with keys being memory types and values being lists of usage
            memory_capacity: Optional list of memory capacity for each device
            
        Returns:
            Matplotlib Figure object
        """
        # Validate input
        if not all(len(v) == len(device_ids) for v in memory_data.values()):
            raise ValueError("All memory data lists must have same length as device_ids")
            
        if memory_capacity and len(memory_capacity) != len(device_ids):
            raise ValueError("memory_capacity list must have same length as device_ids")
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Width of a bar
        bar_width = 0.8 / len(memory_data)
        
        # Positions of the bars on the x-axis
        positions = np.arange(len(device_ids))
        
        # Plot each memory type as a group of bars
        for i, (memory_type, values) in enumerate(memory_data.items()):
            offset = -0.4 + (i + 0.5) * bar_width
            ax.bar(positions + offset, values, bar_width, label=memory_type, alpha=0.8)
        
        # Plot memory capacity line if provided
        if memory_capacity:
            ax.plot(positions, memory_capacity, 'k--', label='Capacity')
        
        # Set labels and title
        ax.set_xlabel('Device ID')
        ax.set_ylabel('Memory Usage (GB)')
        ax.set_title('Memory Usage by Device')
        ax.set_xticks(positions)
        ax.set_xticklabels([str(d) for d in device_ids])
        ax.legend()
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        return fig
    
    def plot_training_progress(self, 
                              epochs: List[int], 
                              train_metric: List[float],
                              val_metric: Optional[List[float]] = None,
                              metric_name: str = 'Loss',
                              log_scale: bool = False) -> plt.Figure:
        """
        Plot training progress over epochs.
        
        Args:
            epochs: List of epoch numbers
            train_metric: List of training metric values
            val_metric: Optional list of validation metric values
            metric_name: Name of the metric being plotted
            log_scale: Whether to use log scale for y-axis
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot training metric
        ax.plot(epochs, train_metric, 'o-', color='tab:blue', label=f'Training {metric_name}')
        
        # Plot validation metric if provided
        if val_metric:
            if len(val_metric) != len(epochs):
                raise ValueError("val_metric list must have same length as epochs")
            ax.plot(epochs, val_metric, 's-', color='tab:orange', label=f'Validation {metric_name}')
        
        # Set labels and title
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric_name)
        ax.set_title(f'Training Progress: {metric_name} vs. Epochs')
        
        # Set log scale if requested
        if log_scale:
            ax.set_yscale('log')
        
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig
    
    def plot_compute_utilization(self,
                                timeline: List[float],
                                utilization_data: Dict[int, List[float]],
                                device_names: Optional[List[str]] = None) -> plt.Figure:
        """
        Plot compute utilization over time.
        
        Args:
            timeline: List of time points
            utilization_data: Dictionary mapping device IDs to lists of utilization values
            device_names: Optional list of device names for legend
            
        Returns:
            Matplotlib Figure object
        """
        # Validate input
        if not all(len(v) == len(timeline) for v in utilization_data.values()):
            raise ValueError("All utilization data lists must have same length as timeline")
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot utilization for each device
        for device_id, utilization in utilization_data.items():
            label = f"Device {device_id}"
            if device_names and device_id < len(device_names):
                label = device_names[device_id]
                
            ax.plot(timeline, utilization, '-', label=label)
        
        # Set labels and title
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Utilization (%)')
        ax.set_title('Compute Utilization Over Time')
        ax.set_ylim([0, 105])  # Utilization is 0-100%
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig
    
    def plot_heatmap(self,
                    data: np.ndarray,
                    x_labels: Optional[List[str]] = None,
                    y_labels: Optional[List[str]] = None,
                    title: str = 'Heatmap',
                    cmap: str = 'viridis',
                    annotate: bool = True) -> plt.Figure:
        """
        Plot a heatmap of data.
        
        Args:
            data: 2D array of values
            x_labels: Optional labels for x-axis
            y_labels: Optional labels for y-axis
            title: Plot title
            cmap: Colormap name
            annotate: Whether to annotate cells with values
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        sns.heatmap(data, annot=annotate, fmt=".2f", cmap=cmap, 
                   xticklabels=x_labels, yticklabels=y_labels, ax=ax)
        
        # Set labels and title
        ax.set_title(title)
        
        return fig
    
    def plot_3d_surface(self,
                       x: np.ndarray,
                       y: np.ndarray,
                       z: np.ndarray,
                       xlabel: str = 'X',
                       ylabel: str = 'Y',
                       zlabel: str = 'Z',
                       title: str = '3D Surface',
                       cmap: str = 'viridis') -> plt.Figure:
        """
        Plot a 3D surface.
        
        Args:
            x: 2D array of x coordinates
            y: 2D array of y coordinates
            z: 2D array of z values
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            zlabel: Label for z-axis
            title: Plot title
            cmap: Colormap name
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create surface plot
        surf = ax.plot_surface(x, y, z, cmap=cmap, edgecolor='none', alpha=0.8)
        
        # Set labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        return fig
    
    def plot_parallel_coordinates(self,
                                data: pd.DataFrame,
                                class_column: str,
                                title: str = 'Parallel Coordinates Plot') -> plt.Figure:
        """
        Plot parallel coordinates visualization.
        
        Args:
            data: DataFrame with features and class column
            class_column: Name of column containing class labels
            title: Plot title
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create parallel coordinates plot
        pd.plotting.parallel_coordinates(data, class_column, ax=ax)
        
        # Set title
        ax.set_title(title)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Adjust layout to prevent clipping of labels
        plt.tight_layout()
        
        return fig
    
    def save_figure(self, fig: plt.Figure, filename: str, dpi: int = 300, bbox_inches: str = 'tight'):
        """
        Save figure to file.
        
        Args:
            fig: Matplotlib Figure to save
            filename: Output filename
            dpi: Resolution in dots per inch
            bbox_inches: Bounding box setting
        """
        fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Figure saved to {filename}")
