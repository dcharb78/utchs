"""
Torus visualization module for the UTCHS framework.

This module implements visualization tools for the toroidal structure of the UTCHS system,
including 3D toroidal geometry and position networks.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union

class TorusVisualizer:
    """
    Visualizes the toroidal structure of the UTCHS system.
    """
    def __init__(self, output_dir: str = 'visualizations'):
        """
        Initialize the torus visualizer.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up figure style
        plt.style.use('dark_background')
        self.dpi = 300
        
        # Color maps
        self.position_cmap = plt.cm.rainbow
        self.structure_cmap = plt.cm.viridis
        self.cycle_cmap = plt.cm.plasma

    def visualize_torus(
        self,
        torus,
        save: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize the toroidal structure.

        Args:
            torus: Torus instance
            save: Whether to save the figure
            filename: Custom filename (default: auto-generated)

        Returns:
            matplotlib Figure
        """
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Torus parameters
        R = torus.major_radius  # Major radius
        r = torus.minor_radius  # Minor radius
        
        # Generate points on torus surface
        u = np.linspace(0, 2*np.pi, 50)
        v = np.linspace(0, 2*np.pi, 50)
        u_grid, v_grid = np.meshgrid(u, v)
        
        x = (R + r * np.cos(v_grid)) * np.cos(u_grid)
        y = (R + r * np.cos(v_grid)) * np.sin(u_grid)
        z = r * np.sin(v_grid)
        
        # Plot torus surface
        ax.plot_surface(x, y, z, color='lightblue', alpha=0.3, edgecolor='gray', linewidth=0.1)
        
        # Plot positions in the torus
        position_colors = self.position_cmap(np.linspace(0, 1, 13))
        
        # Create legend entries for position numbers
        legend_elements = []
        for i in range(13):
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=position_colors[i], 
                          markersize=8, 
                          label=f'Position {i+1}')
            )
        
        # Track all positions for display
        position_list = []
        
        for structure in torus.structures:
            # Get color for this structure
            structure_color = self.structure_cmap(structure.id / len(torus.structures))
            
            for cycle in structure.cycles:
                # Get color for this cycle
                cycle_color = self.cycle_cmap(cycle.id / len(structure.cycles))
                
                for position in cycle.positions:
                    # Calculate parameters for position on torus
                    cycle_angle = 2*np.pi * cycle.id / len(structure.cycles)
                    structure_angle = 2*np.pi * structure.id / len(torus.structures)
                    position_angle = 2*np.pi * position.number / 13
                    
                    # Calculate angle on major circle
                    u_val = structure_angle + cycle_angle / len(structure.cycles)
                    
                    # Calculate angle on minor circle
                    v_val = position_angle
                    
                    # Calculate 3D coordinates on torus
                    x_position = (R + r * np.cos(v_val)) * np.cos(u_val)
                    y_position = (R + r * np.cos(v_val)) * np.sin(u_val)
                    z_position = r * np.sin(v_val)
                    
                    # Store position info
                    position_list.append({
                        'position': position,
                        'coordinates': (x_position, y_position, z_position),
                        'cycle': cycle,
                        'structure': structure
                    })
                    
                    # Scale dot size by energy level
                    size = 30 + 70 * (position.energy_level / 10)
                    
                    # Plot position with color based on position number
                    ax.scatter(
                        x_position, y_position, z_position,
                        color=position_colors[position.number-1],
                        s=size,
                        alpha=0.7,
                        edgecolor='white'
                    )
        
        # Add connections between positions within each cycle
        for structure in torus.structures:
            for cycle in structure.cycles:
                for i in range(len(cycle.positions)):
                    # Connect to next position in cycle
                    position1 = cycle.positions[i]
                    position2 = cycle.positions[(i+1) % 13]
                    
                    # Find coordinates
                    coordinates1 = None
                    coordinates2 = None
                    
                    for position_info in position_list:
                        if position_info['position'] == position1:
                            coordinates1 = position_info['coordinates']
                        if position_info['position'] == position2:
                            coordinates2 = position_info['coordinates']
                            
                    if coordinates1 and coordinates2:
                        ax.plot(
                            [coordinates1[0], coordinates2[0]],
                            [coordinates1[1], coordinates2[1]],
                            [coordinates1[2], coordinates2[2]],
                            color='white',
                            alpha=0.3,
                            linewidth=1
                        )
        
        # Add legend for positions
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Torus {torus.id} Structure')
        
        # Equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Save figure
        if save:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'torus_{torus.id}_{timestamp}.png'
                
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved figure to {filepath}")
            
        return fig

    def visualize_position_network(
        self,
        system,
        save: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize the network of positions across tori.

        Args:
            system: UTCHSSystem instance
            save: Whether to save the figure
            filename: Custom filename (default: auto-generated)

        Returns:
            matplotlib Figure
        """
        # Create figure using networkx
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes for each position in each torus
        node_positions = {}
        node_colors = []
        node_sizes = []
        
        for torus_idx, torus in enumerate(system.tori):
            torus_y = -torus_idx * 2  # Vertical spacing between tori
            
            for structure_idx, structure in enumerate(torus.structures):
                structure_x_offset = (structure_idx - len(structure.cycles)/2) * 5
                
                for cycle_idx, cycle in enumerate(structure.cycles):
                    cycle_y_offset = cycle_idx * 0.5
                    
                    for position in cycle.positions:
                        # Create node ID
                        node_id = f"T{torus.id}_S{structure.id}_C{cycle.id}_P{position.number}"
                        
                        # Calculate node position in visualization
                        angle = 2 * np.pi * position.number / 13
                        radius = 2
                        x = structure_x_offset + radius * np.cos(angle)
                        y = torus_y + cycle_y_offset + radius * np.sin(angle)
                        
                        # Add node
                        G.add_node(node_id)
                        node_positions[node_id] = (x, y)
                        
                        # Color based on position number
                        color_val = (position.number - 1) / 12
                        node_colors.append(plt.cm.rainbow(color_val))
                        
                        # Size based on energy level
                        size = 100 + 500 * (position.energy_level / 10)
                        node_sizes.append(size)
                        
                        # Add edges between adjacent positions within cycle
                        next_pos = (position.number % 13) + 1
                        for other_position in cycle.positions:
                            if other_position.number == next_pos:
                                next_node_id = f"T{torus.id}_S{structure.id}_C{cycle.id}_P{next_pos}"
                                G.add_edge(node_id, next_node_id, color='gray', weight=1)
                                
                        # Add edges for position 10 to position 1 connections (seed transition)
                        if position.number == 10 and torus_idx < len(system.tori) - 1:
                            # Find position 1 in the next torus
                            next_torus = system.tori[torus_idx + 1]
                            next_node_id = f"T{next_torus.id}_S1_C1_P1"
                            if next_node_id in G:
                                G.add_edge(node_id, next_node_id, color='red', weight=2)
        
        # Draw network
        edges = G.edges()
        edge_colors = [G[u][v]['color'] for u, v in edges]
        edge_widths = [G[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx(
            G,
            pos=node_positions,
            with_labels=False,
            node_color=node_colors,
            node_size=node_sizes,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.7
        )
        
        # Add title
        plt.title('UTCHS Position Network')
        plt.axis('off')
        
        # Add legends
        # Position colors
        position_legend_elements = []
        for i in range(1, 14):
            if i in [3, 6, 9]:  # Vortex positions
                marker = '*'
                label = f'Position {i} (Vortex)'
            elif i in [1, 2, 3, 5, 7, 11, 13]:  # Prime positions
                marker = 's'
                label = f'Position {i} (Prime)'
            else:
                marker = 'o'
                label = f'Position {i}'
                
            position_legend_elements.append(
                plt.Line2D([0], [0], marker=marker, color='w', 
                          markerfacecolor=plt.cm.rainbow((i-1)/12), 
                          markersize=8, 
                          label=label)
            )
            
        # Edge types
        edge_legend_elements = [
            plt.Line2D([0], [0], color='gray', lw=1, label='Sequential'),
            plt.Line2D([0], [0], color='red', lw=2, label='Seed Transition')
        ]
        
        # Create two separate legends
        leg1 = plt.legend(handles=position_legend_elements, loc='upper left', 
                        title='Position Types', fontsize=8)
        plt.gca().add_artist(leg1)
        plt.legend(handles=edge_legend_elements, loc='upper right', 
                 title='Connection Types', fontsize=8)
        
        # Save figure
        if save:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'position_network_{timestamp}.png'
                
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved figure to {filepath}")
            
        return fig

    def visualize_resonance_network(
        self,
        system,
        save: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize the resonance network within the UTCHS system.

        Args:
            system: UTCHSSystem instance
            save: Whether to save the figure
            filename: Custom filename (default: auto-generated)

        Returns:
            matplotlib Figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create graph
        G = nx.Graph()
        
        # Collect resonant pairs from all cycles
        resonant_pairs = []
        node_map = {}  # Maps (torus_id, structure_id, cycle_id, position_number) to node_id
        node_counter = 0
        
        # Add nodes and collect resonant pairs
        for torus_idx, torus in enumerate(system.tori):
            for structure_idx, structure in enumerate(torus.structures):
                for cycle_idx, cycle in enumerate(structure.cycles):
                    # Find resonant pairs within this cycle
                    cycle_pairs = cycle.find_resonant_positions()
                    
                    # Add nodes for positions in this cycle
                    for position in cycle.positions:
                        # Create a unique identifier for the position
                        pos_id = (torus.id, structure.id, cycle.id, position.number)
                        
                        # Check if this position is already mapped
                        if pos_id not in node_map:
                            node_id = f"N{node_counter}"
                            node_map[pos_id] = node_id
                            node_counter += 1
                            
                            # Add node to graph
                            G.add_node(
                                node_id, 
                                position=position.number,
                                torus=torus.id,
                                structure=structure.id,
                                cycle=cycle.id,
                                energy=position.energy_level,
                                phase=position.phase,
                                is_vortex=(position.number in [3, 6, 9]),
                                is_prime=(position.number in [1, 2, 3, 5, 7, 11, 13])
                            )
                    
                    # Add resonant pairs from this cycle
                    for pos1_num, pos2_num, resonance in cycle_pairs:
                        resonant_pairs.append({
                            'torus': torus.id,
                            'structure': structure.id,
                            'cycle': cycle.id,
                            'position1': pos1_num,
                            'position2': pos2_num,
                            'resonance': resonance
                        })
        
        # Add edges for resonant pairs
        for pair in resonant_pairs:
            # Get node IDs
            pos1_id = (pair['torus'], pair['structure'], pair['cycle'], pair['position1'])
            pos2_id = (pair['torus'], pair['structure'], pair['cycle'], pair['position2'])
            
            if pos1_id in node_map and pos2_id in node_map:
                node1 = node_map[pos1_id]
                node2 = node_map[pos2_id]
                
                # Add edge with resonance as weight
                G.add_edge(
                    node1, node2,
                    weight=pair['resonance'],
                    color=plt.cm.plasma(pair['resonance'])
                )
        
        # Calculate node positions using spring layout
        pos = nx.spring_layout(G, k=0.3, iterations=50)
        
        # Draw nodes with appropriate colors and sizes
        node_colors = []
        node_sizes = []
        
        for node, data in G.nodes(data=True):
            # Color based on position
            if data['is_vortex']:
                color = 'red'
            elif data['is_prime'] and not data['is_vortex']:
                color = 'green'
            else:
                color = plt.cm.rainbow((data['position']-1)/12)
                
            node_colors.append(color)
            
            # Size based on energy
            size = 100 + 300 * (data['energy'] / 10)
            node_sizes.append(size)
        
        # Draw edges with weights determining width and color
        edges = G.edges()
        edge_colors = [G[u][v]['color'] for u, v in edges]
        edge_weights = [G[u][v]['weight'] * 5 for u, v in edges]
        
        # Draw the network
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.7
        )
        
        nx.draw_networkx_edges(
            G, pos,
            edge_color=edge_colors,
            width=edge_weights,
            alpha=0.5
        )
        
        # Add node labels (position numbers)
        node_labels = {node: data['position'] for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_color='white')
        
        # Add title and turn off axis
        plt.title(f'UTCHS Resonance Network (Pairs: {len(edges)})')
        plt.axis('off')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=10, label='Vortex Position (3, 6, 9)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                      markersize=10, label='Prime Position'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                      markersize=10, label='Regular Position'),
            plt.Line2D([0], [0], color=plt.cm.plasma(0.9), lw=3, label='Strong Resonance'),
            plt.Line2D([0], [0], color=plt.cm.plasma(0.5), lw=2, label='Medium Resonance'),
            plt.Line2D([0], [0], color=plt.cm.plasma(0.1), lw=1, label='Weak Resonance')
        ]
        
        plt.legend(handles=legend_elements, loc='lower right')
        
        # Save figure
        if save:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'resonance_network_{timestamp}.png'
                
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved figure to {filepath}")
            
        return fig

    def visualize_torus_metrics(
        self,
        system,
        save: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize metrics across all tori in the system.

        Args:
            system: UTCHSSystem instance
            save: Whether to save the figure
            filename: Custom filename (default: auto-generated)

        Returns:
            matplotlib Figure
        """
        # Collect metrics from all tori
        torus_ids = []
        coherence_values = []
        stability_values = []
        energy_values = []
        
        for torus in system.tori:
            torus_ids.append(torus.id)
            coherence_values.append(torus.phase_coherence)
            stability_values.append(torus.stability_metric)
            energy_values.append(sum(pos.energy_level for structure in torus.structures 
                                for cycle in structure.cycles for pos in cycle.positions))
            
        # Normalize energy values
        max_energy = max(energy_values) if energy_values else 1.0
        energy_values = [e / max_energy for e in energy_values]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot metrics bar chart
        x = np.arange(len(torus_ids))
        width = 0.25
        
        ax1.bar(x - width, coherence_values, width, label='Phase Coherence', color='blue', alpha=0.7)
        ax1.bar(x, stability_values, width, label='Stability', color='green', alpha=0.7)
        ax1.bar(x + width, energy_values, width, label='Normalized Energy', color='red', alpha=0.7)
        
        ax1.set_xlabel('Torus ID')
        ax1.set_ylabel('Metric Value')
        ax1.set_title('Torus Metrics Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(torus_ids)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot line chart showing metrics progression
        ax2.plot(torus_ids, coherence_values, 'o-', label='Phase Coherence', color='blue')
        ax2.plot(torus_ids, stability_values, 's-', label='Stability', color='green')
        ax2.plot(torus_ids, energy_values, 'D-', label='Normalized Energy', color='red')
        
        ax2.set_xlabel('Torus ID')
        ax2.set_ylabel('Metric Value')
        ax2.set_title('Metrics Evolution Across Tori')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        if save:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'torus_metrics_{timestamp}.png'
                
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved figure to {filepath}")
            
        return fig

    def visualize_prime_anchor_network(
        self,
        system,
        save: bool = True,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize the network of prime torsional anchors.

        Args:
            system: UTCHSSystem instance
            save: Whether to save the figure
            filename: Custom filename (default: auto-generated)

        Returns:
            matplotlib Figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes for prime positions only
        prime_positions = [1, 2, 3, 5, 7, 11, 13]
        node_map = {}  # Maps (torus_id, structure_id, cycle_id, position_number) to node_id
        node_counter = 0
        
        # Collect prime position nodes
        for torus_idx, torus in enumerate(system.tori):
            for structure_idx, structure in enumerate(torus.structures):
                for cycle_idx, cycle in enumerate(structure.cycles):
                    for position in cycle.positions:
                        # Only include prime positions
                        if position.number in prime_positions:
                            # Create a unique identifier for the position
                            pos_id = (torus.id, structure.id, cycle.id, position.number)
                            
                            # Check if this position is already mapped
                            if pos_id not in node_map:
                                node_id = f"N{node_counter}"
                                node_map[pos_id] = node_id
                                node_counter += 1
                                
                                # Add node to graph
                                G.add_node(
                                    node_id, 
                                    position=position.number,
                                    torus=torus.id,
                                    structure=structure.id,
                                    cycle=cycle.id,
                                    energy=position.energy_level,
                                    phase=position.phase
                                )
        
        # Add edges between prime positions that resonate
        for torus in system.tori:
            for structure in torus.structures:
                for cycle in structure.cycles:
                    # Get prime positions in this cycle
                    prime_pos = [p for p in cycle.positions if p.number in prime_positions]
                    
                    # Check resonance between each pair
                    for i, pos1 in enumerate(prime_pos):
                        for pos2 in prime_pos[i+1:]:
                            resonance = pos1.calculate_resonance_with(pos2)
                            
                            # Add edge if resonance is strong enough
                            if resonance > 0.6:
                                pos1_id = (torus.id, structure.id, cycle.id, pos1.number)
                                pos2_id = (torus.id, structure.id, cycle.id, pos2.number)
                                
                                if pos1_id in node_map and pos2_id in node_map:
                                    node1 = node_map[pos1_id]
                                    node2 = node_map[pos2_id]
                                    
                                    G.add_edge(
                                        node1, node2,
                                        weight=resonance,
                                        color=plt.cm.plasma(resonance)
                                    )
        
        # Calculate node positions using a circular layout for better visualization
        pos = nx.spring_layout(G, k=0.5)
        
        # Draw nodes with appropriate colors and sizes
        node_colors = []
        node_sizes = []
        
        # Custom colors for specific prime positions
        prime_colors = {
            1: 'gold',       # Position 1 (first prime)
            2: 'orange',     # Position 2 (second prime)
            3: 'red',        # Position 3 (third prime, also vortex)
            5: 'purple',     # Position 5
            7: 'blue',       # Position 7
            11: 'green',     # Position 11
            13: 'cyan'       # Position 13
        }
        
        for node, data in G.nodes(data=True):
            position_number = data['position']
            color = prime_colors[position_number]
            node_colors.append(color)
            
            # Size based on energy and importance
            size = 100 + 300 * (data['energy'] / 10)
            # Make Position 1 (Prime Unity) larger
            if position_number == 1:
                size *= 1.5
            node_sizes.append(size)
        
        # Draw edges with weights determining width
        edges = G.edges()
        edge_colors = [G[u][v]['color'] for u, v in edges]
        edge_weights = [G[u][v]['weight'] * 5 for u, v in edges]
        
        # Draw the network
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.7,
            edgecolors='white'
        )
        
        nx.draw_networkx_edges(
            G, pos,
            edge_color=edge_colors,
            width=edge_weights,
            alpha=0.5
        )
        
        # Add node labels (position numbers)
        node_labels = {node: data['position'] for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color='white')
        
        # Add title and turn off axis
        plt.title('Prime Torsional Anchor Network')
        plt.axis('off')
        
        # Add legend for prime positions
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                      markersize=10, label=f'Position {num}')
            for num, color in prime_colors.items()
        ]
        
        plt.legend(handles=legend_elements, loc='lower right')
        
        # Save figure
        if save:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'prime_anchor_network_{timestamp}.png'
                
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved figure to {filepath}")
            
        return fig