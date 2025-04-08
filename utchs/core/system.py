"""
System module for the UTCHS framework.

This module implements the UTCHSSystem class, which serves as the main controller
for the entire UTCHS framework, coordinating the evolution of the hierarchical structure
and its interaction with the phase and energy fields.
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import numpy.typing as npt

from ..core.position import Position
from ..core.torus import Torus
from ..math.phase_field import PhaseField
from ..fields.energy_field import EnergyField
from ..utils.logging_config import configure_logging, get_logger, SystemError, ConfigurationError

# Get logger for this module
logger = get_logger(__name__)

class UTCHSSystem:
    """
    Main class representing the complete UTCHS framework.
    
    This class coordinates the evolution of the hierarchical structure (positions, cycles,
    structures, and tori) and manages their interaction with the phase and energy fields.
    
    Attributes:
        config (Dict): Configuration dictionary with simulation parameters
        current_tick (int): Current simulation tick
        tori (List[Torus]): List of torus objects in the system
        current_torus_idx (int): Index of the currently active torus
        phase_field (PhaseField): Phase field manager
        energy_field (EnergyField): Energy field manager
        global_coherence (float): Global coherence metric
        global_stability (float): Global stability metric
        energy_level (float): Current energy level
        phase_recursion_depth (int): Current phase recursion depth
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the UTCHS system with configuration parameters.

        Args:
            config: Configuration dictionary with simulation parameters
            
        Raises:
            ConfigurationError: If required configuration parameters are missing or invalid
        """
        try:
            # Validate configuration
            self._validate_config(config)
            
            self.config = config
            self.current_tick = 0
            self.tori = [Torus(1)]  # Start with one torus
            self.current_torus_idx = 0
            
            # Initialize fields
            self.phase_field = PhaseField(config)
            self.energy_field = EnergyField(config)
            
            # Set up logger
            self.logger = self._setup_logger()
            
            # System metrics
            self.global_coherence = 1.0
            self.global_stability = 0.0
            self.energy_level = 0.0
            self.phase_recursion_depth = 1
            
            # Store history
            self.history_length = config.get('history_length', 100)
            self.state_history = []
            
            # Record initial state
            self._record_state()
            
            self.logger.info("UTCHS system initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize UTCHS system: {str(e)}")
            raise SystemError(f"System initialization failed: {str(e)}") from e

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration dictionary.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        required_params = ['grid_size', 'grid_spacing']
        for param in required_params:
            if param not in config:
                raise ConfigurationError(f"Missing required configuration parameter: {param}")
                
        # Validate grid size
        if not isinstance(config['grid_size'], (tuple, list)) or len(config['grid_size']) != 3:
            raise ConfigurationError("grid_size must be a tuple or list of length 3")
            
        # Validate grid spacing
        if not isinstance(config['grid_spacing'], (int, float)) or config['grid_spacing'] <= 0:
            raise ConfigurationError("grid_spacing must be a positive number")
            
        logger.debug("Configuration validation successful")

    def _setup_logger(self) -> logging.Logger:
        """
        Set up the system logger.
        
        Returns:
            Configured logger instance
        """
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "utchs_simulation.log")
        
        return configure_logging(
            name="utchs.system",
            log_level=self.config.get('log_level', 'INFO'),
            log_file=log_file
        )

    def advance_tick(self) -> Dict:
        """
        Advance the system by one tick.
        
        Returns:
            Dict with current system state
            
        Raises:
            SystemError: If tick advancement fails
        """
        try:
            self.current_tick += 1
            self.logger.debug(f"Advancing to tick {self.current_tick}")
            
            # Get current position in hierarchy
            current_torus = self.tori[self.current_torus_idx]
            current_structure = current_torus.get_current_structure()
            current_cycle = current_structure.get_current_cycle()
            
            # Advance position
            cycle_completed, current_position = current_cycle.advance()
            
            # Check for cycle completion
            if cycle_completed:
                self.logger.info(f"Cycle {current_cycle.id} in Structure {current_structure.id} completed")
                structure_completed = current_structure.advance_cycle()
                
                # Check for structure completion
                if structure_completed:
                    self.logger.info(f"Structure {current_structure.id} in Torus {current_torus.id} completed")
                    torus_completed = current_torus.advance_structure()
                    
                    # Check for torus completion
                    if torus_completed:
                        self.logger.info(f"Torus {current_torus.id} completed")
                        seed_data = current_torus.extract_seed()
                        new_torus = Torus(len(self.tori) + 1, seed_data=seed_data)
                        self.tori.append(new_torus)
                        self.current_torus_idx += 1
                        self.logger.info(f"New Torus {new_torus.id} created from seed")
                        
                        # Increment phase recursion depth at each torus transition
                        self.phase_recursion_depth += 1
            
            # Update fields
            self._update_fields(current_position)
            
            # Apply field effects back to the hierarchical structure
            self._apply_field_effects()
            
            # Update system metrics
            self._update_system_metrics()
            
            # Record state
            state = self._record_state()
            
            self.logger.debug(f"Tick {self.current_tick}: {state}")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error advancing tick {self.current_tick}: {str(e)}")
            raise SystemError(f"Tick advancement failed: {str(e)}") from e

    def _update_fields(self, current_position) -> None:
        """
        Update phase and energy fields based on current position.
        
        Args:
            current_position: Current position in the hierarchy
            
        Raises:
            SystemError: If field update fails
        """
        try:
            # Get Möbius parameters for this tick
            mobius_parameters = self._calculate_mobius_parameters(current_position)
            
            # Update phase field
            self.phase_field.update(current_position, mobius_parameters)
            
            # Update energy field
            self.energy_field.update(self.phase_field)
            
        except Exception as e:
            self.logger.error(f"Error updating fields: {str(e)}")
            raise SystemError(f"Field update failed: {str(e)}") from e

    def _calculate_mobius_parameters(self, position: Position) -> Dict[str, complex]:
        """
        Calculate Möbius transformation parameters based on current position.

        Args:
            position: Current position in the UTCHS hierarchy

        Returns:
            Dictionary containing Möbius transformation parameters (a, b, c, d)
        """
        # Base parameters
        tick = self.current_tick
        amplitude = 1.0
        amplitude_frequency = 0.1
        
        beta_amplitude = 0.3
        beta_frequency = 0.15
        
        gamma_amplitude = 0.2
        gamma_frequency = 0.2
        
        delta_amplitude = 0.25
        delta_frequency = 0.12
        
        # Use current tick and position properties to create dynamic parameters
        normalized_position = position.number / 13
        
        # Position-specific frequency modulation
        if position.number in [3, 6, 9]:  # Vortex positions
            frequency_multiplier = 1.2
        elif position.number in [1, 2, 3, 5, 7, 11, 13]:  # Prime positions
            frequency_multiplier = 1.1
        else:
            frequency_multiplier = 1.0
            
        # Calculate complex parameters
        alpha = complex(
            amplitude * np.cos(tick * amplitude_frequency * frequency_multiplier),
            amplitude * np.sin(tick * amplitude_frequency * frequency_multiplier)
        )
        
        beta = complex(
            beta_amplitude * np.sin(tick * beta_frequency * frequency_multiplier + normalized_position * np.pi),
            beta_amplitude * np.cos(tick * beta_frequency * frequency_multiplier + normalized_position * np.pi)
        )
        
        gamma = complex(
            gamma_amplitude * np.sin(tick * gamma_frequency * frequency_multiplier + 2 * normalized_position * np.pi),
            gamma_amplitude * np.cos(tick * gamma_frequency * frequency_multiplier + 2 * normalized_position * np.pi)
        )
        
        delta = complex(
            delta_amplitude * np.cos(tick * delta_frequency * frequency_multiplier + np.pi),
            delta_amplitude * np.sin(tick * delta_frequency * frequency_multiplier + np.pi)
        )
        
        # Ensure ad - bc != 0 (valid Möbius transformation)
        determinant = alpha * delta - beta * gamma
        if abs(determinant) < 1e-10:
            beta *= 1.1  # Small adjustment to ensure non-zero determinant
            
        return {
            'a': alpha,
            'b': beta,
            'c': gamma,
            'd': delta
        }

    def _apply_field_effects(self) -> None:
        """
        Apply field effects back to the hierarchical structure.
        
        Raises:
            SystemError: If field effects application fails
        """
        try:
            # Apply effects to all tori
            for torus in self.tori:
                torus.apply_field_effects(self.phase_field, self.energy_field)
                
        except Exception as e:
            self.logger.error(f"Error applying field effects: {str(e)}")
            raise SystemError(f"Field effects application failed: {str(e)}") from e

    def _update_system_metrics(self) -> None:
        """
        Update global system metrics.
        
        Raises:
            SystemError: If metrics update fails
        """
        try:
            # Calculate global coherence
            torus_coherences = [torus.phase_coherence for torus in self.tori]
            self.global_coherence = np.mean(torus_coherences) if torus_coherences else 0.0
            
            # Calculate global stability
            torus_stabilities = [torus.stability_metric for torus in self.tori]
            self.global_stability = np.mean(torus_stabilities) if torus_stabilities else 0.0
            
            # Calculate total energy level
            self.energy_level = self.energy_field.calculate_total_energy()
            
            # Log significant changes
            if self.current_tick % 100 == 0 or abs(self.global_coherence - self.state_history[-1]['global_coherence']) > 0.1:
                self.logger.info(f"System metrics - Coherence: {self.global_coherence:.4f}, Stability: {self.global_stability:.4f}, Energy: {self.energy_level:.4f}, Recursion Depth: {self.phase_recursion_depth}")
                
        except Exception as e:
            self.logger.error(f"Error updating system metrics: {str(e)}")
            raise SystemError(f"System metrics update failed: {str(e)}") from e

    def _record_state(self) -> Dict:
        """
        Record the current system state.
        
        Returns:
            State dictionary
            
        Raises:
            SystemError: If state recording fails
        """
        try:
            # Get current position in hierarchy
            current_torus = self.tori[self.current_torus_idx]
            current_structure = current_torus.get_current_structure()
            current_cycle = current_structure.get_current_cycle()
            current_position = current_cycle.get_current_position()
            
            # Create state dictionary
            state = {
                'tick': self.current_tick,
                'position': {
                    'number': current_position.number,
                    'role': current_position.role,
                    'phase': current_position.phase,
                    'energy': current_position.energy_level
                },
                'hierarchy': {
                    'cycle': current_cycle.id,
                    'structure': current_structure.id,
                    'torus': current_torus.id
                },
                'metrics': {
                    'phase_field_energy': self.energy_field.calculate_total_energy(),
                    'torsion_stability': self.phase_field.calculate_torsion_stability() if hasattr(self.phase_field, 'calculate_torsion_stability') else 0.0,
                    'global_coherence': self.global_coherence,
                    'global_stability': self.global_stability,
                    'phase_recursion_depth': self.phase_recursion_depth
                },
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Add to state history
            self.state_history.append(state)
            
            # Limit history length
            if len(self.state_history) > self.history_length:
                self.state_history.pop(0)
                
            return state
            
        except Exception as e:
            self.logger.error(f"Error recording system state: {str(e)}")
            raise SystemError(f"State recording failed: {str(e)}") from e

    def run_simulation(self, num_ticks: int) -> List[Dict]:
        """
        Run the simulation for a specified number of ticks.

        Args:
            num_ticks: Number of ticks to simulate

        Returns:
            List of state dictionaries for each tick
            
        Raises:
            SystemError: If simulation fails
        """
        self.logger.info(f"Starting simulation for {num_ticks} ticks")
        states = []
        
        try:
            for _ in range(num_ticks):
                state = self.advance_tick()
                states.append(state)
                
                # Save checkpoint every 1000 ticks
                if self.current_tick % 1000 == 0:
                    self._save_checkpoint()
                    
        except Exception as e:
            self.logger.error(f"Simulation error at tick {self.current_tick}: {str(e)}")
            raise SystemError(f"Simulation failed: {str(e)}") from e
            
        self.logger.info(f"Simulation completed after {num_ticks} ticks")
        return states

    def _save_checkpoint(self) -> None:
        """
        Save current system state to binary file.
        
        Raises:
            SystemError: If checkpoint saving fails
        """
        try:
            # Create checkpoint directory if it doesn't exist
            checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            filename = os.path.join(checkpoint_dir, f"utchs_checkpoint_{self.current_tick}.npz")
            self.logger.info(f"Saving checkpoint to {filename}")
            
            # Create a serializable state dictionary
            state = {
                'tick': self.current_tick,
                'current_torus_idx': self.current_torus_idx,
                'global_coherence': self.global_coherence,
                'global_stability': self.global_stability,
                'energy_level': self.energy_level,
                'phase_recursion_depth': self.phase_recursion_depth,
                'phase_field': self.phase_field.get_serializable_state(),
                'energy_field': self.energy_field.get_serializable_state()
            }
            
            np.savez_compressed(filename, state=state)
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {str(e)}")
            raise SystemError(f"Checkpoint saving failed: {str(e)}") from e

    def load_checkpoint(self, filename: str) -> None:
        """
        Load system state from a checkpoint file.
        
        Args:
            filename: Path to checkpoint file
            
        Raises:
            SystemError: If checkpoint loading fails
        """
        self.logger.info(f"Loading checkpoint from {filename}")
        
        try:
            data = np.load(filename, allow_pickle=True)
            state = data['state'].item()
            
            self.current_tick = state['tick']
            self.current_torus_idx = state['current_torus_idx']
            self.global_coherence = state['global_coherence']
            self.global_stability = state['global_stability']
            self.energy_level = state['energy_level']
            self.phase_recursion_depth = state['phase_recursion_depth']
            
            self.phase_field.load_state(state['phase_field'])
            self.energy_field.load_state(state['energy_field'])
            
            self.logger.info(f"Checkpoint loaded: tick {self.current_tick}")
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            raise SystemError(f"Checkpoint loading failed: {str(e)}") from e

    def analyze_system_state(self) -> Dict:
        """
        Perform comprehensive analysis of the current system state.
        
        Returns:
            Dictionary with analysis results
            
        Raises:
            SystemError: If analysis fails
        """
        try:
            analysis = {
                "hierarchy_metrics": self._analyze_hierarchy(),
                "field_metrics": self._analyze_fields(),
                "resonance_patterns": self._analyze_resonance(),
                "stability_analysis": self._analyze_stability(),
                "phase_recursion": self._analyze_phase_recursion()
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing system state: {str(e)}")
            raise SystemError(f"System analysis failed: {str(e)}") from e
        
    def _analyze_hierarchy(self) -> Dict:
        """
        Analyze the hierarchical structure.
        
        Returns:
            Dictionary with hierarchy metrics
            
        Raises:
            SystemError: If hierarchy analysis fails
        """
        try:
            metrics = {
                "total_tori": len(self.tori),
                "active_torus": self.current_torus_idx + 1,
                "torus_metrics": [],
                "position_distribution": self._calculate_position_distribution()
            }
            
            # Collect metrics for each torus
            for torus in self.tori:
                torus_metrics = {
                    "id": torus.id,
                    "structures": len(torus.structures),
                    "phase_coherence": torus.phase_coherence,
                    "stability": torus.stability_metric,
                    "energy_distribution": torus.get_energy_distribution().tolist()
                }
                metrics["torus_metrics"].append(torus_metrics)
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing hierarchy: {str(e)}")
            raise SystemError(f"Hierarchy analysis failed: {str(e)}") from e
        
    def _calculate_position_distribution(self) -> Dict:
        """
        Calculate position distribution and characteristics.
        
        Returns:
            Dictionary with position distribution metrics
            
        Raises:
            SystemError: If position distribution calculation fails
        """
        try:
            # Count positions by number
            position_counts = {i+1: 0 for i in range(13)}
            position_energy = {i+1: 0.0 for i in range(13)}
            
            for torus in self.tori:
                for structure in torus.structures:
                    for cycle in structure.cycles:
                        for position in cycle.positions:
                            position_counts[position.number] += 1
                            position_energy[position.number] += position.energy_level
            
            # Calculate percentages
            total = sum(position_counts.values())
            percentages = {
                position: (count / total * 100 if total > 0 else 0) 
                for position, count in position_counts.items()
            }
            
            # Calculate average energy
            average_energy = {
                position: (position_energy[position] / position_counts[position] if position_counts[position] > 0 else 0)
                for position in position_counts.keys()
            }
            
            return {
                "counts": position_counts,
                "percentages": percentages,
                "average_energy": average_energy
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating position distribution: {str(e)}")
            raise SystemError(f"Position distribution calculation failed: {str(e)}") from e
        
    def _analyze_fields(self) -> Dict:
        """
        Analyze the field dynamics.
        
        Returns:
            Dictionary with field metrics
            
        Raises:
            SystemError: If field analysis fails
        """
        try:
            metrics = {
                "phase_field": {
                    "total_energy": self.phase_field.calculate_total_energy(),
                    "singularity_count": len(self.phase_field.singularities),
                    "singularity_distribution": self._analyze_singularity_distribution()
                },
                "energy_field": {
                    "total_energy": self.energy_field.calculate_total_energy(),
                    "peak_energy": self.energy_field.get_peak_energy(),
                    "energy_gradient": self.energy_field.calculate_energy_gradient()
                }
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing fields: {str(e)}")
            raise SystemError(f"Field analysis failed: {str(e)}") from e
        
    def _analyze_singularity_distribution(self) -> Dict:
        """
        Analyze the distribution of phase singularities.
        
        Returns:
            Dictionary with singularity distribution metrics
            
        Raises:
            SystemError: If singularity distribution analysis fails
        """
        try:
            # Count singularities by charge
            positive_count = 0
            negative_count = 0
            
            for singularity in self.phase_field.singularities:
                if singularity['charge'] > 0:
                    positive_count += 1
                else:
                    negative_count += 1
                    
            return {
                "positive_charge": positive_count,
                "negative_charge": negative_count,
                "net_charge": positive_count - negative_count,
                "charge_balance": (positive_count == negative_count)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing singularity distribution: {str(e)}")
            raise SystemError(f"Singularity distribution analysis failed: {str(e)}") from e
        
    def _analyze_resonance(self) -> Dict:
        """
        Analyze resonance patterns in the system.
        
        Returns:
            Dictionary with resonance metrics
            
        Raises:
            SystemError: If resonance analysis fails
        """
        try:
            # Collect resonant position pairs across all tori
            resonant_pairs = []
            
            for torus in self.tori:
                for structure in torus.structures:
                    for cycle in structure.cycles:
                        pairs = cycle.find_resonant_positions()
                        for pos1, pos2, resonance in pairs:
                            resonant_pairs.append({
                                "torus": torus.id,
                                "structure": structure.id,
                                "cycle": cycle.id,
                                "position1": pos1,
                                "position2": pos2,
                                "resonance": resonance
                            })
            
            # Find most common resonant position pairs
            position_pair_count = {}
            for pair in resonant_pairs:
                key = (pair["position1"], pair["position2"])
                if key not in position_pair_count:
                    position_pair_count[key] = 0
                position_pair_count[key] += 1
                
            # Sort by count
            sorted_pairs = sorted(position_pair_count.items(), key=lambda x: x[1], reverse=True)
            
            common_pairs = [
                {"position1": pair[0], "position2": pair[1], "count": count}
                for pair, count in sorted_pairs[:5]  # Top 5
            ]
            
            return {
                "total_resonant_pairs": len(resonant_pairs),
                "average_resonance": np.mean([pair["resonance"] for pair in resonant_pairs]) if resonant_pairs else 0,
                "common_resonant_pairs": common_pairs
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing resonance: {str(e)}")
            raise SystemError(f"Resonance analysis failed: {str(e)}") from e
        
    def _analyze_stability(self) -> Dict:
        """
        Analyze stability metrics.
        
        Returns:
            Dictionary with stability metrics
            
        Raises:
            SystemError: If stability analysis fails
        """
        try:
            # Collect stability history
            stability_history = [state["metrics"]["global_stability"] for state in self.state_history]
            
            # Calculate trends
            stability_trend = 0
            if len(stability_history) > 10:
                recent = np.mean(stability_history[-10:])
                previous = np.mean(stability_history[-20:-10])
                stability_trend = recent - previous
                
            # Analyze vortex points (3-6-9) across all tori
            vortex_metrics = []
            for torus in self.tori:
                for structure in torus.structures:
                    vortex_analysis = structure.analyze_vortex_points()
                    if vortex_analysis:
                        vortex_metrics.append({
                            "torus": torus.id,
                            "structure": structure.id,
                            "energy_ratio": vortex_analysis["energy_ratio"],
                            "phase_coherence": vortex_analysis["phase_coherence"],
                            "stability_factor": vortex_analysis["stability_factor"]
                        })
            
            return {
                "current_stability": self.global_stability,
                "stability_trend": stability_trend,
                "vortex_metrics": vortex_metrics,
                "critical_threshold": self.global_stability < 0.3,  # Flag if stability is low
                "peak_stability": max(stability_history) if stability_history else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing stability: {str(e)}")
            raise SystemError(f"Stability analysis failed: {str(e)}") from e
        
    def _analyze_phase_recursion(self) -> Dict:
        """
        Analyze phase recursion dynamics.
        
        Returns:
            Dictionary with phase recursion metrics
            
        Raises:
            SystemError: If phase recursion analysis fails
        """
        try:
            return {
                "recursion_depth": self.phase_recursion_depth,
                "mobius_iterations": self.current_tick,
                "field_complexity": self._calculate_field_complexity(),
                "self_similarity": self._calculate_self_similarity()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing phase recursion: {str(e)}")
            raise SystemError(f"Phase recursion analysis failed: {str(e)}") from e
        
    def _calculate_field_complexity(self) -> float:
        """
        Calculate complexity measure of the phase field.
        
        Returns:
            Complexity measure
            
        Raises:
            SystemError: If field complexity calculation fails
        """
        try:
            # Simple complexity measure based on singularity count and field gradient
            singularity_count = len(self.phase_field.singularities)
            
            # Calculate average gradient magnitude
            gradient = self.phase_field.calculate_phase_gradient()
            avg_gradient = np.mean(np.sqrt(np.sum(gradient**2, axis=0)))
            
            # Combine measures
            complexity = 0.5 * singularity_count + 0.5 * avg_gradient
            
            return complexity
            
        except Exception as e:
            self.logger.error(f"Error calculating field complexity: {str(e)}")
            raise SystemError(f"Field complexity calculation failed: {str(e)}") from e
        
    def _calculate_self_similarity(self) -> float:
        """
        Calculate measure of self-similarity across scales.
        
        Returns:
            Self-similarity measure
            
        Raises:
            SystemError: If self-similarity calculation fails
        """
        try:
            # Compare energy distributions across tori
            if len(self.tori) < 2:
                return 0.0
                
            similarity_scores = []
            
            # Compare each torus with the next one
            for i in range(len(self.tori) - 1):
                dist1 = self.tori[i].get_energy_distribution()
                dist2 = self.tori[i+1].get_energy_distribution()
                
                # Calculate cosine similarity
                dot_product = np.sum(dist1 * dist2)
                norm1 = np.sqrt(np.sum(dist1**2))
                norm2 = np.sqrt(np.sum(dist2**2))
                
                if norm1 > 0 and norm2 > 0:
                    similarity = dot_product / (norm1 * norm2)
                else:
                    similarity = 0
                    
                similarity_scores.append(similarity)
                
            return np.mean(similarity_scores) if similarity_scores else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating self-similarity: {str(e)}")
            raise SystemError(f"Self-similarity calculation failed: {str(e)}") from e
        
    def get_serializable_state(self) -> Dict:
        """
        Get a serializable representation of the entire system state.
        
        Returns:
            Dictionary with system state
            
        Raises:
            SystemError: If state serialization fails
        """
        try:
            return {
                "tick": self.current_tick,
                "current_torus_idx": self.current_torus_idx,
                "global_coherence": self.global_coherence,
                "global_stability": self.global_stability,
                "energy_level": float(self.energy_level),  # Convert to Python float for JSON compatibility
                "phase_recursion_depth": self.phase_recursion_depth,
                "tori": [torus.get_serializable_state() for torus in self.tori],
                "phase_field": self.phase_field.get_serializable_state(),
                "energy_field": self.energy_field.get_serializable_state()
            }
            
        except Exception as e:
            self.logger.error(f"Error serializing system state: {str(e)}")
            raise SystemError(f"State serialization failed: {str(e)}") from e

    def safe_flatten(self, array_like: Any) -> Union[np.ndarray, List[Any]]:
        """Safely flatten an array-like object.
        
        Args:
            array_like: Array, list, tuple, or scalar to flatten
            
        Returns:
            Flattened array-like object as a 1D numpy array or list
            
        Raises:
            ValueError: If the array has inconsistent shapes
        """
        try:
            if isinstance(array_like, np.ndarray):
                return array_like.flatten()
            elif isinstance(array_like, (list, tuple)):
                # Convert to numpy array first
                try:
                    return np.array(array_like, dtype=object).flatten()
                except ValueError as e:
                    if "inhomogeneous shape" in str(e):
                        # Re-raise with original message for test compatibility
                        raise ValueError("inhomogeneous shape")
                    raise
            return [array_like]  # For scalars, return as single item list
        except Exception as e:
            raise e