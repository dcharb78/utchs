"""
Test system integration with data sufficiency validation.

This module contains tests for the SystemIntegrator class, which implements
data sufficiency validation and tiered calculation approach to prevent
division by zero errors and improve performance.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utchs.core.system_integration import SystemIntegrator, integrate_system_tracking

class TestSystemIntegration(unittest.TestCase):
    """Test suite for system integration with data sufficiency validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock system
        self.mock_system = MagicMock()
        self.mock_system.current_tick = 0
        self.mock_system.phase_recursion_depth = 1
        self.mock_system.tori = []
        
        # Create mock recursion tracker
        self.mock_tracker = MagicMock()
        self.mock_tracker.position_history = {0: [], 1: []}
        self.mock_tracker.get_recursion_transitions.return_value = []
        
        # Create test config
        self.config = {
            'output_dir': 'test_output',
            'analysis_interval': 10,
            'visualization_interval': 20,
            'min_ticks': 20,             # Minimum ticks for advanced analysis
            'min_positions': 10,         # Minimum positions tracked
            'min_transitions': 3,        # Minimum transitions recorded
            'min_depths': 1              # Minimum recursion depths
        }
    
    @patch('utchs.core.recursion_tracker.RecursionTracker.get_instance')
    @patch('utchs.core.transition_analyzer.TransitionAnalyzer')
    @patch('utchs.core.fractal_analyzer.FractalAnalyzer')
    @patch('utchs.visualization.recursion_vis.RecursionVisualizer')
    @patch('utchs.core.meta_pattern_detector.MetaPatternDetector')
    def test_initialization(self, mock_meta, mock_vis, mock_fractal, mock_trans, mock_tracker_get):
        """Test that SystemIntegrator initializes correctly."""
        # Set up mocks
        mock_tracker_get.return_value = self.mock_tracker
        
        # Initialize SystemIntegrator
        integrator = SystemIntegrator(self.mock_system, self.config)
        
        # Check initialization
        self.assertEqual(integrator.system, self.mock_system)
        self.assertEqual(integrator.recursion_tracker, self.mock_tracker)
        self.assertEqual(integrator.min_ticks, 20)
        self.assertEqual(integrator.min_positions, 10)
        self.assertEqual(integrator.min_transitions, 3)
        self.assertEqual(integrator.min_depths, 1)
        
        # Verify mocks were called
        mock_tracker_get.assert_called_once()
        mock_trans.assert_called_once()
        mock_fractal.assert_called_once()
        mock_vis.assert_called_once()
        mock_meta.assert_called_once()
    
    @patch('utchs.core.recursion_tracker.RecursionTracker.get_instance')
    @patch('utchs.core.transition_analyzer.TransitionAnalyzer')
    @patch('utchs.core.fractal_analyzer.FractalAnalyzer')
    @patch('utchs.visualization.recursion_vis.RecursionVisualizer')
    @patch('utchs.core.meta_pattern_detector.MetaPatternDetector')
    def test_insufficient_data_detection(self, mock_meta, mock_vis, mock_fractal, mock_trans, mock_tracker_get):
        """Test that insufficient data is detected correctly."""
        # Set up mocks
        mock_tracker_get.return_value = self.mock_tracker
        
        # Initialize SystemIntegrator
        integrator = SystemIntegrator(self.mock_system, self.config)
        
        # Test with insufficient ticks
        self.mock_system.current_tick = 10  # Below min_ticks (20)
        self.assertFalse(integrator._has_sufficient_data_for_analysis())
        
        # Test with insufficient positions
        self.mock_system.current_tick = 30  # Above min_ticks
        self.mock_tracker.position_history = {0: [1, 2, 3], 1: [1, 2]}  # Total 5 positions, below min_positions (10)
        self.assertFalse(integrator._has_sufficient_data_for_analysis())
        
        # Test with insufficient transitions
        self.mock_system.current_tick = 30  # Above min_ticks
        self.mock_tracker.position_history = {0: list(range(20))}  # 20 positions, above min_positions
        self.mock_tracker.get_recursion_transitions.return_value = [1, 2]  # 2 transitions, below min_transitions (3)
        self.assertFalse(integrator._has_sufficient_data_for_analysis())
        
        # Test with insufficient depths
        self.mock_system.current_tick = 30  # Above min_ticks
        self.mock_tracker.position_history = {0: list(range(20))}  # Only depth 0, need min_depths (1)
        self.mock_tracker.get_recursion_transitions.return_value = list(range(5))  # 5 transitions, above min_transitions
        self.config['min_depths'] = 2  # Require 2 depths
        integrator = SystemIntegrator(self.mock_system, self.config)
        self.assertFalse(integrator._has_sufficient_data_for_analysis())
        
        # Test with sufficient data
        self.mock_system.current_tick = 30  # Above min_ticks
        self.mock_tracker.position_history = {0: list(range(10)), 1: list(range(10))}  # 20 positions, above min_positions
        self.mock_tracker.get_recursion_transitions.return_value = list(range(5))  # 5 transitions, above min_transitions
        self.config['min_depths'] = 1  # Require 1 depth
        integrator = SystemIntegrator(self.mock_system, self.config)
        self.assertTrue(integrator._has_sufficient_data_for_analysis())
    
    @patch('utchs.core.recursion_tracker.RecursionTracker.get_instance')
    @patch('utchs.core.transition_analyzer.TransitionAnalyzer')
    @patch('utchs.core.fractal_analyzer.FractalAnalyzer')
    @patch('utchs.visualization.recursion_vis.RecursionVisualizer')
    @patch('utchs.core.meta_pattern_detector.MetaPatternDetector')
    def test_safely_handle_division_by_zero(self, mock_meta, mock_vis, mock_fractal, mock_trans, mock_tracker_get):
        """Test that division by zero errors are handled safely."""
        # Set up mocks
        mock_tracker_get.return_value = self.mock_tracker
        
        # Initialize SystemIntegrator
        integrator = SystemIntegrator(self.mock_system, self.config)
        
        # Test with empty transitions (potential division by zero)
        self.mock_tracker.get_recursion_transitions.return_value = []
        metrics = integrator._get_recursion_metrics()
        
        # Should not raise division by zero and return valid metrics
        self.assertEqual(metrics['transitions_count'], 0)
        self.assertEqual(metrics['phi_resonance_count'], 0)
        self.assertEqual(metrics['phi_resonance_percentage'], 0.0)
    
    @patch('utchs.core.recursion_tracker.RecursionTracker.get_instance')
    @patch('utchs.core.transition_analyzer.TransitionAnalyzer')
    @patch('utchs.core.fractal_analyzer.FractalAnalyzer')
    @patch('utchs.visualization.recursion_vis.RecursionVisualizer')
    @patch('utchs.core.meta_pattern_detector.MetaPatternDetector')
    def test_analysis_deferral(self, mock_meta, mock_vis, mock_fractal, mock_trans, mock_tracker_get):
        """Test that analysis is deferred until sufficient data exists."""
        # Set up mocks
        mock_tracker = MagicMock()
        mock_tracker.position_history = {0: [], 1: []}
        mock_tracker.get_recursion_transitions.return_value = []
        mock_tracker_get.return_value = mock_tracker
        
        mock_transition_analyzer = MagicMock()
        mock_trans.return_value = mock_transition_analyzer
        
        # Initialize SystemIntegrator
        integrator = SystemIntegrator(self.mock_system, self.config)
        
        # Run analysis with insufficient data
        self.mock_system.current_tick = 5  # Below min_ticks (20)
        integrator._run_analysis(5)
        
        # Verify that advanced analysis was not performed
        mock_transition_analyzer.analyze_p13_seventh_cycle_transformation.assert_not_called()
        mock_transition_analyzer.analyze_octave_transitions.assert_not_called()
        mock_transition_analyzer.detect_phi_resonances.assert_not_called()
        
        # Run analysis with sufficient data
        self.mock_system.current_tick = 30  # Above min_ticks
        mock_tracker.position_history = {0: list(range(10)), 1: list(range(10))}
        mock_tracker.get_recursion_transitions.return_value = list(range(5))
        
        # Mock has_sufficient_data to return True
        with patch.object(integrator, '_has_sufficient_data_for_analysis', return_value=True):
            integrator._run_analysis(30)
            
            # Verify that advanced analysis was performed
            mock_transition_analyzer.analyze_p13_seventh_cycle_transformation.assert_called_once()
            mock_transition_analyzer.analyze_octave_transitions.assert_called_once()
            mock_transition_analyzer.detect_phi_resonances.assert_called_once()
    
    @patch('utchs.core.recursion_tracker.RecursionTracker.get_instance')
    @patch('utchs.core.transition_analyzer.TransitionAnalyzer')
    @patch('utchs.core.fractal_analyzer.FractalAnalyzer')
    @patch('utchs.visualization.recursion_vis.RecursionVisualizer')
    @patch('utchs.core.meta_pattern_detector.MetaPatternDetector')
    def test_tiered_calculation(self, mock_meta, mock_vis, mock_fractal, mock_trans, mock_tracker_get):
        """Test that calculations are performed in tiers based on data availability."""
        # Set up mocks
        mock_tracker = MagicMock()
        mock_tracker.position_history = {0: [], 1: []}
        mock_tracker.get_recursion_transitions.return_value = [
            {'phi_phase_resonance': True, 'phi_energy_resonance': False},
            {'phi_phase_resonance': False, 'phi_energy_resonance': True},
            {'phi_phase_resonance': False, 'phi_energy_resonance': False}
        ]
        mock_tracker_get.return_value = mock_tracker
        
        mock_fractal_analyzer = MagicMock()
        mock_fractal.return_value = mock_fractal_analyzer
        mock_fractal_analyzer.calculate_self_similarity.return_value = {'self_similarity': 0.85}
        mock_fractal_analyzer.calculate_fractal_dimension.return_value = {'dimension': 1.6}
        
        mock_meta_detector = MagicMock()
        mock_meta.return_value = mock_meta_detector
        mock_meta_detector.detect_meta_patterns.return_value = {
            'detected': True,
            'meta_cycle_strength': 0.75
        }
        
        # Initialize SystemIntegrator
        integrator = SystemIntegrator(self.mock_system, self.config)
        
        # Tier 1: Basic metrics with insufficient data
        self.mock_system.current_tick = 5  # Below min_ticks
        with patch.object(integrator, '_has_sufficient_data_for_analysis', return_value=False):
            metrics = integrator._get_recursion_metrics()
            
            # Basic metrics should be calculated
            self.assertEqual(metrics['transitions_count'], 3)
            self.assertEqual(metrics['phi_resonance_count'], 2)
            self.assertEqual(metrics['phi_resonance_percentage'], 2/3)
            
            # Advanced metrics should not be calculated
            self.assertIsNone(metrics['self_similarity'])
            self.assertIsNone(metrics['fractal_dimension'])
            self.assertFalse(metrics['meta_pattern_detected'])
            self.assertEqual(metrics['meta_pattern_strength'], 0.0)
            
            # Verify advanced analysis was not performed
            mock_fractal_analyzer.calculate_self_similarity.assert_not_called()
            mock_fractal_analyzer.calculate_fractal_dimension.assert_not_called()
            mock_meta_detector.detect_meta_patterns.assert_not_called()
        
        # Tier 2: Advanced metrics with sufficient data and on analysis interval
        self.mock_system.current_tick = 40  # Above min_ticks
        self.mock_system.current_tick = integrator.analysis_interval  # On analysis interval
        with patch.object(integrator, '_has_sufficient_data_for_analysis', return_value=True):
            metrics = integrator._get_recursion_metrics()
            
            # Basic metrics should be calculated
            self.assertEqual(metrics['transitions_count'], 3)
            self.assertEqual(metrics['phi_resonance_count'], 2)
            self.assertEqual(metrics['phi_resonance_percentage'], 2/3)
            
            # Advanced metrics should be calculated
            self.assertEqual(metrics['self_similarity'], 0.85)
            self.assertEqual(metrics['fractal_dimension'], 1.6)
            self.assertTrue(metrics['meta_pattern_detected'])
            self.assertEqual(metrics['meta_pattern_strength'], 0.75)
            
            # Verify advanced analysis was performed
            mock_fractal_analyzer.calculate_self_similarity.assert_called_once()
            mock_fractal_analyzer.calculate_fractal_dimension.assert_called_once()
            mock_meta_detector.detect_meta_patterns.assert_called_once()
    
    @patch('utchs.core.recursion_tracker.RecursionTracker.get_instance')
    def test_result_caching(self, mock_tracker_get):
        """Test that results are cached to improve performance."""
        # Set up mocks
        mock_tracker = MagicMock()
        mock_tracker.position_history = {0: [], 1: []}
        mock_tracker.get_recursion_transitions.return_value = [{'phi_phase_resonance': True}]
        mock_tracker_get.return_value = mock_tracker
        
        # Initialize SystemIntegrator with short cache validity
        self.config['cache_validity_ticks'] = 5
        integrator = SystemIntegrator(self.mock_system, self.config)
        
        # First call should calculate and cache metrics
        self.mock_system.current_tick = 10
        with patch.object(mock_tracker, 'get_recursion_transitions', wraps=mock_tracker.get_recursion_transitions) as spy:
            metrics1 = integrator._get_recursion_metrics()
            self.assertEqual(spy.call_count, 1)
            self.assertEqual(integrator.last_cache_tick, 10)
        
        # Second call within cache validity should use cached metrics
        self.mock_system.current_tick = 12  # Within cache validity (10 + 5)
        with patch.object(mock_tracker, 'get_recursion_transitions', wraps=mock_tracker.get_recursion_transitions) as spy:
            metrics2 = integrator._get_recursion_metrics()
            self.assertEqual(spy.call_count, 0)  # Should not call get_recursion_transitions
        
        # Call after cache expiry should recalculate
        self.mock_system.current_tick = 20  # Outside cache validity (10 + 5)
        with patch.object(mock_tracker, 'get_recursion_transitions', wraps=mock_tracker.get_recursion_transitions) as spy:
            metrics3 = integrator._get_recursion_metrics()
            self.assertEqual(spy.call_count, 1)  # Should call get_recursion_transitions
            self.assertEqual(integrator.last_cache_tick, 20)  # Cache timestamp should update

if __name__ == '__main__':
    unittest.main() 