"""
Test module for system integration of recursion tracking components.

This module tests the integration of recursion tracking, phase-locking, coherence gating,
and meta-pattern detection with the main UTCHSSystem.
"""

import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from utchs.core.system import UTCHSSystem
from utchs.core.system_integration import UTCHSSystemIntegrator, integrate_recursion_tracking, create_default_configuration
from utchs.core.recursion_tracker import RecursionTracker
from utchs.core.transition_analyzer import TransitionAnalyzer
from utchs.core.fractal_analyzer import FractalAnalyzer
from utchs.core.meta_pattern_detector import MetaPatternDetector

class TestSystemIntegration:
    """Test class for system integration functionality."""
    
    def setup_method(self):
        """Setup test environment before each test."""
        # Create a mock system
        self.mock_system = MagicMock(spec=UTCHSSystem)
        self.mock_system.current_tick = 0
        self.mock_system.phase_recursion_depth = 1
        self.mock_system.global_coherence = 1.0
        self.mock_system.global_stability = 0.0
        self.mock_system.energy_level = 0.0
        
        # Create a mock position
        self.mock_position = MagicMock()
        self.mock_position.number = 3
        self.mock_position.phase = 0.0
        self.mock_position.energy_level = 1.0
        self.mock_position.absolute_position = {'torus': 1, 'structure': 1, 'cycle': 1, 'position': 3}
        self.mock_position.spatial_location = np.array([0.0, 0.0, 0.0])
        
        # Setup the mock system's hierarchical access
        mock_cycle = MagicMock()
        mock_cycle.id = 1
        mock_cycle.get_current_position.return_value = self.mock_position
        
        mock_structure = MagicMock()
        mock_structure.id = 1
        mock_structure.get_current_cycle.return_value = mock_cycle
        
        mock_torus = MagicMock()
        mock_torus.id = 1
        mock_torus.get_current_structure.return_value = mock_structure
        
        self.mock_system.tori = [mock_torus]
        self.mock_system.current_torus_idx = 0
        
        # Create a test configuration
        self.test_config = {
            'tracking_interval': 1,
            'analysis_interval': 10,
            'visualization_interval': 20,
            'output_dir': 'test_output',
            'recursion': {
                'max_history_length': 100,
                'max_recursion_depth': 3
            }
        }
        
        # Reset RecursionTracker singleton
        RecursionTracker._instance = None
    
    def teardown_method(self):
        """Cleanup after each test."""
        # Reset RecursionTracker singleton
        RecursionTracker._instance = None
        
        # Remove test output directory if it exists
        if os.path.exists('test_output'):
            for file in os.listdir('test_output'):
                os.remove(os.path.join('test_output', file))
            os.rmdir('test_output')
    
    def test_integrator_initialization(self):
        """Test initialization of the system integrator."""
        integrator = UTCHSSystemIntegrator(self.mock_system, self.test_config)
        
        # Check that components are initialized
        assert integrator.system == self.mock_system
        assert integrator.config == self.test_config
        assert isinstance(integrator.recursion_tracker, RecursionTracker)
        assert isinstance(integrator.transition_analyzer, TransitionAnalyzer)
        assert isinstance(integrator.fractal_analyzer, FractalAnalyzer)
        assert isinstance(integrator.meta_pattern_detector, MetaPatternDetector)
        assert integrator.tracking_interval == 1
        assert integrator.analysis_interval == 10
        assert integrator.visualization_interval == 20
        assert not integrator.attached
    
    def test_attach_components(self):
        """Test attaching components to the system."""
        integrator = UTCHSSystemIntegrator(self.mock_system, self.test_config)
        
        # Save original methods before attaching
        original_advance_tick = self.mock_system.advance_tick
        original_record_state = self.mock_system._record_state
        original_update_system_metrics = self.mock_system._update_system_metrics
        original_analyze_system_state = self.mock_system.analyze_system_state
        
        # Attach components
        integrator.attach_components()
        
        # Check that methods are changed
        assert self.mock_system.advance_tick != original_advance_tick
        assert self.mock_system._record_state != original_record_state
        assert self.mock_system._update_system_metrics != original_update_system_metrics
        assert self.mock_system.analyze_system_state != original_analyze_system_state
        
        # Check that new methods are added
        assert hasattr(self.mock_system, 'get_recursion_tracker')
        assert hasattr(self.mock_system, 'get_transition_analyzer')
        assert hasattr(self.mock_system, 'get_fractal_analyzer')
        assert hasattr(self.mock_system, 'get_meta_pattern_detector')
        assert hasattr(self.mock_system, 'generate_recursion_report')
        
        # Check that attached flag is set
        assert integrator.attached
    
    def test_detach_components(self):
        """Test detaching components from the system."""
        integrator = UTCHSSystemIntegrator(self.mock_system, self.test_config)
        
        # Save original methods before attaching
        original_advance_tick = self.mock_system.advance_tick
        original_record_state = self.mock_system._record_state
        original_update_system_metrics = self.mock_system._update_system_metrics
        original_analyze_system_state = self.mock_system.analyze_system_state
        
        # Attach and then detach components
        integrator.attach_components()
        integrator.detach_components()
        
        # Check that methods are restored
        assert self.mock_system.advance_tick == original_advance_tick
        assert self.mock_system._record_state == original_record_state
        assert self.mock_system._update_system_metrics == original_update_system_metrics
        assert self.mock_system.analyze_system_state == original_analyze_system_state
        
        # Check that added methods are removed
        assert not hasattr(self.mock_system, 'get_recursion_tracker')
        assert not hasattr(self.mock_system, 'get_transition_analyzer')
        assert not hasattr(self.mock_system, 'get_fractal_analyzer')
        assert not hasattr(self.mock_system, 'get_meta_pattern_detector')
        assert not hasattr(self.mock_system, 'generate_recursion_report')
        
        # Check that attached flag is cleared
        assert not integrator.attached
    
    def test_enhanced_advance_tick(self):
        """Test the enhanced advance_tick method."""
        integrator = UTCHSSystemIntegrator(self.mock_system, self.test_config)
        
        # Mock the tracking method
        integrator._track_current_position = MagicMock()
        integrator._run_recursion_analysis = MagicMock()
        integrator._run_recursion_visualization = MagicMock()
        
        # Attach components
        integrator.attach_components()
        
        # Run advance_tick at different ticks
        self.mock_system.current_tick = 1
        self.mock_system.advance_tick()
        assert integrator._track_current_position.call_count == 1
        assert integrator._run_recursion_analysis.call_count == 0
        assert integrator._run_recursion_visualization.call_count == 0
        
        self.mock_system.current_tick = 10
        self.mock_system.advance_tick()
        assert integrator._track_current_position.call_count == 2
        assert integrator._run_recursion_analysis.call_count == 1
        assert integrator._run_recursion_visualization.call_count == 0
        
        self.mock_system.current_tick = 20
        self.mock_system.advance_tick()
        assert integrator._track_current_position.call_count == 3
        assert integrator._run_recursion_analysis.call_count == 2
        assert integrator._run_recursion_visualization.call_count == 1
    
    def test_integrate_recursion_tracking(self):
        """Test the main integration function."""
        # Mock the integrator's attach_components method
        with patch('utchs.core.system_integration.UTCHSSystemIntegrator.attach_components') as mock_attach:
            # Integrate recursion tracking
            integrator = integrate_recursion_tracking(self.mock_system, self.test_config)
            
            # Check that components are attached
            mock_attach.assert_called_once()
            assert integrator.system == self.mock_system
            assert integrator.config == self.test_config
    
    def test_default_configuration(self):
        """Test the default configuration creation."""
        config = create_default_configuration()
        
        # Check main configuration entries
        assert 'tracking_interval' in config
        assert 'analysis_interval' in config
        assert 'visualization_interval' in config
        assert 'output_dir' in config
        
        # Check component configurations
        assert 'recursion' in config
        assert 'meta_pattern' in config
        assert 'phase_lock' in config
        assert 'coherence_gate' in config
    
    def test_recursion_tracker_singleton(self):
        """Test that recursion tracker is a singleton across multiple integrators."""
        # Create two integrators
        integrator1 = UTCHSSystemIntegrator(self.mock_system, self.test_config)
        integrator2 = UTCHSSystemIntegrator(self.mock_system, self.test_config)
        
        # Check that they share the same recursion tracker instance
        assert integrator1.recursion_tracker is integrator2.recursion_tracker
    
    def test_track_current_position(self):
        """Test tracking of the current position."""
        integrator = UTCHSSystemIntegrator(self.mock_system, self.test_config)
        
        # Mock the recursion tracker's track_position method
        integrator.recursion_tracker.track_position = MagicMock()
        
        # Track the current position
        integrator._track_current_position(10)
        
        # Check that track_position is called with correct parameters
        integrator.recursion_tracker.track_position.assert_called_once_with(
            self.mock_position, self.mock_system.phase_recursion_depth, 10
        )
    
    def test_run_recursion_analysis(self):
        """Test running recursion analysis."""
        integrator = UTCHSSystemIntegrator(self.mock_system, self.test_config)
        
        # Mock analyzer methods
        integrator.transition_analyzer.analyze_p13_seventh_cycle_transformation = MagicMock()
        integrator.transition_analyzer.analyze_octave_transitions = MagicMock()
        integrator.transition_analyzer.detect_phi_resonances = MagicMock()
        integrator.fractal_analyzer.calculate_fractal_dimension = MagicMock()
        integrator.fractal_analyzer.calculate_multi_scale_entropy = MagicMock()
        integrator.fractal_analyzer.calculate_self_similarity = MagicMock()
        integrator.meta_pattern_detector.detect_meta_patterns = MagicMock()
        
        # Run analysis
        integrator._run_recursion_analysis(100)
        
        # Check that analysis methods are called
        integrator.transition_analyzer.analyze_p13_seventh_cycle_transformation.assert_called_once()
        integrator.transition_analyzer.analyze_octave_transitions.assert_called_once()
        integrator.transition_analyzer.detect_phi_resonances.assert_called_once()
        integrator.fractal_analyzer.calculate_fractal_dimension.assert_called_once()
        integrator.fractal_analyzer.calculate_multi_scale_entropy.assert_called_once()
        integrator.fractal_analyzer.calculate_self_similarity.assert_called_once()
        integrator.meta_pattern_detector.detect_meta_patterns.assert_called_once() 