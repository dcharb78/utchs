"""
Tests for the OmniLens Möbius framework integration.

This module tests the MobiusNode and ResonantField classes that were integrated
from the OmniLens Möbius Framework starter code.
"""

import unittest
import numpy as np
from utchs.math.mobius import MobiusNode, MobiusTransformation
from utchs.fields.resonant_field import ResonantField


class TestMobiusNode(unittest.TestCase):
    """Test cases for the MobiusNode class."""
    
    def test_init(self):
        """Test MobiusNode initialization."""
        node = MobiusNode(signal=complex(0.618, 0), phase=0)
        self.assertEqual(node.signal, complex(0.618, 0))
        self.assertEqual(node.phase, 0)
        self.assertIsNone(node.next)
        self.assertIsNone(node.feedback)
    
    def test_recurse(self):
        """Test the recurse method."""
        node = MobiusNode(signal=complex(0.618, 0), phase=0)
        # Provide a harmonic alignment of 0.5
        coherence = node.recurse(0.5)
        # Coherence should be |signal| * harmonic_alignment
        expected_coherence = abs(complex(0.618, 0)) * 0.5
        self.assertAlmostEqual(coherence, expected_coherence)
        # Phase should be updated
        self.assertAlmostEqual(node.phase, expected_coherence % (2 * np.pi))
    
    def test_apply_mobius(self):
        """Test applying a Möbius transformation to a node."""
        node = MobiusNode(signal=complex(1.0, 0), phase=0)
        # Apply a simple Möbius transformation
        node.apply_mobius(
            a=complex(1.0, 0),
            b=complex(1.0, 0),
            c=complex(0.0, 0),
            d=complex(1.0, 0)
        )
        # Signal should be transformed to (1*1 + 1)/(0*1 + 1) = 2.0
        self.assertEqual(node.signal, complex(2.0, 0))
        
    def test_apply_mobius_zero_denominator(self):
        """Test applying a Möbius transformation with zero denominator."""
        node = MobiusNode(signal=complex(1.0, 0), phase=0)
        # Apply a transformation that results in division by zero
        node.apply_mobius(
            a=complex(1.0, 0),
            b=complex(1.0, 0),
            c=complex(1.0, 0),
            d=complex(-1.0, 0)
        )
        # Signal should be set to infinity
        self.assertEqual(node.signal, float('inf'))


class TestResonantField(unittest.TestCase):
    """Test cases for the ResonantField class."""
    
    def test_init(self):
        """Test ResonantField initialization."""
        field = ResonantField(tuning=144000, dimensions=2, resolution=16)
        self.assertEqual(field.tuning, 144000)
        self.assertEqual(field.dimensions, 2)
        self.assertEqual(field.resolution, 16)
        self.assertEqual(field.field.shape, (16, 16))
    
    def test_harmonic_alignment(self):
        """Test the harmonic alignment function."""
        field = ResonantField()
        
        # Test at various phase values
        test_phases = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
        expected_values = [0, 0.5, 1.0, 0.5, 0]
        
        for phase, expected in zip(test_phases, expected_values):
            with self.subTest(phase=phase):
                alignment = field.harmonic_alignment(phase)
                self.assertAlmostEqual(alignment, expected)
                
        # Test array input
        phases = np.array(test_phases)
        alignments = field.harmonic_alignment(phases)
        np.testing.assert_array_almost_equal(alignments, expected_values)
    
    def test_apply_to_node(self):
        """Test applying field effects to a node."""
        field = ResonantField(dimensions=3, resolution=16)
        node = MobiusNode(signal=complex(1.0, 0), phase=np.pi)  # Max alignment at π
        
        # Apply field at center position
        coherence = field.apply_to_node(node, (0, 0, 0))
        
        # Coherence should be non-negative
        self.assertGreaterEqual(coherence, 0)
        
        # Node phase should be updated
        self.assertNotEqual(node.phase, np.pi)
    
    def test_resonance_scan(self):
        """Test the resonance scan function."""
        field = ResonantField(dimensions=2, resolution=4)
        
        # Create a simple phase field
        phase_field = np.ones((4, 4)) * np.pi  # All values at max alignment
        
        # Calculate resonance
        resonance = field.resonance_scan(phase_field)
        
        # Resonance shape should match phase field
        self.assertEqual(resonance.shape, phase_field.shape)
        
        # All resonance values should be positive
        self.assertTrue(np.all(resonance >= 0))
    
    def test_tune_to_frequency(self):
        """Test retuning the field."""
        field = ResonantField(tuning=144000)
        
        # Save original field
        original_field = field.field.copy()
        
        # Retune to a new frequency
        new_frequency = 432000
        field.tune_to_frequency(new_frequency)
        
        # Tuning should be updated
        self.assertEqual(field.tuning, new_frequency)
        
        # Field pattern should be different
        self.assertFalse(np.array_equal(field.field, original_field))
    
    def test_create_harmonic_overlay(self):
        """Test creating a harmonic overlay field."""
        field = ResonantField(tuning=144000)
        
        # Create a perfect fifth harmonic overlay (ratio 3:2 = 1.5)
        harmonic_field = field.create_harmonic_overlay(1.5)
        
        # Should be a ResonantField
        self.assertIsInstance(harmonic_field, ResonantField)
        
        # Tuning should be scaled by the ratio
        self.assertEqual(harmonic_field.tuning, 144000 * 1.5)
        
        # Dimensions and resolution should match
        self.assertEqual(harmonic_field.dimensions, field.dimensions)
        self.assertEqual(harmonic_field.resolution, field.resolution)


class TestMobiusNodeIntegration(unittest.TestCase):
    """Test integration between MobiusNode, MobiusTransformation, and ResonantField."""
    
    def test_node_transformation_field_integration(self):
        """Test the full integration between nodes, transformations, and fields."""
        # Create a node
        node = MobiusNode(signal=complex(1.0, 0), phase=0)
        
        # Create a Möbius transformation
        mobius = MobiusTransformation(
            a=complex(0.9, 0.1),
            b=complex(0.0, 0.2),
            c=complex(-0.05, 0.05),
            d=complex(1.0, 0.0)
        )
        
        # Create a field
        field = ResonantField(dimensions=3, resolution=16)
        
        # Apply the transformation to the node
        node.apply_mobius(mobius.a, mobius.b, mobius.c, mobius.d)
        
        # Apply field effects to the node
        coherence = field.apply_to_node(node, (0, 0, 0))
        
        # Verify that everything worked together without errors
        self.assertIsNotNone(coherence)
        self.assertGreaterEqual(coherence, 0)
    
    def test_recursive_transformation_sequence(self):
        """Test a sequence of recursive transformations."""
        # Create a node
        node = MobiusNode(signal=complex(0.618, 0), phase=0)
        
        # Create a field
        field = ResonantField()
        
        # Run several iterations
        coherence_values = []
        for _ in range(5):
            # Apply a simple transformation
            node.apply_mobius(
                a=complex(0.9, 0.1),
                b=complex(0.1, 0.1),
                c=complex(-0.05, 0),
                d=complex(1.0, 0)
            )
            
            # Apply field effects
            coherence = field.apply_to_node(node, (0, 0, 0))
            coherence_values.append(coherence)
        
        # The sequence should show some variation
        self.assertGreater(max(coherence_values) - min(coherence_values), 0.01)
        
        # All coherence values should be positive
        self.assertTrue(all(c >= 0 for c in coherence_values))


if __name__ == '__main__':
    unittest.main() 