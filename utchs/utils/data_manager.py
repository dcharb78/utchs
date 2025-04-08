"""Data management utilities for the UTCHS framework."""

import os
import json
import h5py
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataManager:
    """
    Manages data persistence, tracking, and retrieval for the UTCHS framework.
    
    This class handles saving and loading field data, simulation states,
    and metadata in various formats (HDF5, binary, JSON) with proper
    versioning and tracking.
    """
    
    def __init__(self, base_dir: str = "data"):
        """
        Initialize the data manager.
        
        Args:
            base_dir: Base directory for data storage
        """
        self.base_dir = base_dir
        self._ensure_directories()
        self.metadata = {}
        
    def _ensure_directories(self) -> None:
        """Ensure all necessary directories exist."""
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "fields"), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "states"), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "metadata"), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "checkpoints"), exist_ok=True)
        
    def save_field(self, field: np.ndarray, name: str, 
                  metadata: Optional[Dict[str, Any]] = None,
                  format: str = "h5") -> str:
        """
        Save a field to disk.
        
        Args:
            field: Field array to save
            name: Name of the field
            metadata: Additional metadata to store
            format: Format to save in ("h5", "binary", "npy")
            
        Returns:
            Path to the saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.{format}"
        
        if format == "h5":
            path = os.path.join(self.base_dir, "fields", filename)
            with h5py.File(path, 'w') as f:
                f.create_dataset("field", data=field)
                if metadata:
                    for key, value in metadata.items():
                        if isinstance(value, (int, float, str)):
                            f.attrs[key] = value
                        elif isinstance(value, np.ndarray):
                            f.create_dataset(f"metadata/{key}", data=value)
            logger.info(f"Saved field to {path}")
            
        elif format == "binary":
            path = os.path.join(self.base_dir, "fields", filename)
            field.tofile(path)
            if metadata:
                meta_path = os.path.join(self.base_dir, "metadata", f"{name}_{timestamp}.json")
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f)
            logger.info(f"Saved field to {path}")
            
        elif format == "npy":
            path = os.path.join(self.base_dir, "fields", filename)
            np.save(path, field)
            if metadata:
                meta_path = os.path.join(self.base_dir, "metadata", f"{name}_{timestamp}.json")
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f)
            logger.info(f"Saved field to {path}")
            
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        return path
    
    def load_field(self, path: str, format: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load a field from disk.
        
        Args:
            path: Path to the field file
            format: Format of the file (if None, inferred from extension)
            
        Returns:
            Tuple of (field array, metadata dictionary)
        """
        if format is None:
            format = path.split('.')[-1]
            
        metadata = {}
        
        if format == "h5":
            with h5py.File(path, 'r') as f:
                field = f["field"][:]
                # Load metadata
                for key, value in f.attrs.items():
                    metadata[key] = value
                # Load array metadata
                if "metadata" in f:
                    for key in f["metadata"].keys():
                        metadata[key] = f[f"metadata/{key}"][:]
                        
        elif format == "binary":
            field = np.fromfile(path, dtype=np.complex128)
            # Try to load metadata
            base_name = os.path.basename(path).split('.')[0]
            meta_path = os.path.join(self.base_dir, "metadata", f"{base_name}.json")
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                    
        elif format == "npy":
            field = np.load(path)
            # Try to load metadata
            base_name = os.path.basename(path).split('.')[0]
            meta_path = os.path.join(self.base_dir, "metadata", f"{base_name}.json")
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                    
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        logger.info(f"Loaded field from {path}")
        return field, metadata
    
    def save_state(self, state: Dict[str, Any], name: str) -> str:
        """
        Save a simulation state.
        
        Args:
            state: State dictionary to save
            name: Name of the state
            
        Returns:
            Path to the saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.json"
        path = os.path.join(self.base_dir, "states", filename)
        
        with open(path, 'w') as f:
            json.dump(state, f)
            
        logger.info(f"Saved state to {path}")
        return path
    
    def load_state(self, path: str) -> Dict[str, Any]:
        """
        Load a simulation state.
        
        Args:
            path: Path to the state file
            
        Returns:
            State dictionary
        """
        with open(path, 'r') as f:
            state = json.load(f)
            
        logger.info(f"Loaded state from {path}")
        return state
    
    def save_checkpoint(self, system_state: Dict[str, Any], name: str) -> str:
        """
        Save a complete system checkpoint.
        
        Args:
            system_state: Complete system state
            name: Name of the checkpoint
            
        Returns:
            Path to the checkpoint directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = os.path.join(self.base_dir, "checkpoints", f"{name}_{timestamp}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save system state
        with open(os.path.join(checkpoint_dir, "state.json"), 'w') as f:
            json.dump(system_state, f)
            
        # Save fields
        if "phase_field" in system_state:
            self.save_field(
                system_state["phase_field"], 
                "phase_field", 
                {"grid_size": system_state.get("grid_size")},
                "h5"
            )
            
        if "energy_field" in system_state:
            self.save_field(
                system_state["energy_field"], 
                "energy_field", 
                {"grid_size": system_state.get("grid_size")},
                "h5"
            )
            
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
        return checkpoint_dir
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Load a complete system checkpoint.
        
        Args:
            path: Path to the checkpoint directory
            
        Returns:
            System state dictionary
        """
        # Load system state
        with open(os.path.join(path, "state.json"), 'r') as f:
            state = json.load(f)
            
        # Load fields if they exist
        phase_field_path = os.path.join(path, "phase_field.h5")
        if os.path.exists(phase_field_path):
            phase_field, _ = self.load_field(phase_field_path)
            state["phase_field"] = phase_field
            
        energy_field_path = os.path.join(path, "energy_field.h5")
        if os.path.exists(energy_field_path):
            energy_field, _ = self.load_field(energy_field_path)
            state["energy_field"] = energy_field
            
        logger.info(f"Loaded checkpoint from {path}")
        return state
    
    def track_metadata(self, key: str, value: Any) -> None:
        """
        Track metadata for the current session.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
        
    def get_metadata(self, key: Optional[str] = None) -> Union[Dict[str, Any], Any]:
        """
        Get tracked metadata.
        
        Args:
            key: Specific metadata key to retrieve, or None for all metadata
            
        Returns:
            Metadata value or dictionary
        """
        if key is None:
            return self.metadata
        return self.metadata.get(key)
    
    def save_metadata(self, name: str) -> str:
        """
        Save tracked metadata to disk.
        
        Args:
            name: Name for the metadata file
            
        Returns:
            Path to the saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.json"
        path = os.path.join(self.base_dir, "metadata", filename)
        
        with open(path, 'w') as f:
            json.dump(self.metadata, f)
            
        logger.info(f"Saved metadata to {path}")
        return path 