"""Model registry utilities for the UTCHS framework."""

import os
import json
import uuid
import hashlib
import shutil
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime
import logging
import pickle
import h5py

from utchs.utils.data_manager import DataManager

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for tracking and versioning models.
    
    This class manages model versions, metadata, and artifacts,
    ensuring that models can be tracked, compared, and reused.
    """
    
    def __init__(self, base_dir: str = "models"):
        """
        Initialize the model registry.
        
        Args:
            base_dir: Base directory for model storage
        """
        self.base_dir = base_dir
        self.data_manager = DataManager(os.path.join(base_dir, "metadata"))
        
        # Create registry directory
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(os.path.join(base_dir, "versions"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "metadata"), exist_ok=True)
        
        # Load or create registry index
        self.index_path = os.path.join(base_dir, "index.json")
        if os.path.exists(self.index_path):
            with open(self.index_path, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {
                "models": {},
                "last_updated": datetime.now().isoformat()
            }
            self._save_index()
            
        logger.info(f"Initialized model registry at {base_dir}")
        
    def _save_index(self) -> None:
        """Save the registry index to disk."""
        self.index["last_updated"] = datetime.now().isoformat()
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)
            
    def register_model(self, name: str, description: str, 
                      parameters: Dict[str, Any],
                      artifacts: Optional[Dict[str, str]] = None) -> str:
        """
        Register a new model.
        
        Args:
            name: Model name
            description: Model description
            parameters: Model parameters
            artifacts: Dictionary of artifact names and paths
            
        Returns:
            Model ID
        """
        # Generate model ID
        model_id = str(uuid.uuid4())
        
        # Create model entry
        model_entry = {
            "id": model_id,
            "name": name,
            "description": description,
            "parameters": parameters,
            "created_at": datetime.now().isoformat(),
            "versions": [],
            "artifacts": artifacts or {}
        }
        
        # Add to index
        self.index["models"][model_id] = model_entry
        self._save_index()
        
        logger.info(f"Registered model: {name} (ID: {model_id})")
        return model_id
        
    def add_version(self, model_id: str, version_name: str,
                   data: Dict[str, Any],
                   metrics: Optional[Dict[str, float]] = None,
                   parent_version: Optional[str] = None) -> str:
        """
        Add a new version to a model.
        
        Args:
            model_id: Model ID
            version_name: Version name
            data: Version data
            metrics: Version metrics
            parent_version: Parent version ID
            
        Returns:
            Version ID
        """
        # Check if model exists
        if model_id not in self.index["models"]:
            raise ValueError(f"Model not found: {model_id}")
            
        # Generate version ID
        version_id = str(uuid.uuid4())
        
        # Create version entry
        version_entry = {
            "id": version_id,
            "name": version_name,
            "created_at": datetime.now().isoformat(),
            "metrics": metrics or {},
            "parent_version": parent_version
        }
        
        # Save version data
        version_dir = os.path.join(self.base_dir, "versions", version_id)
        os.makedirs(version_dir, exist_ok=True)
        
        # Save version data
        with open(os.path.join(version_dir, "data.pkl"), 'wb') as f:
            pickle.dump(data, f)
            
        # Add to model
        self.index["models"][model_id]["versions"].append(version_entry)
        self._save_index()
        
        logger.info(f"Added version {version_name} to model {model_id}")
        return version_id
        
    def get_model(self, model_id: str) -> Dict[str, Any]:
        """
        Get model information.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model information
        """
        if model_id not in self.index["models"]:
            raise ValueError(f"Model not found: {model_id}")
            
        return self.index["models"][model_id]
        
    def get_version(self, model_id: str, version_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Get version information and data.
        
        Args:
            model_id: Model ID
            version_id: Version ID
            
        Returns:
            Tuple of (version information, version data)
        """
        # Check if model exists
        if model_id not in self.index["models"]:
            raise ValueError(f"Model not found: {model_id}")
            
        # Find version
        version_info = None
        for version in self.index["models"][model_id]["versions"]:
            if version["id"] == version_id:
                version_info = version
                break
                
        if version_info is None:
            raise ValueError(f"Version not found: {version_id}")
            
        # Load version data
        version_dir = os.path.join(self.base_dir, "versions", version_id)
        with open(os.path.join(version_dir, "data.pkl"), 'rb') as f:
            version_data = pickle.load(f)
            
        return version_info, version_data
        
    def save_field(self, model_id: str, version_id: str, 
                  field: np.ndarray, name: str,
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a field for a model version.
        
        Args:
            model_id: Model ID
            version_id: Version ID
            field: Field array to save
            name: Field name
            metadata: Additional metadata
            
        Returns:
            Path to the saved field
        """
        # Check if model and version exist
        if model_id not in self.index["models"]:
            raise ValueError(f"Model not found: {model_id}")
            
        version_exists = False
        for version in self.index["models"][model_id]["versions"]:
            if version["id"] == version_id:
                version_exists = True
                break
                
        if not version_exists:
            raise ValueError(f"Version not found: {version_id}")
            
        # Add model and version metadata
        if metadata is None:
            metadata = {}
        metadata["model_id"] = model_id
        metadata["version_id"] = version_id
        metadata["timestamp"] = datetime.now().isoformat()
        
        # Save field
        version_dir = os.path.join(self.base_dir, "versions", version_id)
        path = os.path.join(version_dir, f"{name}.h5")
        
        with h5py.File(path, 'w') as f:
            f.create_dataset("field", data=field)
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (int, float, str)):
                        f.attrs[key] = value
                    elif isinstance(value, np.ndarray):
                        f.create_dataset(f"metadata/{key}", data=value)
                        
        logger.debug(f"Saved field {name} for model {model_id}, version {version_id}")
        return path
        
    def load_field(self, model_id: str, version_id: str, name: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load a field for a model version.
        
        Args:
            model_id: Model ID
            version_id: Version ID
            name: Field name
            
        Returns:
            Tuple of (field array, metadata dictionary)
        """
        # Check if model and version exist
        if model_id not in self.index["models"]:
            raise ValueError(f"Model not found: {model_id}")
            
        version_exists = False
        for version in self.index["models"][model_id]["versions"]:
            if version["id"] == version_id:
                version_exists = True
                break
                
        if not version_exists:
            raise ValueError(f"Version not found: {version_id}")
            
        # Load field
        version_dir = os.path.join(self.base_dir, "versions", version_id)
        path = os.path.join(version_dir, f"{name}.h5")
        
        if not os.path.exists(path):
            raise ValueError(f"Field not found: {name}")
            
        with h5py.File(path, 'r') as f:
            field = f["field"][:]
            
            # Load metadata
            metadata = {}
            for key, value in f.attrs.items():
                metadata[key] = value
                
            # Load array metadata
            if "metadata" in f:
                for key in f["metadata"].keys():
                    metadata[key] = f[f"metadata/{key}"][:]
                    
        return field, metadata
        
    def compare_versions(self, model_id: str, version_ids: List[str], 
                        metric: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare multiple versions of a model.
        
        Args:
            model_id: Model ID
            version_ids: List of version IDs to compare
            metric: Metric to compare on
            
        Returns:
            Comparison results
        """
        # Check if model exists
        if model_id not in self.index["models"]:
            raise ValueError(f"Model not found: {model_id}")
            
        # Get versions
        versions = []
        for version_id in version_ids:
            version_info = None
            for version in self.index["models"][model_id]["versions"]:
                if version["id"] == version_id:
                    version_info = version
                    break
                    
            if version_info is None:
                raise ValueError(f"Version not found: {version_id}")
                
            versions.append(version_info)
            
        # Compare metrics if specified
        if metric:
            results = {
                "metric": metric,
                "comparison": {}
            }
            
            for version in versions:
                if metric in version["metrics"]:
                    results["comparison"][version["id"]] = version["metrics"][metric]
                    
            return results
            
        # Compare all metrics
        results = {
            "metrics": {},
            "created_at": {}
        }
        
        for version in versions:
            results["created_at"][version["id"]] = version["created_at"]
            
            for metric_name, metric_value in version["metrics"].items():
                if metric_name not in results["metrics"]:
                    results["metrics"][metric_name] = {}
                    
                results["metrics"][metric_name][version["id"]] = metric_value
                
        return results
        
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all registered models.
        
        Returns:
            List of model information
        """
        return list(self.index["models"].values())
        
    def list_versions(self, model_id: str) -> List[Dict[str, Any]]:
        """
        List all versions of a model.
        
        Args:
            model_id: Model ID
            
        Returns:
            List of version information
        """
        if model_id not in self.index["models"]:
            raise ValueError(f"Model not found: {model_id}")
            
        return self.index["models"][model_id]["versions"]
        
    def delete_model(self, model_id: str) -> None:
        """
        Delete a model and all its versions.
        
        Args:
            model_id: Model ID
        """
        if model_id not in self.index["models"]:
            raise ValueError(f"Model not found: {model_id}")
            
        # Delete version directories
        for version in self.index["models"][model_id]["versions"]:
            version_dir = os.path.join(self.base_dir, "versions", version["id"])
            if os.path.exists(version_dir):
                shutil.rmtree(version_dir)
                
        # Remove from index
        del self.index["models"][model_id]
        self._save_index()
        
        logger.info(f"Deleted model: {model_id}")
        
    def delete_version(self, model_id: str, version_id: str) -> None:
        """
        Delete a version of a model.
        
        Args:
            model_id: Model ID
            version_id: Version ID
        """
        if model_id not in self.index["models"]:
            raise ValueError(f"Model not found: {model_id}")
            
        # Find version
        version_index = None
        for i, version in enumerate(self.index["models"][model_id]["versions"]):
            if version["id"] == version_id:
                version_index = i
                break
                
        if version_index is None:
            raise ValueError(f"Version not found: {version_id}")
            
        # Delete version directory
        version_dir = os.path.join(self.base_dir, "versions", version_id)
        if os.path.exists(version_dir):
            shutil.rmtree(version_dir)
            
        # Remove from index
        del self.index["models"][model_id]["versions"][version_index]
        self._save_index()
        
        logger.info(f"Deleted version {version_id} from model {model_id}")
        
    def export_model(self, model_id: str, export_dir: str) -> str:
        """
        Export a model and all its versions.
        
        Args:
            model_id: Model ID
            export_dir: Directory to export to
            
        Returns:
            Path to the exported model
        """
        if model_id not in self.index["models"]:
            raise ValueError(f"Model not found: {model_id}")
            
        # Create export directory
        os.makedirs(export_dir, exist_ok=True)
        
        # Export model information
        model_info = self.index["models"][model_id]
        with open(os.path.join(export_dir, "model.json"), 'w') as f:
            json.dump(model_info, f, indent=2)
            
        # Export versions
        for version in model_info["versions"]:
            version_dir = os.path.join(self.base_dir, "versions", version["id"])
            export_version_dir = os.path.join(export_dir, "versions", version["id"])
            
            if os.path.exists(version_dir):
                shutil.copytree(version_dir, export_version_dir)
                
        logger.info(f"Exported model {model_id} to {export_dir}")
        return export_dir
        
    def import_model(self, import_dir: str) -> str:
        """
        Import a model and all its versions.
        
        Args:
            import_dir: Directory to import from
            
        Returns:
            Model ID
        """
        # Check if model.json exists
        model_json_path = os.path.join(import_dir, "model.json")
        if not os.path.exists(model_json_path):
            raise ValueError(f"Invalid model directory: {import_dir}")
            
        # Load model information
        with open(model_json_path, 'r') as f:
            model_info = json.load(f)
            
        # Generate new model ID
        model_id = str(uuid.uuid4())
        old_model_id = model_info["id"]
        
        # Update model ID
        model_info["id"] = model_id
        
        # Import versions
        for version in model_info["versions"]:
            old_version_id = version["id"]
            new_version_id = str(uuid.uuid4())
            
            # Update version ID
            version["id"] = new_version_id
            
            # Copy version directory
            import_version_dir = os.path.join(import_dir, "versions", old_version_id)
            version_dir = os.path.join(self.base_dir, "versions", new_version_id)
            
            if os.path.exists(import_version_dir):
                shutil.copytree(import_version_dir, version_dir)
                
        # Add to index
        self.index["models"][model_id] = model_info
        self._save_index()
        
        logger.info(f"Imported model from {import_dir} as {model_id}")
        return model_id 