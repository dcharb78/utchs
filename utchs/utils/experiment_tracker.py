"""Experiment tracking utilities for the UTCHS framework."""

import os
import json
import uuid
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
import logging
import platform
import sys
import random
import torch
import git
from pathlib import Path

from utchs.utils.data_manager import DataManager

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Tracks experiments and ensures reproducibility.
    
    This class manages experiment metadata, tracks parameters,
    and ensures that experiments can be reproduced by capturing
    the complete environment state.
    """
    
    def __init__(self, name: str, base_dir: str = "experiments"):
        """
        Initialize the experiment tracker.
        
        Args:
            name: Name of the experiment
            base_dir: Base directory for experiment data
        """
        self.name = name
        self.base_dir = base_dir
        self.experiment_id = str(uuid.uuid4())
        self.start_time = datetime.now()
        self.data_manager = DataManager(os.path.join(base_dir, self.experiment_id))
        
        # Create experiment directory
        os.makedirs(os.path.join(base_dir, self.experiment_id), exist_ok=True)
        
        # Initialize metadata
        self.metadata = {
            "name": name,
            "experiment_id": self.experiment_id,
            "start_time": self.start_time.isoformat(),
            "parameters": {},
            "results": {},
            "environment": self._capture_environment(),
            "git_info": self._capture_git_info(),
            "checkpoints": [],
            "events": []
        }
        
        # Save initial metadata
        self._save_metadata()
        
        logger.info(f"Started experiment: {name} (ID: {self.experiment_id})")
        
    def _capture_environment(self) -> Dict[str, Any]:
        """
        Capture the current environment state.
        
        Returns:
            Dictionary of environment information
        """
        env = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "numpy_version": np.__version__,
            "random_seed": random.getstate(),
            "numpy_random_seed": np.random.get_state(),
        }
        
        # Add PyTorch info if available
        try:
            env["torch_version"] = torch.__version__
            env["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                env["cuda_device_count"] = torch.cuda.device_count()
                env["cuda_device_name"] = torch.cuda.get_device_name(0)
        except ImportError:
            pass
            
        return env
        
    def _capture_git_info(self) -> Dict[str, Any]:
        """
        Capture Git repository information.
        
        Returns:
            Dictionary of Git information
        """
        try:
            repo = git.Repo(search_parent_directories=True)
            return {
                "commit_hash": repo.head.object.hexsha,
                "branch": repo.active_branch.name,
                "is_dirty": repo.is_dirty(),
                "untracked_files": repo.untracked_files,
            }
        except (git.InvalidGitRepositoryError, git.NoSuchPathError):
            return {"error": "Not a Git repository"}
            
    def _save_metadata(self) -> None:
        """Save experiment metadata to disk."""
        path = os.path.join(self.base_dir, self.experiment_id, "metadata.json")
        with open(path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
            
    def log_parameter(self, name: str, value: Any) -> None:
        """
        Log an experiment parameter.
        
        Args:
            name: Parameter name
            value: Parameter value
        """
        self.metadata["parameters"][name] = value
        self._save_metadata()
        logger.debug(f"Logged parameter: {name} = {value}")
        
    def log_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Log multiple experiment parameters.
        
        Args:
            parameters: Dictionary of parameter names and values
        """
        self.metadata["parameters"].update(parameters)
        self._save_metadata()
        logger.debug(f"Logged {len(parameters)} parameters")
        
    def log_result(self, name: str, value: Any) -> None:
        """
        Log an experiment result.
        
        Args:
            name: Result name
            value: Result value
        """
        self.metadata["results"][name] = value
        self._save_metadata()
        logger.debug(f"Logged result: {name} = {value}")
        
    def log_results(self, results: Dict[str, Any]) -> None:
        """
        Log multiple experiment results.
        
        Args:
            results: Dictionary of result names and values
        """
        self.metadata["results"].update(results)
        self._save_metadata()
        logger.debug(f"Logged {len(results)} results")
        
    def log_event(self, name: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an event during the experiment.
        
        Args:
            name: Event name
            data: Additional event data
        """
        event = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "data": data or {}
        }
        self.metadata["events"].append(event)
        self._save_metadata()
        logger.debug(f"Logged event: {name}")
        
    def save_checkpoint(self, name: str, data: Dict[str, Any]) -> str:
        """
        Save a checkpoint during the experiment.
        
        Args:
            name: Checkpoint name
            data: Checkpoint data
            
        Returns:
            Path to the checkpoint
        """
        checkpoint_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        checkpoint_path = self.data_manager.save_checkpoint(data, checkpoint_id)
        
        # Record checkpoint in metadata
        self.metadata["checkpoints"].append({
            "name": name,
            "id": checkpoint_id,
            "path": checkpoint_path,
            "timestamp": datetime.now().isoformat()
        })
        self._save_metadata()
        
        logger.info(f"Saved checkpoint: {name} (ID: {checkpoint_id})")
        return checkpoint_path
        
    def load_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID
            
        Returns:
            Checkpoint data
        """
        # Find checkpoint path
        checkpoint = None
        for cp in self.metadata["checkpoints"]:
            if cp["id"] == checkpoint_id:
                checkpoint = cp
                break
                
        if checkpoint is None:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")
            
        # Load checkpoint
        data = self.data_manager.load_checkpoint(checkpoint["path"])
        logger.info(f"Loaded checkpoint: {checkpoint['name']} (ID: {checkpoint_id})")
        return data
        
    def save_field(self, field: np.ndarray, name: str, 
                  metadata: Optional[Dict[str, Any]] = None,
                  format: str = "h5") -> str:
        """
        Save a field during the experiment.
        
        Args:
            field: Field array to save
            name: Field name
            metadata: Additional metadata
            format: Save format
            
        Returns:
            Path to the saved field
        """
        # Add experiment metadata
        if metadata is None:
            metadata = {}
        metadata["experiment_id"] = self.experiment_id
        metadata["timestamp"] = datetime.now().isoformat()
        
        path = self.data_manager.save_field(field, name, metadata, format)
        logger.debug(f"Saved field: {name}")
        return path
        
    def end_experiment(self, success: bool = True, 
                      summary: Optional[Dict[str, Any]] = None) -> None:
        """
        End the experiment and save final metadata.
        
        Args:
            success: Whether the experiment completed successfully
            summary: Additional summary data
        """
        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["duration"] = (datetime.now() - self.start_time).total_seconds()
        self.metadata["success"] = success
        
        if summary:
            self.metadata["summary"] = summary
            
        self._save_metadata()
        logger.info(f"Ended experiment: {self.name} (ID: {self.experiment_id})")
        
    def get_experiment_path(self) -> str:
        """
        Get the path to the experiment directory.
        
        Returns:
            Experiment directory path
        """
        return os.path.join(self.base_dir, self.experiment_id)
        
    def get_parameter_hash(self) -> str:
        """
        Get a hash of the experiment parameters.
        
        Returns:
            Parameter hash string
        """
        param_str = json.dumps(self.metadata["parameters"], sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()
        
    @staticmethod
    def load_experiment(experiment_id: str, base_dir: str = "experiments") -> Dict[str, Any]:
        """
        Load experiment metadata.
        
        Args:
            experiment_id: Experiment ID
            base_dir: Base directory for experiments
            
        Returns:
            Experiment metadata
        """
        path = os.path.join(base_dir, experiment_id, "metadata.json")
        if not os.path.exists(path):
            raise ValueError(f"Experiment not found: {experiment_id}")
            
        with open(path, 'r') as f:
            metadata = json.load(f)
            
        return metadata
        
    @staticmethod
    def list_experiments(base_dir: str = "experiments") -> List[Dict[str, Any]]:
        """
        List all experiments.
        
        Args:
            base_dir: Base directory for experiments
            
        Returns:
            List of experiment metadata
        """
        experiments = []
        
        if not os.path.exists(base_dir):
            return experiments
            
        for experiment_id in os.listdir(base_dir):
            metadata_path = os.path.join(base_dir, experiment_id, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    experiments.append(metadata)
                    
        return experiments 